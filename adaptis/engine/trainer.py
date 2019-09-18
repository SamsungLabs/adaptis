import os
import cv2
import mxnet as mx
import numpy as np
from mxnet import gluon, autograd
from tqdm import tqdm
import logging

from copy import deepcopy
from gluoncv.utils.viz.segmentation import DeNormalize
from collections import defaultdict

from adaptis.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from adaptis.utils.vis import draw_probmap, draw_points, visualize_mask
from adaptis.utils.misc import save_checkpoint, get_dict_batchify_fn


class AdaptISTrainer(object):
    def __init__(self, args, model, model_cfg, loss_cfg,
                 trainset, valset, optimizer_params,
                 optimizer='adam',
                 image_dump_interval=200,
                 checkpoint_interval=10,
                 tb_dump_period=25,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 train_proposals=False,
                 hybridize_model=True):
        self.args = args
        self.model_cfg = model_cfg
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.hybridize_model = hybridize_model
        self.checkpoint_interval = checkpoint_interval
        self.train_proposals = train_proposals
        self.task_prefix = ''

        self.trainset = trainset
        self.valset = valset

        self.train_data = gluon.data.DataLoader(
            trainset, args.batch_size, shuffle=True,
            last_batch='rollover',
            batchify_fn=get_dict_batchify_fn(args.workers),
            thread_pool=args.thread_pool,
            num_workers=args.workers)

        self.val_data = gluon.data.DataLoader(
            valset, args.val_batch_size,
            batchify_fn=get_dict_batchify_fn(args.workers),
            last_batch='rollover',
            thread_pool=args.thread_pool,
            num_workers=args.workers)

        logger.info(model)
        model.cast(args.dtype)
        model.collect_params().reset_ctx(ctx=args.ctx)

        self.net = model
        self.evaluator = None
        if args.weights is not None:
            if os.path.isfile(args.weights):
                model.load_parameters(args.weights, ctx=args.ctx, allow_missing=True)
                args.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{args.weights}'")

        self.lr_scheduler = None
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(niters=len(self.train_data))
            optimizer_params['lr_scheduler'] = self.lr_scheduler

        kv = mx.kv.create(args.kvstore)
        if not train_proposals:
            train_params = self.net.collect_params()
        else:
            train_params = self.net.proposals_head.collect_params()
            self.task_prefix = 'proposals'

        self.trainer = gluon.Trainer(train_params,
                                     optimizer, optimizer_params,
                                     kvstore=kv, update_on_kvstore=len(args.ctx) > 1)

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        if args.input_normalization:
            self.denormalizator = DeNormalize(args.input_normalization['mean'],
                                              args.input_normalization['std'])
        else:
            self.denormalizator = lambda x: x

        self.sw = None
        self.image_dump_interval = image_dump_interval

    def training(self, epoch):
        if self.sw is None:
            self.sw = SummaryWriterAvg(logdir=str(self.args.logs_path),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100)
        train_loss = 0.0
        hybridize = False

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        for i, batch_data in enumerate(tbar):
            if self.lr_scheduler is not None:
                self.lr_scheduler.update(i, epoch)
            global_step = epoch * len(self.train_data) + i

            losses, losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data)

            autograd.backward(losses)
            self.trainer.step(1, ignore_stale_grad=True)

            batch_loss = sum(loss.asnumpy().mean() for loss in losses) / len(losses)
            train_loss += batch_loss

            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                                   global_step=global_step)
            self.sw.add_scalar(tag=f'{log_prefix}Losses/overall', value=batch_loss, global_step=global_step)

            for k, v in self.loss_cfg.items():
                if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                    v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)

            if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                self.save_visualization(splitted_batch_data, outputs, global_step, prefix='images/train')

            self.sw.add_scalar(tag=f'{log_prefix}States/learning_rate', value=self.trainer.learning_rate,
                               global_step=global_step)

            tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.6f}')
            for metric in self.train_metrics:
                metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)
            mx.nd.waitall()

            if self.hybridize_model and not hybridize:
                self.net.hybridize()
                hybridize = True

        for metric in self.train_metrics:
            self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                               global_step=epoch, disable_avg=True)

        save_checkpoint(self.net, self.args.checkpoints_path, prefix=self.task_prefix, epoch=None)
        if epoch % self.checkpoint_interval == 0:
            save_checkpoint(self.net, self.args.checkpoints_path, prefix=self.task_prefix, epoch=epoch)

    def validation(self, epoch):
        if self.sw is None:
            self.sw = SummaryWriterAvg(logdir=str(self.args.logs_path),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100)

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        num_batches = 0
        val_loss = 0
        losses_logging = defaultdict(list)
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.val_data) + i
            losses, batch_losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data, validation=True)

            for loss_name, loss_values in batch_losses_logging.items():
                losses_logging[loss_name].extend(loss_values)

            batch_loss = sum(loss.asnumpy()[0] for loss in losses) / len(losses)
            val_loss += batch_loss
            num_batches += 1

            tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss/num_batches:.6f}')
            for metric in self.val_metrics:
                metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        for loss_name, loss_values in losses_logging.items():
            self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                               global_step=epoch, disable_avg=True)

        for metric in self.val_metrics:
            self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                               global_step=epoch, disable_avg=True)
        self.sw.add_scalar(tag=f'{log_prefix}Losses/overall', value=val_loss / num_batches,
                           global_step=epoch, disable_avg=True)

    def batch_forward(self, batch_data, validation=False):
        splitted_batch = {k: gluon.utils.split_and_load(v, ctx_list=self.args.ctx, even_split=False)
                               for k, v in batch_data.items()}
        if 'instances' in splitted_batch:
            splitted_batch['instances'] = [masks.reshape(shape=(-3, -2))
                                           for masks in splitted_batch['instances']]

        metrics = self.val_metrics if validation else self.train_metrics

        losses_logging = defaultdict(list)
        with autograd.record(True) if not validation else autograd.pause(False):
            outputs = [self.net(image, points)
                       for image, points in zip(splitted_batch['images'], splitted_batch['points'])]

            losses = []
            for ictx, ctx_output in enumerate(outputs):
                loss = 0.0
                loss = self.add_loss('instance_loss', loss, losses_logging, validation,
                                     lambda: (ctx_output.instances, splitted_batch['instances'][ictx]))
                loss = self.add_loss('segmentation_loss', loss, losses_logging, validation,
                                     lambda: (ctx_output.semantic, splitted_batch['semantic'][ictx]))
                loss = self.add_loss('proposals_loss', loss, losses_logging, validation,
                                     lambda: (ctx_output.instances, ctx_output.proposals, splitted_batch['instances'][ictx]))

                with autograd.pause():
                    for m in metrics:
                        m.update(*(getattr(ctx_output, x) for x in m.pred_outputs),
                                 *(splitted_batch[x][ictx] for x in m.gt_outputs))

                losses.append(loss)

        return losses, losses_logging, splitted_batch, outputs

    def add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = mx.nd.mean(loss)
            losses_logging[loss_name].append(loss.asnumpy())
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix='images'):
        outputs = outputs[0]

        output_images_path = self.args.logs_path / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        images = splitted_batch_data['images']
        points = splitted_batch_data['points']
        instance_masks = splitted_batch_data['instances']

        image_blob, points = images[0][0], points[0][0]
        image = self.denormalizator(image_blob.as_in_context(mx.cpu(0))).asnumpy() * 255
        image = image.transpose((1, 2, 0))

        gt_instance_masks = instance_masks[0].asnumpy()
        predicted_instance_masks = mx.nd.sigmoid(outputs.instances).asnumpy()

        if 'semantic' in splitted_batch_data:
            segmentation_labels = splitted_batch_data['semantic']
            gt_segmentation = segmentation_labels[0][0].asnumpy() + 1
            predicted_label = mx.nd.squeeze(mx.nd.argmax(outputs.semantic[0], 0)).asnumpy() + 1

            if len(gt_segmentation.shape) == 3:
                area_weights = gt_segmentation[1] ** self.loss_cfg.segmentation_loss._area_gamma
                area_weights -= area_weights.min()
                area_weights /= area_weights.max()
                area_weights = draw_probmap(area_weights)

                _save_image('area_weights', area_weights[:, :, ::-1])
                gt_segmentation = gt_segmentation[0]

            gt_mask = visualize_mask(gt_segmentation.astype(np.int32), self.trainset.num_classes + 1)
            predicted_mask = visualize_mask(predicted_label.astype(np.int32), self.trainset.num_classes + 1)
            result = np.hstack((image, gt_mask, predicted_mask)).astype(np.uint8)

            _save_image('semantic_segmentation', result[:, :, ::-1])

        points = points.asnumpy()
        gt_masks = np.squeeze(gt_instance_masks[:points.shape[0]])
        predicted_masks = np.squeeze(predicted_instance_masks[:points.shape[0]])

        viz_image = []
        for gt_mask, point, predicted_mask in zip(gt_masks, points, predicted_masks):
            timage = draw_points(image, [point], (0, 255, 0))
            gt_mask[gt_mask < 0] = 0.25
            gt_mask = draw_probmap(gt_mask)
            predicted_mask = draw_probmap(predicted_mask)
            viz_image.append(np.hstack((timage, gt_mask, predicted_mask)))
        viz_image = np.vstack(viz_image)

        result = viz_image.astype(np.uint8)
        _save_image('instance_segmentation', result[:, :, ::-1])


def init_proposals_head(model, ctx):
    model.hybridize(False)
    model.collect_params().setattr('grad_req', 'null')
    model.add_proposals_head(ctx)
