import os
from copy import deepcopy
from collections import defaultdict

import cv2
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from adaptis.utils import log, vis, misc


class AdaptISTrainer(object):
    def __init__(self, args, model, model_cfg, loss_cfg,
                 trainset, valset,
                 optimizer_params,
                 image_dump_interval=200,
                 checkpoint_interval=10,
                 tb_dump_period=25,
                 num_epochs=1,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 train_proposals=False):
        self.args = args
        self.model_cfg = model_cfg
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period

        self.train_metrics = metrics if metrics is not None else []
        self.val_metrics = deepcopy(self.train_metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.train_proposals = train_proposals
        self.task_prefix = ''
        self.summary_writer = None

        self.trainset = trainset
        self.valset = valset
        self.train_loader = DataLoader(trainset, batch_size=args.batch_size, pin_memory=True,
                                       shuffle=True, num_workers=args.workers, drop_last=True)
        self.val_loader = DataLoader(valset, batch_size=args.val_batch_size, pin_memory=True,
                                     shuffle=False, num_workers=args.workers, drop_last=True)

        self.device = args.device
        log.logger.info(model)
        self.net = model.to(self.device)
        self.evaluator = None
        self._load_weights()

        if train_proposals:
            self.task_prefix = 'proposals'
        self.optim = torch.optim.Adam(self.net.get_trainable_params(), **optimizer_params)
        self.tqdm_out = log.TqdmToLogger(log.logger, level=log.logging.INFO)

        self.lr_scheduler = None
        self.lr = optimizer_params['lr']
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim, T_max=num_epochs * len(self.train_loader))
            if args.start_epoch > 0:
                for _ in range(args.start_epoch):
                    self.lr_scheduler.step()

        if args.input_normalization:
            mean = torch.tensor(args.input_normalization['mean'], dtype=torch.float32)
            std = torch.tensor(args.input_normalization['std'], dtype=torch.float32)

            self.denormalizator = Normalize((-mean / std), (1.0 / std))
        else:
            self.denormalizator = lambda x: x

    def _load_weights(self):
        if self.args.weights is not None:
            if os.path.isfile(self.args.weights):
                self.net.load_weights(self.args.weights)
                self.args.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.args.weights}'")

    def training(self, epoch):
        if self.summary_writer is None:
            self.summary_writer = log.SummaryWriterAvg(log_dir=str(self.args.logs_path),
                                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_loader, file=self.tqdm_out, ncols=100)
        train_loss = 0.0

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_loader) + i

            loss, losses_logging, batch_data, outputs = self.batch_forward(batch_data)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            loss = loss.detach().cpu().numpy().mean()
            train_loss += loss

            for loss_name, loss_values in losses_logging.items():
                self.summary_writer.add_scalar(
                    tag=f'{log_prefix}Losses/{loss_name}',
                    value=np.array(loss_values).mean(),
                    global_step=global_step
                )
            self.summary_writer.add_scalar(
                tag=f'{log_prefix}Losses/overall',
                value=loss,
                global_step=global_step
            )

            for k, v in self.loss_cfg.items():
                if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                    v.log_states(
                        self.summary_writer,
                        f'{log_prefix}Losses/{k}',
                        global_step
                    )

            if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                self.save_visualization(batch_data, outputs, global_step, prefix='train')

            self.summary_writer.add_scalar(
                tag=f'{log_prefix}States/learning_rate',
                value=self.lr if self.lr_scheduler is None else self.lr_scheduler.get_lr(),
                global_step=global_step
            )

            tbar.set_description(f'Epoch {epoch}, training loss {train_loss / (i + 1):.6f}')
            for metric in self.train_metrics:
                metric.log_states(
                    self.summary_writer,
                    f'{log_prefix}Metrics/{metric.name}',
                    global_step
                )

        for metric in self.train_metrics:
            self.summary_writer.add_scalar(
                tag=f'{log_prefix}Metrics/{metric.name}',
                value=metric.get_epoch_value(),
                global_step=epoch, disable_avg=True)

        misc.save_checkpoint(self.net, self.args.checkpoints_path, prefix=self.task_prefix, epoch=None)
        if epoch % self.checkpoint_interval == 0:
            misc.save_checkpoint(self.net, self.args.checkpoints_path, prefix=self.task_prefix, epoch=epoch)

    def validation(self, epoch):
        if self.summary_writer is None:
            self.summary_writer = log.SummaryWriterAvg(log_dir=str(self.args.logs_path),
                                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_loader, file=self.tqdm_out, ncols=100)

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        num_batches = 0
        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.train(False)
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.val_loader) + i
            loss, batch_losses_logging, batch_data, outputs = self.batch_forward(batch_data, validation=True)

            for loss_name, loss_values in batch_losses_logging.items():
                losses_logging[loss_name].extend(loss_values)

            loss = loss.item()
            val_loss += loss
            num_batches += 1

            tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss / num_batches:.6f}')
            for metric in self.val_metrics:
                metric.log_states(
                    self.summary_writer,
                    f'{log_prefix}Metrics/{metric.name}',
                    global_step
                )

        for loss_name, loss_values in losses_logging.items():
            self.summary_writer.add_scalar(
                tag=f'{log_prefix}Losses/{loss_name}',
                value=np.array(loss_values).mean(),
                global_step=epoch, disable_avg=True
            )

        for metric in self.val_metrics:
            self.summary_writer.add_scalar(
                tag=f'{log_prefix}Metrics/{metric.name}',
                value=metric.get_epoch_value(),
                global_step=epoch, disable_avg=True
            )
        self.summary_writer.add_scalar(
            tag=f'{log_prefix}Losses/overall',
            value=val_loss / num_batches,
            global_step=epoch, disable_avg=True
        )
        self.net.train(True)

    def save_visualization(self, batch_data, outputs, global_step, prefix):
        output_images_path = self.args.logs_path / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        images = batch_data['images']
        points = batch_data['points']
        instance_masks = batch_data['instances']

        image_blob, points = images[0], points[0]
        image = self.denormalizator(image_blob).cpu().numpy() * 255
        image = image.transpose((1, 2, 0))

        gt_instance_masks = instance_masks.cpu().numpy()
        predicted_instance_masks = torch.sigmoid(outputs.instances.detach()).cpu().numpy()

        if 'semantic' in batch_data:
            segmentation_labels = batch_data['semantic']
            gt_segmentation = segmentation_labels[0].cpu().numpy() + 1
            predicted_label = torch.argmax(outputs.semantic[0].detach(), dim=0).cpu().numpy() + 1

            if len(gt_segmentation.shape) == 3:
                area_weights = gt_segmentation[1] ** self.loss_cfg.segmentation_loss._area_gamma
                area_weights -= area_weights.min()
                area_weights /= area_weights.max()
                area_weights = vis.draw_probmap(area_weights)

                _save_image('area_weights', area_weights[:, :, ::-1])
                gt_segmentation = gt_segmentation[0]

            gt_mask = vis.visualize_mask(gt_segmentation.astype(np.int32), self.trainset.num_classes + 1)
            predicted_mask = vis.visualize_mask(predicted_label.astype(np.int32), self.trainset.num_classes + 1)
            result = np.hstack((image, gt_mask, predicted_mask)).astype(np.uint8)

            _save_image('semantic_segmentation', result[:, :, ::-1])

        points = points.cpu().numpy()
        gt_masks = np.squeeze(gt_instance_masks[:points.shape[0]])
        predicted_masks = np.squeeze(predicted_instance_masks[:points.shape[0]])

        viz_image = []
        for gt_mask, point, predicted_mask in zip(gt_masks, points, predicted_masks):
            timage = vis.draw_points(image, [point], (0, 255, 0))
            gt_mask[gt_mask < 0] = 0.25
            gt_mask = vis.draw_probmap(gt_mask)
            predicted_mask = vis.draw_probmap(predicted_mask)
            viz_image.append(np.hstack((timage, gt_mask, predicted_mask)))
        viz_image = np.vstack(viz_image)

        result = viz_image.astype(np.uint8)
        _save_image('instance_segmentation', result[:, :, ::-1])

    def batch_forward(self, batch_data, validation=False):
        if 'instances' in batch_data:
            batch_size, num_points, c, h, w = batch_data['instances'].size()
            batch_data['instances'] = batch_data['instances'].view(batch_size * num_points, c, h, w)

        metrics = self.val_metrics if validation else self.train_metrics

        losses_logging = defaultdict(list)
        with torch.set_grad_enabled(not validation):
            image, points = batch_data['images'], batch_data['points']
            output = self.net(image.to(self.device), points.to(self.device))

            loss = 0.0
            loss = self._add_loss('instance_loss', loss, losses_logging, validation,
                                  lambda: (output.instances, batch_data['instances'].to(self.device)))
            loss = self._add_loss('segmentation_loss', loss, losses_logging, validation,
                                  lambda: (output.semantic, batch_data['semantic'].to(self.device)))
            loss = self._add_loss('proposals_loss', loss, losses_logging, validation,
                                  lambda: (output.instances, output.proposals, batch_data['instances'].to(self.device)))

            with torch.no_grad():
                for m in metrics:
                    m.update(*(getattr(output, x) for x in m.pred_outputs),
                             *(batch_data[x].to(self.device) for x in m.gt_outputs))

        return loss, losses_logging, batch_data, output

    def _add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)
            losses_logging[loss_name].append(loss.detach().cpu().numpy())
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss
