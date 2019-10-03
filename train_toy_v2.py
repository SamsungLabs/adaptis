import mxnet as mx
from mxnet.gluon.data.vision import transforms
from functools import partial
from gluoncv.utils import LRScheduler
from easydict import EasyDict as edict
from albumentations import Compose, Flip

from adaptis.engine.trainer import AdaptISTrainer, init_proposals_head
from adaptis.model.toy.models import get_unet_model
from adaptis.model.losses import NormalizedFocalLossSigmoid, NormalizedFocalLossSoftmax, AdaptISProposalsLossIoU
from adaptis.model.metrics import AdaptiveIoU
from adaptis.data.toy import ToyDataset
from adaptis.utils.exp import init_experiment
from adaptis.utils.log import logger


def add_exp_args(parser):
    parser.add_argument('--dataset-path', type=str, help='Path to the dataset')
    return parser


def init_model():
    model_cfg = edict()
    model_cfg.syncbn = True

    model_cfg.input_normalization = {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]
    }

    model_cfg.input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(model_cfg.input_normalization['mean'],
                             model_cfg.input_normalization['std']),
    ])

    if args.ngpus > 1 and model_cfg.syncbn:
        norm_layer = partial(mx.gluon.contrib.nn.SyncBatchNorm, num_devices=args.ngpus)
    else:
        norm_layer = mx.gluon.nn.BatchNorm

    model = get_unet_model(norm_layer)
    model.initialize(mx.init.Xavier(rnd_type='gaussian', magnitude=1), ctx=mx.cpu(0))

    return model, model_cfg


def train(model, model_cfg, args, train_proposals, start_epoch=0):
    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.50, gamma=2)
    loss_cfg.instance_loss_weight = 1.0 if not train_proposals else 0.0

    if not train_proposals:
        num_epochs = 160
        num_points = 12

        loss_cfg.segmentation_loss = NormalizedFocalLossSoftmax(ignore_label=-1, gamma=1)
        loss_cfg.segmentation_loss_weight = 0.75
    else:
        num_epochs = 10
        num_points = 32

        loss_cfg.proposals_loss = AdaptISProposalsLossIoU(args.batch_size)
        loss_cfg.proposals_loss_weight = 1.0

    args.val_batch_size = args.batch_size
    args.input_normalization = model_cfg.input_normalization

    train_augmentator = Compose([
        Flip()
    ], p=1.0)

    trainset = ToyDataset(
        args.dataset_path,
        split='train',
        num_points=num_points,
        augmentator=train_augmentator,
        with_segmentation=True,
        points_from_one_object=train_proposals,
        input_transform=model_cfg.input_transform,
        epoch_len=10000
    )

    valset = ToyDataset(
        args.dataset_path,
        split='test',
        augmentator=None,
        num_points=num_points,
        with_segmentation=True,
        points_from_one_object=train_proposals,
        input_transform=model_cfg.input_transform
    )

    optimizer_params = {
        'learning_rate': 5e-4,
        'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8
    }

    if not train_proposals:
        lr_scheduler = partial(LRScheduler, mode='cosine',
                               baselr=optimizer_params['learning_rate'],
                               nepochs=num_epochs)
    else:
        lr_scheduler = partial(LRScheduler, mode='cosine',
                               baselr=optimizer_params['learning_rate'],
                               nepochs=num_epochs)

    trainer = AdaptISTrainer(args, model, model_cfg, loss_cfg,
                             trainset, valset,
                             optimizer='adam',
                             optimizer_params=optimizer_params,
                             lr_scheduler=lr_scheduler,
                             checkpoint_interval=40 if not train_proposals else 5,
                             image_dump_interval=200 if not train_proposals else -1,
                             train_proposals=train_proposals,
                             hybridize_model=not train_proposals,
                             metrics=[AdaptiveIoU()])

    logger.info(f'Starting Epoch: {start_epoch}')
    logger.info(f'Total Epochs: {num_epochs}')
    for epoch in range(start_epoch, num_epochs):
        trainer.training(epoch)
        trainer.validation(epoch)


if __name__ == '__main__':
    args = init_experiment('toy_v2', add_exp_args, script_path=__file__)

    model, model_cfg = init_model()
    train(model, model_cfg, args, train_proposals=False,
          start_epoch=args.start_epoch)
    init_proposals_head(model, args.ctx)
    train(model, model_cfg, args, train_proposals=True)
