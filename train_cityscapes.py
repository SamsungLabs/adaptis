import mxnet as mx
import random
from mxnet.gluon.data.vision import transforms
from functools import partial
from gluoncv.utils import LRScheduler
from easydict import EasyDict as edict
from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, PadIfNeeded, RandomCrop,
    RGBShift, RandomBrightness, RandomContrast
)

from adaptis.engine.trainer import AdaptISTrainer, init_proposals_head
from adaptis.model.cityscapes.models import get_cityscapes_model
from adaptis.model.losses import NormalizedFocalLossSigmoid, NormalizedFocalLossSoftmax, AdaptISProposalsLossIoU
from adaptis.model.metrics import AdaptiveIoU
from adaptis.data.cityscapes import CityscapesDataset
from adaptis.utils.exp import init_experiment
from adaptis.utils.log import logger


def add_exp_args(parser):
    parser.add_argument('--dataset-path', type=str, help='Path to the dataset')
    return parser


def init_model():
    model_cfg = edict()
    model_cfg.syncbn = True
    model_cfg.crop_size = (400, 720)

    model_cfg.input_normalization = {
        'mean': [.485, .456, .406],
        'std': [.229, .224, .225]
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

    model = get_cityscapes_model(num_classes=19, norm_layer=norm_layer,
                                 backbone='resnet50')
    model.initialize(mx.init.Xavier(rnd_type='gaussian', magnitude=2), ctx=mx.cpu(0))
    model.feature_extractor.load_pretrained_weights()

    return model, model_cfg


def train(model, model_cfg, args, train_proposals, start_epoch=0):
    args.val_batch_size = args.batch_size
    args.input_normalization = model_cfg.input_normalization
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.25, gamma=2)
    loss_cfg.instance_loss_weight = 1.0 if not train_proposals else 0.0

    if not train_proposals:
        num_epochs = 250
        num_points = 6

        loss_cfg.segmentation_loss = NormalizedFocalLossSoftmax(ignore_label=-1, gamma=1)
        loss_cfg.segmentation_loss_weight = 0.75
    else:
        num_epochs = 8
        num_points = 48

        loss_cfg.proposals_loss = AdaptISProposalsLossIoU(args.batch_size)
        loss_cfg.proposals_loss_weight = 1.0

    train_augmentator = Compose([
        HorizontalFlip(),
        ShiftScaleRotate(shift_limit=0.03, scale_limit=0,
                         rotate_limit=(-3, 3), border_mode=0, p=0.75),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightness(limit=(-0.25, 0.25), p=0.75),
        RandomContrast(limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    def scale_func(image_shape):
        return random.uniform(0.85, 1.15)

    trainset = CityscapesDataset(
        args.dataset_path,
        split='train',
        num_points=num_points,
        augmentator=train_augmentator,
        with_segmentation=True,
        points_from_one_object=train_proposals,
        input_transform=model_cfg.input_transform,
        min_object_area=80,
        sample_ignore_object_prob=0.025,
        keep_background_prob=0.05,
        image_rescale=scale_func,
        use_jpeg=False
    )

    valset = CityscapesDataset(
        args.dataset_path,
        split='test',
        augmentator=val_augmentator,
        num_points=num_points,
        with_segmentation=True,
        points_from_one_object=train_proposals,
        input_transform=model_cfg.input_transform,
        min_object_area=80,
        image_rescale=scale_func,
        use_jpeg=False
    )

    if not train_proposals:
        optimizer_params = {
            'learning_rate': 0.01,
            'momentum': 0.9, 'wd': 1e-4
        }
        lr_scheduler = partial(LRScheduler, mode='poly', baselr=optimizer_params['learning_rate'],
                               nepochs=num_epochs)
    else:
        optimizer_params = {
            'learning_rate': 5e-4,
            'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8
        }
        lr_scheduler = partial(LRScheduler, mode='cosine',
                               baselr=optimizer_params['learning_rate'],
                               nepochs=num_epochs)

    trainer = AdaptISTrainer(args, model, model_cfg, loss_cfg,
                             trainset, valset,
                             optimizer='sgd' if not train_proposals else 'adam',
                             optimizer_params=optimizer_params,
                             lr_scheduler=lr_scheduler,
                             checkpoint_interval=40 if not train_proposals else 2,
                             image_dump_interval=100 if not train_proposals else -1,
                             train_proposals=train_proposals,
                             hybridize_model=not train_proposals,
                             metrics=[AdaptiveIoU()])

    logger.info(f'Starting Epoch: {start_epoch}')
    logger.info(f'Total Epochs: {num_epochs}')
    for epoch in range(start_epoch, num_epochs):
        trainer.training(epoch)
        trainer.validation(epoch)


if __name__ == '__main__':
    args = init_experiment('cityscapes', add_exp_args, script_path=__file__)

    model, model_cfg = init_model()
    train(model, model_cfg, args, train_proposals=False,
          start_epoch=args.start_epoch)
    init_proposals_head(model, args.ctx)
    train(model, model_cfg, args, train_proposals=True)
