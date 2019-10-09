from functools import partial

import torch
from torchvision import transforms
from easydict import EasyDict as edict
from albumentations import Compose, Flip

from adaptis.engine.trainer import AdaptISTrainer
from adaptis.model.toy.models import get_unet_model
from adaptis.model.losses import NormalizedFocalLossSigmoid, NormalizedFocalLossSoftmax, AdaptISProposalsLossIoU
from adaptis.model.metrics import AdaptiveIoU
from adaptis.data.toy import ToyDataset
from adaptis.utils import log
from adaptis.model import initializer
from adaptis.utils.exp import init_experiment


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

    # training using DataParallel is not implemented
    norm_layer = torch.nn.BatchNorm2d

    model = get_unet_model(norm_layer=norm_layer)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=1.0))

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
        'lr': 5e-4, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    if not train_proposals:
        lr_scheduler = partial(torch.optim.lr_scheduler.CosineAnnealingLR,
                               last_epoch=-1)
    else:
        lr_scheduler = partial(torch.optim.lr_scheduler.CosineAnnealingLR,
                               last_epoch=-1)

    trainer = AdaptISTrainer(args, model, model_cfg, loss_cfg,
                             trainset, valset,
                             num_epochs=num_epochs,
                             optimizer_params=optimizer_params,
                             lr_scheduler=lr_scheduler,
                             checkpoint_interval=40 if not train_proposals else 5,
                             image_dump_interval=200 if not train_proposals else -1,
                             train_proposals=train_proposals,
                             metrics=[AdaptiveIoU()])

    log.logger.info(f'Starting Epoch: {start_epoch}')
    log.logger.info(f'Total Epochs: {num_epochs}')
    for epoch in range(start_epoch, num_epochs):
        trainer.training(epoch)
        trainer.validation(epoch)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    args = init_experiment('toy_v2', add_exp_args, script_path=__file__)

    model, model_cfg = init_model()
    train(model, model_cfg, args, train_proposals=False,
          start_epoch=args.start_epoch)
    model.add_proposals_head()
    train(model, model_cfg, args, train_proposals=True)
