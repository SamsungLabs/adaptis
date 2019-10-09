import torch
import numpy as np

from adaptis.utils.log import logger


def save_checkpoint(net, checkpoints_path, epoch=None, prefix='', verbose=True):
    if epoch is None:
        checkpoint_name = 'last_checkpoint.params'
    else:
        checkpoint_name = f'{epoch:03d}.params'

    if prefix:
        checkpoint_name = f'{prefix}_{checkpoint_name}'

    if not checkpoints_path.exists():
        checkpoints_path.mkdir(parents=True)

    checkpoint_path = checkpoints_path / checkpoint_name
    if verbose:
        logger.info(f'Save checkpoint to {str(checkpoint_path)}')

    torch.save(net.state_dict(), str(checkpoint_path))


def get_unique_labels(mask):
    return np.nonzero(np.bincount(mask.flatten() + 1))[0] - 1


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims
