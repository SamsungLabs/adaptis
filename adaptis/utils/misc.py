import numpy as np
from .log import logger
from functools import partial
from mxnet.gluon.data.dataloader import default_mp_batchify_fn, default_batchify_fn


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
    net.save_parameters(str(checkpoint_path))


def get_unique_labels(mask):
    return np.nonzero(np.bincount(mask.flatten() + 1))[0] - 1


def get_dict_batchify_fn(num_workers):
    base_batchify_fn = default_mp_batchify_fn if num_workers > 0 else default_batchify_fn

    return partial(dict_batchify_fn, base_batchify_fn=base_batchify_fn)


def dict_batchify_fn(data, base_batchify_fn):
    if isinstance(data[0], dict):
        ret = {k: [] for k in data[0].keys()}
        for x in data:
            for k, v in x.items():
                ret[k].append(v)
        return {k: base_batchify_fn(v) for k, v in ret.items()}
    else:
        return base_batchify_fn(data)
