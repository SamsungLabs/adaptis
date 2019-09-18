import mxnet as mx
import argparse


def get_common_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='Dataloader threads')

    parser.add_argument('--thread-pool', action='store_true', default=False,
                        help='use ThreadPool for dataloader workers')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--ngpus', type=int,
                        default=len(mx.test_utils.list_gpus()),
                        help='number of GPUs')
    parser.add_argument('--gpus', type=str, default='', required=False)

    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')

    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')

    parser.add_argument('--batch-size', type=int, default=8)

    parser.add_argument('--exp-name', type=str, default='',
                        help='experiment name')

    return parser


def get_train_arguments():
    parser = get_common_arguments()
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Start epoch for learning schedule and for logging')

    parser.add_argument('--weights', type=str, default=None,
                        help='Put the path to resuming file if needed')

    parser.add_argument('--val-batch-size', type=int, default=8)

    parser.add_argument('--no-exp', action='store_true', default=False,
                        help="Don't create exps dir")

    return parser
