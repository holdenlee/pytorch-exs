from __future__ import print_function
import argparse
import torch

def standard_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch neural net training')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-freq', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-freq', type=int, default=0, metavar='N',
                        help='how many batches to wait before saving')
    parser.add_argument('--data-dir', type=str, default="/n/fs/scratch/holdenl/data", metavar='DD',
                        help='data location')
    parser.add_argument('--save-file', type=str, default="/n/fs/scratch/holdenl/data/models/model", metavar='SD',
                        help='save location')
    parser.add_argument('--download', type=bool, default=True, metavar='DL',
                        help='whether to download dataset (set False for offline mode)')
    parser.add_argument('-v', type=int, default=1, metavar='V',
                        help='verbosity level')
    return parser

def args_post_process(args):
    #args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print("CUDA available:", args.cuda)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    return args
