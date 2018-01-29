import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision.datasets import *
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
from utils import *

from standard_args import *
from pt_utils import *
from mnist_data import *
from mnist_model import *

DATA_DIR = "/n/fs/scratch/holdenl/data"

if __name__=='__main__':
    parser = standard_parser()
    args = parser.parse_args()
    args = args_post_process(args)

    train_loader, test_loader = load_mnist(data_dir=args.data_dir, download=args.download, train_batch_size=args.batch_size, test_batch_size=args.test_batch_size,cuda=args.cuda)
    save_file = args.save_file
    #save_file = os.path.join(args.save_dir,"mnist")
    net = train(MNISTNet(),
                train_loader,
                test_loader,
                epochs=args.epochs,
                loss_fn=F.nll_loss,
                correct_fn=num_label_matches,
                optimizer = lambda param:optim.Adam(param, lr=1e-4),
                #optimizer=lambda param: optim.SGD(param, lr=args.lr, momentum=args.momentum),
                log_freq=args.log_freq,
                save_freq=args.save_freq,
                save_file=save_file,
                cuda=args.cuda,
                v=args.v)
    torch.save(net.state_dict(), save_file)
