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

if __name__=='__main__':
    parser = standard_parser()
    args = parser.parse_args()
    args = args_post_process(args)
    _, test_loader = load_mnist(data_dir=args.data_dir, download=args.download, train_batch_size=args.batch_size, test_batch_size=args.test_batch_size,cuda=args.cuda)

    net = MNISTNet()
    net.load_state_dict(torch.load(args.save_file))
    net.cuda()
    test(net,test_loader,loss_fn=F.nll_loss,correct_fn=num_label_matches,cuda=args.cuda, v=args.v)
