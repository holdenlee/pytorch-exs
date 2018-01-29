import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision.datasets import *
import torchvision.transforms as transforms

import sys
import os
from utils import *

DATA_DIR = "/n/fs/scratch/holdenl/data"


def load_mnist(data_dir=DATA_DIR,train_batch_size=32,test_batch_size=32,download=False,cuda=False):
    kwargs = {'pin_memory': True} if cuda else {}
    #'num_workers': 1, 
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    trainset = MNIST(root=data_dir, train=True,
                     download=download, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                              shuffle=True, num_workers=2,**kwargs)

    testset = MNIST(root=data_dir, train=False,
                    download=download, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                             shuffle=False, num_workers=2,**kwargs)
    return train_loader, test_loader #, trainset, testset
