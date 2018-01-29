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

from pt_utils import *


class MNISTNet(nn.Module):

    def __init__(self):
        super(MNISTNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # 28*28
        self.conv1 = nn.Conv2d(1, 32, 5,padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5,padding=2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        #print(x.size())
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        #print(x.size())
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #print(x.size())
        x = x.view(-1, num_flat_features(x))
        #print(x.size())
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

