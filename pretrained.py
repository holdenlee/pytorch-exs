import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.datasets import *
import torchvision.transforms as transforms
import torchvision.models as models

from pt_utils import *

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)

#All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.

imagenet_data = torchvision.datasets.ImageFolder(DATA_DIR)
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=2)
data_dir=DATA_DIR
batch_size=32
testset = MNIST(root=data_dir, train=False,
                download=True, transform=normalize)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
