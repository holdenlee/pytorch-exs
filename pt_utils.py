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

DATA_DIR = "/n/fs/scratch/holdenl/data"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train(net,trainloader,epochs=2,loss_fn=nn.CrossEntropyLoss(),optimizer=lambda param: optim.SGD(param, lr=0.01),print_every=100,save_every=100,save_dir=DATA_DIR,verbose=1):
    makedir(os.path.dirname(save_dir))
    optimizer = optimizer(net.parameters())
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 1):
            # get the inputs
            inputs, labels = data
            
            # wrap them in Variable
            xs, ys = Variable(inputs), Variable(labels)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(xs)
            loss = loss_fn(outputs, ys)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.data[0]
            if print_every != 0 and i % print_every == 0:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i, running_loss / print_every))
                sys.stdout.flush()
                running_loss = 0.0
            if save_every !=0 and i % save_every == 0:
                torch.save(net.state_dict(),save_dir)
    return net

def num_label_matches(outputs, labels):
    _, predicted = torch.max(outputs,1)
    return (predicted == labels).sum()

def test(net,testloader,loss_fn=None,correct_fn=num_label_matches):
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        total += labels.size(0)
        correct += correct_fn(outputs.data, labels)
    return correct/total

def load_mnist(data_dir=DATA_DIR,batch_size=32):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = MNIST(root=data_dir, train=True,
                     download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = MNIST(root=data_dir, train=False,
                    download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader #, trainset, testset
