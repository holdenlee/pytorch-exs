import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
#from torchvision.datasets import *
import torchvision.transforms as transforms

import sys
import os
import time
#import argparse
from utils import *

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

def train_epoch(net,train_loader,optimizer,epoch=1,loss_fn=F.nll_loss,log_freq=100,save_freq=100,save_file=DATA_DIR,cuda=True,v=1):
    net.train()
    running_loss = 0.0
    start = time.time()
    startt = time.time()
    for i, data in enumerate(train_loader, 1):
        # get the inputs
        inputs, labels = data
        if cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
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
        if log_freq != 0 and i % log_freq == 0:    # print every 100 mini-batches
            endt = time.time()
            #print('Time: %f s' % (endt-startt))
            printv('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {} s'.format(
                epoch, i * len(inputs), len(train_loader.dataset),
                100. * i / len(train_loader), loss.data[0],
                endt-startt),v,1)
            #print('[%d, %5d] loss: %.3f' %
            #      (epoch + 1, i, running_loss / log_freq))
            sys.stdout.flush()
            running_loss = 0.0
            startt = time.time()
        if save_freq !=0 and i % save_freq == 0:
            torch.save(net.state_dict(),save_file)
    end=time.time()
    printv('Epoch %d time: %f s' % (epoch,end-start),v,0)
    return net


def num_label_matches(outputs, labels):
    _, pred = torch.max(outputs,1)
    return pred.eq(labels.data.view_as(pred)).sum()
    #return (predicted == labels).sum()

def train(net,train_loader,test_loader,epochs=2,loss_fn=F.nll_loss,correct_fn=num_label_matches,optimizer=lambda param: optim.SGD(param, lr=0.01),log_freq=100,save_freq=100,save_file=DATA_DIR,cuda=True,v=1):
    if cuda:
        net = net.cuda()
    makedir(os.path.dirname(save_file))
    optimizer = optimizer(net.parameters())
    start = time.time()
    for epoch in range(1,epochs+1):  # loop over the dataset multiple times
        net = train_epoch(net,train_loader,optimizer,epoch=epoch,loss_fn=loss_fn,log_freq=log_freq,save_freq=save_freq,save_file=save_file,cuda=cuda,v=v)
        test(net,test_loader,loss_fn=loss_fn,correct_fn=correct_fn,cuda=cuda,v=v)
    end = time.time()
    printv('Total time: %f s' % (end-start),v,0)
    return net


def test(net,test_loader,loss_fn=F.nll_loss,correct_fn=num_label_matches,cuda=True,v=1):
    #print("TEST CUDA:", cuda)
    net.eval()
    correct = 0
    #total = 0
    test_loss = 0
    for data in test_loader:
        images, labels = data
        if cuda:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images, volatile=True)
        labels = Variable(labels)
        outputs = net(images)
        test_loss += loss_fn(outputs, labels, size_average=False).data[0]
        #total += labels.size(0)
        correct += correct_fn(outputs.data, labels)
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    printv('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)),v,0)
    return accuracy


