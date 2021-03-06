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
import utils

from standard_args import *

DATA_DIR = "/n/fs/scratch/holdenl/data"

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
        test(net,test_loader,loss_fn=loss_fn,correct_fn=correct_fn,v=v)
    end = time.time()
    printv('Total time: %f s' % (end-start),v,0)
    return net


def test(net,test_loader,loss_fn=F.nll_loss,correct_fn=num_label_matches,v=1):
    net.eval()
    correct = 0
    #total = 0
    test_loss = 0
    for data in test_loader:
        images, labels = data
        if args.cuda:
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

"""
torch.save(the_model.state_dict(), PATH)

the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))
"""
