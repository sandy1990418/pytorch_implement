import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image


## using mnist
train_dataset = datasets.MNIST(root='data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data',
                              train=False,
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=32,
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break



### model construct ###

class Bottleneck(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1):
        super(Bottleneck,self).__init__()

        self.mid_layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size = 3,stride=stride,padding=1,bias =False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace =True),
            nn.Conv2d(out_channel,out_channel,stride=1,kernel_size = 3,padding=1,bias =False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut =nn.Sequential()
        if stride !=1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size = 1,stride=stride,bias =False),
                nn.BatchNorm2d(out_channel))
            

        def forward(self, x):
            out = self.mid_layer(x)
            out = out+self.shortcut(x)
            out = F.relu(out) 
            return out
        


class ResNet18(nn.Module):
    def __init__(self,bottleneck,num_classes):
        super().__init__()
        self.in_channel =1
        # self.bottleneck = bottleneck()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2,padding = 3)
        self.batch = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.max_pooling1 = nn.MaxPool2d(3, stride=2)
        self.layer1 = self.make_layer(bottleneck,64,2,1)
        self.layer2 = self.make_layer(bottleneck,128,2,2)
        self.layer3 = self.make_layer(bottleneck,256,2,2)
        self.layer4 = self.make_layer(bottleneck,512,2,2)
        self.fc = nn.Linear(512,num_classes)

    
    ## combine layers 
    def make_layer(self,bottleneck,use_channel,number_iteration,stride):
        ## create list (append and create list)
        strides = [stride]+[1]*(number_iteration-1)
        ## create empty list
        layers =[]
        for stride in strides:
            layers.append(bottleneck(self.in_channel,use_channel,stride))
            self.in_channel = use_channel
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.batch(self.conv1(x)))
        x = self.max_pooling1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x,4)      
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x 




device ='gpu' if torch.cuda.is_available()==True  else 'cpu'


def ResNet():
    return ResNet18(Bottleneck,10)

#define loss funtion & optimizer
criterion = nn.CrossEntropyLoss()
model =ResNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)




#train
EPOCH=2
for epoch in range(EPOCH):
    print('\nEpoch: %d' % (epoch + 1))
    model.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(train_loader, 0):
        #prepare dataset
        length = len(train_loader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        #forward & backward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #print ac & loss in each batch
        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' 
              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
        
    #get the ac with testdataset in each epoch
    print('Waiting Test...')
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            model.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Test\'s ac is: %.3f%%' % (100 * correct / total))

print('Train has finished, total epoch is %d' % EPOCH)

