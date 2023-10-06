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


##########################
##########################

# using Cifar10 

train_dataset = datasets.CIFAR10(root='data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.CIFAR10(root='data',
                              train=False,
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=64,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=64,
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break


class Bottleneck(nn.Module):
    def __init__(self,in_channel,out_channel,stride):
        super(Bottleneck,self).__init__()

        self.bottle = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size =1,stride =1,padding =0  ,bias =False ),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel,out_channel,kernel_size =3,stride = stride ,padding =1,bias = False),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel,out_channel*4,kernel_size =1 ,stride = 1,padding =0, bias =False),
            nn.BatchNorm2d(out_channel*4)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel*4 :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel,out_channel*4,kernel_size =1,stride = stride,bias = False,padding =0),
                nn.BatchNorm2d(out_channel*4)
            )
    def forward(self,x):
        out = self.bottle(x)
        out = out+self.shortcut(x)
        out =F.relu(out)
        return out

    
class ResNet50(nn.Module):
    def __init__(self,Bottleneck,num_classes):
        super(ResNet50,self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3,64,kernel_size = 7,stride = 2,padding =3)
        self.batchnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace =True)

        self.max_pooling = nn.MaxPool2d(3, stride=2,padding =1 )
        self.block1 = self.make_layer(Bottleneck,64,3,1,3)
        self.block2 = self.make_layer(Bottleneck,128,3,2,4)
        self.block3 = self.make_layer(Bottleneck,256,3,2,6)
        self.block4 = self.make_layer(Bottleneck,512,3,2,3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048,num_classes)

    def make_layer(self,Bottleneck,out_channel,number_iteration,stride,total_run):
        allstrides=([stride] +[1]*(number_iteration-1))*total_run
        layer =[]
        for stride in allstrides:
            layer.append(Bottleneck(self.in_channel ,out_channel,stride))
            self.in_channel = out_channel *4
        return nn.Sequential(*layer)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x =self.fc(x)
        return x



device ='gpu' if torch.cuda.is_available()==True  else 'cpu'


def ResNet():
    return ResNet50(Bottleneck,10)

#define loss funtion & optimizer
criterion = nn.CrossEntropyLoss()
model =ResNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)



#train
EPOCH=2
model =ResNet()
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