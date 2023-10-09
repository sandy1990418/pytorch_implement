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

## reference :https://github.com/CaptainDredge/Variational-AutoEncoder-in-Pytorch/blob/master/train.py


## using Cifar10 
btsize=64

train_dataset = datasets.CIFAR10(root='data',
                                 train =True,
                                 transform = transforms.ToTensor(),
                                 download =True)

test_dataset = datasets.CIFAR10(root ='data',
                                train = False,
                                transform = transforms.ToTensor(),
                                download = True)

train_loader = DataLoader(train_dataset,batch_size = btsize,shuffle = True)
test_loader = DataLoader(test_dataset,batch_size =btsize,shuffle =False)


########################
########## VAE #########
########################


class VAE(nn.Module):
    def __init__(self,in_channel,hidden_layer,hidden_layer_reverse,latent_size):
        super(VAE,self).__init__()
        
        self.latent_temp = latent_size
        ## Encoder 
        layer = [] 
        for hidden in hidden_layer:

            layer.append(nn.Sequential(
                nn.Conv2d(in_channel,hidden,kernel_size =3,stride =1 ,padding =1,bias = False),
                nn.MaxPool2d(2, stride=2),
                nn.BatchNorm2d(hidden),
                nn.GELU()
            ))
            in_channel = hidden
        self.encoder = nn.Sequential(*layer)

        ## latent layer 
        self.mu = nn.Linear(hidden_layer[-1]*16,latent_size)
        self.var = nn.Linear(hidden_layer[-1]*16,latent_size)
        self.decoder_input = nn.Linear(latent_size, hidden_layer[-1] *32*32)
        ## Decoder 
        decoder_layer = [] 
        for i in range(len(hidden_layer_reverse)-1):
            decoder_layer.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_layer_reverse[i],hidden_layer_reverse[i+1],kernel_size =3,stride =1,padding =1,bias = False),
                nn.BatchNorm2d(hidden_layer_reverse[i+1]),
                nn.GELU()
            ))
        self.decoder =nn.Sequential(*decoder_layer)
        self.final = nn.Sequential(
                nn.ConvTranspose2d(hidden_layer_reverse[-1],hidden_layer_reverse[-1],kernel_size =3),
                nn.BatchNorm2d(hidden_layer_reverse[i+1]),
                nn.GELU(),
                nn.Conv2d(hidden_layer_reverse[-1], out_channels=3,
                          kernel_size=3, stride=1,padding =0),
                
                nn.Sigmoid())
        
    def reparameterize(self, mu, logvar) :
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self,x):
        x  = self.encoder(x)
        x  = torch.flatten(x)
        
        
        mu,var = self.mu(x),torch.exp(self.var(x))
        z = self.reparameterize(mu,var)

        z = self.decoder_input(z)
        z = z.view(-1, 64, 32, 32)
        z = self.decoder(z)
        z = self.final(z)
        return z,mu,var
    

device ='gpu' if torch.cuda.is_available()==True  else 'cpu'


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

#define loss funtion & optimizer
model =VAE(3,[16,32,64],[64,32,16],4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=5e-4, momentum=0.9) #

#train
EPOCH=2
from tqdm import tqdm 
for epoch in range(EPOCH):
    print('\nEpoch: %d' % (epoch + 1))
    model.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in tqdm(enumerate(train_loader, 0)):
        #prepare dataset
        length = len(train_loader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        #forward & backward
        x_hat, mean, log_var = model(inputs)
        loss = loss_function(inputs, x_hat, mean, log_var)

        loss.backward()
        optimizer.step()

        #print ac & loss in each batch
        sum_loss += loss.item()
    print("\tEpoch", epoch + 1, "\tAverage Loss: ", sum_loss/len(train_dataset))
        # _, predicted = torch.max(outputs.data, 1)
        # total += labels.size(0)
        # correct += predicted.eq(labels.data).cpu().sum()
        # print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
        #       % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

    # #get the ac with testdataset in each epoch
    # print('Waiting Test...')
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for data in test_loader:
    #         model.eval()
    #         images, labels = data
    #         images, labels = images.to(device), labels.to(device)

    #         x_hat, mean, log_var = model(images)
    #         loss = loss_function(images, x_hat, mean, log_var)
    #         total += loss.item()
    #     print("\tEpoch", epoch + 1, "\t test Average Loss: ", total/len(test_dataset))

print('Train has finished, total epoch is %d' % EPOCH)