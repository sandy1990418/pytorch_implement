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
## reference2:https://hackmd.io/@EUSQi8vWTzuCNlcbMQ_lXw/ry8kn7Fc?type=view

## using mnist
btsize=50
train_dataset = datasets.MNIST(root='data',
                                 train =True,
                                 transform = transforms.ToTensor(),
                                 download =True)

test_dataset = datasets.MNIST(root ='data',
                                train = False,
                                transform = transforms.ToTensor(),
                                download = True)

train_loader = DataLoader(train_dataset,batch_size = btsize,shuffle = True)
test_loader = DataLoader(test_dataset,batch_size =btsize,shuffle =False)


########################
########## VAE #########
########################


class Encoder(nn.Module):
    def __init__(self,in_channel,encoder_layer,latent):
        super(Encoder,self).__init__()
        
        encoder = []
        for hidden in encoder_layer:
            encoder.append(nn.Linear(in_channel,hidden,bias =True))
            nn.BatchNorm1d(hidden)
            nn.LeakyReLU(inplace =True)
            in_channel = hidden

        self.encoder = nn.Sequential(*encoder)
        self.mu = nn.Linear(hidden,latent,bias = True)

        ## to avoid negative value of variance
        self.log_var = nn.Linear(hidden,latent,bias = True)

    def forward(self,x):
        x = self.encoder(x)
        mu,log_var = self.mu(x),self.log_var(x)

        return mu,log_var
    
class Decoder(nn.Module):
    def __init__(self,decoder_layer,latent):
        super(Decoder,self).__init__()
        decoder = []
        for hidden in decoder_layer:
            decoder.append(nn.Linear(latent,hidden,bias =True))
            nn.BatchNorm1d(hidden)
            nn.LeakyReLU(inplace =True)
            latent = hidden

        self.decoder = nn.Sequential(*decoder)        

    def forward(self,x):
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return  x 
    

class VAE(nn.Module):
    def __init__(self,encoder,decoder):
        super(VAE,self).__init__()
        self.encoder = encoder
        self.decoder = decoder 
    
    def reparametric(self,mu,log_var):
        epsilon = torch.randn_like(log_var).to(device)
        z=(mu+torch.exp(0.5 * log_var)*epsilon)
        return z
    
    def forward(self,x):
        mu,log_var=self.encoder(x)
        z = self.reparametric(mu,log_var)
        z_hat = self.decoder(z)

        return z_hat,mu,log_var




def loss_function(x, x_hat, mean, log_var):
    reproduction_loss =  F.mse_loss(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

encoder = Encoder(28*28,[32,64,128,512],256)
decoder = Decoder([512,256,128,64,28*28],256)


device = 'gpu' if torch.cuda.is_available() ==True else 'cpu'

model = VAE(encoder,decoder).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr =0.004, weight_decay=5e-4) #, momentum=0.9


from tqdm import tqdm
EPOCH =10
for epoch in range(EPOCH):
    model.train()
    sum_loss = 0.0
    print('\nEpoch %d' % (epoch+1))
    for i,data in tqdm(enumerate(train_loader)):
        

        input,label = data
        input = input.view(btsize, 28*28)
        input,label = input.to(device),label.to(device)
        
        optimizer.zero_grad()

        zhat,mu,log_var = model(input)
        loss = loss_function(input,zhat,mu,log_var)

        loss.backward()
        optimizer.step()
        
        sum_loss += loss.item()

    print('\t Epoch',epoch+1,'\t train avg_loss',sum_loss/len(train_dataset))

    with torch.no_grad():
        test_sum_loss = 0.0 
        for i ,data in tqdm(enumerate(test_loader)):
            model.eval()
            input,label = data
            input,label = input.to(device),label.to(device)
            input = input.view(btsize, 28*28)
            zhat,mu,log_var = model(input)
            loss = loss_function(input,zhat,mu,log_var)

            test_sum_loss+=loss.item()
    print('\t Epoch',epoch+1,'\t test avg_loss',test_sum_loss/len(test_dataset))



plt.imshow((input[4,:].detach().numpy()).reshape((28,28)))
plt.imshow((zhat[4,:].detach().numpy()).reshape((28,28)))