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

from torchvision.transforms import Compose, Resize, ToTensor,transforms,datasets
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


import matplotlib.pyplot as plt
from PIL import Image



class Patches_Embedding(nn.Module):
    def __init__(self,patch_size):
        super(Patches_Embedding,self).__init__()
        
        # b =batch,c=channel,h=height,w=width
        # split each image into h1*h2 smaller
        # https://einops.rocks/api/rearrange/
        # https://zhuanlan.zhihu.com/p/348849092
        self.pat
        Rearrange('b c (h h1) (w w1) -> (b h1 w1) h w c', h1=patch_size, w1=patch_size)
        

