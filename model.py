# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:29:58 2018

@author: jiaym15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=6):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)
        
        mv = torch.arange(1024)
        X = mv.float()%32
        Y = torch.floor(mv.float()/32)
        x = X/31*2-1
        y = Y/31*2-1
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)

        r = torch.cat([x,y],dim=1)
        r_norm = torch.norm(r,dim=1)
        mask = torch.lt(r_norm,1)
        self.mask = torch.reshape(mask,[32,32])
        


    def forward(self, x):
        mask = self.mask
        mask = mask.expand(x.size(0),1,32,32)
        x1 = self.main(x)
        y = torch.zeros(x1.size()).to(self.device)
        y[mask] = x1[mask]
        return y

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=32):
        super(Discriminator, self).__init__()
               
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        
        block1_1 = []
        block1_1.append(nn.ReLU())
        block1_1.append(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1))
        block1_1.append(nn.Dropout2d(p=0.2))
        self.denseblock1_1 = nn.Sequential(*block1_1)
        
        block1_2 = []
        block1_2.append(nn.ReLU())
        block1_2.append(nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1))
        block1_2.append(nn.Dropout2d(p=0.2))
        self.denseblock1_2 = nn.Sequential(*block1_2)                
        
        trans = []
        trans.append(nn.ReLU())
        trans.append(nn.Conv2d(48, 48, kernel_size=1, stride=1))
        trans.append(nn.Dropout2d(p=0.2))
        trans.append(nn.AvgPool2d(kernel_size=4, stride=2, padding=1))
        self.transition = nn.Sequential(*trans)
        
        block2_1 = []
        block2_1.append(nn.ReLU())
        block2_1.append(nn.Conv2d(48, 16, kernel_size=3, stride=1, padding=1))
        block2_1.append(nn.Dropout2d(p=0.2))
        self.denseblock2_1 = nn.Sequential(*block2_1)
        
        block2_2 = []
        block2_2.append(nn.ReLU())
        block2_2.append(nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1))
        block2_2.append(nn.Dropout2d(p=0.2))
        self.denseblock2_2 = nn.Sequential(*block2_2)        

        self.conv2 = nn.Conv2d(80, 80, kernel_size=1, stride=1, padding=0)
        
        dense = []
        dense.append(nn.Linear(20480, 128))
        dense.append(nn.Linear(128,3))
        self.dense = nn.Sequential(*dense)
#        
    def forward(self, x):
        h1 = self.conv1(x)
        h2 = torch.cat([h1, self.denseblock1_1(h1)], dim=1)
        h3 = torch.cat([h1, self.denseblock1_1(h1), self.denseblock1_2(h2)], dim=1)
        d1 = self.transition(h3)
        d2 = torch.cat([d1, self.denseblock2_1(d1)], dim=1)
        d3 = torch.cat([d1, self.denseblock2_1(d1), self.denseblock2_2(d2)], dim=1)        
        d4 = self.conv2(d3)           
        d = d4.view(-1, self.num_flat_features(d4))
        out_reg = self.dense(d)
        out_reg = self.normalize(out_reg)
        return out_reg.view(out_reg.size(0), out_reg.size(1))
        
    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def normalize(self, x):
        x_norm = torch.norm(x,dim=1)
        x_norm = x_norm.unsqueeze(1)
        x_norm = x_norm.expand(x_norm.size(0),x.size(1))
        return x/x_norm