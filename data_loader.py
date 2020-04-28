# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:21:14 2018

@author: jiaym15
"""

from torch.utils import data
from torchvision import transforms as T
#from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import numpy as np
from math import floor

class Pixels(data.Dataset):
    """Dataset class for the Pixels dataset."""
    def __init__(self, train_image_dir, test_image_dir, transform, mode):
        """Initialize and preprocess the Pixels dataset."""
        self.train_image_dir = train_image_dir
        self.test_image_dir = test_image_dir
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the Pixels attribute file."""
        if self.mode == 'train':
            trainFileList=os.listdir(self.train_image_dir)
            random.seed(1234)
            random.shuffle(trainFileList)
            for i, trainFile in enumerate(trainFileList):
                filename=trainFile
                n = trainFile[:-4].split('_')
        #            ind = int(n[0])            
                nx = float(n[1].split('=')[1])
                ny = float(n[2].split('=')[1])
                nz = float(n[3].split('=')[1])
                normal = [nx,ny,nz]
                self.train_dataset.append([filename, normal])
        
        if self.mode == 'test':
            testFileList=os.listdir(self.test_image_dir)
            f = open(os.path.join(self.test_image_dir,testFileList[0]))
            data = f.read()
            lines = data.split('\n')
            for i in range(len(lines)-1):
                line = lines[i].split('\\')
                ind = int(line[0])
                nx = float(line[1])
                ny = float(line[2])
                nz = float(line[3])
                normal = [nx, ny, nz]
                mapsind = [int(l) for l in line[4:14]]
                maps = [float(l) for l in line[14:24]]
                self.test_dataset.append([ind, normal, mapsind, maps])
                
                
        print('Finished preprocessing the Pixels dataset...')

        

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""

        if self.mode == 'train':
            dataset = self.train_dataset
            filename, normal = dataset[index]
        else :
            dataset = self.test_dataset
            ind, normal, mapsind, maps = dataset[index]

        if self.mode == 'train':
            f = open(os.path.join(self.train_image_dir, filename))
            line=f.readline()
            line=line.split('\\')
            for i in range(len(line)):
                line[i]=float(line[i])
            tgt = np.where(np.array(line)>0)
            tgtrandom = np.random.permutation(tgt[0])
            L = tgtrandom[:10]
            X = np.floor(L/32)
            Y = L%32
            if len(L)<10:
                X=np.pad(X,(0,10-len(L)),'constant')
                Y=np.pad(Y,(0,10-len(L)),'constant')
            imageall = torch.FloatTensor(line)
            image10=torch.zeros(1024)
            image10[L]=imageall[L]
            imageall = torch.reshape(imageall,[1,32,32])
            image10 = torch.reshape(image10,[1,32,32])
            return imageall, image10, torch.FloatTensor(normal), torch.LongTensor(X), torch.LongTensor(Y)
        if self.mode == 'test':
            imageall=torch.zeros(1024)
            imageall[np.array(mapsind)-1]=torch.tensor(maps)
            imageall = torch.reshape(imageall,[1,32,32])
            image10=imageall
            return imageall, image10, torch.FloatTensor(normal), torch.tensor(ind)
        
            
    def __len__(self):
        """Return the number of images."""
        return self.num_images

        
def get_loader(train_image_dir, test_image_dir, image_size=32, 
               batch_size=64, dataset='Pixels', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    transform.append(T.ToTensor())
    transform = T.Compose(transform)


    dataset = Pixels(train_image_dir, test_image_dir, transform, mode)


    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader