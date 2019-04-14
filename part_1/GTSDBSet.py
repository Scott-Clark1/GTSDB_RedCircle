from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np

from skimage import io, transform
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math

import cv2

from random import random

import time

from transformations import *

INPUT_DIR = 'FullIJCNN2013'
AUG_DIR = 'AUGMENTED'



class GTSDBSet(Dataset):
    def __init__(self, path_to_file, path_to_data, max_objects=4, transform=None):
        self.data = pd.read_csv(path_to_file)
        self.data['filename'] = path_to_data + self.data['filename']
        self.unique_images = self.data['filename'].unique()
        self.transform = transform
        self.max_objects = max_objects

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, idx):
        img_name = self.unique_images[idx]
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)
        
        tow = [0]
        if '_' in img_name:
          tow = [1]
        labels = self._getLabels(img_name)
        return img_name, image, labels
    
    def _getLabels(self, img_id):
      labels = np.zeros((self.max_objects,5))
      data = self.data[self.data['filename'] == img_id]
      data['x'] = data['Xmin'] + (data['Xmax'] - data['Xmin']) / 2
      data['y'] = data['Ymin'] + (data['Ymax'] - data['Ymin']) / 2
      data['w'] = data['Xmax'] - data['Xmin']
      data['h'] = data['Ymax'] - data['Ymin']
      data['label'] = 1
      data = data[['label', 'x', 'y', 'w', 'h']]
      data = data.values.reshape((-1,5))
      #print(data.shape)
      labels[0:data.shape[0],:] = data
      return labels
  
  
  
  


if __name__ == '__main__':
  cuda = True

  batch_size = 1
  
  transform = transforms.Compose([
                   transforms.ToPILImage(),
                   #transforms.Resize((416, 416)),
                   #transforms.RandomCrop(185, padding=26),
                   #transforms.RandomHorizontalFlip(p=.5),
                   transforms.ToTensor()
              ])
  train_data = GTSDBSet('FullIJCNN2013/gt_train.txt','FullIJCNN2013/ALL_DATA/',transform = transform)

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

  augmented_data = []
  save = True
  show = False
  for idx, sample in enumerate(train_loader):
    #if idx > 49:
    #  break
    
    img_name = sample[0][0]
    image = np.transpose(sample[1][0,:], [1,2,0])
    labels = sample[2][0]
    
    
    image, labels = flipImageAndLabels(image, labels, p=.5)
    image, labels = rotateImageAndLabels(image, labels, 30, p=.3)
    image = noisifiyImage(image, p=.2)
    image = changeBrightnessImage(image, p=.4)
    
    new_fname = "AUG_" + img_name.split("/")[-1].split(".")[0] + ".png" #gross line that gets the file name, removes its extension, and makes it a png
    
    #print(img_name)
    
    if save:
      plt.imsave(INPUT_DIR + '/' + AUG_DIR + '/' + new_fname, image)
    if show:
      fig, ax = plt.subplots(1)
      ax.imshow(image)
    for i in range(0,4):
      if labels[i,0] != 0:
        if show:
          rect = patches.Rectangle((labels[i,1] - labels[i,3]/2, labels[i,2] - labels[i,4]/2), labels[i,3], labels[i,4], linewidth=1,edgecolor='r',facecolor='none')
          ax.add_patch(rect)
        augmented_data.append([new_fname, labels[i,1] - labels[i,3]/2, labels[i,2] - labels[i,4]/2, labels[i,1] + labels[i,3]/2, labels[i,2] + labels[i,4]/2, labels[i,0]])
    
    if show:
      plt.show()
  #print(pd.DataFrame(augmented_data))
  if save:
    augmented_data = pd.DataFrame(augmented_data)
    augmented_data.columns = ['filename','Xmin','Ymin','Xmax','Ymax','label']
    augmented_data['Xmin'] = augmented_data['Xmin'].astype(int)
    augmented_data['Ymin'] = augmented_data['Ymin'].astype(int)
    augmented_data['Xmax'] = augmented_data['Xmax'].astype(int)
    augmented_data['Ymax'] = augmented_data['Ymax'].astype(int)
    augmented_data['label'] = augmented_data['label'].astype(int)
    augmented_data.to_csv(INPUT_DIR + '/gt_aug_train.txt', index=False)

