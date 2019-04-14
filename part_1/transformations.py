import numpy as np
import cv2
from random import random

def flipImageAndLabels(img, labels, p=.5):
  flipped = np.copy(img)
  new_labels = np.copy(labels)
  if random() < p:
    Xcenter, Ycenter = img.shape[1] / 2, img.shape[0] / 2
    flipped = np.flip(flipped, axis=1)
    new_labels[:,1] = labels[:,1] + (Xcenter - labels[:,1])*2
    
  for i in range(0, labels.shape[0]): 
    if not np.array(labels[i,:]).any(): # we want our labels to stay at zero if that's what they were initially
      new_labels[i,:] = np.zeros(new_labels.shape[1])
  return flipped, new_labels
    
def rotateImageAndLabels(img, labels, angle=10, p=.5):
  #print(img.dtype)
  #print(img.shape)
  img = np.array(img)
  H, W, _ = img.shape
  rotated = np.copy(img)
  new_labels = np.copy(labels)
  if random() < p:
    angle *= (random() - .5)*2
    midX, midY = W//2, H//2
    M = cv2.getRotationMatrix2D((midX,midY), angle, 1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    
    newW = int((H * sin) + (W * cos))
    newH = int((H * cos) + (W * sin))
    M[0,2] += (newW / 2) - midX
    M[1,2] += (newH / 2) - midY
    
    
    difW = (newW - W) / 2
    difH = (newH - H) / 2
    
    adjustedX = labels[:,1] - midX
    adjustedY = labels[:,2] - midY
    cos_rad = np.cos(0.0174533*angle)
    sin_rad = np.sin(0.0174533*angle)
    new_labels[:,1] = midX + cos_rad * adjustedX + sin_rad * adjustedY + difW
    new_labels[:,2] = midY - sin_rad * adjustedX + cos_rad * adjustedY + difH
  
  
    rotated = cv2.warpAffine(img, M, (newW,newH))
  
  for i in range(0, labels.shape[0]):
    if not np.array(labels[i,:]).any(): # we want our labels to stay at zero if that's what they were initially
      new_labels[i,:] = np.zeros(new_labels.shape[1])
    
  return rotated, new_labels
  
  
  
def noisifiyImage(img, p=.2, pNoise=.03):
  noisy = np.copy(img)
  if random() < p:
    Y, X, _ = img.shape
    for i in range(0, Y):
      for j in range(0, X):
        rand = random()
        if rand < pNoise:
          #print(rand)
          noisy[i,j,:] = 255
        elif rand > (1 - pNoise):
          noisy[i,j,:] = 0
  return noisy
  
def changeBrightnessImage(img, p=.5):
  brightened = np.copy(img)
  #print(img.shape)
  if random() < p:
    brightness_change = random() / 2# * (random() - .5)1
    brightened = brightened + brightness_change
  #print(brightened - image)
  return np.clip(brightened, 0, 1)
