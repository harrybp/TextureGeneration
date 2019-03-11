import torch
import os
import numpy as np
import random
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import sqlite3

to_pil_image = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

def crop_texture_images():
    crop = transforms.RandomCrop((256,256))
    for file in os.listdir("textures"):
        if file.endswith(".jpg"):
            print(file)
            texture = Image.open('textures/' + file)
            cropped = crop(texture)
            cropped.save('textures/cropped/' + file)

# Takes a source image tensor, cut some off the bottom and append to the top
def tile_vertical(source):
    size = source.shape[1]
    size1 = random.randint(1,source.shape[1])
    top_half = source[:, 0:size1, :]
    bottom_half = source[:, size1:size,:]
    return torch.cat((bottom_half, top_half), 1)

#Take a source image tensor, cut some off the left and append to the right
def tile_horizontal(source):
    size = source.shape[2]
    size1 = random.randint(1,source.shape[2])
    left_half = source[:,:,0:size1]
    right_half = source[:,:,size1:size]
    return torch.cat((right_half, left_half), 2)



