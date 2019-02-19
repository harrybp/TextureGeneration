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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Get dataloader for a folder of images
def get_folder_dataloader(folder_path, image_size, batch_size=64):
    transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    dataset = torchvision.datasets.ImageFolder(root=folder_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Get a dataloader for a single image (random crops)
#Image size = size of the crop taken
# Image will then be scaled by dividing by imageresize
def get_image_dataloader(image_path, image_size, image_resize=1, dataset_size=4096, batch_size=64):
    dataset = TextureDataset(
        image_size=image_size,
        size=dataset_size,
        image_path=image_path,
        transform=transforms.Compose([
            transforms.Resize((int(image_size/image_resize), int(image_size/image_resize))),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Initialise weights for GAN as done in https://arxiv.org/pdf/1511.06434.pdf
def initialise_weights(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)
        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Create a dataset of n randomly cropped images from a source image
# TODO: handle resize better
class TextureDataset(Dataset):
    def __init__(self, image_size, size, image_path, transform=None):
        self.image = Image.open(image_path)#.resize((400,400), Image.ANTIALIAS)
        self.transform = transform
        self.size = size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        size = self.image_size
        width, height = self.image.size
        x0 = random.randint(0,width - size)
        y0 = random.randint(0,height - size)
        sample = self.image.crop((x0, y0, x0 + size, y0 + size))

        if self.transform:
            sample = self.transform(sample)

        return sample

