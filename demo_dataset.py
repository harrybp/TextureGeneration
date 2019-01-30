import utils
import torch
import torchvision
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import sys
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import PIL.Image, PIL.ImageTk
import argparse
import os



def demo2():
    toPILImage = transforms.ToPILImage()
    myTrans = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    batch_size = 4
    dataset = torchvision.datasets.ImageFolder(root='textures/bricks.jpg', transform=myTrans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    j = iter(dataloader).next()[0]
    

    current_images = toPILImage(vutils.make_grid(j[:1], nrow=1, padding=1, normalize=True).cpu())
    current_images.save('textures/crop/bricks.jpg')



def demo1(name):
    toPILImage = transforms.ToPILImage()
    '''
    source_image = 'textures/van_gogh.jpg'
    #Create dataset and dataloader from source image
    batch_size = 4
    dataset_size = 16384
    dataset = utils.TextureDataset(
        image_size=256,
        size=dataset_size,
        image_path=source_image,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    '''
    dataloader = utils.get_image_dataloader('textures/'+name+'.jpg', 256)
    j = iter(dataloader).next()
    current_images = toPILImage(vutils.make_grid(j[:1], nrow=1, padding=1, normalize=True).cpu())
    current_images.save('textures/cropped/'+name+'.jpg')

if __name__ == "__main__":
   # demo1('painting')
    #demo1('snake')
    demo1('pebbles')