import torch
import numpy as np
import random
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import sqlite3

#Add a GAN to the database
def save_GAN(name, source, iterations, lr, width, height):
    try:
        with sqlite3.connect("database.db") as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute("INSERT INTO gans (name,sourceimg,iterations,lr,width,height) VALUES (?,?,?,?,?,?)", (name,source,iterations,lr,width,height) )
            con.commit()
            msg = "Added to Database"
    except:
        con.rollback()
        msg = "Error adding to Database"
    finally:
        print(msg)
        con.close()

#Get list of all trained GANs in database
def get_GANS():
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    query = 'SELECT name, MAX(iterations) FROM gans GROUP BY name'
    cur.execute(query)
    rows = cur.fetchall()
    return rows

def update_progress(progress):
    try:
        with sqlite3.connect("database.db") as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute('UPDATE gatys SET progress = ' + str(progress) + ' WHERE current = "gatys"')
            con.commit()
            msg = "Added to Database"
    except:
        con.rollback()
        msg = "Error adding to Database"
    finally:
        print(msg)
        con.close()

def update_progress_gan(progress):
    try:
        with sqlite3.connect("database.db") as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute('UPDATE gans_progress SET progress = ' + str(progress) + ' WHERE current = "gans"')
            con.commit()
            msg = "Added to Database"
    except:
        con.rollback()
        msg = "Error adding to Database"
    finally:
        print(msg)
        con.close()

def get_progress_gan():
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    query = 'SELECT progress FROM gans_progress WHERE current = "gans"'
    cur.execute(query)
    rows = cur.fetchall()
    return int(rows[0][0])

def get_progress():
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    query = 'SELECT progress FROM gatys WHERE current = "gatys"'
    cur.execute(query)
    rows = cur.fetchall()
    return int(rows[0][0])
    

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

#Calculate the gram matrix of a single layer
def gram_matrix(layer):
    c, h, w = layer.size()
    layer = layer.view(c, h * w)
    gram = torch.mm(layer, layer.t())
    return gram

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Calculate the gram matrix for each layer of a cnn
def gram_matrix_layers(layers):
    targets = []
    for i in range(len(layers)):
        gram = gram_matrix(layers[i])
        targets.append(gram)
    return targets

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Get the number of variables in each layer
def get_layer_sizes(layers):
    sizes = []
    for i, layer in enumerate(layers):
        c, h, w = layer.size()
        sizes.append(c*h*w)
    return sizes

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Get style loss by comparing two Gram Matrix's
def get_style_loss(grams1, grams2, layer_sizes, weights):
    style_loss = 0
    for i in range(len(grams1)): #for now
        gram1 = grams1[i]
        gram2 = grams2[i]
        style_loss += weights[i] * torch.mean((gram1 - gram2)**2) / layer_sizes[i]
    return style_loss

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Run an input image through the cnn and extract feature map from each layer
def get_feature_layers(input, cnn, layers_select):
    features = []
    prev_feature = input.unsqueeze(0)
    for i, module in enumerate(cnn):
        feature = module(prev_feature)
        if(i in layers_select):
            features.append(feature.squeeze(0))
        prev_feature = feature
    return features

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
def get_image_dataloader(image_path, image_size, dataset_size=4096, batch_size=64):
    dataset = TextureDataset(
        image_size=image_size,
        size=dataset_size,
        image_path=image_path,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

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
# The following generator and discriminator implementations are taken from
# https://github.com/pytorch/examples/tree/master/dcgan
# Create a Generator CNN, which takes a 1D input vector and produces an image
class Generator(nn.Module):
    def __init__(self, input_size, image_size, channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( input_size, image_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(True), 
            nn.ConvTranspose2d(image_size * 8, image_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(True), 
            nn.ConvTranspose2d( image_size * 4, image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(True), 
            nn.ConvTranspose2d( image_size * 2, image_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size),
            nn.ReLU(True),
            nn.ConvTranspose2d( image_size, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Create a Discriminator CNN, which takes an image as 
#   input and returns a 0 (fake) or 1 (real)    
class Discriminator(nn.Module):
    def __init__(self, image_size, channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, image_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size, image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size * 2, image_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size * 4, image_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CycleGenerator and a second discriminator implementation taken from
# http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf
class CycleGenerator(nn.Module):
    def __init__(self, conv_dim=64, init_zero_weights=False):
        super(CycleGenerator, self).__init__()

        #Conv layers decode image and extract features
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)

        #Residual layers transform the image
        self.resnet_block = ResnetBlock(conv_dim * 2)

        #Deconv layers encode features back into an image
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.resnet_block(out))
        out = F.relu(self.deconv1(out))
        out = F.tanh(self.deconv2(out))
        return out


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out

def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class DCDiscriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super(DCDiscriminator, self).__init__()
        self.conv1 = conv(3, conv_dim, 4) 
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.conv4 = conv(conv_dim * 4, 1, 4, padding=0, batch_norm=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.conv4(out).squeeze()
        out = F.sigmoid(out)
        return out
