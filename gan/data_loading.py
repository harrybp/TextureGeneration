'''
Provides methods for obtaining dataloaders for training GAN models
Methods:
    get_folder_dataloader:      get a dataloader for a folder of images
    get_image_dataloader:       get a dataloader for random crops of a single image
Classes:
    TextureDataset:             defines a dataset of random crops of a single image
'''
import random
import os
import torchvision.transforms as transforms
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

def get_dataloader(path, image_size, batch_size=64, scaling_factor=1):
    '''
    Returns a dataloader for the supplied path which can be a path to an image or a folder of images
    Args:
        path:           path of the image file / folder of images to use
        image_size:     size of images to train on, images will be cropped to this size
        scaling_factor: image size will be multiplied by this before cropping, set higher to 'zoom out' images
        batch_size:     size of images batches returned by the dataloader
    '''
    if os.path.isdir(path):
        return get_folder_dataloader(path, image_size, scaling_factor=scaling_factor, batch_size=batch_size)
    return get_image_dataloader(path, image_size, scaling_factor=scaling_factor, batch_size=batch_size)


def get_folder_dataloader(folder_path, image_size, scaling_factor=1, dataset_size=4096, batch_size=64):
    '''
    Get a dataloader for a folder of images
    Args:
        folder_path:    path of the folder of images to use
        image_size:     size of images to train on, images will be cropped to this size
        scaling_factor: image size will be multiplied by this before cropping, set higher to 'zoom out' images
        dataset_size:   minimum size of dataset, (if there are less images in the folder they will be duplicated up to this)
        batch_size:     size of images batches returned by the dataloader
    Returns:
        A dataloader used to get the images in the provided folder
    '''
    dataset = FolderDataset(
        folder_path=folder_path,
        image_size=image_size,
        dataset_size=dataset_size,
        transform=transforms.Compose([
            transforms.RandomCrop(int(image_size * scaling_factor)),
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

def get_image_dataloader(image_path, image_size, scaling_factor=1, dataset_size=4096, batch_size=64):
    '''
    Get a dataloader for random crops of a single image
    Args:
        image_path:     path of the image to use
        image_size:     size of images to train on, image will be cropped to this
        scaling_factor: image size will be multiplied by this before cropping, set higher to 'zoom out' images
        dataset_size:   the total number of random crops of the source image in the dataset
        batch_size:     size of images batches returned by the dataloader
    Returns:
        A dataloader used to get random crops of the provided image
    '''
    dataset = ImageDataset(
        image_size=int(image_size * scaling_factor),
        dataset_size=dataset_size,
        image_path=image_path,
        transform=transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

class ImageDataset(Dataset):
    '''
    Create a Dataset of randomly cropped sections of a source image
    Args:
        image_path:     path of the image to use
        image_size:     size of images to train on, image will be cropped to this
        dataset_size:   the total number of random crops of the source image in the dataset
        transform:        transform to be applied to all images in the dataset
    Returns:
        A DataSet Object
    '''
    def __init__(self, image_path, image_size, dataset_size, transform=None):
        self.image = Image.open(image_path)
        self.transform = transform
        self.size = dataset_size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        #Randomly crop the image
        size = self.image_size
        width, height = self.image.size
        x0 = random.randint(0, width - size)
        y0 = random.randint(0, height - size)
        sample = self.image.crop((x0, y0, x0 + size, y0 + size))

        if self.transform:
            sample = self.transform(sample)

        return sample

class FolderDataset(Dataset):
    '''
    Create a Dataset of randomly cropped sections of images from an image folder
    Args:
        folder_path:    path of the folder of images to use
        image_size:     size of images to train on, image will be cropped to this
        dataset_size:   minimum size of dataset, (if there are less images in the folder they will be duplicated up to this)
        transform:      transform to be applied to all images in the dataset
    Returns:
        A DataSet Object
    '''
    def __init__(self, folder_path, image_size, dataset_size, transform=None):
        self.transform = transform
        self.image_size = image_size
        self.images = []
        for file in os.listdir(folder_path):
            self.images.append(Image.open(folder_path + '/' + file).convert('RGB'))
        self.size = max(dataset_size, len(self.images))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        idx = idx % len(self.images)
        if self.transform:
            image = self.transform(self.images[idx])
        return image