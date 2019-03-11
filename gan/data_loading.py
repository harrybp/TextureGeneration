'''
Provides methods for obtaining dataloaders for training GAN models
Methods:
    get_folder_dataloader:      get a dataloader for a folder of images
    get_image_dataloader:       get a dataloader for random crops of a single image
Classes:
    TextureDataset:             defines a dataset of random crops of a single image
'''
import random
import torchvision.transforms as transforms
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

def get_folder_dataloader(folder_path, image_size, batch_size=64):
    '''
    Get a dataloader for a folder of images
    Args:
        folder_path:    path of the folder of images to use
        image_size:     size of images to train on, images will be resized to this
        batch_size:     size of images batches returned by the dataloader
    Returns:
        A dataloader used to get the images in the provided folder
    '''
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(root=folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

def get_image_dataloader(image_path, image_size, image_resize=1, dataset_size=4096, batch_size=64):
    '''
    Get a dataloader for random crops of a single image
    Args:
        image_path:     path of the image to use
        image_size:     size of images to train on, image will be cropped to this
        image_resize:   images will be scaled down by this factor before cropping
        dataset_size:   the total number of random crops of the source image in the dataset
        batch_size:     size of images batches returned by the dataloader
    Returns:
        A dataloader used to get random crops of the provided image
    '''
    dataset = TextureDataset(
        image_size=image_size,
        dataset_size=dataset_size,
        image_path=image_path,
        transform=transforms.Compose([
            transforms.Resize((int(image_size/image_resize), int(image_size/image_resize))),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

class TextureDataset(Dataset):
    '''
    Create a Dataset of randomly cropped sections of a source image
    Args:
        image_size:     size of images to train on, image will be cropped to this
        dataset_size:   the total number of random crops of the source image in the dataset
        image_path:     path of the image to use
        transform:        transform to be applied to all images in the dataset
    Returns:
        A DataSet Object
    '''
    def __init__(self, image_size, dataset_size, image_path, transform=None):
        self.image = Image.open(image_path)#.resize((400,400), Image.ANTIALIAS)
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
