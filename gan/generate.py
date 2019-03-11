'''
Provides methods for generating images using pretrained GAN models
Methods:
    generate_image: uses a pretrained GAN model to generate an image
    demo_ps_gan:    generate and save an image using pretrained PSGAN
    demo_dc_gan:    generate and save an image using pretrained DCGAN
'''
import os
import torch
import torchvision.transforms as transforms
import config
from .models import PSGenerator, DCGenerator

def demo_ps_gan(name, checkpoint=-1, image_size=256, tile=False, filepath='result.jpg'):
    '''
    Generate an Image using a pre-trained PSGAN model, saves it as result.jpg
    Args:
        name:           name of the gan model, (the name of the outer folder containing the training checkpoints)
        image_size:     size of the image to generate
        checkpoint:     the training checkpoint to use for generating, (if -1, the most recent checkpoint is used)
        tile:           whether the generated image should be tileable
        filepath:       the path to which the image will be saved
    '''
    transform = None
    if tile:
        transform = transforms.CenterCrop((image_size, image_size))
    root_directory = 'models/' + name + '/ps'
    if checkpoint < 0:
        checkpoint = 0
        for root, dirs, files in os.walk(root_directory):
            checkpoint += len(dirs)
    root_directory = root_directory + '/' + str(checkpoint)
    generator = PSGenerator().to(config.device)
    generator.load_state_dict(torch.load(root_directory + '/generator.pt'))
    noise = generator.noise(1, image_size, tile=tile).to(config.device)
    image = generate_image(generator, noise, transform=transform)#
    image.save(filepath)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   Generate an image using a pretrained dc_gan
#       If checkpoint = -1, most recent checkpoint is used
def demo_dc_gan(name, checkpoint=-1, filepath='result.jpg'):
    '''
    Generate an Image using a pre-trained DCGAN model, saves it as result.jpg
    Args:
        name:           name of the gan model, (the name of the outer folder containing the training checkpoints)
        checkpoint:     the training checkpoint to use for generating, (if -1, the most recent checkpoint is used)
        filepath:       the path to which the image will be saved
    '''
    root_directory = 'models/' + name + '/dc'
    if checkpoint < 0:
        checkpoint = 0
        for root, dirs, files in os.walk(root_directory):
            checkpoint += len(dirs)
    root_directory = root_directory + '/' + str(checkpoint)
    generator = DCGenerator(100,64,3).to(config.device)
    generator.load_state_dict(torch.load(root_directory + '/generator.pt'))
    noise = generator.noise(4, 64).to(config.device)
    image = generate_image(generator, noise)
    image.save(filepath)

def generate_image(generator, noise, transform=None):
    '''
    Generate an Image using a given generator and noise vector
    Args:
        generator:      a pretrained generator model
        noise:          the noise vector to be inputted to the generator
        transform:      transform to be applied to the generated image
    Returns:
        The generated image in PIL image format
    '''
    normalise = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
    result = generator(noise).detach().cpu()
    if result.shape[0] > 1: ###Potentially remove
        res1 = torch.cat((result[0], result[1]), 1)
        res2 = torch.cat((result[2], result[3]), 1)
        result = torch.cat((res1, res2), 2) #Make grid of identical noise
    else:
        result = result.squeeze()
    result = transforms.ToPILImage()(normalise(result))
    if transform is not None:
        result = transform(result)
    return result