import gan
import os
import torch
import imageio
#import torchvision
import numpy as np
#import matplotlib.pyplot as plt
#import sys
import torchvision.transforms as transforms
#import torchvision.models as models
#import torch.optim as optim
#import PIL.Image, PIL.ImageTk
import argparse
import torch.nn.functional as F
import models
import utils
import imageio

def generate_gif_2(gan_name, gan_checkpoint, image_size=512, key_frames=200, interpolated_frames=30, fps=40):
    generator = models.PSGenerator()
    generator.load_state_dict(torch.load('models/' + gan_name + '/ps/' + str(gan_checkpoint) + '/generator.pt'))

    images = []
    current_vector = np.random.uniform(-1, 1, (1, 64, int(image_size / 32), int(image_size / 32)))
    add_vector = np.random.uniform(-0.05, 0.05, (1, 64, int(image_size / 32), int(image_size / 32)))
    
    for n in range(key_frames):
        current_vector = np.add(current_vector, add_vector)
        gan.generate_image(generator, 'gif.jpg', noise=torch.Tensor(current_vector))
        images.append(imageio.imread('gif.jpg'))
            
    imageio.mimsave('output.gif', images, fps=fps)


def generate_gif(gan_name, gan_checkpoint, image_size=512, key_frames=4, interpolated_frames=30, fps=20):
    generator = models.PSGenerator()
    generator.load_state_dict(torch.load('models/' + gan_name + '/ps/' + str(gan_checkpoint) + '/generator.pt'))

    images = []
    current_vector = np.random.uniform(-1, 1, (1, 64, int(image_size / 32), int(image_size / 32)))
    start_vector = np.copy(current_vector)
    
    for n in range(key_frames):
        if n == key_frames-1:
            next_vector = start_vector
        else:
            next_vector = np.random.uniform(-1, 1, (1, 64, int(image_size / 32), int(image_size / 32)))
        difference = np.subtract(next_vector, current_vector)
        to_add = np.divide(difference, interpolated_frames)
        i = 0
        for i in range(interpolated_frames):
            gan.generate_image(generator, 'gif.jpg', noise=torch.Tensor(current_vector))
            images.append(imageio.imread('gif.jpg'))
            current_vector = np.add(current_vector, to_add)
            

    imageio.mimsave('output.gif', images, fps=fps)
    
if __name__ == '__main__':
    generate_gif_2('paint_wood', 356)