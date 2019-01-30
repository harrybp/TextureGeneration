import os
import math
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import numpy as np

UNIFOMR_RANGE_MIN = -1.0
UNIFOMR_RANGE_MAX = 1.0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Trains a Generational Adversarial Network on a source image
# The Generator weights are saved in the trained_models folder
def train_GAN(source_images, learning_rate, iterations, generator_name, resume=False, model_name=None, save_intermediates=True, single_image=False, image_size=256 ):
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('models/' + generator_name):
        os.makedirs('models/' + generator_name)
    print(generator_name)
    device = torch.device('cuda') #Can change to cpu if you dont have CUDA
    criterion = nn.BCELoss()
    batch_size = 8

    #Initialise networks and optimisers
    discriminator = models.PSDiscriminator().to(device)
    generator = models.PSGenerator().to(device)
    optimizerG = torch.optim.Adam(generator.parameters(), lr=5e-5, weight_decay=1e-8, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=5e-5, weight_decay=1e-8, betas=(0.5, 0.999))

    #Create dataset and dataloader from source image
    if single_image:
        dataloader = utils.get_image_dataloader(source_images, image_size, batch_size=batch_size)
    else:
        dataloader = utils.get_folder_dataloader(source_images, image_size, batch_size=batch_size)

    epochs = int(iterations / len(dataloader.dataset))

    toPILImage = transforms.ToPILImage()
    true_label = torch.ones(batch_size, int(image_size/32)*int(image_size/32)).to(device)
    false_label = torch.zeros(batch_size, int(image_size/32)*int(image_size/32)).to(device)
    iters = 0
    saves = 0
    for i in range(epochs):
        for j, data in enumerate(dataloader, 0):
            
            if not single_image:
                data = data[0]
            data = data.to(device)
            iters += data.shape[0]
            
            ## Train with all-real batch
            optimizerD.zero_grad() #AM I SURE???
            output = discriminator(data) #Run real data through discriminator
            real_discriminator_error = F.binary_cross_entropy(output, true_label[:data.shape[0]]) #Get error from result and target
            real_discriminator_error.backward() #Backpropogate

            ## Train with all-fake batch
            noise = torch.Tensor(np.random.uniform(-1, 1, (batch_size, 64, int(image_size/32), int(image_size/32)))).to(device) #Generate batch of noise for input
            fake = generator(noise) #Generate fake images from noise with generator
            output = discriminator(fake.detach()) #Run fakes through discriminator
            fake_discriminator_error = F.binary_cross_entropy(output, false_label) #Get error from result and target
            fake_discriminator_error.backward() #Backpropogate

            optimizerD.step()

            # Train generator
            optimizerG.zero_grad()
            output = discriminator(fake) #Run fakes through discriminator again
            generator_error = F.binary_cross_entropy(output, true_label) #Get error from result and target
            generator_error.backward()#Backpropogate
            optimizerG.step()

            print('Iteration %d / %d, Image Batch %d / %d, Disc Loss R&F %f & %f , Gen Loss %f' % (i+1, epochs, j+1, len(dataloader), real_discriminator_error, fake_discriminator_error, generator_error))
        
            if (save_intermediates and iters > 500) or (i == epochs-1 and j == len(dataloader)-1):
                saves += 1
                print('Saving models.. (' + str(iters) + ',' + str(saves) + ')')
                iters -= 500
                os.makedirs('models/' + generator_name + '/' + str(saves))
                torch.save(generator.state_dict(), 'models/' + generator_name + '/' + str(saves) + '/generator.pt')
                if (saves % 10) == 0:
                    torch.save(discriminator.state_dict(), 'models/' + generator_name + '/' + str(saves) + '/discriminator.pt')
                    torch.save(optimizerG.state_dict(), 'models/' + generator_name + '/' + str(saves) + '/generator_optimiser.pt')
                    torch.save(optimizerD.state_dict(), 'models/' + generator_name + '/' + str(saves) + '/discriminator_optimiser.pt')
                utils.save_GAN(generator_name, source_images, saves, learning_rate, 256, 256)
                with torch.no_grad():
                        current_images = generator(noise[:4]).detach().cpu() #Get image batch
                        current_images = toPILImage(vutils.make_grid(current_images[:4], nrow=2, padding=1, normalize=True).cpu())
                        current_images.save('GAN_in_progress.jpg')

from PIL import Image
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Loads a given generator and generates a batch of images
# Images saved into GAN_result.jpg
def demonstrate_GAN(generator_filepath, filepath='temp/GAN_demo', image_size=128, tile=False, noise=None):
    print(filepath)
    toPILImage = transforms.ToPILImage()
    centerCrop = transforms.CenterCrop((image_size,image_size))
    normalise = transforms.Normalize( mean=[-1, -1, -1], std=[2, 2, 2])
    generator = models.PSGenerator()
    generator.load_state_dict(torch.load(generator_filepath))
    input_size = int(image_size / 32)
    if noise is None:
        noise = torch.Tensor(np.random.uniform(-1, 1, (64, input_size, input_size)))#Generate batch of noise for input
    if tile:
        noise_array = torch.cat((noise, noise), 1)
        noise = torch.cat((noise_array, noise_array), 2) #Make grid of identical noise
    result = generator(noise.unsqueeze(0)).detach().squeeze()
    result = toPILImage(normalise(result))
    if tile:
        result = centerCrop(result)
    result.save(filepath + '.jpg')
    print('saved as ' + filepath + '.jpg')
        

import imageio

def visualise(generator_name, iterations, filepath='temp/gan/gan', image_size=256, tile=False):
    if not os.path.exists('temp/GAN_demo'):
        os.makedirs('temp/GAN_demo')
    images = []        
    input_size = int(image_size / 32)
    noise = torch.Tensor(np.random.uniform(-1, 1, (64, input_size, input_size))) #Generate batch of noise for input
    for i in range(iterations):
        utils.update_progress_gan(int( (100/iterations) * i ) + 1)
        #filepath = 'temp/GAN_demo/gif'+str(i)
        demonstrate_GAN('models/' + generator_name + '/' + str(i+1) + '/generator.pt', filepath=filepath, image_size=image_size, tile=tile, noise=noise)
        images.append(imageio.imread(filepath + '.jpg'))
    #imageio.mimsave('test.gif', images)


    




if __name__ == '__main__':
    visualise('water', 92, image_size=256)
    #train_GAN('textures/water.jpg', 0.00002, 140000, 'water', single_image=True)
    #train_GAN('textures/pebbles.jpg', 0.00002, 140000, 'pebbles', single_image=True)

    #train_GAN('textures/water_large.jpg', 0.00002, 400000, 'water', image_size=128, single_image=True)
    #demonstrate_GAN('models/PS_water_64_gen.pt', image_size=512, tile=True)