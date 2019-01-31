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
import database

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Trains a Generational Adversarial Network on a source image
# The Generator weights are saved in the trained_models folder
def train_GAN(source_images, learning_rate, iterations, generator_name, resume=False, model_name=None, save_intermediates=True, single_image=False, image_size=256, batch_size=8, cuda=True ):
    database.save_GAN(generator_name, 0, learning_rate, image_size)
    if not os.path.exists('models/' + generator_name):
        os.makedirs('models/' + generator_name)

    if cuda:
        device = torch.device('cuda') #Can change to cpu if you dont have CUDA
    else:
        device = torch.device('cpu')

    #Initialise networks and optimisers
    discriminator = models.PSDiscriminator().to(device)
    generator = models.PSGenerator().to(device)
    generator_optim = torch.optim.Adam(generator.parameters(), lr=5e-5, weight_decay=1e-8, betas=(0.5, 0.999))
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=5e-5, weight_decay=1e-8, betas=(0.5, 0.999))


    #Create dataloader from source image/images
    if os.path.isfile(source_images):
        dataloader = utils.get_image_dataloader(source_images, image_size, batch_size=batch_size)
    else:
        dataloader = utils.get_folder_dataloader(source_images, image_size, batch_size=batch_size)

    #True label is array of 1's, False is array of 0's
    true_label = torch.ones(batch_size, int(image_size/32)*int(image_size/32)).to(device)
    false_label = torch.zeros(batch_size, int(image_size/32)*int(image_size/32)).to(device)

    iters = 0 #Count how many iterations done
    saves = 0 #Count how many checkpoints saved
    epochs = int(iterations / len(dataloader.dataset))
    for i in range(epochs):
        for j, data in enumerate(dataloader, 0):
            
            if not os.path.isfile(source_images):
                data = data[0]
            data = data.to(device)
            iters += data.shape[0]
            
            ## Train with all-real batch
            discriminator_optim.zero_grad() 
            output = discriminator(data) #Run real data through discriminator
            real_discriminator_error = F.binary_cross_entropy(output, true_label[:data.shape[0]]) #Get error from result and target
            real_discriminator_error.backward() #Backpropogate

            ## Train with all-fake batch
            noise = torch.Tensor(np.random.uniform(-1, 1, (batch_size, 64, int(image_size/32), int(image_size/32)))).to(device) #Generate batch of noise for input
            fake = generator(noise) #Generate fake images from noise with generator
            output = discriminator(fake.detach()) #Run fakes through discriminator
            fake_discriminator_error = F.binary_cross_entropy(output, false_label) #Get error from result and target
            fake_discriminator_error.backward() #Backpropogate

            discriminator_optim.step()

            # Train generator
            generator_optim.zero_grad()
            output = discriminator(fake) #Run fakes through discriminator again
            generator_error = F.binary_cross_entropy(output, true_label) #Get error from result and target
            generator_error.backward()#Backpropogate
            generator_optim.step()

            print('Iteration %d / %d, Image Batch %d / %d, Disc Loss R&F %f & %f , Gen Loss %f' % (i+1, epochs, j+1, len(dataloader), real_discriminator_error, fake_discriminator_error, generator_error))
        
            if (save_intermediates and iters > 500) or (i == epochs-1 and j == len(dataloader)-1):
                saves += 1
                iters -= 500
                save_checkpoint(generator_name, saves, generator, discriminator, generator_optim, discriminator_optim)
                

def save_checkpoint(generator_name, checkpoint_number, generator, discriminator, gen_optimiser, disc_optimiser):
    print('Saving models.. (' + str(checkpoint_number) + ')')
    os.makedirs('models/' + generator_name + '/' + str(checkpoint_number))
    file_path = 'models/' + generator_name + '/' + str(checkpoint_number) + '/'

    #Save generator
    torch.save(generator.state_dict(), file_path + 'generator.pt')
    if (checkpoint_number % 10) == 0: #Save all others every 10 checkpoints
        torch.save(discriminator.state_dict(), file_path + 'discriminator.pt')
        torch.save(gen_optimiser.state_dict(), file_path + 'generator_optimiser.pt')
        torch.save(disc_optimiser.state_dict(), file_path + 'discriminator_optimiser.pt')
    database.update_GAN(generator_name, checkpoint_number)
    demonstrate_GAN(file_path + 'generator.pt', 'temp/GAN_in_progress.jpg')



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Loads a given generator and generates an images
def demonstrate_GAN(generator_filepath, saved_image_filepath='temp/GAN_demo.jpg', image_size=128, tile=False, noise=None):
    center_crop = transforms.CenterCrop((image_size, image_size))
    normalise = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])

    generator = models.PSGenerator()
    generator.load_state_dict(torch.load(generator_filepath))
    if noise is None:
        noise_size = int(image_size / 32)
        noise = torch.Tensor(np.random.uniform(-1, 1, (64, noise_size, noise_size)))
    if tile:
        noise_array = torch.cat((noise, noise), 1)
        noise = torch.cat((noise_array, noise_array), 2) #Make grid of identical noise
    result = generator(noise.unsqueeze(0)).detach().squeeze()
    result = utils.to_pil_image(normalise(result))
    if tile:
        result = center_crop(result)
    result.save(saved_image_filepath)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Runs the same noise vector through every checkpoint of a generator
# To visualise how it improves over time
def visualise(generator_name, iterations, filepath='temp/gan.jpg', image_size=256, tile=False, save_gif=False):
    images = []
    input_size = int(image_size / 32)
    noise = torch.Tensor(np.random.uniform(-1, 1, (64, input_size, input_size)))
    for i in range(iterations):
        database.update_progress(int((100/iterations) * i) + 1, 'gan')
        demonstrate_GAN('models/' + generator_name + '/' + str(i+1) + '/generator.pt', saved_image_filepath=filepath, image_size=image_size, tile=tile, noise=noise)
        if save_gif:
            images.append(imageio.imread(filepath))
    if save_gif:
        imageio.mimsave('test.gif', images)

if __name__ == '__main__':
    #visualise('water', 92, image_size=256)
    train_GAN('textures/water.jpg', 0.00002, 140000, 'water', single_image=True)
    #train_GAN('textures/pebbles.jpg', 0.00002, 140000, 'pebbles', single_image=True)

    #train_GAN('textures/water_large.jpg', 0.00002, 400000, 'water', image_size=128, single_image=True)
    #demonstrate_GAN('models/PS_water_64_gen.pt', image_size=512, tile=True)