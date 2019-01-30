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
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

def loss_helper(x, m):
    return torch.sum((x - 1) ** 2) / m

def loss_helper1(x, m):
    return torch.sum(x ** 2) / m

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Cycle GAN implementation based on 
# http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf
def train_GAN(learning_rate, iterations,cycle_consistency=False):
    device = torch.device('cuda')
    image_size = 32
    batch_size = 16
    transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    #Get dataloader for each set of images
    dataset_A = datasets.ImageFolder('emojis/Apple', transform)
    dataset_B = datasets.ImageFolder('emojis/Windows', transform)
    dataloader_A_og = DataLoader(dataset=dataset_A, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_B_og = DataLoader(dataset=dataset_B, batch_size=batch_size, shuffle=True, num_workers=4)

    #Define the networks
    generator_A2B = utils.CycleGenerator(32).to(device) #Transforms images from domain A to domain B
    generator_B2A = utils.CycleGenerator(32).to(device) #Transforms images from domain B to domain A
    discriminator_A = utils.DCDiscriminator(32).to(device) #Identifies images from domain A
    discriminator_B = utils.DCDiscriminator(32).to(device) #Identifies images from domain B

    #Define the optimisers
    generator_params = list(generator_A2B.parameters()) + list(generator_B2A.parameters())  # Get generator parameters
    discriminator_params = list(discriminator_A.parameters()) + list(discriminator_B.parameters())  # Get discriminator parameters
    g_optimizer = optim.Adam(generator_params, learning_rate, [0.5, 0.999])
    d_optimizer = optim.Adam(discriminator_params, learning_rate, [0.5, 0.999])

    generator_A2B.load_state_dict(torch.load('models/_A2B_gen.pt'))
    generator_B2A.load_state_dict(torch.load('models/_B2A_gen.pt'))
    discriminator_A.load_state_dict(torch.load('models/_A_disc.pt'))
    discriminator_B.load_state_dict(torch.load('models/_B_disc.pt'))
    g_optimizer.load_state_dict(torch.load('models/_gen_optimiser.pt'))
    d_optimizer.load_state_dict(torch.load('models/_disc_optimiser.pt'))

    for i in range(iterations):

        #Create iterable dataloaders each epoch
        dataloader_A = iter(dataloader_A_og)
        dataloader_B = iter(dataloader_B_og)

        for j in range( 0, min( len(dataloader_A), len(dataloader_B) ) ): #Loop over dataloaders

            data_A = next(dataloader_A)[0].to(device).squeeze() #Batch from domain A
            data_B = next(dataloader_B)[0].to(device).squeeze() #Batch from domain B
            if not data_A.shape == data_B.shape:
                break

            #Train discriminators on real images
            d_optimizer.zero_grad()
            output_A = discriminator_A.forward(data_A) #Run real A data thru disc A
            output_B = discriminator_B.forward(data_B) #Run real B data thru disc B
            real_d_error_A = loss_helper(output_A, batch_size) #Get discriminator errors
            real_d_error_B = loss_helper(output_B, batch_size)
            real_d_error = real_d_error_A + real_d_error_B
            real_d_error.backward() 
            d_optimizer.step()

            #Train discriminators on fake images
            d_optimizer.zero_grad()
            fake_B = generator_A2B.forward(data_A) #Generate fake A images
            fake_A = generator_B2A.forward(data_B) #Generate fake B images
            output_A = discriminator_A.forward(fake_A) #Run fake A data thru disc A
            output_B = discriminator_B.forward(fake_B) #Run fake B data thru disc B
            fake_d_error_A = loss_helper1(output_A, batch_size) #Get discriminator errors
            fake_d_error_B = loss_helper1(output_B, batch_size)
            fake_d_error = fake_d_error_A + fake_d_error_B
            fake_d_error.backward()
            d_optimizer.step()

            #Train generator A2B
            g_optimizer.zero_grad()
            fake_B = generator_A2B.forward(data_A) #Generate fake B images
            output = discriminator_B.forward(fake_B) #Run fake B images thru disc B
            gen_B_error = loss_helper(output, batch_size) #Get generator error
            if cycle_consistency:
                reconstructed_A = generator_B2A.forward(fake_B) #Regenerate A images from fake B images
                difference = data_A - reconstructed_A #Get difference with original A images
                reconstructed_A_error = loss_helper1(difference, batch_size) #Get generator error
                gen_B_error += reconstructed_A_error
            gen_B_error.backward()
            g_optimizer.step()

            #Train generator B2A
            g_optimizer.zero_grad()
            fake_A = generator_B2A.forward(data_B) #Generate fake A images
            output = discriminator_A.forward(fake_A) #Run fake A images thru disc A
            gen_A_error = loss_helper(output, batch_size) #Get generator error
            if cycle_consistency:
                reconstructed_B = generator_A2B.forward(fake_A) #Regenerate B images from fake A images
                difference = data_B - reconstructed_B #Get difference with original B images
                reconstructed_B_error = loss_helper1(difference, batch_size) #Get generator error
                gen_A_error += reconstructed_B_error
            gen_A_error.backward()
            g_optimizer.step()

            if j % 20 == 0:
                print('Iteration [{:5d}/{:5d}] | Disc Real Loss: {:6.4f} | Disc Fake Loss: {:6.4f} | Gen A2B Loss: {:6.4f} | '
                      'Gen B2A Loss: {:6.4f} '.format(
                        i, iterations, real_d_error,
                        fake_d_error, gen_B_error, gen_A_error))
    
        toPILImage = transforms.ToPILImage()
        #Data A
        dataloader_A = iter(dataloader_A_og)
        data_A = next(dataloader_A)[0].to(device)
        fake_B = generator_A2B(data_A).cpu()
        current_images = toPILImage(vutils.make_grid(fake_B[:16], nrow=4, padding=1, normalize=True).cpu())
        current_images.save('bit1.png')
        data_A = data_A.cpu()
        current_images = toPILImage(vutils.make_grid(data_A[:16], nrow=4, padding=1, normalize=True).cpu())
        current_images.save('bit0.png')

        #Data B
        dataloader_B = iter(dataloader_B_og)
        data_B = next(dataloader_B)[0].to(device)
        fake_A = generator_B2A(data_B).cpu()
        current_images = toPILImage(vutils.make_grid(fake_A[:16], nrow=4, padding=1, normalize=True).cpu())
        current_images.save('bit2.png')
        data_B = data_B.cpu()
        current_images = toPILImage(vutils.make_grid(data_B[:16], nrow=4, padding=1, normalize=True).cpu())
        current_images.save('bit3.png')

        if (i % 20) == 0:

            torch.save(generator_B2A.state_dict(), 'models/_B2A_gen.pt')
            torch.save(generator_A2B.state_dict(), 'models/_A2B_gen.pt')
            torch.save(discriminator_A.state_dict(), 'models/_A_disc.pt')
            torch.save(discriminator_B.state_dict(), 'models/_B_disc.pt')
            torch.save(g_optimizer.state_dict(), 'models/_gen_optimiser.pt')
            torch.save(d_optimizer.state_dict(), 'models/_disc_optimiser.pt')



if __name__ == "__main__":          
    train_GAN(0.0003, 6000)