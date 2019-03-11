'''
Provides methods for training new GAN models
Methods:
    train_gan:          method for training new GAN models
    train_ps_gan:       calls train_gan to train a PSGAN model
    train_dc_gan:       calls train_gan to train a DCGAN model
    resume:             resumes model state from a saved checkpoint
    save_checkpoint:    saves model state
'''
import os
import torch
from torch.nn.functional import binary_cross_entropy
from config import BASE_DIRECTORY, DEVICE
from .models import PSGenerator, PSDiscriminator, DCGenerator, DCDiscriminator, initialise_weights
from .data_loading import get_image_dataloader
from .generate import generate_image

def train_ps_gan(source_image, generator_name, image_size=256, iterations=44000, batch_size=8, resume_from=None):
    '''
    Train a new PSGAN model using random crops of a source image as the training data
    Args:
        source_image:       the image to train the model on
        generator_name:     unique name for the model (forms part of the save directory path)
        image_size:         size of images to be trained on
        iterations:         total number of images trained on
        batch_size:         number of images in each training batch
        resume_from:        directory path to checkpoints to resume from (if None no resume is done)
    '''
    generator = PSGenerator()
    discriminator = PSDiscriminator()
    dataloader = get_image_dataloader(BASE_DIRECTORY + '/textures/' + source_image, image_size, batch_size=batch_size)
    params = {
        'dataloader':           dataloader,
        'generator':            generator.to(DEVICE),
        'discriminator':        discriminator.to(DEVICE),
        'generator_optim':      torch.optim.Adam(generator.parameters(), lr=5e-5, weight_decay=1e-8, betas=(0.5, 0.999)),
        'discriminator_optim':  torch.optim.Adam(discriminator.parameters(), lr=5e-5, weight_decay=1e-8, betas=(0.5, 0.999)),
        'save_path':            generator_name + '/ps/',
        'iterations':           iterations,
        'resume_from':          resume_from
    }
    train_GAN(**params)

def train_dc_gan(source_image, generator_name, iterations=44000, batch_size=8, resume_from=None):
    '''
    Train a new PSGAN model using random crops of a source image as the training data
    Args:
        source_image:       the image to train the model on
        generator_name:     unique name for the model (forms part of the save directory path)
        iterations:         total number of images trained on
        batch_size:         number of images in each training batch
        resume_from:        directory path to checkpoints to resume from (if None no resume is done)
    '''
    generator = DCGenerator(100, 64, 3).apply(initialise_weights)
    discriminator = DCDiscriminator(64, 3).apply(initialise_weights)
    #dataloader = get_image_dataloader(source_image, 128, image_resize=2, batch_size=batch_size)
    dataloader = get_image_dataloader(BASE_DIRECTORY + '/textures/' + source_image, 64, batch_size=batch_size)
    params = {
        'dataloader':           dataloader,
        'generator':            generator.to(DEVICE),
        'discriminator':        discriminator.to(DEVICE),
        'generator_optim':      torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
        'discriminator_optim':  torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
        'save_path':            generator_name + '/dc/',
        'iterations':           iterations,
        'resume_from':          resume_from
    }
    train_GAN(**params)

def resume(file_path, generator, discriminator, generator_optim, discriminator_optim):
    '''
    Resume models and optimisers to previous saved state
    Args:
        file_path:              the path to the directory containing the checkpoints
        generator, disc...:     the models and optimisers to be resumed
    Returns:
        generator, disc...:     the models and optimisers, now with the resumed state
    '''
    generator.load_state_dict(torch.load(file_path + '/generator.pt'))
    discriminator.load_state_dict(torch.load(file_path + '/discriminator.pt'))
    generator_optim.load_state_dict(torch.load(file_path + '/generator_optimiser.pt'))
    discriminator_optim.load_state_dict(torch.load(file_path + '/generator_optimiser.pt'))
    return generator, discriminator, generator_optim, discriminator_optim


def train_GAN(dataloader, generator, discriminator, generator_optim, discriminator_optim, save_path, iterations, resume_from ):
    '''
    Train a new GAN model
    Args:
        dataloader:             a dataloader supplying the images to train the GAN on
        generator:              a generator model
        discriminator:          a discriminator model
        generator_optim:        an optimiser for training the generator
        discriminator_optim:    an optimiser for training the discriminator
        save_path:              root directory to save all model checkpoints
        iterations:             total number of images trained on
        resume_from:            directory path to checkpoints to resume from (if None no resume is done)
    '''
    if resume_from is not None:
        generator, discriminator, generator_optim, discriminator_optim = resume(resume_from, generator, discriminator, generator_optim, discriminator_optim)

    batch_size = dataloader.batch_size
    image_size = dataloader.dataset.image_size
    iterations_until_checkpoint = 50 #Count how many iterations done
    total_iterations = 0
    saves = 0 #Count how many checkpoints saved
    epochs = int(iterations / len(dataloader.dataset))
    for i in range(epochs):
        for j, data in enumerate(dataloader, 0):
            
            data = data.to(DEVICE)
            iterations_until_checkpoint -= data.shape[0]
            total_iterations += data.shape[0]
            
            ## Train with all-real batch
            discriminator_optim.zero_grad() 
            output = discriminator(data) #Run real data through discriminator
            true_label = torch.full((output.shape), 1, device=DEVICE)
            real_discriminator_error = binary_cross_entropy(output, true_label) #Get error from result and target
            real_discriminator_error.backward() #Backpropogate

            ## Train with all-fake batch
            noise = generator.noise(batch_size, image_size).to(DEVICE)
            fake = generator(noise) #Generate fake images from noise with generator
            output = discriminator(fake.detach()) #Run fakes through discriminator
            false_label = torch.full((output.shape), 0, device=DEVICE)
            fake_discriminator_error = binary_cross_entropy(output, false_label) #Get error from result and target
            fake_discriminator_error.backward() #Backpropogate
            discriminator_optim.step()

            # Train generator
            generator_optim.zero_grad()
            output = discriminator(fake) #Run fakes through discriminator again
            true_label = torch.full((output.shape), 1, device=DEVICE)
            generator_error = binary_cross_entropy(output, true_label) #Get error from result and target
            generator_error.backward()#Backpropogate
            generator_optim.step()

            print('Iteration %d / %d, Image Batch %d / %d, Disc Loss R&F %f & %f , Gen Loss %f' % (i+1, epochs, j+1, len(dataloader), real_discriminator_error, fake_discriminator_error, generator_error))
            if (iterations_until_checkpoint < 0) or (i == epochs-1 and j == len(dataloader)-1):
                saves += 1
                if total_iterations < 10000:
                    iterations_until_checkpoint += 50
                elif total_iterations < 20000:
                    iterations_until_checkpoint += 150
                else:
                    iterations_until_checkpoint += 300
                if (i == epochs-1 and j == len(dataloader)-1) or saves % 50 == 0:
                    save_checkpoint(save_path, saves, generator, discriminator, generator_optim, discriminator_optim, just_generator=False)
                else: #Save only generator unless its final iteration
                    save_checkpoint(save_path, saves, generator, discriminator, generator_optim, discriminator_optim, just_generator=True)
                noise = generator.noise(1, image_size).to(DEVICE)
                with torch.no_grad():
                    image = generate_image(generator, noise)
                    image.save(BASE_DIRECTORY + '/in_progress.jpg')

def save_checkpoint(model_name, checkpoint_number, generator, discriminator, gen_optimiser, disc_optimiser, just_generator=False):
    '''
    Saves a checkpoint of the GAN models and optimisers.
    Checkpoints are saved to models/[model_name]/[checkpoint_number]/[component].pt
    Args:
        model_name:         the name given to the GAN being trained
        checkpoint_number:  iterated every time a checkpoint is saved
        generator:          generator model
        discriminator:      discriminator model
        gen_optimiser:      optimiser for the generator
        disc_optimiser:     optimiser for the discriminator
        just_generator:     if True, only the generator will be saved
    '''
    print('Saving models.. (' + str(checkpoint_number) + ')')
    directory = BASE_DIRECTORY + '/models/' + model_name + '/' + str(checkpoint_number)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(generator.state_dict(), directory + '/generator.pt')
    if not just_generator:
        torch.save(discriminator.state_dict(), directory + '/discriminator.pt')
        torch.save(gen_optimiser.state_dict(), directory + '/generator_optimiser.pt')
        torch.save(disc_optimiser.state_dict(), directory + '/discriminator_optimiser.pt')
