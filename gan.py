import os
import torch
import imageio
import numpy as np
import torchvision.transforms as transforms
import argparse
import torch.nn.functional as F
import models
import utils

device = torch.device('cuda') #Can change to cpu if you dont have CUDA

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Trains a GAN using PSGAN model
# Inputs:
#       - source_image:     image to use for training
#       - image_size:       size of images to train on
#       - generator_name:   models will be saved under this name
#       - iterations:       Total number of images fed trained on
#       - batch_size:       How many images to operate on at once
def train_ps_gan(source_image, image_size, generator_name, iterations=44000, batch_size=8, resume_from=None):
    generator = models.PSGenerator()
    discriminator = models.PSDiscriminator()
    dataloader = utils.get_image_dataloader(source_image, 2048, image_resize=8, batch_size=batch_size)
    params = {
        'dataloader':           dataloader,
        'generator':            generator.to(device),
        'discriminator':        discriminator.to(device),
        'generator_optim':      torch.optim.Adam(generator.parameters(), lr=5e-5, weight_decay=1e-8, betas=(0.5, 0.999)),
        'discriminator_optim':  torch.optim.Adam(discriminator.parameters(), lr=5e-5, weight_decay=1e-8, betas=(0.5, 0.999)),
        'image_size':           image_size,
        'batch_size':           batch_size,
        'save_path':            generator_name + '/ps/',
        'iterations':           iterations,
        'resume_from':          resume_from
    }
    train_GAN(**params)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   Trains a GAN using DCGAN model, image size is fixed to 64x64
# Inputs:
#       - source_image:     image to use for training
#       - generator_name:   models will be saved under this name
#       - iterations:       Total number of images fed trained on
#       - batch_size:       How many images to operate on at once
def train_dc_gan(source_image, generator_name, iterations=44000, batch_size=8):
    generator= models.DCGenerator(100,64,3).apply(utils.initialise_weights)
    discriminator = models.DCDiscriminator(64,3).apply(utils.initialise_weights)
    dataloader = utils.get_image_dataloader(source_image, 128, image_resize=2, batch_size=batch_size)
    params = {
        'dataloader':           dataloader,
        'generator':            generator.to(device),
        'discriminator':        discriminator.to(device),
        'generator_optim':      torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
        'discriminator_optim':  torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
        'image_size':           64,
        'batch_size':           batch_size,
        'save_path':            generator_name + '/dc/',
        'iterations':           iterations
    }
    train_GAN(**params)

def resume(file_path, generator, discriminator, generator_optim, discriminator_optim):
    generator.load_state_dict(torch.load(file_path + '/generator.pt'))
    discriminator.load_state_dict(torch.load(file_path + '/discriminator.pt'))
    generator_optim.load_state_dict(torch.load(file_path + '/generator_optimiser.pt'))
    discriminator_optim.load_state_dict(torch.load(file_path + '/generator_optimiser.pt'))
    return generator, discriminator, generator_optim, discriminator_optim


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Trains a GAN
# Inputs:
#       - dataloader:
#       - generator, discriminator:         The networks
#       - generator/discriminator_optim:    Optimisers for the networks
#       - image_size:                       Width and height of image in pixels
#       - batch_size:                       How many images are passed through the networks at once
#       - generator_name:                   Checkpoints will be saved under this name
#       - iterations:                       How many images are fed through the networks in total
def train_GAN(dataloader, generator, discriminator, generator_optim, discriminator_optim, image_size, batch_size, save_path, iterations, resume_from ):
    if not os.path.exists('models/' + save_path):
        os.makedirs('models/' + save_path)

    if resume_from is not None:
        generator, discriminator, generator_optim, discriminator_optim = resume(resume_from, generator, discriminator, generator_optim, discriminator_optim)

    iterations_until_checkpoint = 50 #Count how many iterations done
    total_iterations = 0
    saves = 0 #Count how many checkpoints saved
    epochs = int(iterations / len(dataloader.dataset))
    for i in range(epochs):
        for j, data in enumerate(dataloader, 0):
            
            data = data.to(device)
            iterations_until_checkpoint -= data.shape[0]
            total_iterations += data.shape[0]
            
            ## Train with all-real batch
            discriminator_optim.zero_grad() 
            output = discriminator(data) #Run real data through discriminator
            true_label = torch.full((output.shape), 1, device=device)
            real_discriminator_error = F.binary_cross_entropy(output, true_label) #Get error from result and target
            real_discriminator_error.backward() #Backpropogate

            ## Train with all-fake batch
            noise = generator.noise(batch_size, image_size).to(device)
            fake = generator(noise) #Generate fake images from noise with generator
            output = discriminator(fake.detach()) #Run fakes through discriminator
            false_label = torch.full((output.shape), 0, device=device)
            fake_discriminator_error = F.binary_cross_entropy(output, false_label) #Get error from result and target
            fake_discriminator_error.backward() #Backpropogate
            discriminator_optim.step()

            # Train generator
            generator_optim.zero_grad()
            output = discriminator(fake) #Run fakes through discriminator again
            true_label = torch.full((output.shape), 1, device=device)
            generator_error = F.binary_cross_entropy(output, true_label) #Get error from result and target
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
                noise = generator.noise(1, image_size).to(device)
                with torch.no_grad():
                    demonstrate_gan(generator, 'in_progress.jpg', noise)
                
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   Saves the models to 'models / generator_name / checkpoint_number' directory
#   Inputs:
#           - generator_name:                   all models are saved under this name
#           - checkpoint_number:                how many checkpoints have been saved so far
#           - generator, discriminator:         models to save
#           - gen_optimiser, disc_optimiser:    optimisers to save
#           - just_generator:                   if True, only the generator is saved to save space
def save_checkpoint(generator_name, checkpoint_number, generator, discriminator, gen_optimiser, disc_optimiser, just_generator):
    print('Saving models.. (' + str(checkpoint_number) + ')')
    directory = 'models/' + generator_name + '/' + str(checkpoint_number)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = 'models/' + generator_name + '/' + str(checkpoint_number) + '/'

    #Save generator
    torch.save(generator.state_dict(), file_path + 'generator.pt')
    if not just_generator:
        torch.save(discriminator.state_dict(), file_path + 'discriminator.pt')
        torch.save(gen_optimiser.state_dict(), file_path + 'generator_optimiser.pt')
        torch.save(disc_optimiser.state_dict(), file_path + 'discriminator_optimiser.pt')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Saves an image generated by using 'noise' as input to 'generator'
# Inputs:
#       - generator:    the generator to be demo'd
#       - filepath:     filepath of the generated image
#       - noise:        the noise vector to be used as generator input
def demonstrate_gan(generator, filepath, noise, transform=None):
    normalise = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
    result = generator(noise).detach().cpu()
    if result.shape[0] > 1:
        res1 = torch.cat((result[0], result[1]), 1)
        res2 = torch.cat((result[2], result[3]), 1)
        result = torch.cat((res1, res2), 2) #Make grid of identical noise
    else:
        result = result.squeeze()
    result = utils.to_pil_image(normalise(result))
    if transform is not None:
        result = transform(result)
    result.save(filepath, quality=60)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Saves an image generated by every checkpoint of a trained generator or just final checkpoint
# Input:
#       - noise:                    the noise vector to be used as input for the generator checpoints
#       - generator_save_directory: the base directory where all checkpoints are saved for the model
#       - generator:                the generator to use
#       - save_directory:           the directory in which to save the generated images
#       - image_size:               the size of the generated images
#       - final_only:               if True only one image is generated using final checkpoint
def visualise(noise, generator_save_directory, generator, save_directory, transform=None, final_only=True):
    iterations = 0
    for root, dirs, files in os.walk(generator_save_directory):
        iterations += len(dirs) #Count number of checkpoints in directory
    for i in range(iterations):
        if not final_only or i == iterations-1:
            generator.load_state_dict(torch.load(generator_save_directory + '/' + str(i+1) + '/generator.pt'))
            demonstrate_gan(generator, save_directory + '/' + str(i) + '.jpg', noise, transform)
def visualise_ps_gan(generator_name, save_directory, image_size, tile=False, final_only=True):
    noise = ps_noise(1, image_size, tile=tile)
    transform = None
    if tile:
        transform = transforms.CenterCrop((image_size, image_size))
    generator = models.PSGenerator().to(device)
    visualise(noise, 'models/' + generator_name + '/ps', generator, save_directory, transform, final_only)
def visualise_dc_gan(generator_name, save_directory, final_only=True):
    generator = models.DCGenerator(100, 64, 3).to(device)
    noise = dc_noise(4, 64)
    visualise(noise, 'models/' + generator_name + '/dc', generator, save_directory, None, final_only)


if __name__ == '__main__':
    #generator = models.PSGenerator().to(device)
    #generator.load_state_dict(torch.load('models/kilburn/ps/383/generator.pt'))
    #noise = ps_noise(1, 512, True)
    #transform = transforms.CenterCrop((512, 512))
    #demonstrate_gan(generator, 'yee.jpg', noise, transform=transform)
    # visualise_ps_gan('pebbles', 'images/boop', 256, tile=True)
    train_ps_gan('textures/kilburn.jpg', 128, 'kilburn', batch_size=64, iterations=120000)
     #images = ['snake', 'pebbles', 'water', 'lava', 'camo', 'painting', 'check']#,'bricks']
    #for image in images:
    #    print(image)
        #train_dc_gan('textures/'+ image +'.jpg', 'dc_' + image)
    #    visualise_dc_gan('dc_' + image, 'images/' + image + '/dc')



'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate texture using gatys et al method.')
    parser.add_argument('source',  help='the source image for texture style')
    parser.add_argument('target',  help='the filename for the created image')
    parser.add_argument('--lr', nargs='?', const=0.8, default=0.8, type=float, help='the learning rate for the optimiser')
    parser.add_argument('--iter', nargs='?' , const=150, default=150, type=int, help='the number of iterations')
    parser.add_argument('--tile', nargs='?' , const=False, default=False, type=bool, help='make the resulting texture tileable')
    args = parser.parse_args()
    generate_texture(args.source, args.target, args.lr, args.iter, tileable=args.tile, save_intermediates=False)'''

'''
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
    #result.save(saved_image_filepath, optimize=True,quality=60)
    result.save(saved_image_filepath, optimize=True,quality=60)'''