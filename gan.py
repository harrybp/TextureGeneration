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
#       - resume_from:      Path to a folder containing checkpoints to resume from, (or None)
def train_ps_gan(source_image, image_size, generator_name, iterations=44000, batch_size=8, image_resize=1, resume_from=None):
    generator = models.PSGenerator()
    discriminator = models.PSDiscriminator()
    dataloader = utils.get_image_dataloader(source_image, image_size, image_resize=image_resize, batch_size=batch_size)
    params = {
        'dataloader':           dataloader,
        'generator':            generator.to(device),
        'discriminator':        discriminator.to(device),
        'generator_optim':      torch.optim.Adam(generator.parameters(), lr=5e-5, weight_decay=1e-8, betas=(0.5, 0.999)),
        'discriminator_optim':  torch.optim.Adam(discriminator.parameters(), lr=5e-5, weight_decay=1e-8, betas=(0.5, 0.999)),
        'image_size':           image_size / image_resize,
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   Generate an image using pretrained ps_gan
#       If checkpoint = -1, most recent checkpoint is used
def demo_ps_gan(name, image_size=256, checkpoint=-1, tile=False):
    transform = None
    if tile:
        transform = transforms.CenterCrop((image_size, image_size))
    root_directory = 'models/' + name + '/ps'
    if checkpoint < 0:
        checkpoint = 0
        for root, dirs, files in os.walk(root_directory):
            checkpoint += len(dirs)
    root_directory = root_directory + '/' + str(checkpoint)
    generator = models.PSGenerator().to(device)
    generator.load_state_dict(torch.load(root_directory + '/generator.pt'))
    noise = generator.noise(1, image_size, tile=tile).to(device)
    generate_image(generator, 'result.jpg', noise, transform=transform)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   Generate an image using a pretrained dc_gan
#       If checkpoint = -1, most recent checkpoint is used
def demo_dc_gan(name, checkpoint=-1):
    root_directory = 'models/' + name + '/dc'
    if checkpoint < 0:
        checkpoint = 0
        for root, dirs, files in os.walk(root_directory):
            checkpoint += len(dirs)
    root_directory = root_directory + '/' + str(checkpoint)
    generator = models.DCGenerator(100,64,3).to(device)
    generator.load_state_dict(torch.load(root_directory + '/generator.pt'))
    noise = generator.noise(4, 64).to(device)
    generate_image(generator, 'result.jpg', noise)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Resume models and optimisers from specified folder 
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
                    generate_image(generator, 'in_progress.jpg', noise)
                
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
def generate_image(generator, filepath, noise, transform=None):
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Train and demonstrate GANs for texture generation.")
    subparsers = parser.add_subparsers(title="actions", help="train new GAN or demo existing GAN", dest='action')

    #Train new GAN
    parser_train = subparsers.add_parser("train", help = "train new GAN")
    parser_train.add_argument('model',  help='GAN model to use, either "ps" or "dc"')
    parser_train.add_argument('source',  help='path to the source image')
    parser_train.add_argument('name',  help='checkpoints will be saved under this name')
    parser_train.add_argument('--image_size', nargs='?', const=256, default=256, type=int,   help='size of image to train on')
    parser_train.add_argument('--batch_size', nargs='?', const=8, default=8, type=int, help='how many images to train on concurrently')
    parser_train.add_argument('--iterations', nargs='?' , const=44000, default=44000, type=int, help='the number of iterations over the training data')

    #Demo GAN
    parser_update = subparsers.add_parser ("demo", help = "demo existing")
    parser_update.add_argument('model',  help='GAN model to use, either "ps" or "dc"')
    parser_update.add_argument('source',  help='the name of the trained GAN to demo')
    parser_update.add_argument('--image_size', nargs='?', const=256, default=256, type=int, help='size of the generated image (where applicable)')
    parser_update.add_argument('--checkpoint', nargs='?', const=-1, default=-1, type=int, help='choose a specific saved checkpoint')
    parser_update.add_argument('--tile', nargs='?', const=False, default=False, type=bool, help='True if the generated image should be tileable (where applicable)')
    args = parser.parse_args()

    if args.action == 'train':
        print('Model %s, Src %s, Name %s, size %d, batch %d, iters %d' % (args.model, args.source, args.name, args.image_size, args.batch_size, args.iterations))
        if args.model == 'ps':
            gan.train_ps_gan(args.source, args.image_size, args.name, args.iterations, args.batch_size)
        elif args.model == 'dc':
            gan.train_dc_gan(args.source, args.name, args.iterations, args.batch_size)
        else:
            print('Please select a valid GAN model (ps/dc)')
    elif args.action == 'demo':
        if args.model == 'ps':
            gan.demo_ps_gan(args.source, args.image_size, args.checkpoint, args.tile)
        elif args.model == 'dc':
            gan.train_dc_gan(args.source, args.checkpoint)
        else:
            print('Please select a valid GAN model (ps/dc)')
