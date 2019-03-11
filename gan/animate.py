'''
Defines methods for creating animated texture GIFs using a pretrained generator model
Methods:
    generate_gif:                   generate a GIF from an array of input noise frames
    generate_sin_noise:             generate frames of input noise using a SIN curve 
    generate_interpolated_noise:    generate frames of input noise by interpolating between random vectors
    concatonate_noise               attaches a graphical representation of a noise vector to an image
'''
import os
import torch
import numpy as np
import torchvision.transforms as transforms
import PIL
from config import BASE_DIRECTORY
from .models import PSGenerator

def generate_sin_noise(generator, frames, image_size, tile=False):
    '''
    Generate a series of noise vectors which will create a smooth GIF. Each frame the noise vector is incremented and sin() of the vector is taken.
    Args:
        generator:      the generator used to get the input noise vector
        frames:         the number of frames the GIF will have
        image_size:     the height and width in pixels of the GIF
        tile:           whether the GIF should be tileable 
    Returns:
        A noise vector with first dimension being the frames 
    '''
    current_vector = generator.noise(1, image_size, tile=tile)
    add_vector = np.full_like(current_vector, 2 * np.pi/frames)
    noise_vectors = np.zeros((frames,current_vector.shape[1], current_vector.shape[2], current_vector.shape[3]))
    for n in range(frames):
        current_vector = np.add(current_vector, add_vector)
        noise_vectors[n] = np.sin(current_vector)
    return torch.Tensor(noise_vectors)

def generate_interpolated_noise(generator, frames, image_size, tile=False):
    '''
    Generate a series of noise vectors which will create a smooth GIF. Random vectors are interpolated between.
    Args:
        generator:      the generator used to get the input noise vector
        frames:         the number of frames the GIF will have
        image_size:     the height and width in pixels of the GIF
        tile:           whether the GIF should be tileable 
    Returns:
        A noise vector with first dimension being the frames 
    '''
    interpolated_frames = 20
    key_frames = int(frames / interpolated_frames)
    print(key_frames)
    current_vector = generator.noise(1, image_size, tile=tile)
    start_vector = np.copy(current_vector)
    noise_vectors = np.zeros((key_frames*interpolated_frames, current_vector.shape[1], current_vector.shape[2], current_vector.shape[3]))
    for n in range(key_frames):
        if n == key_frames-1:
            next_vector = start_vector
        else:
            next_vector = generator.noise(1, image_size, tile=tile)
        difference = np.subtract(next_vector, current_vector)
        to_add = np.divide(difference, interpolated_frames)
        for i in range(interpolated_frames):
            noise_vectors[(n*interpolated_frames) + i] = current_vector
            current_vector = np.add(current_vector, to_add)
    return torch.Tensor(noise_vectors)

def generate_gif(gan_name, gan_checkpoint=-1, noise_type='sin', image_size=512, frames=200, frame_duration=70, show_noise=False, tile=False):
    '''
    Generate an animated GIF using a pretrained generator and save it as output.gif
    Args:
        gan_name:           the name of the generator to use
        gan_checkpoint:     the training checkpoint of the GAN to use, (if -1, the most recent checkpoint is used)
        noise_type:         the method used to generate the noise for creating the GIF (either "sin" or "interpolated")
        image_size:         the width and height in pixels of the GIF
        frames:             how many frames will the GIF have
        frame_duration:     time in ms for each frame in the GIF
        show_noise:         whether a graphical representation of the noise vector should be included in the GIF
        tile:               whether the GIF should be tileable
    '''
    generator = PSGenerator()
    root_directory = BASE_DIRECTORY + '/models/' + gan_name + '/ps'
    if gan_checkpoint < 0:
        gan_checkpoint = 0
        for root, dirs, files in os.walk(root_directory):
            gan_checkpoint += len(dirs)
    generator.load_state_dict(torch.load(root_directory + '/' + str(gan_checkpoint) + '/generator.pt'))
    if noise_type == 'interpolated':
        noise_vectors = generate_interpolated_noise(generator, frames, image_size, tile)
    else:
        noise_vectors = generate_sin_noise(generator, frames, image_size, tile)
    images = []
    for n in range(len(noise_vectors)):
        print('Generating frame %d of %d' % (n+1, len(noise_vectors)))
        image_vector = generator(noise_vectors[n].unsqueeze(0)).detach()
        image = transforms.ToPILImage()(transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])(image_vector[0]))
        if tile:
            image = transforms.CenterCrop((image_size, image_size))(image)
        if show_noise:
            image = concatonate_noise(image, noise_vectors[n])
        images.append(image)
    images[0].save(BASE_DIRECTORY + '/output.gif', format='GIF', append_images=images[1:], save_all=True, duration=frame_duration, loop=0)

def concatonate_noise(image, noise):
    '''
    Concatonate an image with a noise vector to create a new image for demonstration purposes
    Args:
        image:      the image in PIL image form
        noise:      the noise vector (which created the image) as a Tensor
    Returns:
        A new image (twice the size) with the original image and a graphical representation of the noise vector
    '''
    image_size = image.height
    #Normalise the noise
    noise_tensor = torch.Tensor(noise)
    noise_tensor = np.add(noise_tensor, 1)
    noise_tensor *= 1.0/noise_tensor.max()  
    noise_tensor = noise_tensor.view((4, int(image_size/8), int(image_size/8)))
    noise_image = transforms.ToPILImage()(noise_tensor)
    noise_image = noise_image.resize((image_size,image_size), PIL.Image.NEAREST)   

    #Create new canvas
    new_image = PIL.Image.new('RGBA', (image_size * 2, image_size))
    new_image.paste(noise_image, (image_size, 0))
    new_image.paste(image, (0, 0))
    return new_image