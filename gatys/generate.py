'''
Contains all the methods required for generating images as done by gatys et al.
Methods:
    generate_texture:       generate a texture image from a source image by optimising on the image
    gram_matrix:            calculates the gram matrix of a CNN feature layer
    gram_matrix_layers:     calculates the gram matrix for all layers of a CNN
    get_layer_sizes:        calculates the size of each layer of a CNN (how many parameters)
    get_style_loss:         calculates the style loss between two images by comparing the gram matrices of each
    get_feature_layers:     run an input image through a CNN and extract the feature layers
    tile_vertical:          split an image into randomly sized top and bottom 'halves' and switch the halves around
    tile_horizontal:        split an image into randomly sized left and right 'halves' and switch the halves around
'''
from random import randint
import os
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
from config import DEVICE, BASE_DIRECTORY

NORMALISE = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
DE_NORMALISE = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
STYLE_IMAGE_CROP = transforms.RandomCrop((256, 256))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Creates a texture image as done by gatys et al.
def generate_texture(source, learning_rate, iterations, image_size=124, tile=False):
    '''
    Generate a texture image from a source image by optimising on the image
    Args:
        source_image:   name of the source texture image
        learning_rate:  the learning rate used by the optimiser
        iterations:     the number of iterations to potimise the image for
        image_size:     width and height of the output image in pixels
        tile:       whether the finished image should be tileable
    '''
    source = os.path.join(BASE_DIRECTORY, 'textures', source)
    vgg16 = models.vgg16(pretrained=True).features.eval().to(DEVICE) #Load in pretrained CNN

    #Read in source image and noise image as Tensors and normalise
    imarray = np.random.rand(image_size, image_size, 3) * 255
    noise_image = NORMALISE(transforms.ToTensor()(Image.fromarray(imarray.astype('uint8')).convert('RGB'))).to(DEVICE)
    style_image = NORMALISE(transforms.ToTensor()(STYLE_IMAGE_CROP(Image.open(source).convert("RGB")))).to(DEVICE)

    #Get target gram matrixes 
    layers = [0, 5, 10, 19, 28]
    weights = [200, 100, 50, 25, 12]
    style_layers = get_feature_layers(style_image, vgg16, layers)
    source_image_grams = gram_matrix_layers(style_layers)
    layer_sizes = get_layer_sizes(style_layers)
    
    #Set up optimiser
    noise_image.requires_grad = True
    optimizer = torch.optim.Adam([noise_image], lr=learning_rate) #Set up the optimizer
    for i in range(iterations):
        print('Iteration %d / %d' % (i+1, iterations))

        noise_image.clamp(-1.5, 1.5)
        optimizer.zero_grad() #Reset all gradients for each iteration

        #Get the gram matrix's for the noise image
        if tile:
            noise_image_v = tile_vertical(noise_image) #Randomly tile the image vertically
            noise_image_h = tile_horizontal(noise_image_v) # and horizontally
            noise_layers = get_feature_layers(noise_image_h, vgg16, layers)   
        else:
            noise_layers = get_feature_layers(noise_image, vgg16, layers)


        #Get the gram matrix's for the noise image
        noise_grams = gram_matrix_layers(noise_layers)

        #Calculate the loss and backpropogate
        loss = get_style_loss(noise_grams, source_image_grams, layer_sizes, weights)
        loss.backward(retain_graph=True)
        optimizer.step()
    
    result = transforms.ToPILImage()(DE_NORMALISE(noise_image.cpu()).clamp(0, 1))
    result.save(os.path.join(BASE_DIRECTORY, 'output.jpg'))

def gram_matrix(layer):
    '''
    Calculates the gram matrix of a feature layer
    Args:
        layer:  a single feature layer of a CNN
    Returns:
        The gram matrix of the layer
    '''
    channels, height, width = layer.size()
    layer = layer.view(channels, height * width)
    gram = torch.mm(layer, layer.t())
    return gram

def gram_matrix_layers(layers):
    '''
    Calculates the gram matrix for all layers of a CNN
    Args:
        layers:     a list containing all feature layers of a CNN
    Returns:
        A list of gram matrices corresponding to the layers
    '''
    grams = []
    for layer in layers:
        gram = gram_matrix(layer)
        grams.append(gram)
    return grams

def get_layer_sizes(layers):
    '''
    Calculates the size of each layer of a CNN (how many parameters)
    Args:
        layers:     a list containing all feature layers of a CNN
    Returns:
        A list of integers representing the number of parameters in each layer
    '''
    sizes = []
    for layer in layers:
        channels, height, width = layer.size()
        sizes.append(channels*height*width)
    return sizes

def get_style_loss(grams_1, grams_2, layer_sizes, weights):
    '''
    Calculates the style loss between two images by comparing the gram matrices of each
    Args:
        grams_1:        a list containing the gram matrix of each layer in image 1
        grams_2:        a list containing the gram matrix of each layer in image 2
        layer_sizes:    the number of parameters in each layer
        weights:        weightings to determine how much each layer impacts the loss
    Returns:
        The style loss between the two images
    '''
    style_loss = 0
    for i in range(len(grams_1)): #for now
        style_loss += weights[i] * torch.mean((grams_1[i] - grams_2[i])**2) / layer_sizes[i]
    return style_loss

def get_feature_layers(input_image, cnn, layers_select):
    '''
    Run an input image through a CNN and extract the feature layers
    Args:
        input_image:    the image to run through the CNN 
        cnn:            the CNN to extract the feature layers from
        layers_select:  only the layers in layers_select will be extracted
    Returns:
        A list of feature layers
    '''
    feature_layers = []
    layer = input_image.unsqueeze(0)
    for i, module in enumerate(cnn):
        next_layer = module(layer)
        if i in layers_select:
            feature_layers.append(next_layer.squeeze(0))
        layer = next_layer
    return feature_layers

def tile_vertical(source):
    '''
    Split an image into randomly sized top and bottom 'halves' and switch the halves around
    Args:
        source:     the image to change
    Returns:
        A new image, the same size as the original
    '''
    image_size = source.shape[1]
    split_point = randint(1, source.shape[1]-1)
    top_half = source[ :, 0:split_point, : ]
    bottom_half = source[ :, split_point:image_size, : ]
    return torch.cat((bottom_half, top_half), 1)

def tile_horizontal(source):
    '''
    Split an image into randomly sized left and right 'halves' and switch the halves around
    Args:
        source:     the image to change
    Returns:
        A new image, the same size as the original
    '''
    image_size = source.shape[2]
    split_point = randint(1, source.shape[2]-1)
    left_half = source[:, :, 0:split_point]
    right_half = source[:, :, split_point:image_size]
    return torch.cat((right_half, left_half), 2)


