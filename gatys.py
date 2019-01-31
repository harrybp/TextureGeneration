import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import utils
import PIL.Image, PIL.ImageTk
import sys
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import database


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Creates a texture image as done by gatys et al.
def generate_texture(source, target, learning_rate, iterations, image_size=128, tileable=False, save_intermediates=True, cuda=True):
    print('Generating texture file "%s" from source file "%s"' % (target, source))
    database.update_progress(0, 'gatys')
    

    #Define gatys-specific transforms
    vgg_de_normalise = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    process_image = transforms.Compose([
        transforms.RandomCrop((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) #Normalised same as images vgg was trained on
    ])

    if cuda:
        device = torch.device('cuda') #Can change to cpu if you dont have device
    else:
        device = torch.device('cpu')

    vgg16 = models.vgg16(pretrained=True).features.eval().to(device) #Load in pretrained CNN
    #Read in source image and noise image as Tensors and normalise
    style_image = process_image(PIL.Image.open(source).convert("RGB")).to(device)
    noise = np.random.rand(image_size,image_size,3) * 255
    noise_image = process_image(PIL.Image.fromarray(noise.astype('uint8')).convert('RGB')).to(device)

    #Get target gram matrixes 
    layers = [0,5,10,19,28]
    weights = [10000, 128, 32, 4, 1]
    style_layers = get_feature_layers(style_image, vgg16, layers)
    source_image_grams = gram_matrix_layers(style_layers)
    layer_sizes = get_layer_sizes(style_layers)
    
    #Set up optimiser
    noise_image.requires_grad = True
    optimizer = torch.optim.Adam([noise_image], lr=learning_rate) #Set up the optimizer
    
    for i in range(iterations):
        sys.stdout.write('\rIteration %d / %d' % (i+1, iterations))
        sys.stdout.flush()
        
        noise_image.clamp(-1.5, 1.5)
        optimizer.zero_grad() #Reset all gradients for each iteration

        #Get the gram matrix's for the noise image
        if tileable:
            noise_image_v = utils.tile_vertical(noise_image) #Randomly tile the image vertically
            noise_image_h = utils.tile_horizontal(noise_image_v) # and horizontally
            noise_layers = get_feature_layers(noise_image_h, vgg16, layers)   
        else:
            noise_layers = get_feature_layers(noise_image, vgg16, layers)
        noise_grams = gram_matrix_layers(noise_layers)

        #Calculate the loss and backpropogate
        loss = get_style_loss(noise_grams, source_image_grams, layer_sizes, weights)
        loss.backward(retain_graph=True)
        optimizer.step()

        if save_intermediates or i == iterations-1:
            current_image = utils.to_pil_image(vgg_de_normalise(noise_image.cpu()).clamp(0, 1))
            current_image.save('temp/' +  target  +'.jpg')
            database.update_progress(int( (100/iterations) * i ) + 1, 'gatys')

#Calculate the gram matrix of a single layer
def gram_matrix(layer):
    c, h, w = layer.size()
    layer = layer.view(c, h * w)
    gram = torch.mm(layer, layer.t())
    return gram

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Calculate the gram matrix for each layer of a cnn
def gram_matrix_layers(layers):
    targets = []
    for i in range(len(layers)):
        gram = gram_matrix(layers[i])
        targets.append(gram)
    return targets

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Get the number of variables in each layer
def get_layer_sizes(layers):
    sizes = []
    for i, layer in enumerate(layers):
        c, h, w = layer.size()
        sizes.append(c*h*w)
    return sizes

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Get style loss by comparing two Gram Matrix's
def get_style_loss(grams1, grams2, layer_sizes, weights):
    style_loss = 0
    for i in range(len(grams1)): #for now
        gram1 = grams1[i]
        gram2 = grams2[i]
        style_loss += weights[i] * torch.mean((gram1 - gram2)**2) / layer_sizes[i]
    return style_loss

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Run an input image through the cnn and extract feature map from each layer
def get_feature_layers(input, cnn, layers_select):
    features = []
    prev_feature = input.unsqueeze(0)
    for i, module in enumerate(cnn):
        feature = module(prev_feature)
        if(i in layers_select):
            features.append(feature.squeeze(0))
        prev_feature = feature
    return features

'''if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate texture using gatys et al method.')
    parser.add_argument('source',  help='the source image for texture style')
    parser.add_argument('target',  help='the filename for the created image')
    parser.add_argument('--lr', nargs='?', const=0.8, default=0.8, type=float, help='the learning rate for the optimiser')
    parser.add_argument('--iter', nargs='?' , const=400, default=400, type=int, help='the number of iterations')
    parser.add_argument('--tile', nargs='?' , const=False, default=False, type=bool, help='make the resulting texture tileable')
    args = parser.parse_args()
    generate_texture(args.source, args.target, args.lr, args.iter, tileable=args.tile)'''