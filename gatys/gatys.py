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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Creates a texture image as done by gatys et al.
def generate_texture(source, target, learning_rate, iterations, image_size=124, tileable=False, save_intermediates=True, cuda=False):
    print('Generating texture file "%s" from source file "%s"' % (target, source))

    #Transforms to normalise in the same way as the images vgg was trained on
    normalise = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    de_normalise = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    #Transforms to switch from Tensor to PIL Image
    crop = transforms.RandomCrop((256,256))
    to_Tensor = transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor()])
    to_PIL = transforms.Compose([transforms.ToPILImage()])

    cnn = models.vgg16(pretrained=True).features.eval() #Load in pretrained CNN
    if cuda:
        cuda = torch.device('cuda') #Can change to cpu if you dont have CUDA
    else:
        cuda = torch.device('cpu')
    #Read in source image and noise image as Tensors and normalise
    style_image = normalise(to_Tensor(crop(PIL.Image.open(source).convert("RGB"))))
    imarray = np.random.rand(image_size,image_size,3) * 255
    noise_image = normalise(to_Tensor(PIL.Image.fromarray(imarray.astype('uint8')).convert('RGB')))

    #Move everything to GPU
    style_image = style_image.to(cuda)
    noise_image = noise_image.to(cuda)
    vgg16 = cnn.to(cuda)

    #Get target gram matrixes 
    layers = [0,5,10,19,28]
    weights = [200, 100, 50, 25, 12]
    style_layers = get_feature_layers(style_image, vgg16, layers)
    source_image_grams = gram_matrix_layers(style_layers)
    layer_sizes = get_layer_sizes(style_layers)
    
    #Set up optimiser
    noise_image.requires_grad = True
    optimizer = torch.optim.Adam([noise_image], lr=learning_rate) #Set up the optimizer
    saved_images = 0
    for i in range(iterations):
        sys.stdout.write('\rIteration %d / %d' % (i+1, iterations))
        sys.stdout.flush()

        noise_image.clamp(-1.5, 1.5)
        optimizer.zero_grad() #Reset all gradients for each iteration

        #Get the gram matrix's for the noise image
        noise_layers = get_feature_layers(noise_image, vgg16, layers)
        noise_grams = gram_matrix_layers(noise_layers)

        #Calculate the loss and backpropogate
        loss = get_style_loss(noise_grams, source_image_grams, layer_sizes, weights)
        loss.backward(retain_graph=True)
        optimizer.step()

        if save_intermediates:
            with torch.no_grad():
                if i < 50 or (i < 100 and i % 5 == 0) or (i % 10 == 0):
                    current_image = to_PIL(de_normalise(noise_image.cpu()).clamp(0, 1))
                    current_image.save(target + '.jpg')
                    saved_images+=1
    
    if not save_intermediates:
        result = to_PIL(de_normalise(noise_image.cpu()).clamp(0, 1))
        result.save(target + '.jpg')

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate texture using gatys et al method.')
    parser.add_argument('source',  help='the source image for texture style')
    parser.add_argument('target',  help='the filename for the created image')
    parser.add_argument('--lr', nargs='?', const=0.8, default=0.8, type=float, help='the learning rate for the optimiser')
    parser.add_argument('--iter', nargs='?' , const=150, default=150, type=int, help='the number of iterations')
    parser.add_argument('--tile', nargs='?' , const=False, default=False, type=bool, help='make the resulting texture tileable')
    args = parser.parse_args()
    generate_texture(args.source, args.target, args.lr, args.iter, tileable=args.tile, save_intermediates=False)


