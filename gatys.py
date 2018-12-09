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
def generate_texture(source, target, learning_rate, iterations, tileable=False, save_intermediates=False):
    image_size = 256
    print('Generating texture file "%s" from source file "%s"' % (target, source))

    #Transforms to normalise in the same way as the images vgg was trained on
    normalise = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    de_normalise = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    #Transforms to switch from Tensor to PIL Image
    to_Tensor = transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor()])
    to_PIL = transforms.Compose([transforms.ToPILImage()])

    cnn = models.vgg16(pretrained=True).features.eval() #Load in pretrained CNN
    cuda = torch.device('cuda') #Can change to cpu if you dont have CUDA

    #Read in source image and noise image as Tensors and normalise
    style_image = normalise(to_Tensor(PIL.Image.open(source).convert("RGB")))
    imarray = np.random.rand(image_size,image_size,3) * 255
    noise_image = normalise(to_Tensor(PIL.Image.fromarray(imarray.astype('uint8')).convert('RGB')))

    #Move everything to GPU
    style_image = style_image.to(cuda)
    noise_image = noise_image.to(cuda)
    vgg16 = cnn.to(cuda)

    #Get target gram matrixes 
    layers = [0,5,10,19,28]
    weights = [10000, 128, 32, 4, 1]
    style_layers = utils.get_feature_layers(style_image, vgg16, layers)
    source_image_grams = utils.gram_matrix_layers(style_layers)
    layer_sizes = utils.get_layer_sizes(style_layers)
    
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
            noise_layers = utils.get_feature_layers(noise_image_h, vgg16, layers)   
        else:
            noise_layers = utils.get_feature_layers(noise_image, vgg16, layers)

        noise_grams = utils.gram_matrix_layers(noise_layers)

        #Calculate the loss and backpropogate
        loss = utils.get_style_loss(noise_grams, source_image_grams, layer_sizes, weights)
        loss.backward(retain_graph=True)
        optimizer.step()

        if save_intermediates:
            if not os.path.exists('temp'):
                os.makedirs('temp')
            current_image = to_PIL(de_normalise(noise_image.cpu()).clamp(0, 1))
            current_image.save('temp/' +  target  + str(i) + '.jpg')
            f=open("temp/gatys.txt", "a+")
            f.write("%d" % (0))
            f.close()
    
    if not save_intermediates:
        current_image = to_PIL(de_normalise(noise_image.cpu()).clamp(0, 1))
        current_image.save(target + '.jpg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate texture using gatys et al method.')
    parser.add_argument('source',  help='the source image for texture style')
    parser.add_argument('target',  help='the filename for the created image')
    parser.add_argument('--lr', nargs='?', const=0.8, default=0.8, type=float, help='the learning rate for the optimiser')
    parser.add_argument('--iter', nargs='?' , const=400, default=400, type=int, help='the number of iterations')
    parser.add_argument('--tile', nargs='?' , const=False, default=False, type=bool, help='make the resulting texture tileable')
    args = parser.parse_args()
    generate_texture(args.source, args.target, args.lr, args.iter, tileable=args.tile)