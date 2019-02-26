import torch.nn as nn
import torch
import numpy as np

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GAN MODELS
# Generators and Discriminators need a 'forward' method
# Generators should provide a method for generating the input noise vector 'noise(batch_size, image_size)'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Generator for Periodic-Spatial GAN
# https://arxiv.org/pdf/1705.06566.pdf
class PSGenerator(nn.Module):
    def __init__(self, channels=[64, 512, 256, 128, 64, 3], kernel_size=4):
        super(PSGenerator, self).__init__()

        #Construct layers with in-channels = channels array
        layers = []
        for i in range(1, len(channels)-1):
            layers.append(nn.ConvTranspose2d(channels[i-1], channels[i], kernel_size=kernel_size, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(channels[i]))
            layers.append(nn.ReLU())
        
        #Final layer uses Tanh rather than Batchnorm and Relu
        layers.append(nn.ConvTranspose2d(channels[-2], channels[-1], kernel_size=kernel_size, stride=2, padding=1))
        layers.append(nn.Tanh())

        self.generate = nn.Sequential(*layers)

        #Initialise weights with std=0.02
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, input):
        return self.generate(input)

    def noise(self, batch_size, image_size, tile=False):
        noise =- torch.Tensor(np.random.uniform(-1, 1, (batch_size, 64, int(image_size/32), int(image_size/32)))) #Generate batch of noise for input
        if tile:
            noise_array = torch.cat((noise, noise), 2)
            noise = torch.cat((noise_array, noise_array), 3) #Make grid of identical noise
        return noise

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Discriminator for Periodic-Spatial GAN
# https://arxiv.org/pdf/1705.06566.pdf
class PSDiscriminator(nn.Module):
    def __init__(self, channels=[3, 64, 128, 256, 512, 1], kernel_size=4):
        super(PSDiscriminator, self).__init__()

        #Construct layers with in-channels = channels array
        layers = []

        #First layer has no batchnorm
        layers.append(nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, stride=2, padding=1))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        for i in range(2, len(channels)-1):
            layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=kernel_size, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(channels[i]))
            layers.append(nn.LeakyReLU(negative_slope=0.2))

        #Final  layer has sigmoid instead of batchnorm and lrelu
        layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_size=kernel_size, stride=2, padding=1))
        layers.append(nn.Sigmoid())

        self.discriminate = nn.Sequential(*layers)

        #Initialise weights with std=0.02
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, input):
        result = self.discriminate(input)
        return result.view(result.shape[0], result.shape[2]*result.shape[2])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Generator for Deep Convolutional GAN
# https://github.com/pytorch/examples/tree/master/dcgan
class DCGenerator(nn.Module):
    def __init__(self, input_size, image_size, channels):
        super(DCGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( input_size, image_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(True), 
            nn.ConvTranspose2d(image_size * 8, image_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(True), 
            nn.ConvTranspose2d( image_size * 4, image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(True), 
            nn.ConvTranspose2d( image_size * 2, image_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size),
            nn.ReLU(True),
            nn.ConvTranspose2d( image_size, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
    def noise(self, batch_size, image_size):
            return torch.randn(batch_size, 100, 1, 1) #Generate batch of noise for input


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Discriminator for Deep Convolutional GAN
# https://github.com/pytorch/examples/tree/master/dcgan 
class DCDiscriminator(nn.Module):
    def __init__(self, image_size, channels):
        super(DCDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, image_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size, image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size * 2, image_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size * 4, image_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(image_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)