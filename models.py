import torchvision.models as models
import torch.nn as nn

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
# The following generator and discriminator implementations are taken from
# https://github.com/pytorch/examples/tree/master/dcgan
# Create a Generator CNN, which takes a 1D input vector and produces an image
class DCGenerator(nn.Module):
    def __init__(self, channels=[100, 512, 256, 128, 64, 3], kernel_size=4):
        super(DCGenerator, self).__init__()
        #Build layers
        layers = []

        #First layer has stride=1, padding=0 so it's different from the rest
        layers.append(nn.ConvTranspose2d( channels[0], channels[1], kernel_size=kernel_size, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(channels[1]))
        layers.append(nn.ReLU(inplace=True))
        for i in range(2, len(channels)-1):
            layers.append(nn.ConvTranspose2d( channels[i-1], channels[i], kernel_size=kernel_size, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(channels[i]))
            layers.append(nn.ReLU(inplace=True))
        #Final layer has Tanh rather than batchnorm and relu
        layers.append(nn.ConvTranspose2d( channels[-2], channels[-1], kernel_size=kernel_size, stride=2, padding=1, bias=False))
        layers.append(nn.Tanh())
        self.generate = nn.Sequential(*layers)

    def forward(self, input):
        return self.generate(input)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Create a Discriminator CNN, which takes an image as 
#   input and returns a 0 (fake) or 1 (real)    
class DCDiscriminator(nn.Module):
    def __init__(self, channels=[3, 64, 128, 256, 512, 1], kernel_size=4):
        super(DCDiscriminator, self).__init__()
        layers = []
        #First layer doesnt have batchnorm
        layers.append(nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, stride=2, padding=1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        for i in range(2, len(channels)-1):
            layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=kernel_size, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(image_size * 2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        #Final layer has different stride and padding, and sigmoid rather than batchnorm and lrelu
        layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_size=kernel_size, stride=1, padding=0, bias=False))
        layers.append(nn.Sigmoid())
        
        self.discriminate = nn.Sequential(*layers)

    def forward(self, input):
        return self.discriminate(input)