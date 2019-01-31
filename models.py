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
