# Image texture and style synthesis by convolutional neural network
### Final Year Project 

I have implemented two different methods for generating textures using convolutional neural networks.
* Optimising on image pixels as done by [gatys et al](https://arxiv.org/pdf/1505.07376.pdf)
* Training a Deep Convolutional GAN to generate textures as done [here](https://arxiv.org/pdf/1511.06434.pdf)

## Gaty's et al Method
This method involves optimising on the pixels of a noise image. On each iteration, the image is run through a pre-trained convolutional neural network and the features in each layer of the network are extracted. The loss function used aims to minimise the difference between the features of this image and a source image. The Gram Matrix of each set of features is calculated first in order to exclude spatial information and focus on style.

The textures produced can be tiled if specified. Below we can see the input image on the left and resulting tiled texture on the right.

![Input image ](http://harrybp.github.io/texture_images/gatys_input.jpg) 
![Resulting tiled texture ](http://harrybp.github.io/texture_images/gatys_tiled.jpg)

To run: `python gatys.py -h`

## DCGAN Method
This method involves training a Deep Convolutional Generational Adversarial Network. The dataset is comprised of randomly cropped 64x64 sections of the source image. The images produced are 64x64. Generation of images from a trained generator is almost instant but training can take hours.

Below we can see the input image and four examples of images created after training a GAN on 64 iterations over dataset of 16384 random crops.

![Input image](http://harrybp.github.io/texture_images/GAN_input.jpg)
![Resulting tiled texture](http://harrybp.github.io/texture_images/GAN_result.jpg)

To run `python GAN.py -h`

## Web Interface
Both implementations can also be run from within a web interface, this allows the training and generation process to be observed while it is happening.

To run `python webInterface.py` then navigate to `localhost:5000`
