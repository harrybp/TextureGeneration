# Image texture and style synthesis by convolutional neural network
### Final Year Project 

The aim of this project is to explore methods of generating unique, realistic textures without needing a large source dataset.
I have implemented two different methods for generating textures using convolutional neural networks.
* Optimising on image pixels as done by [gatys et al](https://arxiv.org/pdf/1505.07376.pdf)
* Training a GAN to generate textures using models from [here](https://arxiv.org/pdf/1511.06434.pdf) and [here](https://arxiv.org/pdf/1705.06566.pdf) although the code can be easily extended to support any generator & discriminator models.

## Gaty's et al Method
This method involves optimising on the pixels of a noise image. On each iteration, the image is run through a pre-trained convolutional neural network and the features in each layer of the network are extracted. The loss function used aims to minimise the difference between the features of this image and a source image. The Gram Matrix of each set of features is calculated first in order to exclude spatial information and focus on style.

The textures produced can be tiled if specified. Below we can see the input image on the left and resulting tiled texture on the right.

![Input image ](http://harrybp.github.io/texture_images/gatys_input.jpg) 
![Resulting tiled texture ](http://harrybp.github.io/texture_images/gatys_tiled.jpg)

To run: `python gatys.py -h`

## GAN Method
This method involves training a Generational Adversarial Network. The dataset is comprised of randomly cropped sections of the source image. The images produced can be of any size. Generation of images from a trained generator is almost instant but training can take some time (Around 30 minutes on a NVIDIA GTX970).

It is possible to generate tileable images using some GAN models. It also can be possible to animate the textures in a smooth and natural way. Examples of this can be found [here](https://harrybp.github.io/texture_generation_demo/).

Below we can see the input image and an example of a generated outuput after training a PSGAN model for around half an hour.

![Input image](http://harrybp.github.io/texture_generation_demo/textures/snake/cropped.jpg)
![Resulting tiled texture](https://harrybp.github.io/texture_generation_demo/textures/snake/gan/336.jpg)

To run `python gan.py -h`

## Comparison of Approaches
I have set up a website [here](https://harrybp.github.io/texture_generation_demo/) to compare the generation process between approaches and to showcase some generated images. 
