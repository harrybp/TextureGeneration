# Texture Generation using Convolutional Neural Networks

The aim of this project is to explore methods of generating unique, realistic or artistic textures without needing a large source dataset.
I have implemented two different methods for generating textures using convolutional neural networks.
* Training a GAN to generate textures using two different models: [DCGAN](https://arxiv.org/pdf/1511.06434.pdf) and [PSGAN](https://arxiv.org/pdf/1705.06566.pdf). (The code is designed to be easily extended to support additional generator & discriminator models.)
* Optimising on image pixels as done by [Gatys et al.](https://arxiv.org/pdf/1505.07376.pdf)

----

## Installation
Prerequisites:
* Python3
* Pip
* [PyTorch](https://pytorch.org/)

Clone and unzip this repository. 
Then simply run `pip install TextureGeneration-master`

**You must have installed PyTorch from [here](https://pytorch.org/) before installing this application.**

If you want to change where the application will store models and textures then edit the file `config/variables.py` prior to installation.

----

## Generating Textures

Run `python setup.py test` before starting to run the test suite

| Command                  | Result                                                |
| ------------------------ | ----------------------------------------------------- |
| `texture_gan train -h`   | to train a new GAN model                              |
| `texture_gan demo -h`    | to generate an image using a trained model            |
| `texture_gan animate -h` | to generate an animated GIF using a trained model     |
| `texture_gatys -h`       | to generate an image from the style of a source image |

----

## Results

An example of the kind of results which this application can achieve. Both the generated images shown below are fully tile-able.

| Method | Source Image             | Result                                                |
| ------ | ------------------------ | ----------------------------------------------------- |
| `texture_gatys` | ![Input image ](http://harrybp.github.io/texture_images/gatys_input.jpg) | ![Resulting tiled texture ](http://harrybp.github.io/texture_images/gatys_tiled.jpg) |
| `texture_gan`   | ![Input image](http://harrybp.github.io/texture_generation_demo/textures/snake/cropped.jpg) | ![Resulting tiled texture](https://harrybp.github.io/texture_generation_demo/textures/snake/gan/336.jpg) |



I have set up a website [here](https://harrybp.github.io/texture_generation_demo/) to compare the generation process between approaches and to showcase some generated images and GIFs. 
