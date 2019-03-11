'''
Initialisation for GAN application
Creates necessary directories, 'textures' and 'models'
'''
from pathlib import Path
import os

BASE_DIRECTORY = str(Path.home()) + '/texture_generation'
if not os.path.exists(BASE_DIRECTORY + '/textures'):
    os.makedirs(BASE_DIRECTORY + '/textures')
if not os.path.exists(BASE_DIRECTORY + '/models'):
    os.makedirs(BASE_DIRECTORY + '/models')
