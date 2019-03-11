'''
Initialisation for the GAN module of the application
Creates necessary folders: textures and models
'''
import os
from config import BASE_DIRECTORY

if not os.path.exists(BASE_DIRECTORY + '/textures'):
    os.makedirs(BASE_DIRECTORY + '/textures')
if not os.path.exists(BASE_DIRECTORY + '/models'):
    os.makedirs(BASE_DIRECTORY + '/models')