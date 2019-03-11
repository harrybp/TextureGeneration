'''
Initialisation for the Gatys module of the application
Creates necessary folders: textures
'''
import os
from config import BASE_DIRECTORY

if not os.path.exists(BASE_DIRECTORY + '/textures'):
    os.makedirs(BASE_DIRECTORY + '/textures')