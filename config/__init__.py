'''
Initialisation for application
Creates the root directory and defines computation device (cpu/cuda)
'''
import os
from pathlib import Path
import torch

DEVICE = torch.device('cuda')
BASE_DIRECTORY = str(Path.home()) + '/texture_generation'
if not os.path.exists(BASE_DIRECTORY):
    os.makedirs(BASE_DIRECTORY)
