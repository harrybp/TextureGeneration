'''
Initialisation for application
Creates the root directory and defines computation device (cpu/cuda)
'''
import os
from pathlib import Path
import torch
from .variables import BASE_DIRECTORY

'''
DEVICE:             if you do not have CUDA then DEVICE must be set to cpu
BASE DIRECTORY:     change this to your chosen dpythirectory
'''
DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')

if not os.path.exists(BASE_DIRECTORY):
    os.makedirs(BASE_DIRECTORY)
