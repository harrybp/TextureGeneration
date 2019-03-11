'''
Initialisation for application
Creates the root directory and defines computation device (cpu/cuda)
'''
import os
from pathlib import Path
import torch

'''
DEVICE:             if you do not have CUDA then DEVICE must be set to cpu
BASE DIRECTORY:     change this to your chosen directory
'''
DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
BASE_DIRECTORY = str(Path.home()) + '/texture_generation'

if not os.path.exists(BASE_DIRECTORY):
    os.makedirs(BASE_DIRECTORY)
