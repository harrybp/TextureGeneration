from torchvision import models
import torch
from gan import generate, models
import unittest

class GanTest(unittest.TestCase):

    def test_generate_image(self):
        #Test PSGAN
        test_model = models.PSGenerator()
        test_noise = torch.zeros((1, 64, 4, 4))
        assert generate.generate_image(test_model, test_noise).verify() == None
        #Test DCGAN
        test_model = models.DCGenerator(100, 64, 3)
        test_noise = torch.zeros((1, 100, 1, 1))
        assert generate.generate_image(test_model, test_noise).verify() == None
