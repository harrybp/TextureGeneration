import unittest
import torch
from torchvision import models
import numpy as np
from gatys import generate

class GatysTest(unittest.TestCase):

    def test_gram_matrix(self):
        params = torch.ones((1,1,1))
        expected_result = torch.FloatTensor([1])
        assert torch.eq(generate.gram_matrix(params), expected_result)

    def test_gram_matrix_layers(self):
        params = [torch.ones((1, 1, 1))]
        expected_result = [torch.FloatTensor([1])]
        assert torch.eq(params[0], generate.gram_matrix_layers(params)[0])

    def test_get_layer_sizes(self):
        params = [
            torch.ones((1, 1, 1)),
            torch.ones((2, 2, 2)),
            torch.ones((5, 5, 5))
        ]
        expected_result = [
            1 * 1 * 1,
            2 * 2 * 2,
            5 * 5 * 5
        ]
        assert expected_result == generate.get_layer_sizes(params)

    def test_get_style_loss(self):
        gram_matrix_zeros = [
            torch.zeros((1,1)),
            torch.zeros((2,2)),
            torch.zeros((5,5))
        ]
        gram_matrix_ones = [
            torch.ones((1,1)),
            torch.ones((2,2)),
            torch.ones((5,5))
        ]
        layer_sizes = [
            1 * 1,
            2 * 2,
            3 * 3
        ]
        weights = [1, 1, 1]
        assert 0 == generate.get_style_loss(gram_matrix_zeros, gram_matrix_zeros, layer_sizes, weights)
        assert 0 == generate.get_style_loss(gram_matrix_ones, gram_matrix_ones, layer_sizes, weights)
        assert 0 < generate.get_style_loss(gram_matrix_ones, gram_matrix_zeros, layer_sizes, weights)

    def test_get_feature_layers(self):
        test_image = torch.ones((3,64,64))
        vgg16 = models.vgg16().features.eval() #VGG16 is a known CNN with 31 layers
        assert len(generate.get_feature_layers(test_image, vgg16, np.arange(50))) == 31
        assert len(generate.get_feature_layers(test_image, vgg16, np.arange(10))) == 10
        assert len(generate.get_feature_layers(test_image, vgg16, [])) == 0

    def test_tile_vertical(self):
        test_image = torch.Tensor([[2, 2], [3, 3]]).unsqueeze(0)
        expected_result = torch.Tensor([[3, 3], [2, 2]]).unsqueeze(0)
        assert torch.all(torch.eq(expected_result, generate.tile_vertical(test_image)))

    def test_tile_horizontal(self):
        test_image = torch.Tensor([[2, 3], [2, 3]]).unsqueeze(0)
        expected_result = torch.Tensor([[3, 2], [3, 2]]).unsqueeze(0)
        assert torch.all(torch.eq(expected_result, generate.tile_horizontal(test_image)))
