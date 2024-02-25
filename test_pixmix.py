import unittest
from PixMix import PixMix
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import torch


class TestPixMix(unittest.TestCase):

    def setUp(self):
        # This method is called before each test
        self.image = Image.open('./data/TroubledWaters.jpg')
        self.pixmix = PixMix()

    def test_output_type(self):
        # Test that the output type is still a PIL Image
        transformed_image = self.pixmix(self.image)
        self.assertIsInstance(transformed_image, Image.Image, "The PixMix transform should return a PIL Image.")

    def test_output_shape(self):
        # Test that the output shape is unchanged
        original_size = self.image.size
        transformed_image = self.pixmix(self.image)
        self.assertEqual(transformed_image.size, original_size,
                         "The PixMix transform should not change the image size.")

    def test_tensor_conversion(self):
        # Test conversion to tensor after applying PixMix
        # This test ensures that the transform can be seamlessly integrated into a preprocessing pipeline
        transformed_image = self.pixmix(self.image)
        tensor_image = F.to_tensor(transformed_image)
        self.assertIsInstance(tensor_image, torch.Tensor,
                              "The transformed image should be convertible to a torch.Tensor.")
        self.assertEqual(tensor_image.dim(), 3,
                         "The tensor representation of the image should have 3 dimensions (C, H, W).")


if __name__ == '__main__':
    unittest.main()
