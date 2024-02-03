import torch
import random
from pathlib import Path

from torchvision.transforms import RandomRotation, RandomPerspective, RandomErasing, RandomVerticalFlip, RandomAffine, \
    RandomHorizontalFlip, RandomPosterize


class PixMix:
    def __init__(self, mixing_dir: Path = Path("./data/fractals/"), k=4, beta=3, augment_ops=None, mix_ops=None):
        """
        Args:
            mixing_dir (list of PIL.Image): directory of images to be used for mixing.
            k (int): Maximum number of mixing rounds.
            beta (int): Parameter controlling the strength of mixing operations.
            augment_ops (list of functions): List of augmentation operations.
            mix_ops (list of functions): List of mixing operations (additive, multiplicative).
        """
        self.mixing_dir = mixing_dir
        self.k = k
        self.beta = beta
        self.augment_ops = augment_ops if augment_ops is not None else [self.default_augment]
        self.mix_ops = mix_ops if mix_ops is not None else self.default_mix_op

    def __call__(self, xorig):
        xpixmix = random.choice([self.augment(xorig), xorig])
        for _ in range(random.choice(range(self.k + 1))):  # random count of mixing rounds
            mix_image = random.choice([self.augment(xorig), self._get_random_mixer()])
            mix_op = random.choice(self.mix_ops)
            xpixmix = mix_op(xpixmix, mix_image, self.beta)
        return xpixmix

    def _get_random_mixer(self):
        return

    def augment(self, x):
        aug_op = random.choice(self.augment_ops)
        return aug_op(x)

    @staticmethod
    def default_augment(x):
        return x

    @staticmethod
    def default_mix_op():
        return [RandomRotation(360),
                RandomPerspective(),
                RandomErasing(),
                RandomVerticalFlip(),
                RandomAffine(degrees=15, translate=0.1, scale=(0.8, 1.2), shear=10),
                RandomHorizontalFlip(),
                RandomPosterize]
