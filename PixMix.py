import torch
import random
from pathlib import Path
from PIL import Image, ImageStat

from torchvision.transforms import RandomRotation, RandomPerspective, RandomErasing, RandomVerticalFlip, RandomAffine, \
    RandomHorizontalFlip


def alpha_blend(image_a: Image, image_b: Image, alpha=0.5):
    image_a = image_a.convert("RGBA")
    image_b = image_b.convert("RGBA")

    image_b = image_b.resize(image_a.size)

    return Image.blend(image_a, image_b, alpha)


def adaptive_blend(image_a: Image, image_b: Image, alpha=0.5):
    image_a = image_a.convert("RGBA")
    image_b = image_b.convert("RGBA")

    image_b = image_b.resize(image_a.size)

    brightness_a = sum(ImageStat.Stat(image_a).mean[:3]) / 3
    brightness_b = sum(ImageStat.Stat(image_b).mean[:3]) / 3

    beta = alpha + (brightness_a - brightness_b) / 255 / 2
    beta = min(max(beta, 0), 1)
    return Image.blend(image_a, image_b, beta)


class PixMix:
    default_augment = [RandomRotation(360),
                       RandomPerspective(),
                       RandomErasing(),
                       RandomVerticalFlip(),
                       RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
                       RandomHorizontalFlip()]

    default_mix_op = [alpha_blend,
                      adaptive_blend]

    def __init__(self, mixing_dir: Path = Path("./data/fractals/"), k=4, beta=3, augment_ops=None, mix_ops=None):
        """
        Args:
            mixing_dir (list of PIL.Image): directory of images to be used for mixing.
            k (int): Maximum number of mixing rounds.
            beta (int): Parameter controlling the strength of mixing operations.
            augment_ops (list of functions): List of augmentation operations.
            mix_ops (list of functions): List of mixing operations (additive, multiplicative).
        """
        assert augment_ops != []
        assert isinstance(augment_ops, list) or augment_ops is None
        assert mix_ops != []
        assert isinstance(mix_ops, list) or mix_ops is None
        self.mixing_dir = mixing_dir
        self.k = k
        self.beta = beta
        self.augment_ops = augment_ops if augment_ops is not None else self.default_augment
        self.mix_ops = mix_ops if mix_ops is not None else self.default_mix_op
        self.fractal_files = list(self.mixing_dir.iterdir())

    def __call__(self, xorig):
        print(type(xorig))
        xpixmix = random.choice([self.augment(xorig), xorig])
        for _ in range(random.choice(range(self.k + 1))):
            mix_image = random.choice([self.augment(xorig), self._get_random_mixer()])
            mix_op = random.choice(self.mix_ops)
            xpixmix = mix_op(xpixmix, mix_image, self.beta)
        return xpixmix

    def _get_random_mixer(self):
        random_file = random.choice(self.fractal_files)
        with Image.open(random_file.__str__()) as img:
            return img.copy()

    def augment(self, x):
        aug_op = random.choice(self.augment_ops)
        return aug_op(x)
