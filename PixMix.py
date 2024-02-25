import torch
import random
from pathlib import Path
from PIL import Image, ImageStat, ImageDraw


class RandomRotation:
    def __init__(self, degrees=360):
        self.degrees = degrees

    def __call__(self, img):
        angle = random.randint(-self.degrees, self.degrees)
        return img.rotate(angle)


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomAffine:
    def __init__(self, degrees, translate=None):
        self.degrees = degrees
        self.translate = translate

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        if self.translate:
            max_dx = self.translate[0] * img.size[0]
            max_dy = self.translate[1] * img.size[1]
            translations = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
        else:
            translations = (0, 0)
        return img.rotate(angle, translate=translations)


class RandomPerspective:
    def __call__(self, img):
        width, height = img.size
        # Simplified: skewing the image slightly for a faux perspective effect
        # For a true perspective effect, more complex calculations are needed
        coefficients = self.find_coefficients(
            [(0, 0), (width, 0), (width, height), (0, height)],
            [(random.randint(0, int(width * 0.1)), random.randint(0, int(height * 0.1))),
             (width - random.randint(0, int(width * 0.1)), random.randint(0, int(height * 0.1))),
             (width - random.randint(0, int(width * 0.1)), height - random.randint(0, int(height * 0.1))),
             (random.randint(0, int(width * 0.1)), height - random.randint(0, int(height * 0.1)))]
        )
        return img.transform((width, height), Image.PERSPECTIVE, coefficients, Image.BICUBIC)

    def find_coefficients(self, from_pts, to_pts):
        # This method calculates the transformation matrix for the perspective transform.
        # In a full implementation, this would handle the complex math to calculate the matrix.
        matrix = []
        # Simplified for demonstration; full implementation needed for true perspective
        return matrix


class RandomErasing:
    def __init__(self, p=0.5, ratio=(0.02, 0.33), value=0):
        self.p = p
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        if random.random() < self.p:
            width, height = img.size
            area = width * height

            erase_area = random.uniform(self.ratio[0], self.ratio[1]) * area
            aspect_ratio = random.uniform(0.3, 1 / 0.3)

            erase_height = int(round((erase_area * aspect_ratio) ** 0.5))
            erase_width = int(round((erase_area / aspect_ratio) ** 0.5))

            x = random.randint(0, width - erase_width)
            y = random.randint(0, height - erase_height)

            img.paste(self.value, (x, y, x + erase_width, y + erase_height))
        return img


def alpha_blend(image_a: Image, image_b: Image, alpha=0.5):
    image_a = image_a.convert("RGBA")
    image_b = image_b.convert("RGBA")

    image_b = image_b.resize(image_a.size)

    return Image.blend(image_a, image_b, alpha).convert("RGB")


def adaptive_blend(image_a: Image, image_b: Image, alpha=0.5):
    image_a = image_a.convert("RGBA")
    image_b = image_b.convert("RGBA")

    image_b = image_b.resize(image_a.size)

    brightness_a = sum(ImageStat.Stat(image_a).mean[:3]) / 3
    brightness_b = sum(ImageStat.Stat(image_b).mean[:3]) / 3

    beta = alpha + (brightness_a - brightness_b) / 255 / 2
    beta = min(max(beta, 0), 1)
    return Image.blend(image_a, image_b, beta).convert("RGB")


class PixMix:
    default_augment = [RandomRotation(360),
                       # RandomPerspective(),
                       RandomErasing(),
                       RandomVerticalFlip(1),
                       RandomAffine(degrees=15, translate=(0.1, 0.1)),
                       RandomHorizontalFlip(1)]

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
