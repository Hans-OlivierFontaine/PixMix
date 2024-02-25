from PIL import Image
from PixMix import PixMix
from pathlib import Path


def visualize_augmentation(img: Image, aug):
    augmented_image = aug(img.copy())
    width, height = img.size
    combined_image = Image.new("RGB", (width * 2, height))
    combined_image.paste(img, (0, 0))
    combined_image.paste(augmented_image, (width, 0))
    combined_image.save("./data/beforeAndAfter.jpg")


if __name__ == "__main__":
    image_path = Path('./data/TroubledWaters.jpg')
    image = Image.open(image_path.__str__()).convert("RGB")
    augmentation = PixMix(k=7)
    visualize_augmentation(image, augmentation)
