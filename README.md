# PixMix
A simple implementation of PixMix for Pytorch
Below is a template for a `README.md` file for a project that includes PixMix augmentation and a training loop for a ResNet model on the CIFAR-100 dataset, as well as `setup.sh` operations for setting up the environment and downloading datasets from Kaggle.

---

# PixMix Augmentation with ResNet on CIFAR-100

This project demonstrates how to implement and use PixMix augmentation in a PyTorch training loop for image classification with a ResNet model on the CIFAR-100 dataset. Additionally, it includes a `setup.sh` script to prepare the environment and download necessary datasets from Kaggle.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.6+
- Pip package manager

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Hans-OlivierFontaine/PixMix.git
cd PixMix
```

2. **Install required Python packages:**

```bash
pip install -r requirements.txt
```

3. **Setup Kaggle API (Optional):**

To download datasets from Kaggle, ensure you have a Kaggle account and obtain your API credentials (`kaggle.json`). Place this file in `~/.kaggle/` to use the Kaggle API for dataset downloads.

4. **Run `setup.sh`:**

Make sure to give execution permissions and run `setup.sh` to download and prepare the CIFAR-100 dataset and any other necessary data.

```bash
chmod +x setup.sh
./setup.sh
```

### Training the Model

To train the ResNet model with PixMix augmentation on the CIFAR-100 dataset, run:

```bash
python train.py
```

Ensure `train.py` includes the training loop, model definition, and data loaders as discussed previously.

## Project Structure

- `pixmix_transform.py`: Contains the implementation of the PixMix augmentation.
- `train.py`: The main script for training the model.
- `requirements.txt`: Specifies all necessary Python packages.
- `setup.sh`: A script to setup the environment and download datasets.
- `README.md`: This file.

## Contributing

Please read [CONTRIBUTING.md](https://github.com/your-OlivierFontaine/PixMix/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

- **Hans-Olivier Fontaine** - *Initial work* - [Hans-OlivierFontaine](https://github.com/Hans-OlivierFontaine)

See also the list of [contributors](https://github.com/your-OlivierFontaine/PixMix/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

- Original paper: [PixMix](https://arxiv.org/pdf/2112.05135.pdf)
- Kaggle Fractal Mixing Set for PixMix: [Dataset](https://www.kaggle.com/datasets/tomandjerry2005/fractal-mixing-set-pixmix)

---