#!/bin/bash

# Check if Kaggle API is installed, if not install it
if ! command -v kaggle > /dev/null; then
    echo "Kaggle CLI not found, installing..."
    pip install kaggle
fi

# Ensure the Kaggle API token is in place
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Kaggle API token not found. Please ensure ~/.kaggle/kaggle.json is present."
    exit 1
fi

# Make sure the permissions for the token are correct
chmod 600 ~/.kaggle/kaggle.json

# Create a directory for the dataset and navigate into it
DATASET_DIR="./data/"
mkdir -p $DATASET_DIR
cd $DATASET_DIR

# Download the dataset (adjust the URL slug as necessary)
kaggle datasets download tomandjerry2005/fractal-mixing-set-pixmix -d --version 1

# Unzip the dataset and remove the zip file
unzip -q fractal-mixing-set-pixmix.zip
rm fractal-mixing-set-pixmix.zip

echo "Dataset downloaded and extracted to: $(pwd)"
