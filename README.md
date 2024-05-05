# Super-Resolution GAN Project

## Overview
This project implements a Super-Resolution Generative Adversarial Network (SRGAN) to enhance low-resolution images to high-resolution. The implementation leverages PyTorch for building and training neural network models.

## Installation
To set up the project environment:
```bash
pip install -r requirements.txt
```

## Usage
To run the project:

Modify the configuration file to specify paths and settings for datasets and model parameters.
Run the main.py script to start the training process:
```bash
python main.py --config configs/SRResNet.yaml
```
or
```bash
python train_srgan.py --config configs/SRGAN.yaml
```

## Weights

Download weights from https://cityuni-my.sharepoint.com/:f:/g/personal/youssef_shaarawy_city_ac_uk/Er3_6Gi4x4RPtzVQbQ6CpEwBTQwqJYfA6QB30QyrkFjoNQ?e=9lKtx3

Add them to the weights folder.

## Dataset

Download dataset from kaggle: https://www.kaggle.com/datasets/ambityga/imagenet100