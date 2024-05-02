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
