import time
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from models import SRResNet
from dataset import SRDataset
from utils import *
from logger import Logger
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from evaluate import evaluate


device = torch.device("cuda" if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else "cpu")


cudnn.benchmark = True


def main():
    """
    Training.
    """
    print(f'{device = }')

    args = parse_arguments()

    # Read settings from the YAML file
    settings = read_settings(args.config)

    # Access and use the settings as needed
    generator_settings = settings.get('generator', {})
    train_settings = settings.get('train', {})
    dataset_settings = settings.get('dataset', {})
    train_dataloader_settings = settings.get('train_dataloader', {})
    test_dataloader_settings = settings.get('test_dataloader', {})

    print(f'{generator_settings = }\n{train_settings = }\n{dataset_settings = }\n'
          f'{train_dataloader_settings = }\n{test_dataloader_settings = }')

    train_dataset = SRDataset(**dataset_settings,
                              stage='train',
                              lr_img_type='imagenet-norm',
                              hr_img_type='[-1, 1]')
    # note that we're passing the collate function here
    train_loader = DataLoader(
        train_dataset, **train_dataloader_settings, pin_memory=True)

    test_dataset = SRDataset(**dataset_settings,
                             stage='test',
                             lr_img_type='imagenet-norm',
                             hr_img_type='[-1, 1]')
    # note that we're passing the collate function here
    test_loader = DataLoader(
        test_dataset, **test_dataloader_settings, pin_memory=True)

    model = SRResNet(**generator_settings,
                     scaling_factor=dataset_settings['scaling_factor'])

    # Logger
    logger = Logger(settings, str(model),
                    'INM705-SuperResolution')

    # Train
    train(model, train_loader, test_loader, logger, **train_settings)


def train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, logger: Logger, lr: float, epochs: int,
          grad_clip=None):
    # Initialize the optimizer
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=float(lr))
    # Move to default device
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # Epochs
    for epoch in range(1, epochs + 1):
        # One epoch's training
        epoch_start = time.perf_counter()
        epoch_loss = train_epoch(model=model,
                                 train_loader=train_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 grad_clip=grad_clip)
        train_end = time.perf_counter()
        psnrs, ssims, wandb_images = evaluate(model, test_loader)
        eval_end = time.perf_counter()
        logger.log({'epoch_loss': epoch_loss,
                   'epoch_train_time': train_end - epoch_start,
                    'epoch_eval_time': eval_end - train_end,
                    'mean_psnr': np.mean(psnrs),
                    'mean_ssim': np.mean(ssims),
                    ** wandb_images
                    })
        print(f'{epoch = }\n'
              f'{epoch_loss = }\n'
              f'epoch_train_time: {train_end - epoch_start}\n'
              f'epoch_eval_time: {eval_end - train_end}\n'
              f'mean_psnr: {np.mean(psnrs)}\n'
              f'mean_ssim: {np.mean(ssims)}\n'
              )

        # Save checkpoint
        save_checkpoint(epoch, model, str(model), optimizer)


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, grad_clip=None):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: content loss function (Mean Squared-Error loss)
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables batch normalization

    total_loss = 0

    # Batches
    for lr_imgs, hr_imgs in tqdm(train_loader):
        # Move to default device
        # (batch_size (N), 3, 24, 24), imagenet-normed
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), in [-1, 1]

        # Forward prop.
        sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

        # Loss
        loss = criterion(sr_imgs, hr_imgs)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, float(grad_clip))

        # Update model
        optimizer.step()

        # Keep track of loss
        total_loss += loss.item()

    del lr_imgs, hr_imgs, sr_imgs  # free some memory since their histories may be stored

    return total_loss / len(train_loader)


if __name__ == '__main__':
    main()
