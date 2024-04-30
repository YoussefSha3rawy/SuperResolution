import time
import torch.backends.cudnn as cudnn
from torch import nn
from models import SRResNet, TruncatedVGG19, Discriminator
from dataset import SRDataset
from utils import *
from logger import Logger
import numpy as np
from tqdm import tqdm
from torchvision.models import efficientnet_b0
from evaluate import evaluate
from torch.utils.data import DataLoader

# Default device
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
    discriminator_settings = settings.get('discriminator', {})
    train_settings = settings.get('train', {})
    dataset_settings = settings.get('dataset', {})
    train_dataloader_settings = settings.get('train_dataloader', {})
    test_dataloader_settings = settings.get('test_dataloader', {})

    print(f'{generator_settings = }\n{discriminator_settings = }\n{train_settings = }\n{dataset_settings = }\n'
          f'{train_dataloader_settings = }\n{test_dataloader_settings = }')

    # Initialize model or load checkpoint
    # Generator
    generator = SRResNet(**generator_settings,
                         scaling_factor=dataset_settings['scaling_factor'])

    if discriminator_settings['discriminator_type'] == 'EfficientNet':
        discriminator = efficientnet_b0()
        # for param in discriminator.parameters():
        #     param.requires_grad = False
        discriminator.classifier[1] = nn.Linear(
            discriminator.classifier[1].in_features, 1)
    else:
        del discriminator_settings['discriminator_type']
        discriminator = Discriminator(**discriminator_settings)

    # Truncated VGG19 network to be used in the loss calculation
    truncated_vgg19 = TruncatedVGG19(i=36)
    truncated_vgg19.eval()

    # Move to default device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    truncated_vgg19 = truncated_vgg19.to(device)

    # Custom dataloaders
    train_dataset = SRDataset(**dataset_settings,
                              stage='train',
                              lr_img_type='imagenet-norm',
                              hr_img_type='imagenet-norm')
    train_loader = DataLoader(train_dataset, **train_dataloader_settings,
                              pin_memory=True)

    test_dataset = SRDataset(**dataset_settings,
                             stage='test',
                             lr_img_type='imagenet-norm',
                             hr_img_type='[-1, 1]')
    test_loader = DataLoader(
        test_dataset, **test_dataloader_settings, pin_memory=True)

    logger = Logger(settings, discriminator.__class__.__name__,
                    'INM705-SuperResolution')

    train(generator, discriminator, truncated_vgg19,
          train_loader, test_loader, logger, **train_settings)


def train(generator: nn.Module, discriminator: nn.Module, truncated_vgg19: nn.Module, train_loader: DataLoader,
          test_loader: DataLoader, logger: Logger, epochs: int, lr_g: float, lr_d: float, loss_type='Default', **kwargs):
    lr_g = float(lr_g)
    lr_d = float(lr_d)
    # Initialize generator's optimizer
    optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()),
                                   lr=lr_g)
    # Initialize discriminator's optimizer
    optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()),
                                   lr=lr_d)
    # Loss functions
    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)

    # Epochs
    for epoch in range(1, epochs+1):
        # At the halfway point, reduce learning rate to a tenth
        if epoch == int(epochs // 2):
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)

        epoch_start = time.perf_counter()
        # One epoch's training
        epoch_loss = train_epoch(train_loader=train_loader,
                                 generator=generator,
                                 discriminator=discriminator,
                                 truncated_vgg19=truncated_vgg19,
                                 content_loss_criterion=content_loss_criterion,
                                 adversarial_loss_criterion=adversarial_loss_criterion,
                                 optimizer_g=optimizer_g,
                                 optimizer_d=optimizer_d,
                                 loss_type=loss_type,
                                 **kwargs)
        train_end = time.perf_counter()
        psnrs, ssims, images_dict = evaluate(generator, test_loader)
        eval_end = time.perf_counter()
        logger.log({**epoch_loss,
                    'epoch_train_time': train_end - epoch_start,
                    'epoch_eval_time': eval_end - train_end,
                    'mean_psnr': np.mean(psnrs),
                    'mean_ssim': np.mean(ssims),
                    **images_dict
                    })
        print({'epoch': epoch,
               **epoch_loss,
               'epoch_train_time': train_end - epoch_start,
               'epoch_eval_time': eval_end - train_end,
               'mean_psnr': np.mean(psnrs),
               'mean_ssim': np.mean(ssims)
               })

        # Save checkpoint
        save_checkpoint(
            epoch, generator, f'{str(generator)}_{discriminator.__class__.__name__}{f"_{loss_type}" if loss_type != "Default" else ""}', optimizer_g)
        save_checkpoint(
            epoch, discriminator, f'{discriminator.__class__.__name__}{f"_{loss_type}" if loss_type != "Default" else ""}', optimizer_d)


def train_epoch(train_loader, generator, discriminator, truncated_vgg19, content_loss_criterion, adversarial_loss_criterion,
                optimizer_g, optimizer_d, beta, alpha, loss_type, grad_clip=None, lambda_gp=10):
    """
    One epoch's training.

    :param train_loader: train dataloader
    :param generator: generator
    :param discriminator: discriminator
    :param truncated_vgg19: truncated VGG19 network
    :param content_loss_criterion: content loss function (Mean Squared-Error loss)
    :param adversarial_loss_criterion: adversarial loss function (Binary Cross-Entropy loss)
    :param optimizer_g: optimizer for the generator
    :param optimizer_d: optimizer for the discriminator
    :param epoch: epoch number
    """
    beta = float(beta)
    alpha = float(alpha)
    generator.train()
    discriminator.train()

    total_content_loss = 0
    total_mse_loss = 0
    total_adversarial_loss = 0
    total_perceptual_loss = 0
    total_discriminator_loss = 0
    for lr_imgs, hr_imgs in tqdm(train_loader):
        # (batch_size (N), 3, 24, 24), imagenet-normed
        lr_imgs = lr_imgs.to(device)
        # (batch_size (N), 3, 96, 96), imagenet-normed
        hr_imgs = hr_imgs.to(device)

        # GENERATOR UPDATE

        # Generate
        sr_imgs = generator(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]
        # (N, 3, 96, 96), imagenet-normed
        sr_imgs = convert_image(
            sr_imgs, source='[-1, 1]', target='imagenet-norm')

        # Calculate VGG feature maps for the super-resolved (SR) and high resolution (HR) images
        sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
        hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()

        # Discriminate super-resolved (SR) images
        sr_discriminated = discriminator(sr_imgs)  # (N)

        # Calculate the Perceptual loss
        content_loss = content_loss_criterion(
            sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
        mse_loss = content_loss_criterion(sr_imgs, hr_imgs)
        if loss_type == 'WGAN':
            adversarial_loss = -sr_discriminated.mean()
        else:
            adversarial_loss = adversarial_loss_criterion(
                sr_discriminated, torch.ones_like(sr_discriminated))
        perceptual_loss = content_loss + beta * adversarial_loss + alpha * mse_loss

        # Back-prop.
        optimizer_g.zero_grad()
        perceptual_loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer_g, grad_clip)

        optimizer_g.step()

        # DISCRIMINATOR UPDATE

        # Discriminate super-resolution (SR) and high-resolution (HR) images
        hr_discriminated = discriminator(hr_imgs)
        sr_discriminated = discriminator(sr_imgs.detach())

        # Binary Cross-Entropy loss
        if loss_type == 'WGAN':
            gradient_penalty = calculate_gradient_penalty(
                discriminator, hr_imgs, sr_imgs.detach())

            # Wasserstein GAN loss with gradient penalty
            discriminator_loss = torch.mean(sr_discriminated) - torch.mean(
                hr_discriminated) + lambda_gp * gradient_penalty
        else:
            discriminator_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                adversarial_loss_criterion(
                    hr_discriminated, torch.ones_like(hr_discriminated))

        # Back-prop.
        optimizer_d.zero_grad()
        discriminator_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_d, grad_clip)

        # Update discriminator
        optimizer_d.step()

        total_content_loss += content_loss
        total_adversarial_loss += adversarial_loss
        total_mse_loss += mse_loss
        total_perceptual_loss += perceptual_loss
        total_discriminator_loss += discriminator_loss

    del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated

    return {'perceptual_loss': total_perceptual_loss / len(train_loader),
            'content_loss': total_content_loss / len(train_loader),
            'adversial_loss': total_adversarial_loss / len(train_loader),
            'discriminator_loss': total_discriminator_loss / len(train_loader),
            'mse_loss': total_mse_loss / len(train_loader)}


if __name__ == '__main__':
    main()
