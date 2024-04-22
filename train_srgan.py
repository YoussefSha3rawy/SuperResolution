import time
import torch.backends.cudnn as cudnn
from torch import nn
from models import SRResNet, TruncatedVGG19, Discriminator
from dataset import SRDataset
from utils import *
from torch.utils.data import DataLoader
from logger import Logger
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import wandb
from collections import deque
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, resnet50, ResNet50_Weights

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
    # note that we're passing the collate function here
    test_loader = DataLoader(
        test_dataset, **test_dataloader_settings, pin_memory=True)

    logger = Logger(settings, discriminator.__class__.__name__,
                    'INM705-SuperResolution')

    train(generator, discriminator, truncated_vgg19,
          train_loader, test_loader, logger, **train_settings)


def train(generator: nn.Module, discriminator: nn.Module, truncated_vgg19: nn.Module, train_loader: DataLoader,
          test_loader: DataLoader, logger: Logger, epochs: int, lr_g: float, lr_d: float, beta=1e-3, grad_clip=None, early_stopping=0):
    lr_g = float(lr_g)
    lr_d = float(lr_d)
    beta = float(beta)
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
        epoch_perceptual_loss, epoch_content_loss, epoch_adversial_loss, epoch_discriminator_loss = train_epoch(train_loader=train_loader,
                                                                                                                generator=generator,
                                                                                                                discriminator=discriminator,
                                                                                                                truncated_vgg19=truncated_vgg19,
                                                                                                                content_loss_criterion=content_loss_criterion,
                                                                                                                adversarial_loss_criterion=adversarial_loss_criterion,
                                                                                                                optimizer_g=optimizer_g,
                                                                                                                optimizer_d=optimizer_d,
                                                                                                                grad_clip=grad_clip,
                                                                                                                beta=beta)
        train_end = time.perf_counter()
        psnrs, ssims = evaluate(generator, test_loader, logger)
        eval_end = time.perf_counter()
        logger.log({'perceptual_loss': epoch_perceptual_loss,
                    'content_loss': epoch_content_loss,
                    'adversial_loss': epoch_adversial_loss,
                    'discriminator_loss': epoch_discriminator_loss,
                    'epoch_train_time': train_end - epoch_start,
                    'epoch_eval_time': eval_end - train_end,
                    'mean_psnr': np.mean(psnrs),
                    'mean_ssim': np.mean(ssims)
                    })
        print(f'perceptual_loss: {epoch_perceptual_loss}\n'
              f'content_loss: {epoch_content_loss}\n'
              f'adversial_loss: {epoch_adversial_loss}\n'
              f'discriminator_loss: {epoch_discriminator_loss}\n'
              f'epoch_train_time: {train_end - epoch_start}\n'
              f'epoch_eval_time: {eval_end - train_end}\n'
              f'mean_psnr: {np.mean(psnrs)}\n'
              f'mean_ssim: {np.mean(ssims)}'
              )

        # Save checkpoint
        save_checkpoint(
            epoch, generator, f'{str(generator)}_{discriminator.__class__.__name__}', optimizer_g)
        save_checkpoint(
            epoch, discriminator, discriminator.__class__.__name__, optimizer_d)


def train_epoch(train_loader, generator, discriminator, truncated_vgg19, content_loss_criterion, adversarial_loss_criterion,
                optimizer_g, optimizer_d, beta, grad_clip=None):
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
    # Set to train mode
    generator.train()
    discriminator.train()  # training mode enables batch normalization

    total_content_loss = 0
    total_adversial_loss = 0
    total_perceptual_loss = 0
    total_discriminator_loss = 0
    # Batches
    for lr_imgs, hr_imgs in tqdm(train_loader):
        # Move to default device
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
        # detached because they're constant, targets
        hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()

        # Discriminate super-resolved (SR) images
        sr_discriminated = discriminator(sr_imgs)  # (N)

        # Calculate the Perceptual loss
        content_loss = content_loss_criterion(
            sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
        adversarial_loss = adversarial_loss_criterion(
            sr_discriminated, torch.ones_like(sr_discriminated))
        perceptual_loss = content_loss + beta * adversarial_loss

        total_content_loss += content_loss
        total_adversial_loss += adversarial_loss
        total_perceptual_loss += perceptual_loss

        # Back-prop.
        optimizer_g.zero_grad()
        perceptual_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_g, grad_clip)

        # Update generator
        optimizer_g.step()

        # DISCRIMINATOR UPDATE

        # Discriminate super-resolution (SR) and high-resolution (HR) images
        hr_discriminated = discriminator(hr_imgs)
        sr_discriminated = discriminator(sr_imgs.detach())
        # But didn't we already discriminate the SR images earlier, before updating the generator (G)? Why not just use that here?
        # Because, if we used that, we'd be back-propagating (finding gradients) over the G too when backward() is called
        # It's actually faster to detach the SR images from the G and forward-prop again, than to back-prop. over the G unnecessarily
        # See FAQ section in the tutorial

        # Binary Cross-Entropy loss
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
            adversarial_loss_criterion(
                hr_discriminated, torch.ones_like(hr_discriminated))
        total_discriminator_loss += adversarial_loss

        # Back-prop.
        optimizer_d.zero_grad()
        adversarial_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_d, grad_clip)

        # Update discriminator
        optimizer_d.step()
    # free some memory since their histories may be stored
    del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated

    return total_perceptual_loss / len(train_loader), total_content_loss / len(train_loader), total_adversial_loss / len(train_loader), total_discriminator_loss / len(train_loader)


def evaluate(model: nn.Module, test_loader: DataLoader, logger: Logger):
    psnrs = []
    ssims = []
    model.eval()
    num_images_to_plot = 3
    imgs_to_plot = deque(maxlen=num_images_to_plot)
    with torch.no_grad():
        # Batches
        for lr_imgs, hr_imgs in tqdm(test_loader):
            # Move to default device
            # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
            lr_imgs = lr_imgs.to(device)
            # (batch_size (1), 3, w, h), in [-1, 1]
            hr_imgs = hr_imgs.to(device)

            # Forward prop.
            sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]
            # Calculate PSNR and SSIM
            sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(
                0)  # (w, h), in y-channel
            hr_imgs_y = convert_image(
                hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
            psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                           data_range=255.)
            ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                         data_range=255.)
            psnrs.append(psnr)
            ssims.append(ssim)
            imgs_to_plot.append(
                (lr_imgs[0].cpu(), hr_imgs[0].cpu(), sr_imgs[0].cpu()))

        for i, (lr_img, hr_img, sr_img) in enumerate(imgs_to_plot):
            logger.log({
                f'lr_{i}': wandb.Image(convert_image(lr_img, source='imagenet-norm', target='pil')),
                f'hr_{i}': wandb.Image(convert_image(hr_img, source='[-1, 1]', target='pil')),
                f'sr_{i}': wandb.Image(convert_image(sr_img, source='[-1, 1]', target='pil')),
            })
    return psnrs, ssims


if __name__ == '__main__':
    main()
