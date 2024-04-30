import wandb
from utils import *
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from collections import deque
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else "cpu")


def evaluate(model: nn.Module, test_loader: DataLoader):
    psnrs = []
    ssims = []
    model.eval()
    num_images_to_plot = 3
    imgs_to_plot = deque(maxlen=num_images_to_plot)
    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(test_loader):
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

        logged_images_dict = {}
        for i, (lr_img, hr_img, sr_img) in enumerate(imgs_to_plot):
            logged_images_dict[f'lr_{i}'] = wandb.Image(
                convert_image(lr_img, source='imagenet-norm', target='pil'))
            logged_images_dict[f'hr_{i}'] = wandb.Image(
                convert_image(hr_img, source='[-1, 1]', target='pil'))
            logged_images_dict[f'sr_{i}'] = wandb.Image(
                convert_image(sr_img, source='[-1, 1]', target='pil'))
        # logger.log(logged_images_dict)
    return psnrs, ssims, logged_images_dict
