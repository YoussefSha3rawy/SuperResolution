import socket
import time
import argparse
import yaml
from PIL import Image
import os
import random
import torchvision.transforms.functional as FT
import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else "cpu")

# Some constants
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor(
    [0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor(
    [0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(
    device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(
    device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


def convert_image(img, source, target):
    """
    Convert an image from a source format to a target format.

    :param img: image
    :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
    :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),
                   'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
    :return: converted image
    """
    assert source in {
        'pil', '[0, 1]', '[-1, 1]', 'imagenet-norm'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    elif source == 'imagenet-norm':
        if img.ndimension() == 3:
            img = img * imagenet_std + imagenet_mean
        elif img.ndimension() == 4:
            img = img * imagenet_std_cuda + imagenet_mean_cuda

    # Convert from [0, 1] to target
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # already in [0, 1]

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)
                           [:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

    return img


class ImageTransforms(object):
    """
    Image transformation pipeline.
    """

    def __init__(self, split, crop_size, scaling_factor, lr_img_type, hr_img_type, max_test_size=100000):
        """
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of HR images
        :param scaling_factor: LR images will be downsampled from the HR images by this factor
        :param lr_img_type: the target format for the LR image; see convert_image() above for available formats
        :param hr_img_type: the target format for the HR image; see convert_image() above for available formats
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.max_test_size = max_test_size if max_test_size is not None else 100000

        if self.split == 'train':
            self.crop = transforms.RandomCrop(
                self.crop_size, padding=0, pad_if_needed=True)

        assert self.split in {'train', 'test'}

    def __call__(self, img):
        """
        :param img: a PIL source image from which the HR image will be cropped, and then downsampled to create the LR image
        :return: LR and HR images in the specified format
        """

        # Crop
        if self.split == 'train':
            # Take a random fixed-size crop of the image, which will serve as the high-resolution (HR) image
            hr_img = self.crop(img)
        else:
            # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
            original_width, original_height = img.size

            # Calculate the new dimensions that are divisible by the given divisor
            new_width = min([original_width - (original_width %
                            self.scaling_factor), self.max_test_size])
            new_height = min([original_height - (original_height %
                             self.scaling_factor), self.max_test_size])

            # Calculate left, top, right, and bottom coordinates for center crop
            left = (original_width - new_width) // 2
            top = (original_height - new_height) // 2
            right = left + new_width
            bottom = top + new_height
            hr_img = img.crop((left, top, right, bottom))

        # Downsize this crop to obtain a low-resolution version of it
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor), int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)

        # Sanity check
        assert hr_img.width == lr_img.width * \
            self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor

        # Convert the LR and HR image to the required type
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)

        return lr_img, hr_img


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(state, filename):
    """
    Save model checkpoint.

    :param state: checkpoint contents
    """

    torch.save(state, filename)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def save_checkpoint(epoch, model, model_name, optimizer):
    ckpt = {'epoch': epoch, 'model_weights': model.state_dict(
    ), 'optimizer_state': optimizer.state_dict()}
    file_name = f"{model_name}.pth"

    directory_name = 'weights'
    os.makedirs(directory_name, exist_ok=True)
    torch.save(ckpt, os.path.join(directory_name, file_name))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Process settings from a YAML file.')
    parser.add_argument('--config', type=str, default='configs/SRGAN.yaml',
                        help='Path to YAML configuration file')
    return parser.parse_args()


def read_settings(config_path):
    with open(config_path, 'r') as file:
        settings = yaml.safe_load(file)

    hostname = socket.gethostname()

    if hostname.endswith('local'):  # Example check for local machine names
        print("Running on Macbook locally")
        settings['dataset']['data_folder'] = settings['dataset']['data_folder_local']
    else:
        print(f"Running on remote server: {hostname}")
        settings['dataset']['data_folder'] = settings['dataset']['data_folder_hyperion']

    del settings['dataset']['data_folder_local'], settings['dataset']['data_folder_hyperion']
    return settings


def prime_factors(n):
    factors = []
    # While n is divisible by 2, add 2 as a factor and divide n by 2
    while n % 2 == 0:
        factors.append(2)
        n //= 2

    # Now n must be odd. Start checking for factors from 3 up to the square root of n
    for i in range(3, int(n**0.5) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n //= i

    # If n is still greater than 2, it means it's a prime number itself
    if n > 2:
        factors.append(n)

    return factors


def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(
            f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result
    return wrapper


class Timer:
    def __init__(self) -> None:
        self.timestamps = []

    def step(self, label: str):
        self.timestamps.append((label, time.perf_counter()))

    def reset(self):
        self.timestamps = []

    def display(self):
        if len(self.timestamps) < 2:
            print("No timestamps to display.")
            return

        for i in range(1, len(self.timestamps)):
            print(
                f"{self.timestamps[i][0]}: {self.timestamps[i][1] - self.timestamps[i-1][1]:.6f} seconds")

        self.reset()


def calculate_gradient_penalty(discriminator, real_imgs, fake_imgs):
    """
    Calculates the gradient penalty for Wasserstein GAN training.

    Args:
        discriminator: The discriminator network.
        real_imgs: A tensor of real images.
        fake_imgs: A tensor of fake images.

    Returns:
        A tensor representing the gradient penalty loss.
    """
    # Sample epsilon for interpolation
    epsilon = torch.rand(real_imgs.size(0), 1, 1, 1).to(device)

    # Interpolate between real and fake images
    mixed_images = epsilon * real_imgs + (1 - epsilon) * fake_imgs.detach()
    mixed_images.requires_grad = True

    # Compute gradients of the critic's output with respect to the mixed images
    mixed_discriminated = discriminator(mixed_images)
    mixed_gradients = torch.autograd.grad(
        outputs=mixed_discriminated,
        inputs=mixed_images,
        grad_outputs=torch.ones_like(mixed_discriminated),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Calculate gradient penalty
    gradient_penalty = ((mixed_gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
