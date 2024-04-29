import torch.nn as nn
import torch
import os
from torch import nn
from utils import prime_factors
from torchvision.models import vgg19, VGG19_Weights
import functools


class _ResidualBlock(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int, num_heads: int) -> None:
        super().__init__()
        if num_heads > 0:
            self.attention = nn.MultiheadAttention(
                num_channels, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_channels, num_channels,
                      kernel_size, 1, int(kernel_size / 2)),
            nn.BatchNorm2d(num_channels),
            nn.PReLU(),
            nn.Conv2d(num_channels, num_channels,
                      kernel_size, 1, int(kernel_size / 2)),
            nn.BatchNorm2d(num_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = x
        # Apply attention mechanism if available
        if hasattr(self, 'attention'):
            batch_size, num_channels, width, height = x.size()
            # Reshape for attention mechanism
            x_reshaped = x.view(batch_size, num_channels,
                                width * height).permute(0, 2, 1)
            # Apply attention
            out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
            # Reshape back to original shape
            out = out.permute(0, 2, 1).view(
                batch_size, num_channels, width, height)

        # Apply convolutional block
        out = self.conv_block(out)
        # Add residual connection
        out += residual
        return out


class _SubPixelConvolutionalBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, scaling) -> None:
        super().__init__()
        self.upscale_block = nn.Sequential(
            nn.Conv2d(num_channels, num_channels *
                      (scaling ** 2), kernel_size, 1, int(kernel_size / 2)),
            nn.PixelShuffle(scaling),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.upscale_block(x)
        return x


class SRResNet(nn.Module):
    def __init__(self, scaling_factor, num_heads=0, checkpoint=None) -> None:
        super().__init__()

        assert scaling_factor in [2, 3, 4, 6, 8]

        self.num_heads = num_heads

        self.channel_size = 64
        self.small_kernel_size = 3
        self.large_kernel_size = 9
        self.num_residual_blocks = 16
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, self.channel_size, self.large_kernel_size, 1,
                      int(self.large_kernel_size / 2)),
            nn.PReLU()
        )

        residual_block_f = functools.partial(
            _ResidualBlock, num_channels=self.channel_size, kernel_size=self.small_kernel_size, num_heads=num_heads)

        self.residual_blocks = self._make_layer(
            residual_block_f, self.num_residual_blocks)

        self.block_2 = nn.Sequential(
            nn.Conv2d(self.channel_size, self.channel_size,
                      self.small_kernel_size, 1, int(self.small_kernel_size / 2)),
            nn.BatchNorm2d(self.channel_size)
        )

        self.upscale_blocks = self._make_upscale_blocks(scaling_factor)

        self.block_3 = nn.Sequential(
            nn.Conv2d(self.channel_size, 3, self.large_kernel_size,
                      1, int(self.large_kernel_size / 2)),
            nn.Tanh()
        )

        if checkpoint is not None and checkpoint != 'None':
            self.load_checkpoint(checkpoint)

    def forward(self, x):
        out = self.block_1(x)
        residual = out
        out = self.residual_blocks(out)
        out = self.block_2(out)
        out += residual
        out = self.upscale_blocks(out)
        out = self.block_3(out)

        return out

    def _make_layer(self, block, n_blocks):
        layers = [block() for _ in range(n_blocks)]
        return nn.Sequential(*layers)

    def _make_upscale_blocks(self, scaling):
        upscale_factors = prime_factors(scaling)
        layers = [_SubPixelConvolutionalBlock(
            self.channel_size, self.small_kernel_size, upscale_factor) for upscale_factor in upscale_factors]
        return nn.Sequential(*layers)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(os.path.join(
            'weights', checkpoint), map_location='cpu')
        model_weights = ckpt['model_weights']
        self.load_state_dict(model_weights)
        print("Model's pretrained weights loaded!")

    def __str__(self) -> str:
        return f'{self.__class__.__name__}{"_attention" if self.num_heads > 0 else ""}'


class TruncatedVGG19(nn.Module):
    def __init__(self, i):
        super(TruncatedVGG19, self).__init__()

        vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        self.truncated_vgg19 = nn.Sequential(
            *list(vgg.features.children())[:i])

    def forward(self, input):
        output = self.truncated_vgg19(
            input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

        return output


class Discriminator(nn.Module):
    """
    The discriminator network in the SRGAN model.
    """

    def __init__(self, in_channels=3, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024, checkpoint=None):
        """
        Initialize the discriminator.

        Args:
            in_channels (int, optional): Number of input channels. Default is 3.
            kernel_size (int, optional): Kernel size for convolutional blocks. Default is 3.
            n_channels (int, optional): Number of output channels in the first convolutional block. Default is 64.
            n_blocks (int, optional): Number of convolutional blocks. Default is 8.
            fc_size (int, optional): Size of the first fully connected layer. Default is 1024.
        """
        super().__init__()

        layers = []
        layers.extend([
            nn.Conv2d(in_channels, n_channels, kernel_size,
                      1, int(kernel_size / 2)),
            nn.LeakyReLU(0.2)
        ])

        # Subsequent convolutional blocks
        in_channels = n_channels
        for i in range(1, n_blocks):
            out_channels = in_channels * 2 if i % 2 == 0 else in_channels
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          2 if i % 2 == 1 else 1, int(kernel_size / 2)),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            ])
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*layers)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)
        self.lrelu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(fc_size, 1)

        if checkpoint is not None and checkpoint != 'None':
            self.load_checkpoint(checkpoint)

    def forward(self, x):
        """
        Forward propagation.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W).

        Returns:
            torch.Tensor: A score (logit) for whether the input is a high-resolution image, of shape (N).
        """
        for block in self.conv_blocks:
            x = block(x)
        x = self.adaptive_pool(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.lrelu(x)
        logit = self.fc2(x)

        return logit

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(os.path.join(
            'weights', checkpoint), map_location='cpu')
        model_weights = ckpt['model_weights']
        self.load_state_dict(model_weights)
        print("Model's pretrained weights loaded!")
