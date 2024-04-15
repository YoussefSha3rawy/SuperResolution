import torch
import os
from torch import nn
from utils import prime_factors
from torchvision.models import vgg19, VGG19_Weights


class _ResidualBlock(nn.Module):
    def __init__(self, num_channels, kernel_size) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_channels, num_channels,
                      kernel_size, 1, int(kernel_size / 2)),
            nn.BatchNorm2d(num_channels),
            nn.PReLU(),
            nn.Conv2d(num_channels, num_channels,
                      kernel_size, 1, int(kernel_size / 2)),
            nn.BatchNorm2d(num_channels)
        )

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
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
    def __init__(self, scaling_factor, checkpoint=None) -> None:
        super().__init__()

        assert scaling_factor in [2, 3, 4, 6, 8]

        self.channel_size = 64
        self.small_kernel_size = 3
        self.large_kernel_size = 9
        self.num_residual_blocks = 16
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, self.channel_size, self.large_kernel_size, 1,
                      int(self.large_kernel_size / 2)),
            nn.PReLU()
        )

        self.residual_blocks = self._make_residual_blocks(
            self.num_residual_blocks)

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

    def _make_residual_blocks(self, n_blocks):
        layers = [_ResidualBlock(
            self.channel_size, self.small_kernel_size) for _ in range(n_blocks)]
        return nn.Sequential(*layers)

    def _make_upscale_blocks(self, scaling):
        upscale_factors = prime_factors(scaling)
        layers = [_SubPixelConvolutionalBlock(
            self.channel_size, self.small_kernel_size, upscale_factor) for upscale_factor in upscale_factors]
        return nn.Sequential(*layers)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(os.path.join('weights', checkpoint))
        model_weights = ckpt['model_weights']
        self.load_state_dict(model_weights)
        print("Model's pretrained weights loaded!")


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
