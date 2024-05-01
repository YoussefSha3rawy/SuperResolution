import torch.nn as nn
import torch
import os
from torch import nn
from utils import prime_factors
from torchvision.models import vgg19, VGG19_Weights
import functools


class _ResidualBlock(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int, num_heads: int) -> None:
        """
        Initializes a residual block with optional multi-head attention.

        :param num_channels: Number of channels in the input and output.
        :param kernel_size: Size of the convolution kernel.
        :param num_heads: Number of heads in the multi-head attention; if zero, attention is skipped.
        """
        super().__init__()
        # Conditional initialization of the attention layer if num_heads is greater than 0
        if num_heads > 0:
            self.attention = nn.MultiheadAttention(
                num_channels, num_heads=num_heads, dropout=0.1, batch_first=True)
        # Standard convolution block with two convolutions, batch normalization, and PReLU activation
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
        """
        Passes the input through the block, applying attention (if initialized) and a convolution block.

        :param x: Input tensor.
        :return: Output tensor after processing through the residual block.
        """
        residual = x  # Save original input for residual connection
        out = x
        # Apply attention if the module has been initialized with it
        if hasattr(self, 'attention'):
            batch_size, num_channels, width, height = x.size()
            # Reshape and permute for the multi-head attention input
            x_reshaped = x.view(batch_size, num_channels,
                                width * height).permute(0, 2, 1)
            out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
            out = out.permute(0, 2, 1).view(
                batch_size, num_channels, width, height)
        out = self.conv_block(out)  # Apply convolutional layers
        out += residual  # Add the original input to the output
        return out


class _SubPixelConvolutionalBlock(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int, scaling: int) -> None:
        """
        Initializes a sub-pixel convolution block used for upscaling.

        :param num_channels: Number of input channels.
        :param kernel_size: Size of the convolution kernel.
        :param scaling: Scaling factor for pixel shuffling, e.g., 2 for doubling resolution.
        """
        super().__init__()
        # Upscaling block with convolution, pixel shuffle for resolution increase, and PReLU activation
        self.upscale_block = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * (scaling ** 2),
                      kernel_size, 1, int(kernel_size / 2)),
            nn.PixelShuffle(scaling),
            nn.PReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor through the sub-pixel convolution block to upscale it.

        :param x: Input tensor.
        :return: Upscaled output tensor.
        """
        x = self.upscale_block(x)
        return x


class SRResNet(nn.Module):
    def __init__(self, scaling_factor: int, num_heads: int = 0, checkpoint: str = None) -> None:
        """
        Initializes the SRResNet model with scaling factor and optional attention.

        :param scaling_factor: Factor to upscale the input image.
        :param num_heads: Number of attention heads; zero if attention is not used.
        :param checkpoint: Path to the pre-trained model weights.
        """
        super().__init__()
        assert scaling_factor in [
            2, 3, 4, 6, 8], "Scaling factor must be one of the specified values."

        # Initial configuration and the first block with a convolution and PReLU activation
        self.num_heads = num_heads
        self.channel_size = 64
        self.small_kernel_size = 3
        self.large_kernel_size = 9
        self.num_residual_blocks = 16
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, self.channel_size, self.large_kernel_size,
                      1,     int(self.large_kernel_size / 2)),
            nn.PReLU()
        )

        # Use functools.partial to configure residual blocks with predefined parameters
        residual_block_f = functools.partial(
            _ResidualBlock, num_channels=self.channel_size, kernel_size=self.small_kernel_size, num_heads=num_heads)

        # Create multiple residual blocks
        self.residual_blocks = self._make_layer(
            residual_block_f, self.num_residual_blocks)

        # Second block with a convolution and batch normalization
        self.block_2 = nn.Sequential(
            nn.Conv2d(self.channel_size, self.channel_size,
                      self.small_kernel_size, 1, int(self.small_kernel_size / 2)),
            nn.BatchNorm2d(self.channel_size)
        )

        # Scaling blocks for upscaling the image
        self.upscale_blocks = self._make_upscale_blocks(scaling_factor)

        # Final block with a convolution to reduce to 3 channels and Tanh activation for image output
        self.block_3 = nn.Sequential(
            nn.Conv2d(self.channel_size, 3, self.large_kernel_size, 1,
                      int(self.large_kernel_size / 2)),
            nn.Tanh()
        )

        # Load pretrained model weights if a checkpoint is provided
        if checkpoint is not None and checkpoint != 'None':
            self.load_checkpoint(checkpoint)

    def forward(self, x):
        """
        Forward pass of the SRResNet model.

        :param x: Input tensor (image).
        :return: Processed tensor (image) after super-resolution.
        """
        out = self.block_1(x)
        residual = out  # Store output of the first block for adding back later
        out = self.residual_blocks(out)
        out = self.block_2(out)
        out += residual  # Add back the first block's output to introduce a residual connection
        out = self.upscale_blocks(out)
        out = self.block_3(out)
        return out

    def _make_layer(self, block, n_blocks):
        """
        Helper method to create a sequence of blocks.

        :param block: Block function or constructor.
        :param n_blocks: Number of blocks to create.
        :return: Sequential module containing the blocks.
        """
        layers = [block() for _ in range(n_blocks)]
        return nn.Sequential(*layers)

    def _make_upscale_blocks(self, scaling):
        """
        Helper method to create upscaling blocks based on the scaling factor.

        :param scaling: Scaling factor for the image.
        :return: Sequential module containing upscaling blocks.
        """
        upscale_factors = prime_factors(
            scaling)
        layers = [_SubPixelConvolutionalBlock(
            self.channel_size, self.small_kernel_size, upscale_factor) for upscale_factor in upscale_factors]
        return nn.Sequential(*layers)

    def load_checkpoint(self, checkpoint):
        """
        Loads model weights from a checkpoint.

        :param checkpoint: Filepath to the checkpoint.
        """
        ckpt = torch.load(os.path.join(
            'weights', checkpoint), map_location='cpu')
        model_weights = ckpt['model_weights']
        self.load_state_dict(model_weights)
        print("Model's pretrained weights loaded!")

    def __str__(self) -> str:
        """
        String representation of the model, indicating if attention is used.

        :return: Model name with or without '_attention'.
        """
        return f'{self.__class__.__name__}{"_attention" if self.num_heads > 0 else ""}'


class TruncatedVGG19(nn.Module):
    def __init__(self, i):
        """
        Initializes a truncated VGG19 model up to the ith layer.

        :param i: Index of the last layer to include from the VGG19 model.
        """
        super(TruncatedVGG19, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        # Truncate the VGG19 at the specified layer
        self.truncated_vgg19 = nn.Sequential(
            *list(vgg.features.children())[:i])

    def forward(self, input):
        """
        Forward pass through the truncated VGG19 model.

        :param input: Input tensor (image).
        :return: Feature map from the truncated VGG19 model.
        """
        output = self.truncated_vgg19(input)
        return output


class Discriminator(nn.Module):
    """
    The Discriminator network for SRGAN. This network is designed to distinguish between 
    super-resolved images generated by the generator and original high-resolution images.
    """

    def __init__(self, in_channels=3, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024, checkpoint=None):
        """
        Initializes the Discriminator network with specified parameters for convolutional and fully connected layers.

        Args:
            in_channels (int, optional): Number of input channels for the images. Defaults to 3 (RGB images).
            kernel_size (int, optional): Kernel size for the convolution layers. Defaults to 3.
            n_channels (int, optional): Number of output channels in the initial convolution layer. Defaults to 64.
            n_blocks (int, optional): Number of convolutional blocks to use in the network. Defaults to 8.
            fc_size (int, optional): Size of the output feature dimension for the first fully connected layer. Defaults to 1024.
            checkpoint (str, optional): Path to a pre-trained model checkpoint for loading pre-existing weights.
        """
        super().__init__()

        # Define the initial layer of the discriminator
        layers = [
            nn.Conv2d(in_channels, n_channels, kernel_size,
                      1, int(kernel_size / 2)),
            nn.LeakyReLU(0.2)
        ]

        # Dynamically create the convolutional blocks, doubling the number of channels every second block
        in_channels = n_channels
        for i in range(1, n_blocks):
            out_channels = in_channels * 2 if i % 2 == 0 else in_channels
            # Stride affects the dimensionality reduction, stride of 2 reduces size
            stride = 2 if i % 2 == 1 else 1
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride, int(kernel_size / 2)),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            ])
            in_channels = out_channels

        # Sequential container for convolutional blocks
        self.conv_blocks = nn.Sequential(*layers)

        # Adaptive average pooling to reduce spatial dimensions to a fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        # Fully connected layers to transform the feature maps into a single output score
        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)
        self.lrelu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(fc_size, 1)  # Output is a single score

        # Load weights from checkpoint if provided
        if checkpoint is not None and checkpoint != 'None':
            self.load_checkpoint(checkpoint)

    def forward(self, x):
        """
        Defines the forward pass through the discriminator.

        Args:
            x (torch.Tensor): The input tensor representing images, shape (N, in_channels, H, W).

        Returns:
            torch.Tensor: Returns a tensor of shape (N,), containing a score (logit) indicating 
                          whether each input image is a real high-resolution image or a fake generated one.
        """
        for block in self.conv_blocks:
            x = block(x)  # Pass input through each convolution block
        # Apply adaptive pooling to standardize output size
        x = self.adaptive_pool(x)
        # Flatten the features for the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc1(x)  # First fully connected layer
        x = self.lrelu(x)  # Apply LeakyReLU activation
        # Final fully connected layer to produce the output score
        logit = self.fc2(x)

        return logit

    def load_checkpoint(self, checkpoint):
        """
        Loads model weights from a specified checkpoint.

        Args:
            checkpoint (str): Path to the checkpoint file.
        """
        ckpt = torch.load(os.path.join(
            'weights', checkpoint), map_location='cpu')
        model_weights = ckpt['model_weights']
        self.load_state_dict(model_weights)
        print("Model's pretrained weights loaded!")
