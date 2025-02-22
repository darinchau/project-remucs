# Implements and archives the various discriminator models experimented with in the training of VQVAE
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from .vae import VQVAEConfig
from ..constants import TARGET_FEATURES
import torch.nn.functional as F
from torch import Tensor


class SpectrogramPatchModel(nn.Module):
    """This uses the idea of PatchGAN but changes the architecture to use Conv2d layers on each bar (4, 128, 512) patches

    Assumes input is of shape (B, 4, 512, 512), outputs a tensor of shape (B, 4, 4)"""

    def __init__(self):
        super(SpectrogramPatchModel, self).__init__()
        # Define a simple CNN architecture for each patch
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)  # Output: (B, 16, 128, 512)
        self.pool11 = nn.AdaptiveMaxPool2d((128, 256))
        self.pool12 = nn.AdaptiveAvgPool2d((64, 256))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Output: (B, 32, 64, 256)
        self.pool21 = nn.AdaptiveMaxPool2d((64, 128))
        self.pool22 = nn.AdaptiveAvgPool2d((32, 128))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: (B, 64, 32, 128)
        self.pool31 = nn.AdaptiveMaxPool2d((32, 32))
        self.pool32 = nn.AdaptiveAvgPool2d((8, 32))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: (B, 128, 8, 32)
        self.fc = nn.Conv2d(128, 4, (8, 32))  # Equivalent to FC layers over each channel

    def forward(self, x: Tensor):
        # x shape: (B, 4, 512, 512)
        # Splitting along the T axis into 4 patches
        patches = x.unflatten(2, (x.size(2) // 128, 128))  # Output: (B, 4, 4, 128, 512)

        # Process each patch
        batch_size, num_patches, channels, height, width = patches.size()
        patches = patches.reshape(-1, channels, height, width)  # Flatten patches for batch processing

        # Apply CNN
        x = self.conv1(patches)
        x = F.relu(x)
        x = self.pool11(x)
        x = self.pool12(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool21(x)
        x = self.pool22(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool31(x)
        x = self.pool32(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.fc(x)
        x = x.view(batch_size, num_patches, channels, -1).squeeze(-1).squeeze(-1)
        return x


class DiscriminatorDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn

        # First convolution layers with spectral normalization
        self.resnet_conv_first = nn.ModuleList()
        # Second convolution layers with spectral normalization
        self.resnet_conv_second = nn.ModuleList()
        # Residual convolution layers with spectral normalization
        self.residual_input_conv = nn.ModuleList()

        for i in range(num_layers):
            # First conv in ResNet block
            first_conv = nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                spectral_norm(nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )),
            )
            self.resnet_conv_first.append(first_conv)

            # Second conv in ResNet block
            second_conv = nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                spectral_norm(nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )),
            )
            self.resnet_conv_second.append(second_conv)

            # Residual connection
            res_conv = spectral_norm(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=1
            ))
            self.residual_input_conv.append(res_conv)

        # Attention mechanisms if enabled
        if self.attn:
            self.attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(num_layers)
            ])
            self.attentions = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True, bias=False)
                for _ in range(num_layers)
            ])

        # Downsample layer
        self.down_sample_conv = spectral_norm(
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        ) if down_sample else nn.Identity()

    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            # ResNet block
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out += self.residual_input_conv[i](resnet_input)

            # Attention block
            if self.attn:
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w).transpose(1, 2)
                in_attn = self.attention_norms[i](in_attn)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out += out_attn

        # Downsample
        out = self.down_sample_conv(out)
        return out


class ResnetDiscriminator(nn.Module):
    def __init__(self, in_channels: int, config: VQVAEConfig):
        super().__init__()
        # Initial convolution
        self.initial_conv = spectral_norm(
            nn.Conv2d(in_channels, config.down_channels[0], kernel_size=3, padding=1)
        )

        # Down blocks mirroring the VQVAE encoder structure
        self.down_blocks = nn.ModuleList()
        for i in range(len(config.down_channels) - 1):
            in_ch = config.down_channels[i]
            out_ch = config.down_channels[i+1]
            down_sample = config.down_sample[i] if i < len(config.down_sample) else False
            attn = config.attn_down[i] if i < len(config.attn_down) else False

            self.down_blocks.append(
                DiscriminatorDownBlock(
                    in_ch, out_ch,
                    down_sample=down_sample,
                    num_heads=config.num_heads,
                    num_layers=config.num_down_layers,
                    attn=attn,
                    norm_channels=config.norm_channels
                )
            )

        # Final layers
        self.final_norm = nn.GroupNorm(config.norm_channels, config.down_channels[-1])
        self.silu = nn.SiLU()
        self.fc = spectral_norm(nn.Linear(config.down_channels[-1], 1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Input shape: (B, 1, 512, 512)
        x = self.initial_conv(x)  # (B, C, 512, 512)

        for block in self.down_blocks:
            x = block(x)

        # Final processing
        x = self.final_norm(x)
        x = self.silu(x)

        # Global average pooling
        x = x.mean(dim=[2, 3])  # (B, C)
        x = self.fc(x)  # (B, 1)

        return x.squeeze(-1)  # (B,)
