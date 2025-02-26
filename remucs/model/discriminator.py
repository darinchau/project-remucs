# Implements and archives the various discriminator models experimented with in the training of VQVAE
# Assuming except SpectrogramPatchModel, all other models take input (b, 512, 512) and output (b,)
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from .vae import VAEConfig
import torch.nn.functional as F
from torch import Tensor


class DiscriminatorDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample, num_layers, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample

        self.resnet_conv_first = nn.ModuleList()
        self.resnet_conv_second = nn.ModuleList()
        self.residual_input_conv = nn.ModuleList()

        for i in range(num_layers):
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
    def __init__(self, config: VAEConfig):
        super().__init__()
        # Initial convolution
        self.initial_conv = spectral_norm(
            nn.Conv2d(1, config.down_channels[0], kernel_size=3, padding=1)
        )

        # Down blocks mirroring the VQVAE encoder structure
        self.down_blocks = nn.ModuleList()
        for i in range(len(config.down_channels) - 1):
            in_ch = config.down_channels[i]
            out_ch = config.down_channels[i+1]
            down_sample = config.down_sample[i] if i < len(config.down_sample) else False

            self.down_blocks.append(
                DiscriminatorDownBlock(
                    in_ch, out_ch,
                    down_sample=down_sample,
                    num_layers=config.num_down_layers,
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

        x = self.initial_conv(x)

        for block in self.down_blocks:
            x = block(x)

        x = self.final_norm(x)
        x = self.silu(x)
        x = x.mean(dim=[2, 3])  # (B, C)
        x = self.fc(x)  # (B, 1)

        return x.squeeze(-1)  # (B,)
