# Implements VQVAE encoder and decoder model to convert spectrogram to latent space and back to spectrogram.
# Reference: https://github.com/explainingai-code/StableDiffusion-PyTorch

import torch
from torch import nn, Tensor
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from AutoMasher.fyp.audio.base.audio_collection import DemucsCollection
import yaml
from ..preprocess import spectro, ispectro
from typing import NamedTuple


@dataclass
class VAEConfig:
    down_channels: list[int]
    """List of number of channels in the downsample blocks."""
    mid_channels: list[int]
    """List of number of channels in the mid blocks."""
    down_sample: list[int]
    """List of boolean values to indicate if the downsample block should downsample."""
    num_down_layers: int
    """Number of layers in the downsample block."""
    num_mid_layers: int
    """Number of layers in the mid block"""
    num_up_layers: int
    """Number of layers in the upsample block."""
    nsources: int
    """Number of stems in the input audio"""
    nchannels: int
    """Number of channels in the input audio"""
    norm_channels: int
    """Number of channels in the group normalization layer."""
    num_heads: int
    """Number of heads in the multihead attention layer."""
    gradient_checkpointing: bool
    """If true, then gradient checkpointing should be used."""
    kl_mean: bool
    """If true, then a mean reduction strategy over the KL loss is used, otherwise a sum is used"""

    def __post_init__(self):
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert all(x <= 1 or x & 1 == 0 for x in self.down_sample)  # Otherwise upsample will not work


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down_sample: int,
        num_layers: int,
        norm_channels: int,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1),
            )
            for i in range(num_layers)
        ])

        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1),
            )
            for _ in range(num_layers)
        ])

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 2 * self.down_sample, self.down_sample, self.down_sample // 2) if self.down_sample > 1 else nn.Identity()

    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

        # Downsample
        out = self.down_sample_conv(out)
        return out


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        up_sample: int,
        num_layers: int,
        norm_channels: int,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1),
            )
            for i in range(num_layers)
        ])

        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1),
            )
            for _ in range(num_layers)
        ])

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])

        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, 2 * self.up_sample, self.up_sample, self.up_sample // 2) if self.up_sample > 1 else nn.Identity()

    def forward(self, x):
        # Upsample
        x = self.up_sample_conv(x)

        out = x
        for i in range(self.num_layers):
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
        return out


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
        norm_channels: int,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            for i in range(num_layers + 1)
        ])

        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            for _ in range(num_layers + 1)
        ])

        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(norm_channels, out_channels)
            for _ in range(num_layers)
        ])

        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True, bias=False)
            for _ in range(num_layers)
        ])

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers + 1)
        ])

    def forward(self, x):
        out = x

        # First resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):
            # Attention Block
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            if self.use_gradient_checkpointing:
                out_attn, _ = checkpoint(self.attentions[i], in_attn, in_attn, in_attn)  # type: ignore
            else:
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)
        return out


class VAEOutput(NamedTuple):
    out: Tensor | None
    z: Tensor | None
    mean: Tensor | None
    logvar: Tensor | None
    kl_loss: Tensor | None
    out_spec: Tensor | None
    in_spec: Tensor | None


def _should_compute(x: dict, compute: str) -> bool:
    return compute not in x or x[compute] is not None


class VAE(nn.Module):
    def __init__(self, model_config: VAEConfig):
        super().__init__()
        self.down_channels = model_config.down_channels
        self.mid_channels = model_config.mid_channels
        self.down_sample = model_config.down_sample
        self.num_down_layers = model_config.num_down_layers
        self.num_mid_layers = model_config.num_mid_layers
        self.num_up_layers = model_config.num_up_layers
        self.gradient_checkpointing = model_config.gradient_checkpointing
        self.kl_mean = model_config.kl_mean

        # Latent Dimension
        self.nsources = model_config.nsources
        self.norm_channels = model_config.norm_channels
        self.num_heads = model_config.num_heads
        self.nchannels = model_config.nchannels

        # Reverse the downsample list to get upsample list
        self.up_sample = list(reversed(self.down_sample))

        ##################### Encoder ######################
        # in_channles = source * (C = 2) * 2
        self.encoder_conv_in = nn.Conv2d(self.nsources * self.nchannels * 2, self.down_channels[0], kernel_size=3, padding=(1, 1))

        # Downblock + Midblock
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(DownBlock(self.down_channels[i], self.down_channels[i + 1],
                                                 down_sample=self.down_sample[i],
                                                 num_layers=self.num_down_layers,
                                                 norm_channels=self.norm_channels,
                                                 use_gradient_checkpointing=self.gradient_checkpointing))

        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1],
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels,
                                              use_gradient_checkpointing=self.gradient_checkpointing))

        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], self.nsources, kernel_size=3, padding=1)
        self.pre_encode_conv = nn.Conv2d(self.nsources, 2*self.nsources, kernel_size=1)

        ##################### Decoder ######################

        self.post_encode_conv = nn.Conv2d(self.nsources, self.nsources, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.nsources, self.mid_channels[-1], kernel_size=3, padding=(1, 1))

        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i - 1],
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels,
                                              use_gradient_checkpointing=self.gradient_checkpointing))

        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(UpBlock(self.down_channels[i], self.down_channels[i - 1],
                                               up_sample=self.down_sample[i - 1],
                                               num_layers=self.num_up_layers,
                                               norm_channels=self.norm_channels,
                                               use_gradient_checkpointing=self.gradient_checkpointing))

        # out_channels = C * 2
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], self.nchannels * 2, kernel_size=3, padding=1)

    def _preprocess(self, x: Tensor, check: bool = True):
        # input shape: (batch, source, channel, time)
        assert len(x.shape) == 4
        if check:
            assert x.shape[1] == self.nsources
            assert x.shape[2] == self.nchannels
        Tx = x.shape[-1]
        z = spectro(x)
        B, S, C, Fq, T = z.shape
        x = torch.view_as_real(z).permute(0, 1, 2, 5, 3, 4)
        x = x.reshape(B, S, C * 2, Fq, T)
        mean = x.mean(dim=(1, 3, 4), keepdim=True)
        std = x.std(dim=(1, 3, 4), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        x = x.reshape(B, S * C * 2, Fq, T)

        # Output x: (B, S * C*2, F, T)
        if check:
            assert x.shape == (B, S * C * 2, Fq, T), f"Expected {(B, S * C * 2, Fq, T)}, got {x.shape}"
        return x, mean, std, (B, Fq, T, Tx)

    def _postprocess(self, x: Tensor, mean, std, shapes):
        B, Fq, T, Tx = shapes
        x = x * std[:, 0] + mean[:, 0]
        # x shape: (B, C*2, F, T)
        assert x.shape == (B, self.nchannels * 2, Fq, T), f"Expected {(B, self.nchannels * 2, Fq, T)}, got {x.shape}"
        # Reverse the process
        x = x.reshape(B, self.nchannels, 2, Fq, T).permute(0, 1, 3, 4, 2)
        z = torch.view_as_complex(x.contiguous())
        x = ispectro(z, length=Tx)
        return x

    def encode(self, x):
        out = self.encoder_conv_in(x)
        for idx, down in enumerate(self.encoder_layers):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_encode_conv(out)
        mean, logvar = torch.chunk(out, 2, dim=1)
        if self.kl_mean:
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        else:
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return mean, logvar, kl_loss

    def decode(self, z):
        out = z
        out = self.post_encode_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out)

        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out

    def forward(self, x, **kwargs) -> VAEOutput:
        """Specify in the kwargs x=None to not return x. For example, forward(x, mean=None)"""
        in_spec, mx, std, shapes = self._preprocess(x)
        mean, logvar, kl_loss = self.encode(in_spec)
        z = mean + torch.exp(0.5 * logvar) * torch.randn(mean.shape).to(device=in_spec.device)

        compute_out = _should_compute(kwargs, "out")
        compute_out_spec = _should_compute(kwargs, "out_spec") or compute_out
        if compute_out_spec:
            out_spec = self.decode(z)
            if compute_out:
                out = self._postprocess(out_spec, mx, std, shapes)
            else:
                out = None
        else:
            out = None
            out_spec = None
        return VAEOutput(
            out=out if _should_compute(kwargs, "out") else None,
            z=z if _should_compute(kwargs, "z") else None,
            mean=mean if _should_compute(kwargs, "mean") else None,
            logvar=logvar if _should_compute(kwargs, "logvar") else None,
            kl_loss=kl_loss if _should_compute(kwargs, "kl_loss") else None,
            out_spec=out_spec if _should_compute(kwargs, "out_spec") else None,
            in_spec=in_spec if _should_compute(kwargs, "in_spec") else None
        )


def read_config(config_path: str):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error reading config file: {exc}")
    try:
        return VAEConfig(**config['autoencoder_params'])
    except KeyError:
        try:
            return VAEConfig(
                down_channels=config["down_channels"],
                mid_channels=config["mid_channels"],
                down_sample=config["down_sample"],
                num_down_layers=config["num_down_layers"],
                num_mid_layers=config["num_mid_layers"],
                num_up_layers=config["num_up_layers"],
                nsources=config["nsources"],
                nchannels=config["nchannels"],
                norm_channels=config["norm_channels"],
                num_heads=config["num_heads"],
                gradient_checkpointing=config["gradient_checkpointing"],
                kl_mean=config["kl_mean"]
            )
        except KeyError:
            raise ValueError("Config file does not contain the correct keys")
