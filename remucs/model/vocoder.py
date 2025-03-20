from torch.utils.data import Dataset
from scipy.io.wavfile import read
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
from argparse import Namespace
import random
import math
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from typing import Any
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import Tuple, Optional
from scipy.signal import get_window
import glob
import json
import os
import torch
from sys import stderr
from typing import Optional
from torch.nn.utils import weight_norm
import os
import shutil
from pathlib import Path
from dataclasses import dataclass

from AutoMasher.fyp import Audio

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Shut up


@dataclass
class WandbConfig:
    project: str
    log_every_n_steps: int


@dataclass
class DistConfig:
    dist_backend: str
    dist_addr: str
    dist_port: str
    world_size: int


@dataclass
class VocoderConfig:
    resblock: str
    num_gpus: int
    batch_size: int
    learning_rate: float
    adam_b1: float
    adam_b2: float
    lr_decay: float
    seed: int
    upsample_rates: list[int]
    upsample_kernel_sizes: list[int]
    upsample_initial_channel: int
    resblock_kernel_sizes: list[int]
    resblock_dilation_sizes: list[list[int]]
    istft_filter_length: int
    istft_hop_length: int
    discriminator_periods: list[int]
    segment_size: int
    num_mels: int
    n_fft: int
    hop_size: int
    win_size: int
    sampling_rate: int
    fmin: int
    fmax: int
    fm_scale_factor: int
    num_workers: int
    wandb: WandbConfig
    dist_config: DistConfig

    @staticmethod
    def load(file_path: str) -> 'VocoderConfig':
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Convert the JSON dictionary into the dataclass
        wandb_config = WandbConfig(**data['wandb'])
        dist_config = DistConfig(**data['dist_config'])
        # Remove 'wandb' and 'dist_config' from data as they need special handling
        del data['wandb']
        del data['dist_config']
        return VocoderConfig(**data, wandb=wandb_config, dist_config=dist_config)


def build_env(config_path: str, ckpt_path: str):
    """Copies config to the checkpoint directory"""
    config_name = Path(config_path).name
    target_path = Path(ckpt_path).joinpath(config_name)
    if config_path != target_path:
        os.makedirs(ckpt_path, exist_ok=True)
        shutil.copyfile(config_path, Path(ckpt_path).joinpath(config_name))


def init_weights(m: torch.nn.Module, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m: torch.nn.Module):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def load_checkpoint(filepath: str, device) -> Optional[dict]:
    logger.info(f"Loading {filepath}")
    assert os.path.isfile(filepath)
    checkpoint_dict = torch.load(filepath, map_location=device)
    logger.info("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    logger.info(f"Saving checkpoint to {filepath}")
    torch.save(obj, filepath)
    logger.info("Complete.")


def scan_checkpoint(cp_dir: str, prefix: str) -> Optional[str]:
    """
    Returns the latest checkpoint from the directory.
    """
    pattern = os.path.join(cp_dir, prefix + "????????")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


class TorchSTFT(torch.nn.Module):
    def __init__(self, istft_filter_length: int, istft_hop_length: int, window: str = "hann", **_):
        super().__init__()
        self.hop_length = istft_hop_length
        self.win_length = istft_filter_length
        self.filter_length = istft_filter_length
        self.window = torch.from_numpy(get_window(window, istft_filter_length, fftbins=True).astype(np.float32))

    def transform(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        forward_transform = torch.stft(
            input_data, self.filter_length, self.hop_length, self.win_length, window=self.window, return_complex=True
        )
        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window.to(magnitude.device),
        )
        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        magnitude, phase = self.transform(input_data)
        reconstruction = self.inverse(magnitude, phase)
        return reconstruction


LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for i in range(3)
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, config: VocoderConfig):
        super(Generator, self).__init__()
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, config.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if config.resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        config.upsample_initial_channel // (2**i),
                        config.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.post_n_fft = config.istft_filter_length
        self.conv_post = weight_norm(Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))  # type: ignore
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            for n, j in enumerate(range(self.num_kernels)):
                if n == 0:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)  # type: ignore
            x = xs / self.num_kernels  # type: ignore
        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])

        return spec, phase

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> tuple[list[torch.Tensor], ...]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> tuple[list[torch.Tensor], ...]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r: list[torch.Tensor], fmap_g: list[torch.Tensor]) -> torch.Tensor:
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss  # type: ignore


def discriminator_loss(
    disc_real_outputs: list[torch.Tensor], disc_generated_outputs: list[torch.Tensor]
) -> tuple[Any, ...]:
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses  # type: ignore


def spectral_normalize(magnitudes: torch.Tensor, clip_val: float = 1e-5):
    output = torch.log(torch.clamp(magnitudes, min=clip_val))
    return output


def get_mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False, **kwargs
) -> torch.Tensor:
    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel).float().to(y.device)
    hann_window = torch.hann_window(win_size).to(y.device)
    pad_value = int((n_fft - hop_size) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad_value, pad_value), mode="reflect").squeeze(1)
    spectrogram = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )
    spectrogram = torch.sqrt(spectrogram.pow(2).sum(-1) + 1e-9)
    mel_spectrogram = torch.matmul(mel_basis, spectrogram)
    normalized_mel_spectrogram = spectral_normalize(mel_spectrogram)
    return normalized_mel_spectrogram


def get_dataset_filelist(args: Namespace) -> tuple[list, list]:
    with open(args.input_training_file, "r", encoding="utf-8") as f:
        training_files = [i[:-1] for i in f.readlines()]
    with open(args.input_validation_file, "r", encoding="utf-8") as f:
        validation_files = [i[:-1] for i in f.readlines()]
    return training_files, validation_files


class MelDataset(Dataset):
    def __init__(
        self,
        training_files,
        segment_size,
        n_fft,
        num_mels,
        hop_size,
        win_size,
        sampling_rate,
        fmin,
        fmax,
        seed,
        split=True,
        device=None,
        fmax_loss=None,
        fine_tuning=False,
        input_mels_dir=None,
        **kwargs,
    ):
        random.seed(seed)
        self.fmin = fmin
        self.fmax = fmax
        self.split = split
        self.n_fft = n_fft
        self.device = device
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmax_loss = fmax_loss
        self.fine_tuning = fine_tuning
        self.audio_files = training_files
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.base_mels_path = input_mels_dir
        self.frames_per_sec = math.ceil(segment_size / hop_size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        filename = self.audio_files[index]
        audio = Audio.load(filename).resample(self.sampling_rate).to_nchannels(1)

        if not self.fine_tuning:
            if self.split:
                if audio.nframes >= self.segment_size:
                    max_audio_start = audio.nframes - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio.slice_frames(audio_start, audio_start + self.segment_size)
                else:
                    audio = audio.pad(target=self.segment_size, front=False)

            mel = get_mel_spectrogram(
                audio,
                self.n_fft,
                self.num_mels,
                self.sampling_rate,
                self.hop_size,
                self.win_size,
                self.fmin,
                self.fmax,
                center=False,
            )
        else:
            raise NotImplementedError("Fine tuning not implemented (yet)")

        mel_loss = get_mel_spectrogram(
            audio.data,
            self.n_fft,
            self.num_mels,
            self.sampling_rate,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax_loss,
            center=False,
        )

        return mel.squeeze(), audio.data[0], filename, mel_loss.squeeze()

    def cut_mel_audio(self, mel: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mel_start = random.randint(0, mel.size(2) - self.frames_per_sec - 1)
        mel = mel[:, :, mel_start: mel_start + self.frames_per_sec]
        audio = audio[:, mel_start * self.hop_size: (mel_start + self.frames_per_sec) * self.hop_size]
        return mel, audio

    def pad_mel_audio(self, mel: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mel = torch.nn.functional.pad(mel, (0, self.frames_per_sec - mel.size(2)), "constant")
        audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), "constant")
        return mel, audio

    def __len__(self) -> int:
        return len(self.audio_files)
