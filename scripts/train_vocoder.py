# Referenced from wavenet_vocoder/train.py
import matplotlib
from remucs.constants import TARGET_FEATURES, TARGET_NFRAMES, TARGET_TIME_FRAMES, SAMPLE_RATE
import numpy as np
from glob import glob
import json
import random
from datetime import datetime
from tqdm import tqdm
from os.path import dirname, join, expanduser, exists
from dataclasses import dataclass, field
import wandb
from torch.utils.data import DataLoader
from warnings import warn
from matplotlib import cm
import librosa.display
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from nnmnkwii import preprocessing as P
from torch.utils.data.sampler import Sampler
from torch.utils import data as data_utils
import torch.backends.cudnn as cudnn
from torch import optim
from torch.nn import functional as F
from torch import nn
import torch
import matplotlib.pyplot as plt
from docopt import docopt
import torch.nn as nn
import torch.nn.functional as F


matplotlib.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class HParams:
    sample_rate: int
    nframes: int  # Target nframes for the unraveled audio
    in_time_frames: int
    in_mel_bins: int

    out_channels: int
    layers: int
    stacks: int
    residual_channels: int
    gate_channels: int
    skip_out_channels: int
    cin_channels: int
    gin_channels: int
    dropout: float
    kernel_size: int
    cin_pad: int
    upsample_conditional_features: bool
    upsample_params: dict
    scalar_input: bool
    output_distribution: str
    learning_rate: float
    max_train_steps: int
    checkpoint_interval: int


class WaveNetVocoder(nn.Module):
    def __init__(self, mel_bins, num_time_frames, num_audio_samples, num_layers, kernel_size, residual_channels, dilation_channels, skip_channels):
        super(WaveNetVocoder, self).__init__()
        self.mel_bins = mel_bins
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.num_time_frames = num_time_frames
        self.num_audio_samples = num_audio_samples

        # Initial causal convolution
        self.causal_conv = nn.Conv1d(mel_bins, residual_channels, kernel_size=1)

        # Dilated convolutions
        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        dilation = 1
        for i in range(num_layers):
            self.dilated_convs.append(nn.Conv1d(residual_channels, dilation_channels, kernel_size, dilation=dilation))
            self.residual_convs.append(nn.Conv1d(dilation_channels, residual_channels, kernel_size=1))
            self.skip_convs.append(nn.Conv1d(dilation_channels, skip_channels, kernel_size=1))
            dilation *= 2

        # Post-processing layers
        self.post_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.post_conv2 = nn.Conv1d(skip_channels, 1, kernel_size=1)

    def forward(self, x):
        # x = x.permute(0, 2, 1)  # Change to (B, time_frames, mel_bins)
        x = self.causal_conv(x)

        skip_connections = []

        for dil_conv, res_conv, skip_conv in zip(self.dilated_convs, self.residual_convs, self.skip_convs):
            filtered = dil_conv(x)
            filtered = torch.tanh(filtered)
            skip = skip_conv(filtered)
            skip_connections.append(skip)

            residual = res_conv(filtered)
            x = x + residual

        x = torch.sum(torch.stack(skip_connections), dim=0)
        x = F.tanh(x)
        x = self.post_conv1(x)
        x = F.tanh(x)
        x = self.post_conv2(x)

        return x.squeeze(1)


def main(hparams_path: str):
    raise NotImplementedError("This script is not yet implemented")


if __name__ == "__main__":
    main("wavenet_vocoder/egs/mol/conf/mol_wavenet_demo.json")
