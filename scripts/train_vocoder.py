# Referenced from wavenet_vocoder/train.py

from dataclasses import dataclass, field
import wandb
from torch.utils.data import DataLoader
from wavenet_vocoder.hparams import hparams, hparams_debug_string
import wavenet_vocoder.audio
from wavenet_vocoder.wavenet_vocoder.mixture import sample_from_mix_gaussian
from wavenet_vocoder.wavenet_vocoder.mixture import mix_gaussian_loss
from wavenet_vocoder.wavenet_vocoder.mixture import sample_from_discretized_mix_logistic
from wavenet_vocoder.wavenet_vocoder.mixture import discretized_mix_logistic_loss
from wavenet_vocoder.wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw, is_scalar_input
from wavenet_vocoder.wavenet_vocoder import WaveNet
from warnings import warn
from matplotlib import cm
from tensorboardX import SummaryWriter
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
import wavenet_vocoder.lrschedule
import matplotlib.pyplot as plt
from docopt import docopt

import sys

import os
from os.path import dirname, join, expanduser, exists
from tqdm import tqdm
from datetime import datetime
import random
import json
from glob import glob

import numpy as np

import matplotlib
matplotlib.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class HParams:
    out_channels: int
    layers: int
    stacks: int
    residual_channels: int
    gate_channels: int
    skip_out_channels: int
    cin_channels: int
    gin_channels: int
    n_speakers: int
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


class DiscretizedMixturelogisticLoss(nn.Module):
    def __init__(self, quantize_channels, log_scale_min):
        super(DiscretizedMixturelogisticLoss, self).__init__()
        self.quantize_channels = quantize_channels
        self.log_scale_min = log_scale_min

    def forward(self, input, target):
        # input and target should already be of size (B, F, D)
        # where B is batch size, F is number of frequency bins, and D is number of time steps

        # Compute the discretized mix logistic loss
        losses = discretized_mix_logistic_loss(
            input, target, num_classes=self.quantize_channels,
            log_scale_min=self.log_scale_min, reduce=False)

        # Ensure the size of losses matches the target size
        assert losses.size() == target.size()

        # Calculate the mean loss over all dimensions
        return losses.mean()


def load_hparams(path: str):
    import json
    with open(path) as f:
        data = json.load(f)
    return HParams(
        out_channels=data["out_channels"],
        layers=data["layers"],
        stacks=data["stacks"],
        residual_channels=data["residual_channels"],
        gate_channels=data["gate_channels"],
        skip_out_channels=data["skip_out_channels"],
        cin_channels=data["cin_channels"],
        gin_channels=data["gin_channels"],
        n_speakers=data["n_speakers"],
        dropout=data["dropout"],
        kernel_size=data["kernel_size"],
        cin_pad=data["cin_pad"],
        upsample_conditional_features=data["upsample_conditional_features"],
        upsample_params=data["upsample_params"],
        scalar_input=data["scalar_input"],
        output_distribution=data["output_distribution"],
        learning_rate=data["learning_rate"],
        max_train_steps=data["max_train_steps"],
        checkpoint_interval=data["checkpoint_interval"]
    )


def build_model(hparams: HParams):
    model = WaveNet(
        out_channels=hparams.out_channels,
        layers=hparams.layers,
        stacks=hparams.stacks,
        residual_channels=hparams.residual_channels,
        gate_channels=hparams.gate_channels,
        skip_out_channels=hparams.skip_out_channels,
        cin_channels=hparams.cin_channels,
        gin_channels=hparams.gin_channels,
        n_speakers=hparams.n_speakers,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        cin_pad=hparams.cin_pad,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_params=hparams.upsample_params,
        scalar_input=True,
        output_distribution=hparams.output_distribution,
    )
    return model


def train_loop(device, model, data_loader, optimizer, criterion, global_step, use_wandb=False, checkpoint_dir=None):
    model.train()
    for step, (x, y, c, g, input_lengths) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        c = c.to(device) if c is not None else None
        g = g.to(device) if g is not None else None

        optimizer.zero_grad()
        y_hat = model(x, c, g, False)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        if use_wandb:
            wandb.log({"train_loss": loss.item()})

        print(f"Step: {global_step+step}, Loss: {loss.item()}")

        if (global_step + step + 1) % hparams.checkpoint_interval == 0:
            save_checkpoint(checkpoint_dir, model, optimizer, global_step + step)

    return global_step + len(data_loader)


def save_checkpoint(checkpoint_dir, model, optimizer, step):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step{step:09d}.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    criterion = DiscretizedMixturelogisticLoss()

    # Initialize Weights & Biases
    if hparams.use_wandb:
        wandb.init(project="wavenet_training", config=hparams)

    # TODO: Define your data loaders
    train_loader = None  # Replace with your DataLoader

    checkpoint_dir = "./checkpoints"
    global_step = 0
    while global_step < hparams.max_train_steps:
        global_step = train_loop(
            device,
            model,
            train_loader,
            optimizer,
            criterion,
            global_step,
            checkpoint_dir=checkpoint_dir
        )


if __name__ == "__main__":
    main()
