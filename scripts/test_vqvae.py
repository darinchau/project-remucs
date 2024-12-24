# Runs the test suite on the trained vqvae model

import yaml
import argparse
import torch
import random
import os
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torch import nn, Tensor
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import wandb
import pickle
from accelerate import Accelerator
from remucs.model.vae import VQVAE, VQVAEConfig, vae_output_to_audio, gla_loss
from remucs.model.lpips import load_lpips
from remucs.dataset import load_dataset
from AutoMasher.fyp.audio.base.audio_collection import DemucsCollection
import torch.nn.functional as F

from .calculate import SpectrogramCollection, TARGET_FEATURES, TARGET_SR, NFFT, SPEC_MAX_VALUE, SPEC_POWER, TARGET_NFRAMES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_config(config_path: str):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)
    return config

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

def evaluate(config_path: str, dataset_dir: str, lookup_table_path: str, model_path: str, reconstructions: int = 3, batch_size: int = 32, first_n: int = 3):
    """Tests the VQVAE model on the test dataset

    Args:
        config_path (str): Path to the configuration file
        dataset_dir (str): Path to the dataset directory
        lookup_table_path (str): Path to the lookup table
        model_path (str): Path to the model checkpoint
        reconstructions (int, optional): Number of reconstructions to save. Defaults to 3.
        batch_size (int, optional): Batch size for the data loader. Defaults to 32.
        first_n (int, optional): Number of batches to evaluate. Set to -1 to evaluate all. Defaults to 3."""
    config = read_config(config_path)

    vae_config = VQVAEConfig(**config['autoencoder_params'])

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    set_seed(train_config['seed'])

    model = VQVAE(im_channels=dataset_config['im_channels'], model_config=vae_config).to(device)

    im_dataset = load_dataset(
        lookup_table_path=lookup_table_path,
        local_dataset_dir=dataset_dir,
        credentials_path=dataset_config['credentials_path'],
        bucket_name=dataset_config['bucket_name'],
        cache_dir=dataset_config['cache_dir'],
        backup_dataset_first_n=1
    )

    print('Dataset size: {}'.format(len(im_dataset)))

    data_loader = DataLoader(im_dataset,
                             batch_size=batch_size,
                             num_workers=train_config['num_workers_dl'],
                             shuffle=False)

    # Create output directories
    reconstruction_loss = torch.nn.MSELoss()
    perceptual_loss = load_lpips().eval().to(device)

    # Freeze perceptual loss parameters
    for param in perceptual_loss.parameters():
        param.requires_grad = False

    # Loads from checkpoint
    model_sd = torch.load(model_path)
    model.load_state_dict(model_sd)

    recon_losses = []
    perceptual_losses = []
    codebook_losses = []
    losses = {}

    if first_n < 0:
        first_n = len(data_loader)
        reconstruction_idxs = random.sample(range(len(im_dataset)), k=reconstructions)
    else:
        reconstruction_idxs = random.sample(range(batch_size * first_n), k=reconstructions)

    print("Reconstructing: ", reconstruction_idxs)

    with torch.no_grad():
        for i, im in tqdm(enumerate(data_loader), total=first_n):
            im = im.float().to(device)
            im = im.mean(dim=2)

            model_output = model(im)
            output, z, quantize_losses = model_output

            recon_loss = reconstruction_loss(output, im)
            recon_losses.append(recon_loss.item())

            lpips_loss = torch.mean(perceptual_loss(output, im))
            perceptual_losses.append(lpips_loss.item())

            codebook_losses.append(quantize_losses['codebook_loss'].item())

            griffin_lim_loss = gla_loss(output, im).mean()

            losses[i] = {
                'path': im_dataset.path_bar[i][0],
                'bar': im_dataset.path_bar[i][1],
                'recon_loss': recon_loss.item(),
                'perceptual_loss': lpips_loss.item(),
                'codebook_loss': quantize_losses['codebook_loss'].item(),
                'griffin_lim_loss': griffin_lim_loss.item()
            }

            for j in range(batch_size):
                if i * batch_size + j in reconstruction_idxs:
                    output = output[j].detach().cpu()
                    im = im[j].detach().cpu()

                    resonstructed_audio = vae_output_to_audio(output)
                    original_audio = vae_output_to_audio(im)

                    resonstructed_audio.save(f"reconstructed_{i * batch_size + j}.wav")
                    original_audio.save(f"original_{i * batch_size + j}.wav")

            print('Reconstruction Loss : {:.4f} | Perceptual Loss : {:.4f} | Codebook Loss : {:.4f} | Griffin-Lim Loss : {:.4f}'
                .format(recon_loss.item(), lpips_loss.item(), quantize_losses['codebook_loss'].item(), griffin_lim_loss.item()))

            if i == first_n - 1:
                break

    print('Reconstruction Loss : {:.4f} | Perceptual Loss : {:.4f} | Codebook Loss : {:.4f} | Griffin-Lim Loss : {:.4f}'
        .format(np.mean(recon_losses), np.mean(perceptual_losses), np.mean(codebook_losses), np.mean(griffin_lim_loss)))

    print(losses)
