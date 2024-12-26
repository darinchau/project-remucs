# This script is used for hot-fixing my discriminator model. It is used to retrain the discriminator model that should be more powerful
# than the current one

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
import torch.nn.functional as F
import wandb
import pickle
from accelerate import Accelerator
from remucs.model.vae import VQVAE, VQVAEConfig
from remucs.model.vae import PatchGAN as Discriminator
from remucs.model.lpips import load_lpips
from remucs.dataset import load_dataset


from remucs.model.vae import VQVAE, VQVAEConfig
from .train_vqvae import read_config

def load_vae(vae_ckpt_path: str, vae_config_path: str, device: torch.device):
    # Load the VQVAE model

    config = read_config(vae_config_path)
    vae_config = VQVAEConfig(**config['autoencoder_params'])
    model = VQVAE(im_channels=config['dataset_params']['im_channels'], model_config=vae_config).to(device)
    sd = torch.load(vae_ckpt_path, map_location=device)
    model.load_state_dict(sd)
    return model

def train(vae_ckpt_path: str, vae_config_path: str, local_dataset_dir: str, base_dir: str, *, start_from_iter: int = 0,
          discriminator: nn.Module | None = None,
          dataset_params = None, train_params = None, autoencoder_params = None):
    """Retrains the discriminator. If discriminator is None, a new discriminator is created based on the PatchGAN architecture."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = read_config(vae_config_path)
    if dataset_params is not None:
        config['dataset_params'].update(dataset_params)
    if train_params is not None:
        config['train_params'].update(train_params)
    if autoencoder_params is not None:
        config['autoencoder_params'].update(autoencoder_params)

    model = load_vae(vae_ckpt_path, vae_config_path, device)
    model.eval()

    # Print the model parameters and bail if necessary
    print(f"Retraining discriminator - Starting from iteration {start_from_iter}")

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    # Create the dataset
    im_dataset = load_dataset(
        lookup_table_path=dataset_config["train_lookup_table_path"],
        local_dataset_dir=local_dataset_dir,
        credentials_path=dataset_config["credentials_path"],
        bucket_name=dataset_config["bucket_name"],
        cache_dir=dataset_config["cache_dir"],
        nbars=dataset_config["nbars"],
        backup_dataset_first_n=dataset_config["backup_dataset_first_n_train"]
    )

    val_dataset = load_dataset(
        lookup_table_path=dataset_config["val_lookup_table_path"],
        local_dataset_dir=local_dataset_dir,
        credentials_path=dataset_config["credentials_path"],
        bucket_name=dataset_config["bucket_name"],
        cache_dir=dataset_config["cache_dir"],
        nbars=dataset_config["nbars"],
        backup_dataset_first_n=dataset_config["backup_dataset_first_n_val"]
    )

    print('Dataset size: {}'.format(len(im_dataset)))

    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             num_workers=train_config['num_workers_dl'],
                             shuffle=False) # Bad machine learning practice but saves so much on my cloud bill + the loaded dataset is essentially random

    val_data_loader = DataLoader(val_dataset,
                                 batch_size=train_config['autoencoder_batch_size'],
                                 num_workers=train_config['num_workers_dl'],
                                 shuffle=False)

    val_count = dataset_config['val_count']

    # Create output directories
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    num_epochs = train_config['epochs']

    if discriminator is None:
        discriminator = Discriminator(im_channels=dataset_config['im_channels'])
    discriminator = discriminator.to(device)

    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))

    accelerator = Accelerator(mixed_precision="bf16")

    model, data_loader, optimizer_d = accelerator.prepare(
        model, data_loader, optimizer_d
    )

    step_count = 0

    # This is for accumulating gradients incase the images are huge
    acc_steps = train_config['autoencoder_acc_steps']
    val_steps = train_config['val_steps']

    # Reload checkpoint
    if start_from_iter > 0:
        disc_save_path = os.path.join(base_dir, f"discriminator_{start_from_iter}_{train_config['vqvae_autoencoder_ckpt_name']}")
        disc_sd = torch.load(disc_save_path)
        discriminator.load_state_dict(disc_sd)
        step_count = start_from_iter

    model_save_steps = train_config['autoencoder_img_save_steps']

    wandb.init(
        # set the wandb project where this run will be logged
        project="discriminator-training-1",
        config=config
    )

    for epoch_idx in range(num_epochs):
        optimizer_d.zero_grad()

        for im in tqdm(data_loader):
            step_count += 1
            im = im.float().to(device).mean(dim=2)

            # Fetch autoencoders output(reconstructions)
            with torch.no_grad():
                with autocast('cuda'):
                    model_output = model(im)
            output: Tensor
            output, _, _ = model_output

            # Save the model
            if step_count % model_save_steps == 0:
                disc_save_path = os.path.join(base_dir, f"discriminator_{step_count}_{train_config['vqvae_autoencoder_ckpt_name']}")
                torch.save(discriminator.state_dict(), disc_save_path)

            fake = output

            disc_fake_pred: Tensor = discriminator(fake.detach())
            disc_fake_loss = F.binary_cross_entropy_with_logits(disc_fake_pred, torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device))
            accelerator.backward(disc_fake_loss / acc_steps)

            disc_real_pred: Tensor = discriminator(im)
            disc_real_loss = F.binary_cross_entropy_with_logits(disc_real_pred, torch.ones(disc_real_pred.shape, device=disc_real_pred.device))
            accelerator.backward(disc_real_loss / acc_steps)

            if step_count % acc_steps == 0:
                optimizer_d.step()
                optimizer_d.zero_grad()

            # Log losses
            wandb.log({
                "Real MSE Loss": disc_real_loss.item(),
                "Fake MSE Loss": disc_fake_loss.item(),
            }, step=step_count)

            ########### Perform Validation #############
            val_count_ = 0
            if step_count % val_steps == 0:
                model.eval()
                discriminator.eval()
                with torch.no_grad():
                    real_loss = []
                    fake_loss = []
                    for val_im in tqdm(val_data_loader, f"Performing validation (step={step_count})", total=min(val_count, len(val_data_loader))):
                        val_count_ += 1
                        if val_count_ > val_count:
                            break
                        val_im = val_im.float().to(device).mean(dim=2)
                        with autocast('cuda'):
                            model_output = model(val_im)
                        output: Tensor
                        output, _, _ = model_output
                        fake = output
                        disc_fake_pred: Tensor = discriminator(fake.detach())
                        disc_fake_loss = F.binary_cross_entropy_with_logits(disc_fake_pred, torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device))

                        disc_real_pred: Tensor = discriminator(val_im)
                        disc_real_loss = F.binary_cross_entropy_with_logits(disc_real_pred, torch.ones(disc_real_pred.shape, device=disc_real_pred.device))

                        real_loss.append(disc_real_loss.item())
                        fake_loss.append(disc_fake_loss.item())
                    wandb.log({
                        "Validation Real MSE Loss": np.mean(real_loss),
                        "Validation Fake MSE Loss": np.mean(fake_loss),
                    }, step=step_count)
                discriminator.train()


        # End of epoch. Clean up the gradients and losses and save the model
        optimizer_d.step()
        optimizer_d.zero_grad()
        disc_save_path = os.path.join(base_dir, f"discriminator_epoch_{epoch_idx}_{step_count}_{train_config['vqvae_autoencoder_ckpt_name']}")
        torch.save(discriminator.state_dict(), disc_save_path)

    wandb.finish()
    print('Done Training...')
