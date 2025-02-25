# This script is used to train the VQ-VAE model with a discriminator for adversarial loss
# Use the config file in resources/config/vqvae.yaml to set the parameters for training
# Adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/blob/main/tools/train_vqvae.py
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
import wandb
import pickle
from accelerate import Accelerator
from remucs.model.vae import VQVAE, VAEConfig
from remucs.model.discriminator import AudioSpectrogramDiscriminator as Discriminator
from remucs.model.lpips import load_lpips
from remucs.spectrogram import load_dataset
import torch.nn.functional as F

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


def sanity_check(config: dict):
    credentials_path = config['dataset_params']['credentials_path']
    if not os.path.exists(credentials_path):
        raise ValueError(f"Credentials file not found at {credentials_path}")

    # Check cache path
    cache_dir = config['dataset_params']['cache_dir']
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Check lookup tables
    for split in ['train', 'val']:
        lookup_table_path = config['dataset_params'][f"{split}_lookup_table_path"]
        if not os.path.exists(lookup_table_path):
            raise ValueError(f"Lookup table not found at {lookup_table_path}")

    # Check training config
    _available_disc_losses = ['bce', 'mse', 'wasserstein']
    if config['train_params']['disc_loss'] not in _available_disc_losses:
        raise ValueError(f"Discriminator loss must be one of {_available_disc_losses}")

    dataset_dir = config['dataset_params']['dataset_dir']
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory not found at {dataset_dir}")


def get_loss(
    im: Tensor,
    model: VQVAE,
    train_config: dict,
    reconstruction_loss: nn.Module,
    perceptual_loss: nn.Module,
    disc_loss: nn.Module,
    discriminator: Discriminator | None = None,
    calculate_losses: bool = True
):
    im = im.float().to(device)

    # im is (B, 5, 2, 512, 512) -> take the mean of two channels
    # (nbatches, ninstruments, nchannels, T, nfeatures)
    im = im.mean(dim=2)

    inputs = im[:, :-1]  # (B, 4, 512, 512)
    target = im[:, -1]  # (B, 512, 512)

    batch = inputs.shape[0]

    with autocast('cuda'):
        model_output = model(inputs)
    output, _, quantize_losses = model_output
    output = output.squeeze(1)

    # assert output.shape == (batch, 512, 512)
    # assert target.shape == (batch, 512, 512)

    losses = {}

    if not calculate_losses:
        return output, target, losses

    ######### Optimize Generator ##########
    # L2 Loss
    with autocast('cuda'):
        recon_loss = reconstruction_loss(output, target)
    codebook_loss = quantize_losses['codebook_loss']
    commitment_loss = quantize_losses['commitment_loss']
    losses['recon_loss'] = recon_loss
    losses['codebook_loss'] = codebook_loss
    losses['commitment_loss'] = commitment_loss

    if discriminator is not None:
        disc_fake_pred: Tensor = discriminator(output)  # (b,)
        if train_config['disc_loss'] == 'wasserstein':
            # Wasserstein WGAN increases the value of discriminator output
            disc_fake_loss = disc_fake_pred.mean()
        else:
            disc_fake_loss = disc_loss(disc_fake_pred, torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device))
        losses['disc_fake_loss'] = disc_fake_loss
    else:
        losses['disc_fake_loss'] = torch.zeros(1, device=device)

    # Perceptual Loss
    lpips_loss = torch.mean(perceptual_loss(output, target, normalize=True))
    losses['lpips_loss'] = lpips_loss

    return output, target, losses


def cycle_dl(dl):
    # Helper function to cycle through a dataloader
    # This prevents O(n) storage of the dataset in memory compared to itertools.cycle
    while True:
        for x in dl:
            yield x


def compute_gradient_penalty(discriminator, x_fake, x_real, lambda_gp):
    d_real = discriminator(x_real)
    d_fake = discriminator(x_fake)
    assert d_real.shape == d_fake.shape == (x_real.shape[0],)
    x_fake = x_fake.squeeze(1)
    x_real = x_real.squeeze(1)
    lip_dist = ((x_fake - x_real) ** 2).mean(2).mean(1) ** 0.5 + 1e-8
    lip_est = (x_fake - x_real).abs().mean() / lip_dist
    lip_loss = (1. - lip_est) ** 2
    disc_loss_ = (d_real - d_fake) + lip_loss * lambda_gp
    return disc_loss_.mean()


def train(config_path: str, output_dir: str, *, start_from_iter: int = 0,
          dataset_params=None, train_params=None, autoencoder_params=None):
    """Retrains the discriminator. If discriminator is None, a new discriminator is created based on the PatchGAN architecture."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = read_config(config_path)
    if dataset_params is not None:
        config['dataset_params'].update(dataset_params)
    if train_params is not None:
        config['train_params'].update(train_params)
    if autoencoder_params is not None:
        config['autoencoder_params'].update(autoencoder_params)

    dataset_dir = config['dataset_params']['dataset_dir']

    # Sanity checks
    sanity_check(config)

    vae_config = VAEConfig(**config['autoencoder_params'])

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    set_seed(train_config['seed'])

    # Create the model and dataset #
    model: VQVAE = VQVAE(
        im_channels_in=dataset_config['im_channels'],
        im_channels_out=1,
        model_config=vae_config
    ).to(device)
    print(f"Starting from iteration {start_from_iter}")

    # Count the number of parameters
    numel = 0
    for p in model.parameters():
        numel += p.numel()
    print('Total number of parameters: {}'.format(numel))
    del numel

    # Create the dataset
    im_dataset = load_dataset(
        lookup_table_path=dataset_config["train_lookup_table_path"],
        local_dataset_dir=dataset_dir,
        credentials_path=dataset_config["credentials_path"],
        bucket_name=dataset_config["bucket_name"],
        cache_dir=dataset_config["cache_dir"],
        nbars=dataset_config["nbars"],
        backup_dataset_first_n=dataset_config["backup_dataset_first_n_train"]
    )

    val_dataset = load_dataset(
        lookup_table_path=dataset_config["val_lookup_table_path"],
        local_dataset_dir=dataset_dir,
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
                             shuffle=False)  # Bad machine learning practice but saves so much on my cloud bill + the loaded dataset is essentially random

    val_data_loader = DataLoader(val_dataset,
                                 batch_size=train_config['autoencoder_batch_size'],
                                 num_workers=train_config['num_workers_dl'],
                                 shuffle=False)

    val_count = dataset_config['val_count']

    # Create output directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_epochs = train_config['epochs']

    reconstruction_loss = torch.nn.MSELoss()
    disc_loss = torch.nn.BCEWithLogitsLoss() if train_config['disc_loss'] == 'bce' else torch.nn.MSELoss()
    perceptual_loss = load_lpips(channels=1).eval().to(device)
    for param in perceptual_loss.parameters():
        param.requires_grad = False

    discriminator = Discriminator(vae_config).to(device)

    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))

    accelerator = Accelerator(mixed_precision="bf16")

    model, optimizer_g, data_loader, optimizer_d = accelerator.prepare(
        model, optimizer_g, data_loader, optimizer_d
    )

    disc_step_start: int = train_config['disc_start']
    step_count = 0

    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    model_save_steps = train_config['autoencoder_img_save_steps']

    val_steps = train_config['val_steps']

    # Reload checkpoint
    if start_from_iter > 0:
        model_save_path = os.path.join(output_dir, f"vqvae_{start_from_iter}_{train_config['vqvae_autoencoder_ckpt_name']}")
        model_sd = torch.load(model_save_path)
        model.load_state_dict(model_sd)
        disc_save_path = os.path.join(output_dir, f"discriminator_{start_from_iter}_{train_config['vqvae_autoencoder_ckpt_name']}")
        disc_sd = torch.load(disc_save_path)
        discriminator.load_state_dict(disc_sd)
        step_count = start_from_iter

    max_discr_to_gen_ratio = train_config['max_discr_to_gen_ratio']
    nbatches = num_epochs * len(data_loader)

    wandb.init(
        # set the wandb project where this run will be logged
        project=train_config['run_name'],
        config=config
    )

    recon_losses = []
    codebook_losses = []
    perceptual_losses = []
    commit_losses = []
    disc_losses = []
    gen_losses = []
    losses = []

    progress = tqdm(desc='Training (nsteps)')
    steps_progress = tqdm(desc='Training (noptims)', total=nbatches)
    dataloader_iter = cycle_dl(data_loader)
    while step_count < nbatches:
        step_count += 1
        steps_progress.update(1)
        if step_count > disc_step_start:
            disc_to_gen_ratio = 1
        else:
            disc_to_gen_ratio = 0

        optimizer_d.zero_grad()
        optimizer_g.zero_grad()

        for _ in range(disc_to_gen_ratio):
            #### Optimize Discriminator ####
            for param in discriminator.parameters():
                param.requires_grad = True

            im = next(dataloader_iter)
            progress.update(1)
            with torch.no_grad():
                output, target, _ = get_loss(
                    im,
                    model,
                    train_config,
                    reconstruction_loss,
                    perceptual_loss,
                    disc_loss,
                    None
                )

            disc_fake_pred: Tensor = discriminator(output)
            disc_real_pred: Tensor = discriminator(target)
            if train_config['disc_loss'] == 'wasserstein':
                # Implements WGAN-GP with Lipschitz penalty
                disc_loss_ = compute_gradient_penalty(discriminator, output, target, train_config['wasserstein_regularizer'])
            else:
                disc_fake_loss = disc_loss(disc_fake_pred, torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device))
                disc_real_loss = disc_loss(disc_real_pred, torch.ones(disc_real_pred.shape, device=disc_real_pred.device))
                disc_loss_: Tensor = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
            disc_losses.append(disc_loss_.item())
            disc_loss_ = disc_loss_ / acc_steps
            accelerator.backward(disc_loss_)
            if step_count % acc_steps == 0:
                optimizer_d.step()
                optimizer_d.zero_grad()

            # Disable grad for discriminator
            for param in discriminator.parameters():
                param.requires_grad = False

        if step_count % model_save_steps == 0:
            model_save_path = os.path.join(output_dir, f"vqvae_{step_count}_{train_config['vqvae_autoencoder_ckpt_name']}")
            disc_save_path = os.path.join(output_dir, f"discriminator_{step_count}_{train_config['vqvae_autoencoder_ckpt_name']}")
            torch.save(model.state_dict(), model_save_path)
            torch.save(discriminator.state_dict(), disc_save_path)

        im = next(dataloader_iter)
        progress.update(1)

        output, target, losses = get_loss(
            im,
            model,
            train_config,
            reconstruction_loss,
            perceptual_loss,
            disc_loss,
            discriminator if step_count > disc_step_start else None
        )

        ######### Optimize Generator ##########
        recon_losses.append(losses['recon_loss'].item())
        codebook_losses.append(losses['codebook_loss'].item() * train_config['codebook_weight'])
        perceptual_losses.append(losses['lpips_loss'].item())
        commit_losses.append(losses['commitment_loss'].item() * train_config['commitment_beta'])
        gen_losses.append(losses['disc_fake_loss'].item() * train_config['disc_weight'])

        g_loss = (
            losses['recon_loss'] +
            losses['codebook_loss'] * train_config['codebook_weight'] +
            losses['commitment_loss'] * train_config['commitment_beta'] +
            losses['lpips_loss'] * train_config['perceptual_weight'] +
            losses['disc_fake_loss'] * train_config['disc_weight']
        ) / acc_steps
        accelerator.backward(g_loss)
        #####################################

        if step_count % acc_steps == 0:
            optimizer_g.step()
            optimizer_g.zero_grad()

        # Log losses
        wandb.log({
            "Reconstruction Loss": recon_losses[-1],
            "Perceptual Loss": perceptual_losses[-1],
            "Codebook Loss": codebook_losses[-1],
            "Generator Loss": gen_losses[-1] if gen_losses else 0,
            "Discriminator Loss": disc_losses[-1] if disc_losses else 0,
            "Commitment Loss": commit_losses[-1],
            "Discriminator steps": disc_to_gen_ratio
        }, step=step_count)

        ########### Perform Validation #############
        val_count_ = 0
        if step_count % val_steps == 0:
            model.eval()
            with torch.no_grad():
                val_recon_losses = []
                val_perceptual_losses = []
                val_codebook_losses = []
                val_commit_losses = []
                for val_im in tqdm(val_data_loader, f"Performing validation (step={step_count})", total=min(val_count, len(val_data_loader))):
                    val_count_ += 1
                    if val_count_ > val_count:
                        break
                    output, target, val_losses = get_loss(
                        val_im,
                        model,
                        train_config,
                        reconstruction_loss,
                        perceptual_loss,
                        disc_loss,
                        None
                    )
                    val_recon_losses.append(val_losses['recon_loss'].item())
                    val_perceptual_losses.append(val_losses['lpips_loss'].item())
                    val_codebook_losses.append(val_losses['codebook_loss'].item())
                    val_commit_losses.append(val_losses['commitment_loss'].item())

            wandb.log({
                "Val Reconstruction Loss": np.mean(val_recon_losses),
                "Val Perceptual Loss": np.mean(val_perceptual_losses),
                "Val Codebook Loss": np.mean(val_codebook_losses),
                "Val Commitment Loss": np.mean(val_commit_losses),
            }, step=step_count)

            tqdm.write(f"Validation complete: Reconstruction loss: {np.mean(val_recon_losses)}, Perceptual Loss: {np.mean(val_perceptual_losses)}, Codebook loss: {np.mean(val_codebook_losses)}")
            model.train()
        ############################################

    wandb.finish()
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path', default='resources/config/vqvae.yaml', type=str)
    parser.add_argument('--output-dir', dest='output_dir', type=str, default='resources/models/vqvae')
    parser.add_argument('--start-iter', dest='start_iter', type=int, default=0)
    args = parser.parse_args()
    train(args.config_path, args.output_dir, start_from_iter=args.start_iter)
