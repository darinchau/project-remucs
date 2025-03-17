# This script is used to train the VAE model with a discriminator for adversarial loss
# As of right now, due to time constraints, this code is held together by duct tape
# I promise I can do better :))
# Use the config file in resources/config/vqvae.yaml to set the parameters for training
from typing import List
from dataclasses import dataclass, replace, asdict
import yaml
import argparse
import torch
import random
import os
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torch import nn, Tensor
from torch.amp.autocast_mode import autocast
import wandb
import pickle
from accelerate import Accelerator
from remucs.model.vae import VAE, VAEConfig
from remucs.model.vggish import Vggish
from remucs.preprocess import spectro, ispectro
from remucs.constants import TRAIN_SPLIT, VALIDATION_SPLIT
import torch.nn.functional as F
from AutoMasher.fyp import SongDataset, YouTubeURL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class Config:
    nbars: int
    nsources: int
    num_workers_ds: int
    dataset_dir: str
    output_dir: str
    val_count: int
    sample_rate: int
    splice_size: int
    nchannels: int
    down_channels: List[int]
    mid_channels: List[int]
    down_sample: List[int]
    norm_channels: int
    num_heads: int
    num_down_layers: int
    num_mid_layers: int
    num_up_layers: int
    gradient_checkpointing: bool
    seed: int
    num_workers_dl: int
    autoencoder_batch_size: int
    disc_start: int
    disc_weight: float
    kl_weight: float
    perceptual_weight: float
    wasserstein_regularizer: float
    epochs: int
    autoencoder_lr: float
    autoencoder_acc_steps: int
    save_steps: int
    vqvae_autoencoder_ckpt_name: str
    run_name: str
    disc_loss: str
    val_steps: int


def get_vae_config(config: Config) -> VAEConfig:
    return VAEConfig(
        down_channels=config.down_channels,
        mid_channels=config.mid_channels,
        down_sample=config.down_sample,
        norm_channels=config.norm_channels,
        num_heads=config.num_heads,
        num_down_layers=config.num_down_layers,
        num_mid_layers=config.num_mid_layers,
        num_up_layers=config.num_up_layers,
        gradient_checkpointing=config.gradient_checkpointing,
        nsources=config.nsources,
        nchannels=config.nchannels,
    )


def load_config_from_yaml(file_path: str) -> Config:
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

    # Create an instance of the Config data class using parsed YAML data
    config = Config(**config)
    return config


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


class Inference:
    def __init__(self, model: VAE, config: Config):
        self.model = model

        self.vggish = Vggish().to(device)
        self.config = config

        for p in self.vggish.model.parameters():
            p.requires_grad = False

    @property
    def sr(self):
        return self.config.sample_rate

    def __call__(self, im: Tensor, target: Tensor):
        # im is (batch, source, channel, time)
        im = im.float().to(device)
        B, S, C, T = im.shape

        assert target.shape == (B, C, T), f"Expected {(B, C, T)}, got {target.shape}"
        assert S == self.config.nsources, f"Expected {self.config.nsources}, got {S}"
        assert C == self.config.nchannels, f"Expected {self.config.nchannels}, got {C}"
        assert T == self.config.splice_size, f"Expected {self.config.splice_size}, got {T}"

        # Preprocess
        with torch.autocast("cuda"):
            out, _, _, _, kl_loss = self.model(im)

        # out.shape = (batch, channel, time)
        assert out.shape == (B, C, T), f"Expected {(B, C, T)}, got {out.shape}"

        losses = {}
        t_features = self.vggish((target, self.sr))
        s_features = self.vggish((out, self.sr))
        with torch.autocast("cuda"):
            perceptual_loss = F.mse_loss(t_features, s_features)
            recon_loss = F.mse_loss(out, target)

        # TODO do a adversarial loss over s and t features
        losses['recon_loss'] = recon_loss
        losses['perceptual_loss'] = perceptual_loss
        losses['kl_loss'] = kl_loss
        return out, losses


class TrainDataset(Dataset):
    def __init__(self, sd: SongDataset, urls: list[YouTubeURL], splice_size: int):
        self.urls = urls
        self.sd = sd
        self.splice_size = splice_size

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx: int):
        url = self.urls[idx]
        audio = self.sd.get_audio(url)
        start_frame = random.randint(0, audio.nframes - self.splice_size)

        parts = self.sd.get_parts(url)

        # Stack in VDIBN convention
        x = [
            aud.slice_frames(start_frame, start_frame + self.splice_size).data
            for aud in (parts.vocals, parts.drums, parts.other, parts.bass, audio)
        ]
        audio = torch.stack(x)
        return audio


def _show_num_params(vae: nn.Module):
    numel = 0
    for p in vae.parameters():
        numel += p.numel()
    print('Total number of parameters: {}'.format(numel))


def save_model(inference: Inference, output_dir: str, step: int):
    model_save_path = os.path.join(output_dir, f"vqvae_{step}_{inference.config.vqvae_autoencoder_ckpt_name}")
    torch.save(inference.model.state_dict(), model_save_path)


def train(config: Config):
    vae_config = get_vae_config(config)
    vae = VAE(vae_config).to(device)
    set_seed(config.seed)

    _show_num_params(vae)

    sd = SongDataset(config.dataset_dir)
    print('Dataset size: {}'.format(len(sd)))

    songs = sd.list_urls("audio")
    train_urls = [url for url in songs if url not in sd.read_info_urls(TRAIN_SPLIT)]
    val_urls = [url for url in songs if url not in sd.read_info_urls(VALIDATION_SPLIT)]

    print('Train size: {}'.format(len(train_urls)))
    print('Val size: {}'.format(len(val_urls)))

    train_ds = TrainDataset(sd, train_urls, config.splice_size)
    val_ds = TrainDataset(sd, val_urls, config.splice_size)

    train_dl = DataLoader(train_ds, batch_size=config.autoencoder_batch_size, shuffle=True, num_workers=config.num_workers_dl)
    val_dl = DataLoader(val_ds, batch_size=config.autoencoder_batch_size, shuffle=False, num_workers=config.num_workers_dl)

    # Create output directories
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    optimizer_g = Adam(vae.parameters(), lr=config.autoencoder_lr, betas=(0.5, 0.999))

    accelerator = Accelerator(mixed_precision="bf16")

    vae, optimizer_g, train_dl, val_dl = accelerator.prepare(
        vae, optimizer_g, train_dl, val_dl
    )

    inference = Inference(vae, config)

    wandb.init(
        # set the wandb project where this run will be logged
        project=config.run_name,
        config=asdict(config),
    )

    recon_losses = []
    perceptual_losses = []
    kl_losses = []
    disc_losses = []
    gen_losses = []

    step_count = 0

    for epoch in range(config.epochs):
        optimizer_g.zero_grad()
        for im in train_dl:
            step_count += 1

            if step_count % config.save_steps == 0:
                save_model(inference, config.output_dir, step_count)

            _, loss = inference(im[:, :-1], im[:, -1])
            recon_losses.append(loss['recon_loss'].item())
            perceptual_losses.append(loss['perceptual_loss'].item())
            kl_losses.append(loss['kl_loss'].item())

            g_loss = (
                loss['recon_loss'] +
                loss['kl_loss'] * config.kl_weight +
                loss['perceptual_loss'] * config.perceptual_weight
            )
            accelerator.backward(g_loss)
            optimizer_g.step()

            wandb.log({
                "Reconstruction Loss": recon_losses[-1],
                "Perceptual Loss": perceptual_losses[-1],
                "KL Loss": kl_losses[-1]
            }, step=step_count)

            if step_count % config.val_steps == 0:
                vae.eval()
                with torch.no_grad():
                    val_recon_losses = []
                    val_perceptual_losses = []
                    val_kl_losses = []
                    for val_im in val_dl:
                        _, val_loss = inference(val_im[:, :-1], val_im[:, -1])
                        val_recon_losses.append(val_loss['recon_loss'].item())
                        val_perceptual_losses.append(val_loss['perceptual_loss'].item())
                        val_kl_losses.append(val_loss['kl_loss'].item())

                wandb.log({
                    "Val Reconstruction Loss": np.mean(val_recon_losses),
                    "Val Perceptual Loss": np.mean(val_perceptual_losses),
                    "Val KL Loss": np.mean(val_kl_losses)
                }, step=step_count)

                print(f"Validation complete: Reconstruction loss: {np.mean(val_recon_losses)}, Perceptual Loss: {np.mean(val_perceptual_losses)}, KL loss: {np.mean(val_kl_losses)}")
                vae.train()

    wandb.finish()
    print('Done Training...')

    save_model(inference, config.output_dir, step_count)


def main():
    parser = argparse.ArgumentParser(description='Train VAE model with discriminator')
    parser.add_argument('--config', type=str, help='Path to the config file', required=True)
    args = parser.parse_args()

    config = load_config_from_yaml(args.config)
    train(config)


if __name__ == '__main__':
    main()
