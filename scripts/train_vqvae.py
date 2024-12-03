# This script is used to train the VQ-VAE model with a discriminator for adversarial loss
# Use the config file in resources/config/vqvae.yaml to set the parameters for training
# Adapted from https://github.com/explainingai-code/StableDiffusion-PyTorch/blob/main/tools/train_vqvae.py
import yaml
import argparse
import torch
import random
import torchvision
import os
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from torchvision.utils import make_grid
from torch import nn, Tensor
from torch.amp import autocast
from torch.amp import GradScaler
import wandb
import pickle
from accelerate import Accelerator
from remucs.model import VQVAE, VQVAEConfig
from remucs.model.lpips import LPIPS
from remucs.dataset import SpectrogramDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Discriminator(nn.Module):
    """Implements patchGAN for adversarial loss"""
    def __init__(self, im_channels=3,
                 conv_channels=[64, 128, 256],
                 kernels=[4,4,4,4],
                 strides=[2,2,2,1],
                 paddings=[1,1,1,1]):
        super().__init__()
        self.im_channels = im_channels
        activation = nn.LeakyReLU(0.2)
        layers_dim = [self.im_channels] + conv_channels + [1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(layers_dim[i], layers_dim[i + 1],
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i],
                          bias=False if i !=0 else True),
                nn.BatchNorm2d(layers_dim[i + 1]) if i != len(layers_dim) - 2 and i != 0 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Identity()
            )
            for i in range(len(layers_dim) - 1)
        ])

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

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

def train(config_path: str, base_dir: str, dataset_dirs: list[str], *, bail = False):
    """Trains the VAE

    config_path: str - Path to the config file
    base_dir: str - Path to the directory where the model checkpoints will be saved
    dataset_dirs: list[str] - Paths to the directory where the dataset is stored"""
    # Read the config file
    config = read_config(config_path)

    vae_config = VQVAEConfig(**config['autoencoder_params'])

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    set_seed(train_config['seed'])

    # Create the model and dataset #
    model = VQVAE(im_channels=dataset_config['im_channels'], model_config=vae_config).to(device)

    # Print the model parameters and bail if necessary
    print(model)
    numel = 0
    for p in model.parameters():
        numel += p.numel()
    print('Total number of parameters: {}'.format(numel))
    if bail:
        return

    # Create the dataset
    datasets = []
    for dataset_dir in dataset_dirs:
        if os.path.isdir(dataset_dir):
            ds = SpectrogramDataset(dataset_dir, nbars=dataset_config['nbars'], num_workers=dataset_config["num_workers_ds"])
            pickle.dump(ds, open(dataset_dir + ".pkl", 'wb'))
        else:
            ds = pickle.load(open(dataset_dir, 'rb'))
        datasets.append(ds)
    im_dataset = ConcatDataset(datasets)

    print('Dataset size: {}'.format(len(im_dataset)))

    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             num_workers=train_config['num_workers_dl'],
                             shuffle=True)

    # Create output directories
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    num_epochs = train_config['epochs']

    # TODO: Can try BCE with logits loss for discriminator???
    reconstruction_loss = torch.nn.MSELoss()
    discriminator_loss = torch.nn.MSELoss()
    perceptual_loss = LPIPS(
        means = [0.1885, 0.1751, 0.1698, 0.0800],
        stds = [0.1164, 0.1066, 0.1065, 0.0672]
    ).eval().to(device)

    # Freeze perceptual loss parameters
    for param in perceptual_loss.parameters():
        param.requires_grad = False

    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)

    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    scaler = GradScaler()

    accelerator = Accelerator(mixed_precision="bf16")

    model, optimizer_g, data_loader, scaler, optimizer_d = accelerator.prepare(
        model, optimizer_g, data_loader, scaler, optimizer_d
    )

    disc_step_start = train_config['disc_start']
    step_count = 0

    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0

    wandb.init(
        # set the wandb project where this run will be logged
        project="vqvae_training",
        config=config
    )

    for epoch_idx in range(num_epochs):
        recon_losses = []
        codebook_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        for im in tqdm(data_loader):
            step_count += 1
            im = im.float().to(device)

            # im is (4, 4, 2, 512, 512) -> take only the magnitude
            im = im [:, :, 0]

            # Fetch autoencoders output(reconstructions)
            with autocast('cuda'):
                model_output = model(im)
            output, z, quantize_losses = model_output

            # Image Saving Logic, disabled for now ## TODO : Enable this
            if step_count % image_save_steps == 0 or step_count == 1:
                torch.save(model.state_dict(), os.path.join(base_dir, f"vqvae_epoch_{epoch_idx}_{step_count}_{train_config['vqvae_autoencoder_ckpt_name']}"))
                torch.save(discriminator.state_dict(), os.path.join(base_dir, f"discriminator_epoch_{epoch_idx}_{step_count}_{train_config['vqvae_autoencoder_ckpt_name']}"))

                # sample_size = min(8, im.shape[0])
                # save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
                # save_output = ((save_output + 1) / 2)
                # save_input = ((im[:sample_size] + 1) / 2).detach().cpu()

                # grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                # img = torchvision.transforms.ToPILImage()(grid)
                # if not os.path.exists(os.path.join(base_dir,'vqvae_autoencoder_samples')):
                #     os.mkdir(os.path.join(base_dir, 'vqvae_autoencoder_samples'))
                # img.save(os.path.join(base_dir,'vqvae_autoencoder_samples', 'current_autoencoder_sample_{}.png'.format(img_save_count)))
                # img_save_count += 1
                # img.close()

            ######### Optimize Generator ##########
            # L2 Loss
            with autocast('cuda'):
                recon_loss = reconstruction_loss(output, im)
            recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            g_loss: torch.Tensor = (recon_loss +
                      (train_config['codebook_weight'] * quantize_losses['codebook_loss'] / acc_steps) +
                      (train_config['commitment_beta'] * quantize_losses['commitment_loss'] / acc_steps))
            codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())

            # Adversarial loss only if disc_step_start steps passed
            if step_count > disc_step_start:
                disc_fake_pred = discriminator(model_output[0])
                disc_fake_loss = discriminator_loss(disc_fake_pred,
                                                torch.ones(disc_fake_pred.shape,
                                                           device=disc_fake_pred.device))
                gen_losses.append(train_config['disc_weight'] * disc_fake_loss.item())
                g_loss += train_config['disc_weight'] * disc_fake_loss / acc_steps

            # Perceptual Loss
            lpips_loss = torch.mean(perceptual_loss(output, im)) / acc_steps
            perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
            g_loss += train_config['perceptual_weight']*lpips_loss / acc_steps

            losses.append(g_loss.item())
            accelerator.backward(g_loss)
            #####################################

            ######### Optimize Discriminator #######
            if step_count > disc_step_start:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(im)
                disc_fake_loss = discriminator_loss(disc_fake_pred,
                                                torch.zeros(disc_fake_pred.shape,
                                                            device=disc_fake_pred.device))
                disc_real_loss = discriminator_loss(disc_real_pred,
                                                torch.ones(disc_real_pred.shape,
                                                           device=disc_real_pred.device))
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / acc_steps
                accelerator.backward(disc_loss)
                if step_count % acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
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
                "Discriminator Loss": disc_losses[-1] if disc_losses else 0
            }, step=step_count)

        # End of epoch. Clean up the gradients and losses and save the model
        optimizer_d.step()
        optimizer_d.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()
        if len(disc_losses) > 0:
            print(
                'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                'Codebook : {:.4f} | G Loss : {:.4f} | D Loss {:.4f}'.
                format(epoch_idx + 1,
                       np.mean(recon_losses),
                       np.mean(perceptual_losses),
                       np.mean(codebook_losses),
                       np.mean(gen_losses),
                       np.mean(disc_losses)))
        else:
            print('Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | Codebook : {:.4f}'.
                  format(epoch_idx + 1,
                         np.mean(recon_losses),
                         np.mean(perceptual_losses),
                         np.mean(codebook_losses)))

        torch.save(model.state_dict(), os.path.join(base_dir, f"vqvae_epoch_{epoch_idx}_{train_config['vqvae_autoencoder_ckpt_name']}"))
        torch.save(discriminator.state_dict(), os.path.join(base_dir, f"discriminator_epoch_{epoch_idx}_{train_config['vqvae_autoencoder_ckpt_name']}"))

    wandb.finish()
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--dataset_dir', dest='dataset_dir', type=str, default='resources/dataset')
    parser.add_argument('--config', dest='config_path', default='resources/config/vqvae.yaml', type=str)
    parser.add_argument('--base_dir', dest='base_dir', type=str, default='resources/ckpts/vqvae')
    args = parser.parse_args()
    train(args.config_path, args.base_dir, args.dataset_dir)
