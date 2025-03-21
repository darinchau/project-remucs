import argparse
import itertools
import os
import time
import warnings

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import trange, tqdm

from torch.optim.adamw import AdamW

from remucs.model.vocoder import MelDataset, get_mel_spectrogram, get_dataset_filelist
from remucs.model.vocoder import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
    TorchSTFT,
    scan_checkpoint,
    load_checkpoint,
    save_checkpoint,
    VocoderConfig,
    build_env,
)

from dataclasses import asdict

warnings.simplefilter(action="ignore", category=FutureWarning)

torch.backends.cudnn.benchmark = True


def _load_checkpoint(
    args: argparse.Namespace,
    device: torch.device,
    generator: Generator,
    mpd: MultiPeriodDiscriminator,
    msd: MultiScaleDiscriminator,
    optim_g: AdamW,
    optim_d: AdamW,
):
    if os.path.isdir(args.checkpoint_path):
        checkpoint_generator = scan_checkpoint(args.checkpoint_path, "g_")
        checkpoint_discriminator = scan_checkpoint(args.checkpoint_path, "do_")
    else:
        checkpoint_generator, checkpoint_discriminator = None, None

    if checkpoint_generator is None or checkpoint_discriminator is None:
        state_dict_do = None
        last_epoch = -1
        steps = 0
    else:
        state_dict_g = load_checkpoint(checkpoint_generator, device)
        state_dict_do = load_checkpoint(checkpoint_discriminator, device)
        assert state_dict_do is not None
        assert state_dict_g is not None
        generator.load_state_dict(state_dict_g["generator"])
        mpd.load_state_dict(state_dict_do["mpd"])
        msd.load_state_dict(state_dict_do["msd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]
        print(f"Continuing training from epoch: {last_epoch} /n Continuing training from steps: {steps}.")
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    return generator, mpd, msd, optim_g, optim_d, steps, last_epoch


def _save_checkpoint(
    args: argparse.Namespace,
    generator: Generator,
    config: VocoderConfig,
    steps: int,
    mpd: MultiPeriodDiscriminator,
    msd: MultiScaleDiscriminator,
    optim_g: AdamW,
    optim_d: AdamW,
    epoch: int,
):
    checkpoint_path = f"{args.checkpoint_path}/g_{steps:08d}"
    save_checkpoint(
        checkpoint_path, {"generator": generator.state_dict()}
    )
    checkpoint_path = f"{args.checkpoint_path}/do_{steps:08d}"
    save_checkpoint(
        checkpoint_path,
        {
            "mpd": mpd.state_dict(),
            "msd": msd.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "steps": steps,
            "epoch": epoch,
        },
    )


@torch.no_grad()
def loss_logging(
    y_mel: torch.Tensor, y_g_hat_mel: torch.Tensor, loss_gen_all: torch.Tensor, steps: int, start_b: float
):
    mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
    wandb.log({"train_loss/gen_total_loss": loss_gen_all})
    wandb.log({"train_loss/mel_error": mel_error})

    print(
        f"Steps : {steps}, Gen Loss Total : {np.round(loss_gen_all.item(), 3)}, "
        f"Mel-Spec. Error : {np.round(mel_error, 3)}, s/b : {time.time() - start_b}"
    )


@torch.no_grad()
def validation(generator, validation_loader, device, config, steps, args, stft):
    generator.eval()
    torch.cuda.empty_cache()
    val_err_tot = 0
    val_length = -1
    for j, batch in enumerate(validation_loader):
        val_length = j
        x, y, _, y_mel = batch
        spec, phase = generator(x.to(device))
        y_g_hat = stft.inverse(spec, phase)
        y_mel = y_mel.to(device)
        y_g_hat_mel = get_mel_spectrogram(y_g_hat.squeeze(1), **asdict(config))
        val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

        if steps == 0:
            wandb.log(
                {
                    f"audio/{j}_gt": wandb.Audio(
                        y[0].detach().cpu().numpy(), caption=f"audio/{j}_gt", sample_rate=config.sampling_rate
                    )
                }
            )
        if steps % args.log_audio_interval == 0:
            wandb.log(
                {
                    f"audio/{j}_generated": wandb.Audio(
                        y_g_hat[0].squeeze(0).detach().cpu().numpy(),
                        caption=f"audio/{j}_generated",
                        sample_rate=config.sampling_rate,
                    )
                }
            )

    val_err = val_err_tot / (val_length + 1)  # If the loop is not entered, this will divide by 0 because why not
    wandb.log({"val_loss/mel_error": val_err})


def _log_checkpoint_info(generator, args):
    print(generator)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    print(f"checkpoints directory : {args.checkpoint_path}")


def _init_models(config, device):
    generator = Generator(config).to(device)
    mpd = MultiPeriodDiscriminator(config.discriminator_periods).to(device)
    msd = MultiScaleDiscriminator().to(device)
    stft = TorchSTFT(
        istft_filter_length=config.istft_filter_length,
        istft_hop_length=config.istft_hop_length,
    ).to(device)
    optim_g = AdamW(generator.parameters(), config.learning_rate, betas=(config.adam_b1, config.adam_b2))
    optim_d = AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        config.learning_rate,
        betas=(config.adam_b1, config.adam_b2),
    )
    return generator, mpd, msd, stft, optim_g, optim_d


def _init_schedulers(optim_g, optim_d, config, last_epoch):
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config.lr_decay, last_epoch=last_epoch)
    return scheduler_g, scheduler_d


def _init_dataloader(fileset, num_workers, shuffle, sampler, batch_size):
    loader = DataLoader(
        fileset,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def _get_dataloaders(args, device, config: VocoderConfig):
    train_wavs, val_wavs = get_dataset_filelist(args)
    train_set = MelDataset(train_wavs, device=device, **asdict(config))
    train_sampler = None
    train_shuffle = True
    train_loader = _init_dataloader(train_set, config.num_workers, train_shuffle, train_sampler, config.batch_size)
    val_set = MelDataset(val_wavs, split=False, shuffle=False, **asdict(config))
    validation_loader = _init_dataloader(val_set, 0, False, None, 1)
    wandb.init(
        project=config.wandb.project,
        config=asdict(config),
    )
    return train_sampler, train_loader, validation_loader


def _setup_rank_train(config, args, device):
    torch.cuda.manual_seed(config.seed)
    generator, mpd, msd, stft, optim_g, optim_d = _init_models(config, device)
    _log_checkpoint_info(generator, args)
    generator, mpd, msd, optim_g, optim_d, steps, last_epoch = _load_checkpoint(
        args, device, generator, mpd, msd, optim_g, optim_d
    )
    generator.train()
    mpd.train()
    msd.train()
    return generator, mpd, msd, last_epoch, stft, optim_d, optim_g, steps


def _generation_step(batch, device, config, stft, generator):
    x, y, _, y_mel = batch
    x = x.to(device)
    y = y.to(device)
    y_mel = y_mel.to(device)
    y = y.unsqueeze(1)
    spec, phase = generator(x)
    y_g_hat = stft.inverse(spec, phase)
    y_g_hat_mel = get_mel_spectrogram(y_g_hat.squeeze(1), **asdict(config))
    return y, y_g_hat, y_mel, y_g_hat_mel


def _train_step(y, y_g_hat, y_mel, y_g_hat_mel, config, optim_d, mpd, msd, optim_g):
    optim_d.zero_grad()
    y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
    loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
    y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
    loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
    loss_disc_all = loss_disc_s + loss_disc_f
    loss_disc_all.backward()
    optim_d.step()
    optim_g.zero_grad()
    loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
    loss_fm_f = config.fm_scale_factor * feature_loss(fmap_f_r, fmap_f_g)
    loss_fm_s = config.fm_scale_factor * feature_loss(fmap_s_r, fmap_s_g)
    loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
    loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
    loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
    return loss_gen_all


def train(args: argparse.Namespace, config: VocoderConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, mpd, msd, last_epoch, stft, optim_d, optim_g, steps = _setup_rank_train(config, args, device)
    scheduler_g, scheduler_d = _init_schedulers(optim_g, optim_d, config, last_epoch)
    _, train_loader, validation_loader = _get_dataloaders(args, device, config)
    for epoch in trange(max(0, last_epoch), args.training_epochs):
        start = time.time()
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}"):
            start_b = time.time()
            y, y_g_hat, y_mel, y_g_hat_mel = _generation_step(batch, device, config, stft, generator)
            loss_gen_all = _train_step(y, y_g_hat, y_mel, y_g_hat_mel, config, optim_d, mpd, msd, optim_g)
            loss_gen_all.backward()
            optim_g.step()
            if steps % args.wandb_log_interval == 0:
                loss_logging(y_mel, y_g_hat_mel, loss_gen_all, steps, start_b)  # type: ignore
            if steps % args.checkpoint_interval == 0 and steps != 0:
                _save_checkpoint(args, generator, config, steps, mpd, msd, optim_g, optim_d, epoch)
            if steps % args.validation_interval == 0:
                validation(generator, validation_loader, device, config, steps, args, stft)
                generator.train()
            steps += 1
        scheduler_g.step()
        scheduler_d.step()
        print(f"Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n")  # type: ignore


def main():
    print("Initializing Training Process...")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", help="path to config/config.json", default="./resources/config/vocoder.json")
    parser.add_argument("--datapath", type=str, help="Path to the wav files")
    parser.add_argument("--checkpoint_path", default="/app/new_checkpoints")
    parser.add_argument("--training_epochs", default=1, type=int)
    parser.add_argument("--wandb_log_interval", default=1, type=int, help="Once per n steps")
    parser.add_argument("--checkpoint_interval", default=1, type=int, help="Once per n steps")
    parser.add_argument("--log_audio_interval", default=1, type=int, help="Once per n steps")
    parser.add_argument("--validation_interval", default=1, type=int, help="Once per n steps")

    args = parser.parse_args()
    config = VocoderConfig.load(args.config_path)
    build_env(args.config_path, args.checkpoint_path)

    torch.manual_seed(config.seed)
    print(f"Batch size per GPU : {config.batch_size}")
    train(args, config)


if __name__ == "__main__":
    main()
