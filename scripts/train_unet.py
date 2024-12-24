# Modified from https://github.com/zyinghua/uncond-image-generation-ldm/blob/main/src/pipeline.py
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path
from dataclasses import dataclass
import yaml

import accelerate
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
import torch
from torch import nn

import diffusers
from diffusers import StableDiffusionPipeline
from diffusers import DDPMScheduler, UNet2DModel, UNet2DConditionModel #type: ignore
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor

from remucs.dataset import SpectrogramDataset, SpectrogramDatasetFromCloud
from remucs.model.vae import VQVAE, VQVAEConfig

import wandb

from .test_vqvae import vae_output_to_audio

def ass():
    import torch
    from diffusers import StableDiffusionPipeline

    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"


    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]

    image.save("astronaut_rides_horse.png")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

# The scaling factor for the VAE latents. This is the value used in the original DDPM paper.
# https://github.com/huggingface/diffusers/blob/c4b5d2ff6b529ac0f895cedb04fef5b25e89c412/src/diffusers/models/autoencoders/vq_model.py
# seems to use this value as well
# Don't ask me how this value was derived, I have no idea D:
VAE_SCALING_FACTOR = 0.18215

class PromptEmbed(nn.Module):
    def __init__(self, tensor, regularizer: float = 3):
        """A simple class to update the embeddings during training time
        The regularizer controls the margins of the norm of the updated embeddings, the higher the regularizer, the more lenient the controls"""
        super(PromptEmbed, self).__init__()
        self.tensor = nn.Parameter(tensor)
        self.initial_l2 = self.get_norm().detach()
        self.regularizer = regularizer

    def get_norm(self):
        l2_norms = torch.norm(self.tensor, p=2, dim=2)
        average_l2_norm = l2_norms.mean()
        return average_l2_norm

    def forward(self):
        regularizer_loss = ((self.get_norm() - self.initial_l2) / self.regularizer).square()
        return self.tensor.data, regularizer_loss

# Add typing to the arguments
@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str
    train_batch_size: int
    eval_batch_size: int
    dataloader_num_workers: int
    num_epochs: int
    save_model_steps: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler: str
    lr_warmup_steps: int
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: float
    adam_epsilon: float
    use_ema: bool
    ema_inv_gamma: float
    ema_power: float
    ema_max_decay: float
    pe_norm_loss_weight: float
    mixed_precision: str
    checkpointing_steps: int
    checkpoints_total_limit: int
    resume_from_checkpoint: str | None
    enable_xformers_memory_efficient_attention: bool
    acc_seed: int

    # Dataset params
    train_lookup_table_path: str # Path to the cloud dataset lookup table
    val_lookup_table_path: str # Path to the cloud dataset lookup table
    credentials_path: str # Path to the cloud credentials
    bucket_name: str # Name of the google cloud bucket for the big dataset
    cache_dir: str # Directory for storing the cache
    nbars: int
    dataset_dir: str # The dir for the local batch
    load_first_n_data: int # Number of data to load first, set to -1 to load all data

    # VQVAE params
    vae_ckpt_path: str

    # Loading the VQVAE model
    model_id: str
    initial_prompt: str # By freezing the prompt, we specify the initial prompt for the model
    freeze_initial_prompt: bool

    # Project params
    project_name: str

def parse_args(path: str, vae_ckpt_path: str, train_lookup_table_path: str = "./resources/lookup_table_train.json",
               val_lookup_table_path: str = "./resources/lookup_table_val.json", **kwargs) -> TrainingConfig:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    config['vae_ckpt_path'] = vae_ckpt_path
    config['train_lookup_table_path'] = train_lookup_table_path
    config['val_lookup_table_path'] = val_lookup_table_path
    config.update(kwargs)
    return TrainingConfig(**config)

def load_vae(args_path: str, args: TrainingConfig, device):
    # Load the VQVAE model
    from .train_vqvae import read_config

    vae_config_path = os.path.join(os.path.dirname(args_path), "vqvae.yaml")
    config = read_config(vae_config_path)
    vae_config = VQVAEConfig(**config['autoencoder_params'])
    model = VQVAE(im_channels=config['dataset_params']['im_channels'], model_config=vae_config).to(device)
    sd = torch.load(args.vae_ckpt_path, map_location=device)
    model.load_state_dict(sd)
    return model

def load_unet_model(args: TrainingConfig, device):
    # Loads the UNet model and scheduler.
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"


    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]

    image.save("astronaut_rides_horse.png")

    prompt = args.initial_prompt

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        lora_scale=None,
        clip_skip=None,
    )

    pe = PromptEmbed(prompt_embeds)
    unet: UNet2DConditionModel = pipe.unet
    scheduler: DDPMScheduler = pipe.scheduler

    if args.freeze_initial_prompt:
        pe.requires_grad_(False)

    return unet, scheduler, pe

def load_train_dataset(args: TrainingConfig):
    im_dataset = SpectrogramDatasetFromCloud(
        lookup_table_path=args.train_lookup_table_path,
        default_specs=SpectrogramDataset(dataset_dir=args.dataset_dir, num_workers=0, load_first_n=10),
        credentials_path=args.credentials_path,
        bucket_name=args.bucket_name,
        cache_dir=args.cache_dir,
        nbars=args.nbars,
    )
    return im_dataset

def load_val_dataset(args: TrainingConfig):
    im_dataset = SpectrogramDatasetFromCloud(
        lookup_table_path=args.val_lookup_table_path,
        default_specs=SpectrogramDataset(dataset_dir=args.dataset_dir, num_workers=0, load_first_n=3),
        credentials_path=args.credentials_path,
        bucket_name=args.bucket_name,
        cache_dir=args.cache_dir,
        nbars=args.nbars,
    )
    return im_dataset

def sample(unet: UNet2DConditionModel, prompt_embeds: PromptEmbed, scheduler: DDPMScheduler, vae: VQVAE, device, sample_prefix: str = "sample_"):
    scheduler.set_timesteps(50, device)
    timesteps = scheduler.timesteps
    embeds, _ = prompt_embeds()
    with torch.no_grad():
        latents = randn_tensor((1, 4, 64, 64), dtype = embeds.dtype, device = device)
        for i, t in enumerate(timesteps):
            latent_model_input = scheduler.scale_model_input(latents, t)
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        latents = latents / VAE_SCALING_FACTOR
        images = vae.decode(latents)
    audio = vae_output_to_audio(images)
    audio.save(f"{sample_prefix}{t}.mp3")

def save_model(accelerator: Accelerator,
               args: TrainingConfig,
               unet: UNet2DConditionModel,
               prompt_embeds: PromptEmbed,
               ema_model: EMAModel,
               filename: str):
    unet = accelerator.unwrap_model(unet)
    prompt_embeds = accelerator.unwrap_model(prompt_embeds)

    if args.use_ema:
        ema_model.store(unet.parameters())
        ema_model.copy_to(unet.parameters())

    path = os.path.join(args.output_dir, filename)
    torch.save({
        "unet": unet.state_dict(),
        "prompt_embeds": prompt_embeds.state_dict(),
    }, path)
    print(f"Saved UNet model to {path}")

    if args.use_ema:
        ema_model.restore(unet.parameters())

def main(config_path: str,
         vae_ckpt_path: str,
         lookup_table_path: str = "./resources/lookup_table_train.json",
         val_lookup_table_path: str = "./resources/lookup_table_val.json",
         **kwargs):

    args = parse_args(config_path, vae_ckpt_path, lookup_table_path, val_lookup_table_path, **kwargs)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    if args.acc_seed is not None:
        set_seed(args.acc_seed)

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load pretrained VAE model
    vae = load_vae(config_path, args, accelerator.device)
    vae.requires_grad_(False)

    # Initialize the model
    model, noise_scheduler, prompt_embeds = load_unet_model(args, accelerator.device)

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=model.__class__,
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        object.__setattr__(args, "mixed_precision", "fp16")
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        object.__setattr__(args, "mixed_precision", "bf16")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            if version.parse(xformers.__version__) == version.parse("0.0.16"):
                print("xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.")
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Initialize the optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = load_train_dataset(args)

    print(f"Dataset size: {len(dataset)}")

    train_dataloader = DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler, prompt_embeds = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, prompt_embeds
    )

    vae = vae.to(accelerator.device, dtype=weight_dtype)

    if args.use_ema:
        ema_model.to(accelerator.device)

    run = os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    print("***** Running training *****")
    print(f"  Num examples = {len(dataset)}")
    print(f"  Num Epochs = {args.num_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            object.__setattr__(args, "resume_from_checkpoint", None)
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            first_epoch = global_step // num_update_steps_per_epoch

    wandb.init(
        # set the wandb project where this run will be logged
        project=args.project_name,
        config=vars(args),
    )

    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        mse_losses = []
        regularizer_losses = []
        losses = []

        model.train()

        optimizer.zero_grad()

        for im in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            global_step += 1

            # im is (B, 4, 2, 512, 512) -> take the mean of the 2 channels
            im = im.to(weight_dtype).mean(dim=2)
            latents, _ = vae.encode(im)
            latents = latents * VAE_SCALING_FACTOR

            noise = torch.randn(latents.shape, dtype=weight_dtype, device=latents.device)
            timesteps = torch.randint(
                0, noise_scheduler.config["num_train_timesteps"], (latents.shape[0],), device=im.device
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) #type: ignore

            # Forward pass
            with accelerator.autocast():
                encoder_hidden_states, regularizer_loss = prompt_embeds()
                encoder_hidden_states = encoder_hidden_states.expand(noisy_latents.shape[0], -1, -1)
                model_output = model(noisy_latents, timesteps, encoder_hidden_states).sample
                mse = F.mse_loss(model_output.float(), noise.float())

            # Checkpoint every args.checkpointing_steps steps
            if global_step % args.checkpointing_steps == 0:
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        print(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                        print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)

            # Compute backward pass
            mse_losses.append(mse.item())
            regularizer_losses.append(regularizer_loss.item())
            loss = mse + args.pe_norm_loss_weight * regularizer_loss
            accelerator.backward(loss)

            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update the EMA model
            if args.use_ema:
                ema_model.step(model.parameters())

            logs = {
                "mse": mse.detach().item(),
                "regularizer": regularizer_loss.detach().item(),
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value

            wandb.log(logs, step=global_step)

        accelerator.wait_for_everyone()

        # Save the model after the epoch
        save_model(accelerator, args, model, prompt_embeds, ema_model, f"model-{epoch}.pt")
    accelerator.end_training()

if __name__ == "__main__":
    config_path = "./resources/config/unet.yaml"
    vae_ckpt_path = "./resources/ckpts/vqvae.pt"
    main(config_path, vae_ckpt_path)
