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
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
import torch
from torch import nn

import diffusers
from diffusers import DDPMScheduler, UNet2DModel, UNet2DConditionModel, StableDiffusionPipeline #type: ignore
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor

from remucs.dataset import SpectrogramDataset, SpectrogramDatasetFromCloud
from remucs.model.vae import VQVAE, VQVAEConfig

import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__, log_level="INFO")

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
        self.initial_l2 = self.get_norm()
        self.regularizer = regularizer

    def get_norm(self):
        l2_norms = torch.norm(self.tensor, p=2, dim=2)
        average_l2_norm = l2_norms.mean()
        return average_l2_norm

    def forward(self):
        regularizer_loss = ((self.get_norm() - self.initial_l2) / self.regularizer).square()
        return self.tensor.data, regularizer_loss

# Add typing to the arguments
class TrainingConfig:
    output_dir: str
    train_batch_size: int
    eval_batch_size: int
    dataloader_num_workers: int
    num_epochs: int
    save_model_epochs: int
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
    logger: str
    logging_dir: str
    local_rank: int
    mixed_precision: str
    checkpointing_steps: int
    checkpoints_total_limit: int
    resume_from_checkpoint: str | None
    enable_xformers_memory_efficient_attention: bool
    acc_seed: int

    # Dataset params
    lookup_table_path: str # Path to the cloud dataset lookup table
    credentials_path: str # Path to the cloud credentials
    bucket_name: str # Name of the google cloud bucket for the big dataset
    cache_dir: str # Directory for storing the cache
    nbars: int
    dataset_dir: str # The dir for the local batch

    vae_ckpt_path: str # Path to the VAE checkpoint dir
    im_channels: int # Number of channels in our image (for VQVAE, it is your responsibility to make sure VQVAE dims == DDPM dims)
    vae_config: dict

    model_id: str
    initial_prompt: str # By freezing the prompt, we specify the initial prompt for the model
    freeze_initial_prompt: bool

def parse_args(path = "./resources/config/unet.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return TrainingConfig(**config)

def load_vae(args: TrainingConfig, device):
    vae_config = VQVAEConfig(**args.vae_config)
    model = VQVAE(im_channels=args.im_channels, model_config=vae_config).to(device)
    sd = torch.load(args.vae_ckpt_path, map_location=device)
    model.load_state_dict(sd)
    return model

def load_unet_model(args: TrainingConfig, device):
    # Loads the UNet model and scheduler.
    # Implement later...
    model_id = args.model_id

    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16) # type: ignore
    pipe = pipe.to(device)

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

    assert unet.config["sample_size"] == args.vae_config["sample_size"], f"UNet ({unet.config['sample_size']}) and VQVAE ({args.vae_config['sample_size']}) sample sizes must match"
    return unet, scheduler, pe

def load_train_dataset(args: TrainingConfig):
    im_dataset = SpectrogramDatasetFromCloud(
        lookup_table_path=args.lookup_table_path,
        default_specs=SpectrogramDataset(dataset_dir=args.dataset_dir, num_workers=0, load_first_n=10),
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
        raise NotImplementedError

def main(args: TrainingConfig):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.acc_seed is not None:
        set_seed(args.acc_seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load pretrained VAE model
    vae = load_vae(args, accelerator.device)
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
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
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

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset = load_train_dataset(args)

    logger.info(f"Dataset size: {len(dataset)}")

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
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    vae = vae.to(accelerator.device, dtype=weight_dtype)

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

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
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            first_epoch = global_step // num_update_steps_per_epoch

    step_count = 0 #TODO implement save every n steps

    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"].to(weight_dtype)
            latents, _ = vae.encode(clean_images)
            latents = latents * VAE_SCALING_FACTOR

            # Sample noise that we'll add to the images
            noise = torch.randn(latents.shape, dtype=weight_dtype, device=latents.device)
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config["num_train_timesteps"], (latents.shape[0],), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) #type: ignore

            with accelerator.accumulate(model):
                # Predict the noise residual
                model_output = model(noisy_latents, timesteps).sample

                loss = F.mse_loss(model_output.float(), noise.float())  # this could have different weights!

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Save the model and post-process stuff
        if accelerator.is_main_process:
            # Do logging if necessary
            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                # Save the unet
                unet_path = os.path.join(args.output_dir, f"unet-{epoch}.pt")
                torch.save(unet.state_dict(), unet_path)
                logger.info(f"Saved UNet model to {unet_path}")

                if args.use_ema:
                    ema_model.restore(unet.parameters())

    accelerator.end_training()

if __name__ == "__main__":
    main(parse_args())
