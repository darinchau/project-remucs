# Looks at the model folder really hard and pray it doesnt explode

import os
import sys
import time
import yaml

base_dir = "./resources/models/vqvae"
config_path = "./resources/config/vqvae.yaml"
FILE_LIMIT = 10


def read_config(config_path: str):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)
    return config


def delete_stuff(a: int, config):
    train_config = config['train_params']
    model_save_path = os.path.join(output_dir, f"vqvae_{step_count}_{train_config['vqvae_autoencoder_ckpt_name']}")
    disc_save_path = os.path.join(output_dir, f"discriminator_{step_count}_{train_config['vqvae_autoencoder_ckpt_name']}")

    if os.path.isfile(discr):
        try:
            os.remove(discr)
        except Exception as e:
            print(e)
    if os.path.isfile(model):
        try:
            os.remove(model)
        except Exception as e:
            print(e)


def main():
    config = read_config(config_path)

    while True:
        time.sleep(10)
        if len(os.listdir(base_dir)) < FILE_LIMIT:
            continue

        files = os.listdir(base_dir)
        indices = [int(file.split('_')[1]) for file in files if file.startswith('vqvae') or file.startswith('discriminator')]

        # Delete the oldest model
        a = min(indices)
        delete_stuff(a, config)
        print(f"Deleted model {a}")


if __name__ == "__main__":
    main()
