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

def main():
    try:
        a = int(sys.argv[1])
    except Exception as e:
        raise ValueError("Usage: python -m scripts.monitor [START_ITER]")

    config = read_config(config_path)

    print(f"Starting with a = {a}")
    while True:
        time.sleep(10)
        if len(os.listdir(base_dir)) < FILE_LIMIT:
            continue
        model = os.path.join(base_dir, f"vqvae_{a}_{config['train_params']['vqvae_autoencoder_ckpt_name']}")
        discr = os.path.join(base_dir, f"vqvae_{a}_{config['train_params']['vqvae_discriminator_ckpt_name']}")

        print("Doing a = ", a)
        if os.path.isfile(model) and os.path.isfile(discr):
            try:
                os.remove(model)
                os.remove(discr)
            except Exception as e:
                print(e)
        elif os.path.isfile(discr):
            try:
                os.remove(discr)
            except Exception as e:
                print(e)
        elif os.path.isfile(model):
            try:
                os.remove(model)
            except Exception as e:
                print(e)
        a += config['train_params']['autoencoder_img_save_steps']


if __name__ == "__main__":
    main()
