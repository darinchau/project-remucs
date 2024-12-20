# Looks at the model folder really hard and pray it doesnt explode

import os
import sys
import time

base_dir = "./resources/models/vqvae"
FILE_LIMIT = 10

def main():
    try:
        a = int(sys.argv[1])
    except Exception as e:
        raise ValueError("Please specify a value to start monitoring")
    print(f"Starting with a = {a}")
    while True:
        time.sleep(10)
        if len(os.listdir(base_dir)) < FILE_LIMIT:
            continue
        model = os.path.join(base_dir, f"vqvae_{a}_vqvae_autoencoder_ckpt.pth")
        discr = os.path.join(base_dir, f"discriminator_{a}_vqvae_autoencoder_ckpt.pth")
        if not os.path.isfile(model) or not os.path.isfile(discr):
            continue
        print("Doing a = ", a)
        try:
            os.remove(model)
            os.remove(discr)
        except Exception as e:
            print(e)
        a += 512

if __name__ == "__main__":
    main()
