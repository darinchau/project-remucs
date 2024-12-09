# Looks at the model folder really hard and pray it doesnt explode


import os
import time

base_dir = "./resources/models/vqvae"
FILE_LIMIT = 15

a = 139264
while True:
    time.sleep(10)
    if len(os.listdir(base_dir)) < FILE_LIMIT:
        continue
    model = os.path.join(base_dir, f"vqvae_epoch_0_{a}_vqvae_autoencoder_ckpt.pth")
    discr = os.path.join(base_dir, f"discriminator_epoch_0_{a}_vqvae_autoencoder_ckpt.pth")
    if not os.path.isfile(model) or not os.path.isfile(discr):
        continue
    print("Doing a = ", a)
    try:
        os.remove(model)
        os.remove(discr)
    except Exception as e:
        print(e)
    a += 512
