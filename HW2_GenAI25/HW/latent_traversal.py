import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def latent_traversal(vae, device, z_min=-3, z_max=3, steps=15):
    #---------------------------------------------------------------
    # To-Do: Perform latent traversal and decode


    #---------------------------------------------------------------
    imgs = recon.squeeze(1).numpy()
    fig, axs = plt.subplots(1, steps, figsize=(steps, 2))
    for i in range(steps):
        axs[i].imshow(imgs[i], cmap="gray")
        axs[i].axis("off")
    plt.suptitle("Latent traversal (z from left to right)")
    # plt.show()
    plt.savefig("Latent_Traversal.png")