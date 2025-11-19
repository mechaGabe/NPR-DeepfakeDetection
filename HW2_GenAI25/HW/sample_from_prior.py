import torch
import numpy as np
import matplotlib.pyplot as plt

@torch.no_grad()
def sample_from_prior(vae, device, n=16):
    #---------------------------------------------------------------
    # To-Do: Sample from prior and decode


    #---------------------------------------------------------------
    nrow = int(np.sqrt(n))
    fig, axs = plt.subplots(nrow, nrow, figsize=(nrow, nrow))
    idx = 0
    for i in range(nrow):
        for j in range(nrow):
            axs[i, j].imshow(recon[idx], cmap="gray")
            axs[i, j].axis("off")
            idx += 1
    plt.suptitle("Samples from prior (decoded)")
    plt.savefig("Samples from prior.png")