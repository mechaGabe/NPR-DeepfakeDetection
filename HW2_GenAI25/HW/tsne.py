import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

@torch.no_grad()
def tsne_latent_mnist(vae, device, n_samples=2000):
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    loader = DataLoader(mnist, batch_size=n_samples, shuffle=True)
    
    xs, ys = next(iter(loader))
    xs = xs.to(device)
    
    #---------------------------------------------------------------
    # To-Do: Encode to latent space


    #---------------------------------------------------------------
    z = z.cpu().numpy()
    ys = ys.numpy()

    #---------------------------------------------------------------
    # To-Do: Apply TSNE to obtain z_2d

    #---------------------------------------------------------------

    plt.figure(figsize=(8,8))
    scatter = plt.scatter(z_2d[:,0], z_2d[:,1], c=ys, cmap="tab10", s=10, alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Digits")
    plt.title("t-SNE of VAE Latent Space (MNIST)")
    plt.axis("off")
    # plt.show()
    plt.savefig("Latent_TSNE_MNIST.png")