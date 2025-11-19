from VAE import VAE
from train import train_epoch
from latent_traversal import latent_traversal
from sample_from_prior import sample_from_prior
from tsne import tsne_latent_mnist

if __name__ == "__main__":
    import torch
    import torch.optim as optim
    from torchvision import datasets, transforms

    # ---------------- Training Setup ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 1
    vae = VAE(latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=128, shuffle=True
    )

    epochs = 5
    for epoch in range(epochs):
        train_loss = train_epoch(vae, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}, Loss {train_loss:.4f}")


    latent_traversal(vae, device)

    sample_from_prior(vae, device)
    
    tsne_latent_mnist(vae, device, n_samples=2000)