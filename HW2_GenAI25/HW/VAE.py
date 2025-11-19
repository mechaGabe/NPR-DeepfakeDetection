import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Encoder ----------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)   # [B,32,14,14]
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)  # [B,64,7,7]


        #--------------------------------------------------------------
        # To-DO: Define the fully connected layers for mu and logvar


        #--------------------------------------------------------------

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h.view(h.size(0), -1)

        #--------------------------------------------------------------
        # To-DO: Compute mu and logvar


        #--------------------------------------------------------------

        return mu, logvar

# ---------------- Decoder ----------------
class Decoder(nn.Module):
    def __init__(self, latent_dim=1):
        super().__init__()

        #--------------------------------------------------------------
        # To-DO: Define the fully connected layer and deconvolutional layers

        #--------------------------------------------------------------

        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # [B,32,14,14]
        self.deconv2 = nn.ConvTranspose2d(32, 1, 4, 2, 1)   # [B,1,28,28]

    def forward(self, z):
        h = F.relu(self.fc(z))
        h = h.view(z.size(0), 64, 7, 7)
        h = F.relu(self.deconv1(h))

        #--------------------------------------------------------------
        # To-DO: Compute the reconstructed mean x_mu
        # no sigmoid; use Gaussian likelihood

        #--------------------------------------------------------------

        return x_mu

# ---------------- VAE ----------------
class VAE(nn.Module):
    def __init__(self, latent_dim=1):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):

        #--------------------------------------------------------------
        # To-DO: Implement the reparameterization trick


        
        #--------------------------------------------------------------

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_mu = self.decoder(z)
        return x_mu, mu, logvar, z