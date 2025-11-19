import torch
import torch.optim as optim
from GAN import Generator, Discriminator
from sample import sample_real_data
from plotting import plot_generated

# ------------------------------
# Train discriminator for one step
# ------------------------------
def train_discriminator(D, G, d_optimizer, batch_size, device, mode="gaussian"):
    real_data = sample_real_data(batch_size, mode=mode).to(device)
    fake_input = torch.randn(batch_size, 2).to(device)

    #--------------------------------------------------------------
    # To-Do: Generate fake samples using the generator
    # Make sure to detach() so gradients don’t flow into G here!!
    #--------------------------------------------------------------

    #--------------------------------------------------------------

    # Forward pass
    d_real = D(real_data)
    d_fake = D(fake_data)

    #--------------------------------------------------------------
    # To-Do: Compute discriminator loss
    # Goal: D(real) → 1, D(fake) → 0
    #--------------------------------------------------------------



    #--------------------------------------------------------------

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    return d_loss, d_real, d_fake


# ------------------------------
# Train generator for one step
# ------------------------------
def train_generator(G, D, g_optimizer, batch_size, device):
    fake_input = torch.randn(batch_size, 2).to(device)

    #--------------------------------------------------------------
    # To-Do: Generate fake samples (do NOT detach)
    #--------------------------------------------------------------

    #--------------------------------------------------------------

    # Forward pass
    d_fake = D(fake_data)

    #--------------------------------------------------------------
    # To-Do: Compute generator loss
    #--------------------------------------------------------------

    #--------------------------------------------------------------

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    return g_loss, d_fake


# ------------------------------
# Main GAN training loop
# ------------------------------
def train_gan(device, num_steps=10000, batch_size=128, plot_every=1000, mode="gaussian"):
    G = Generator().to(device)
    D = Discriminator().to(device)
    g_optimizer = optim.Adam(G.parameters(), lr=1e-3)
    d_optimizer = optim.Adam(D.parameters(), lr=1e-3)
    
    losses = {'D': [], 'G': [], 'D(x)': [], 'D(G(z))': []}

    for step in range(1, num_steps + 1):
        for _ in range(2): # <- To-Do: Modify the training loop here
            d_loss, d_real, d_fake = train_discriminator(D, G, d_optimizer, batch_size, device, mode)

        g_loss, d_fake = train_generator(G, D, g_optimizer, batch_size, device)

        if step % plot_every == 0:
            print(f"[{step}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, "
                  f"D(x): {d_real.mean().item():.2f}, D(G(z)): {d_fake.mean().item():.2f}")
            losses['D'].append(d_loss.item())
            losses['G'].append(g_loss.item())
            losses['D(x)'].append(d_real.mean().item())
            losses['D(G(z))'].append(d_fake.mean().item())

            plot_generated(G, mode, step)

    return losses