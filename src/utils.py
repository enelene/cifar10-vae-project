import torch
import torch.nn.functional as F

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction Loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence Loss
    # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total Loss
    total_loss = recon_loss + (beta * kld_loss)
    
    return total_loss, recon_loss, kld_loss
