import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (Shortcut)
        self.shortcut = nn.Sequential()
        # If the input and output shapes don't match (due to stride or channels), 
        # we need to transform the shortcut to match.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # <--- The Magic Skip Connection
        out = F.relu(out)
        return out

class ResNetVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ResNetVAE, self).__init__()
        self.latent_dim = latent_dim

        # --- ENCODER ---
        # Input: 3 x 32 x 32
        self.encoder = nn.Sequential(
            # Initial lift
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            # Downsampling Blocks
            ResBlock(32, 64, stride=2),  # -> 16x16
            ResBlock(64, 128, stride=2), # -> 8x8
            ResBlock(128, 256, stride=2),# -> 4x4
            nn.Flatten()
        )
        
        # Latent Space
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(256 * 4 * 4, latent_dim)

        # --- DECODER ---
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            # We use ConvTranspose2d to upsample
            # Note: Implementing ResNet Decoder is tricky, so we often use 
            # ConvTranspose layers with good capacity for the decoder part 
            # to mirror the encoder's power.
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 4->8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 16->32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.Conv2d(32, 3, kernel_size=3, padding=1), # Final projection
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_var(encoded)
        z = self.reparameterize(mu, logvar)
        
        z_projected = self.decoder_input(z)
        
        # FIX IS HERE: Reshape to match the Decoder's first layer input (256 channels)
        z_reshaped = z_projected.view(-1, 256, 4, 4) 
        
        reconstruction = self.decoder(z_reshaped)
        
        return reconstruction, mu, logvar
