import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction Loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence Loss
    # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total Loss
    total_loss = recon_loss + (beta * kld_loss)
    
    return total_loss, recon_loss, kld_loss

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # 1. Load VGG19 pre-trained on ImageNet
        # We use the 'features' part of VGG, which contains the Convolutions
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # 2. Slice VGG to extract features at different depths
        # We want features from early layers (edges) and middle layers (textures)
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        
        # VGG19 Layer Mapping:
        # Relu1_2 is at index 3 (captures fine details)
        # Relu2_2 is at index 8 (captures textures)
        # Relu3_4 is at index 17 (captures parts of objects)
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg[x])
            
        # 3. Freeze VGG (We are NOT training this network)
        for param in self.parameters():
            param.requires_grad = False
            
        # Normalization for VGG (ImageNet stats)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

    def forward(self, recon_x, x):
        # VGG expects normalized inputs. 
        # Assuming inputs are in [0, 1], we normalize them.
        x_norm = self.normalize(x)
        recon_norm = self.normalize(recon_x)

        # Get features for Original Image
        h1 = self.slice1(x_norm)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)

        # Get features for Reconstructed Image
        h1_recon = self.slice1(recon_norm)
        h2_recon = self.slice2(h1_recon)
        h3_recon = self.slice3(h2_recon)

        # Calculate MSE loss between the FEATURE MAPS, not the pixels
        loss1 = nn.functional.mse_loss(h1_recon, h1)
        loss2 = nn.functional.mse_loss(h2_recon, h2)
        loss3 = nn.functional.mse_loss(h3_recon, h3)
        
        # Weighted sum: Early layers (loss1) are crucial for sharpness
        return loss1 + loss2 + loss3
