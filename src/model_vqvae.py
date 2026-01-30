import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        # This is the "Codebook" (e.g., 512 distinct vectors)
        # We initialize it uniformly
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # inputs shape: [Batch, Channels, Height, Width]
        # Convert to: [Batch, Height, Width, Channels] for easier math
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input: [N*H*W, Channels]
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances between input vectors and ALL codebook vectors
        # Formula: (a-b)^2 = a^2 + b^2 - 2ab
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding: Find the nearest codebook vector index (argmin)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Create one-hot encoding
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize: Replace input with the nearest codebook vector
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # --- LOSS CALCULATION ---
        # 1. Codebook Loss: Move codebook vectors closer to inputs
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        
        # 2. Commitment Loss: Prevent encoder from growing too huge
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # --- STRAIGHT THROUGH ESTIMATOR (The Magic) ---
        # We cannot backprop through "argmin" (it's discrete). 
        # So we manually copy the gradients from 'quantized' directly to 'inputs'
        quantized = inputs + (quantized - inputs).detach()
        
        # Reshape back to [Batch, Channels, Height, Width]
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encodings

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels, out_channels=num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, num_residual_layers):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens) for _ in range(self._num_residual_layers)])
    
    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class VQVAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        
        # Encoder
        self._encoder = nn.Sequential(
            nn.Conv2d(3, num_hiddens // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1),
            ResidualStack(num_hiddens, num_hiddens, num_residual_hiddens, num_residual_layers)
        )
        
        # Pre-VQ Conv (Project to embedding dim)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
        
        # The VQ Layer
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # Decoder
        self._decoder = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1),
            ResidualStack(num_hiddens, num_hiddens, num_residual_hiddens, num_residual_layers),
            nn.ConvTranspose2d(in_channels=num_hiddens, out_channels=num_hiddens // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=num_hiddens // 2, out_channels=3, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        
        return loss, x_recon, quantized
