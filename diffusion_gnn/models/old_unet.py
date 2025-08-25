import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet(nn.Module):
    """
    A simplified U-Net architecture for denoising, i.e. the reverse
    process of diffusion. This is ε_θ(x_t, t) - the neural network
    that predicts noise.
    Key insight: we predict the noise ε, instead of directly predicting
    x_{t-1}. This is more stable and allows for better training.
    The network needs to know the timestep t to predict appropriate
    noise level.

    The architecture consists of:
    - Downsampling layers to capture low-level features.
    - Bottleneck layer that combines features with time conditioning.
    - Upsampling layers to reconstruct the original input.
    - Positional encoding for time steps to condition the network.
    - Group normalization for stable training.
    - GELU activation for non-linearity.

    The input is a 1D tensor (e.g., molecular graph features) and the
    output is also a 1D tensor of the same shape, representing the
    denoised features at the previous time step.

    Args:
        dim (int): Base dimension of the model. Default is 64.
        channels (int): Number of input channels. Default is 1.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, channels, seq_length).
    """
    def __init__(self, dim:int=64, channels:int=1) -> torch.Tensor:
        super().__init__()
        self.channels = channels
        self.dim = dim

        # Time embedding
        # Maps timestep t to a learned embedding
        time_dim = dim * 4  # 4x the base dimension
        self.time_mlp = nn.Sequential(
            PositionalEncoding(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Downsampling layers
        self.down1 = nn.Sequential(
            nn.Conv1d(channels, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )

        self.down2 = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 3, padding=1, stride=2),
            nn.GroupNorm(8, dim * 2),
            nn.GELU(),
        )

        # Bottleneck layer with time conditioning
        self.bottleneck = nn.Sequential(
            nn.Conv1d(dim * 2, dim * 4, 3, padding=1),
            nn.GroupNorm(8, dim * 4),
            nn.GELU(),
            nn.Conv1d(dim * 4, dim * 4, 3, padding=1),
            nn.GroupNorm(8, dim * 4),
            nn.GELU(),
        )

        # Time projection for bottleneck
        self.time_proj = nn.Linear(time_dim, dim * 4)

        # Upsampling layers
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(dim * 4, dim * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, dim * 2),
            nn.GELU(),
        )

        self.up2 = nn.Sequential(
            nn.Conv1d(dim * 4, dim, 3, padding=1), # dim*4 due to skip connections
            nn.GroupNorm(8, dim),
            nn.GELU(),
        )

        # Output layer
        self.out = nn.Conv1d(dim * 2, channels, 1)  # dim*2 due to skip connections

    def forward(self, x:torch.Tensor, timestep:torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.
        x shape: (batch_size, channels, seq_length)
        timestep shape: (batch_size,)

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_length).
            timestep (torch.Tensor): Time step tensor of shape (batch_size,).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, seq_length).
        """
        # # Ensure x is [batch_size, channels, seq_length]
        # if x.shape[1] != self.channels:
        #     # Permute from [batch_size, seq_length, channels] to [batch_size, channels, seq_length]
        #     x = x.permute(0, 2, 1)

        # Time embedding
        t = self.time_mlp(timestep)

        # Downsampling with skip connections
        h1 = self.down1(x)
        h2 = self.down2(h1)

        # Bottleneck with time conditioning
        h = self.bottleneck(h2)

        # At time information to bottleneck features
        # This is crucial - the network needs to know what timestep it's denoising
        t_proj = self.time_proj(t).unsqueeze(-1)  # add spatial dimension
        h = h + t_proj

        # Upsampling with skip connections (U-Net style)
        h = self.up1(h)

        # Handle spatial dimension mismatch before skip connection
        if h.shape[-1] != h2.shape[-1]:
            h = F.interpolate(h, size=h2.shape[-1], mode='linear', align_corners=False)

        h = torch.cat([h, h2], dim=1)  # skip connection
        h = self.up2(h)

        # Handle spatial dimension mismatch before second skip connection
        if h.shape[-1] != h1.shape[-1]:
            h = F.interpolate(h, size=h1.shape[-1], mode='linear', align_corners=False)

        h = torch.cat([h, h1], dim=1)  # skip connection

        return self.out(h)



class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for time steps.
    Maps an integer time step t to a continuous embedding.
    This is used to condition the U-Net on the time step, and
    helps the network understand the noise level.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings