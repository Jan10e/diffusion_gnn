import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timesteps
    Similar to transformer positional encodings
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: [batch_size] tensor of timesteps

        Returns:
            [batch_size, dim] embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    """
    Time embedding with learnable projection
    """

    def __init__(self, dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbeddings(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: [batch_size] tensor of timesteps

        Returns:
            [batch_size, dim] time embeddings
        """
        emb = self.sinusoidal(time)
        emb = self.mlp(emb)
        return emb