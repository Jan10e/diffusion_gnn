import torch
import math
from typing import Dict, Union

class NoiseScheduler:
    """
    Pure noise scheduling logic.
    Single responsibility: precompute and provide diffusion coefficients.
    """
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        # Precompute all diffusion coefficients
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.num_timesteps = num_timesteps
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def to(self, device: Union[str, torch.device]):
        """Move all precomputed tensors to the specified device."""
        for attr in self.__dict__:
            if isinstance(self.__dict__[attr], torch.Tensor):
                self.__dict__[attr] = self.__dict__[attr].to(device)
        return self

    def get_parameters(self) -> Dict:
        """Returns a dictionary of all precomputed tensors for use by samplers."""
        return {
            'num_timesteps': self.num_timesteps,
            'betas': self.betas,
            'alphas': self.alphas,
            'alphas_cumprod': self.alphas_cumprod,
            'alphas_cumprod_prev': self.alphas_cumprod_prev,
            'sqrt_alphas_cumprod': self.sqrt_alphas_cumprod,
            'sqrt_one_minus_alphas_cumprod': self.sqrt_one_minus_alphas_cumprod,
            'posterior_variance': self.posterior_variance
        }