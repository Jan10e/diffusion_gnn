import torch
import numpy as np


def polynomial_schedule(num_steps: int, power: int = 2, precision: float = 1e-5) -> torch.Tensor:
    """
    Polynomial noise schedule (used in EDM paper)

    Args:
        num_steps: Number of diffusion steps
        power: Polynomial power (default 2)
        precision: Minimum beta value

    Returns:
        Beta schedule tensor
    """
    t = torch.linspace(0, 1, num_steps)
    alphas = (1 - 2 * t + t ** power) ** power
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = torch.cat([torch.tensor([0.0]), betas])
    return betas.clamp(min=precision, max=0.999)


def linear_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear noise schedule

    Args:
        num_steps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value

    Returns:
        Beta schedule tensor
    """
    return torch.linspace(beta_start, beta_end, num_steps)


def cosine_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule (from improved DDPM paper)

    Args:
        num_steps: Number of diffusion steps
        s: Small offset to prevent beta from being too small

    Returns:
        Beta schedule tensor
    """
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps)
    alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

