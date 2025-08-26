import torch
from typing import Optional


class NoiseScheduler:
    """
    Pure noise scheduling logic.
    Single responsibility: handle diffusion math.
    """

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        # Precompute all diffusion coefficients
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # Store coefficients (will be moved to device when needed)
        self.num_timesteps = num_timesteps
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def to(self, device):
        """Move tensors to device"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def q_sample_step(self, x_start: torch.Tensor, t: torch.Tensor,
                      noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Handle broadcasting
        while sqrt_alpha_cumprod_t.dim() < x_start.dim():
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)

        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    # Add to diffusion_gnn/models/noise_scheduler.py
    def p_sample_step(self, model, x, edge_index, edge_attr, batch, t):
        """Proper DDPM reverse sampling step"""
        # Predict noise
        noise_pred = model(x, edge_index, edge_attr, batch, t)

        # DDPM coefficients
        alpha_t = 1 - self.betas[t]
        alpha_bar_t = self.alphas_cumprod[t]

        # Mean calculation
        coeff1 = 1 / torch.sqrt(alpha_t)
        coeff2 = self.betas[t] / torch.sqrt(1 - alpha_bar_t)

        mean = coeff1[batch].unsqueeze(-1) * (x - coeff2[batch].unsqueeze(-1) * noise_pred)

        # Add noise (except final step)
        if t[0] > 0:
            posterior_var = self.betas[t] * (1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t])
            noise = torch.randn_like(x)
            return mean + torch.sqrt(posterior_var[batch].unsqueeze(-1)) * noise
        else:
            return mean