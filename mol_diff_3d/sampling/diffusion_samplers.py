import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class DDPMQSampler:
    """
    Forward diffusion process (q-sampler).
    Responsibility: Add noise to the data based on a given noise schedule.
    """

    def __init__(self, scheduler_params: Dict):
        self.alphas_cumprod = scheduler_params['alphas_cumprod']
        self.sqrt_alphas_cumprod = scheduler_params['sqrt_alphas_cumprod']
        self.sqrt_one_minus_alphas_cumprod = scheduler_params['sqrt_one_minus_alphas_cumprod']
        self.num_timesteps = len(self.alphas_cumprod)

    def q_sample_step(self, x_start: torch.Tensor, t: torch.Tensor,
                      noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion for discrete features.
        MolDiff Paper (Eq. 7): q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        while sqrt_alpha_cumprod_t.dim() < x_start.dim():
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)

        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def q_sample_pos_step(self, pos_start: torch.Tensor, t: torch.Tensor,
                          noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion for 3D positions.
        """
        if noise is None:
            noise = torch.randn_like(pos_start)

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1)

        return sqrt_alpha_cumprod_t * pos_start + sqrt_one_minus_alpha_cumprod_t * noise


class DDPMPsampler:
    """
    Reverse diffusion process (p-sampler).
    Responsibility: Denoise data using a trained model and a given noise schedule.
    """

    def __init__(self, scheduler_params: Dict):
        self.betas = scheduler_params['betas']
        self.alphas_cumprod = scheduler_params['alphas_cumprod']
        self.num_timesteps = len(self.betas)

    def p_sample_step(self, model, x_t: torch.Tensor, pos_t: torch.Tensor, edge_index: torch.Tensor,
                      batch: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse DDPM sampling step, now for both features and positions.
        """
        noise_pred_x, noise_pred_pos = model(x_t, edge_index, pos_t, batch, t)

        alpha_t = 1 - self.betas[t]

        mean_pos = (pos_t - (1 - alpha_t).sqrt() * noise_pred_pos) / alpha_t.sqrt()

        x_t_minus_1_pos = mean_pos
        if t > 0:
            x_t_minus_1_pos += torch.sqrt(self.betas[t]) * torch.randn_like(pos_t)

        mean_x = (x_t - (1 - alpha_t).sqrt() * noise_pred_x) / alpha_t.sqrt()
        x_t_minus_1_x = mean_x
        if t > 0:
            x_t_minus_1_x += torch.sqrt(self.betas[t]) * torch.randn_like(x_t)

        return x_t_minus_1_x, x_t_minus_1_pos