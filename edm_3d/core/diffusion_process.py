import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import deepchem as dc
from deepchem.models.torch_models import TorchModel


class DiffusionProcess:
    """
    Implements the forward and reverse diffusion process for EDM
    """

    def __init__(
            self,
            num_steps: int = 1000,
            noise_schedule: str = 'polynomial_2',
            noise_precision: float = 1e-5
    ):
        self.num_steps = num_steps
        self.noise_schedule = noise_schedule

        # Compute noise schedule (beta values)
        self.betas = self._get_noise_schedule(noise_precision)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def _get_noise_schedule(self, precision: float) -> torch.Tensor:
        """
        Generate noise schedule

        Polynomial_2 schedule from EDM paper
        """
        t = torch.linspace(0, 1, self.num_steps)

        if self.noise_schedule == 'polynomial_2':
            alphas = (1 - 2 * t + t ** 2) ** 2
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = torch.cat([torch.tensor([0.0]), betas])
        else:
            # Linear schedule
            betas = torch.linspace(precision, 0.02, self.num_steps)

        return betas.clamp(min=precision, max=0.999)

    def forward_diffusion(
            self,
            x0: torch.Tensor,
            h0: torch.Tensor,
            t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add noise to molecule (forward process)

        q(x_t, h_t | x_0, h_0)

        Args:
            x0: Original coordinates [N, 3]
            h0: Original atom types [N, num_atom_types]
            t: Timestep [batch_size]

        Returns:
            Noisy coordinates, noisy atom types, noise_x, noise_h
        """
        # Get alpha_bar for timestep t
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)

        # Sample noise
        noise_x = torch.randn_like(x0)
        noise_h = torch.randn_like(h0)

        # Add noise (reparameterization trick)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise_x
        ht = torch.sqrt(alpha_bar_t) * h0 + torch.sqrt(1 - alpha_bar_t) * noise_h

        # Center coordinates (maintain E(3) equivariance)
        xt = xt - xt.mean(dim=1, keepdim=True)

        return xt, ht, noise_x, noise_h

    def reverse_diffusion_step(
            self,
            xt: torch.Tensor,
            ht: torch.Tensor,
            pred_noise_x: torch.Tensor,
            pred_noise_h: torch.Tensor,
            t: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single reverse diffusion step (denoising)

        p(x_{t-1} | x_t)
        """
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        beta_t = self.betas[t]

        # Predict x0 from xt
        x0_pred = (xt - torch.sqrt(1 - alpha_bar_t) * pred_noise_x) / torch.sqrt(alpha_bar_t)
        h0_pred = (ht - torch.sqrt(1 - alpha_bar_t) * pred_noise_h) / torch.sqrt(alpha_bar_t)

        if t > 0:
            # Add noise for stochastic sampling
            noise_x = torch.randn_like(xt)
            noise_h = torch.randn_like(ht)

            # Compute x_{t-1}
            xt_1 = (
                    torch.sqrt(self.alpha_bars[t - 1]) * beta_t / (1 - alpha_bar_t) * x0_pred +
                    torch.sqrt(alpha_t) * (1 - self.alpha_bars[t - 1]) / (1 - alpha_bar_t) * xt +
                    torch.sqrt(beta_t) * noise_x
            )

            ht_1 = (
                    torch.sqrt(self.alpha_bars[t - 1]) * beta_t / (1 - alpha_bar_t) * h0_pred +
                    torch.sqrt(alpha_t) * (1 - self.alpha_bars[t - 1]) / (1 - alpha_bar_t) * ht +
                    torch.sqrt(beta_t) * noise_h
            )
        else:
            # Final step: no noise
            xt_1 = x0_pred
            ht_1 = h0_pred

        # Center coordinates
        xt_1 = xt_1 - xt_1.mean(dim=1, keepdim=True)

        return xt_1, ht_1

