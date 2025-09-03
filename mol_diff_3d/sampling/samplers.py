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
        Forward diffusion for continuous positions.
        """
        if noise is None:
            noise = torch.randn_like(pos_start)

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Ensure broadcasting is correct
        while sqrt_alpha_cumprod_t.dim() < pos_start.dim():
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)

        return sqrt_alpha_cumprod_t * pos_start + sqrt_one_minus_alpha_cumprod_t * noise


class DDPMPsampler:
    """
    Reverse diffusion process (p-sampler).
    Responsibility: Denoise data using a trained model and a given noise schedule.
    """

    def __init__(self, scheduler_params: Dict):
        self.betas = scheduler_params['betas']
        self.alphas_cumprod = scheduler_params['alphas_cumprod']
        self.alphas_cumprod_prev = scheduler_params['alphas_cumprod_prev']
        self.posterior_variance = scheduler_params['posterior_variance']
        self.num_timesteps = len(self.betas)

    def p_sample_step(self, model, x_t: torch.Tensor, pos_t: torch.Tensor, edge_index: torch.Tensor,
                      batch: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse DDPM sampling step.
        """
        # Ensure model is in evaluation mode for sampling
        model.eval()

        # Get noise prediction from the model for both features and positions
        noise_pred_x, noise_pred_pos = model(x_t, edge_index, pos_t, batch, t)

        # --- Positional sampling (Continuous Data) ---
        # The reverse process for continuous positions is a Gaussian.
        # It's a key property of DDPMs that this can be calculated directly.

        # The mean of the reverse Gaussian distribution
        # Eq. 9 in DDPM paper (Ho et al.) and implied in MolDiff.
        mean_pos = (
                1 / (1 - self.betas[t]).sqrt().unsqueeze(-1)
                * (pos_t - self.betas[t].unsqueeze(-1) / (1 - self.alphas_cumprod[t]).sqrt().unsqueeze(
            -1) * noise_pred_pos)
        )

        # The variance of the reverse Gaussian distribution
        # Precomputed in noise_scheduler.py as self.posterior_variance
        variance_pos = self.posterior_variance[t].unsqueeze(-1)

        # If t > 0, we add noise to the mean to get the next state.
        if t[0] > 0:
            x_t_minus_1_pos = mean_pos + variance_pos.sqrt() * torch.randn_like(pos_t)
        else:
            x_t_minus_1_pos = mean_pos  # No noise added at the final step (t=0)

        # --- Feature Sampling (Discrete Data) ---
        # The reverse process for discrete features is a multinomial distribution.
        # The model's noise prediction for features serves as logits for this distribution.

        # The MolDiff paper's approach for discrete features is to use the
        # predicted noise to estimate the probability of each atom type at
        # the previous timestep. The simplest, and most common, way to
        # do this is to take the argmax of the predicted logits.
        # For more complex cases, you might use a Gumbel-Softmax trick during training,
        # but for sampling, argmax is the standard.

        # This is where your implementation needs a key correction.
        # We don't add noise here. We predict the discrete value directly.

        # The simplest way to handle this during sampling is to directly predict
        # the final atom type (one-hot vector). This works because the reverse
        # process is typically very deterministic for discrete data.

        # We take the argmax to get the predicted class index (e.g., C, N, O)
        x_t_minus_1_x = noise_pred_x.argmax(dim=-1)

        # One-hot encode the predicted class index
        x_t_minus_1_x = F.one_hot(x_t_minus_1_x, num_classes=x_t.shape[-1]).float()

        return x_t_minus_1_x, x_t_minus_1_pos