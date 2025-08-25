import torch
import torch.nn.functional as F
from diffusion_1d.core.noise_scheduler_1d import DDPMScheduler

class DDPM:
    """
    Denoising Diffusion Probabilistic Model (DDPM).
    Implements the forward and reverse diffusion processes from Ho et al. (2020).

    Ho et al (2020) showed that the loss function ||ε - ε_θ(x_t, t)||² works
    well as the full variational bound but is much simpler to implement.

    Args:
        model (torch.nn.Module): The neural network model used for denoising.
        scheduler (DDPMScheduler): Scheduler for managing the diffusion process.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:


    """
    def __init__(self, model:torch.nn.Module, scheduler: DDPMScheduler, device: torch.device):
        self.model = model
        self.scheduler = scheduler
        self.device = device

    def train_loss(self, x_0):
        """
        Comput training loss from Algorith 1 in Ho (2020).

        Algorithm 1 (Training):
        1. Sample t ~ Uniform(1, T)
        2. Sample ε ~ N(0, I)
        3. Take gradient step on ∇_θ ||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||²

        This is the simplified loss (Eq. 14) rather than the full variational bound
        """
        # FIXME: debugging
        print(f"DDPM train_loss input shape: {x_0.shape}")

        batch_size = x_0.shape[0]

        # Step 1: Sample t uniformly
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device).long()

        # Step 2: Sample noise ε
        noise = torch.randn_like(x_0)

        # Step 3: Forward process - get x_t
        x_t = self.scheduler.q_sample(x_0, t, noise=noise)

        # Step 4: Predict noise using the model
        predicted_noise = self.model(x_t, t)

        # Step 5: MSE loss between true and predicted noise
        loss = F.mse_loss(noise, predicted_noise)

        return loss

    @torch.no_grad()
    def sample(self, shape:tuple, return_all_timesteps:bool=False):
        """
        Sampling procedure from Algorithm 2 in Ho et al.

        Algorithm 2 (Sampling):
        1. x_T ~ N(0, I)
        2. for t = T, ..., 1:
           x_{t-1} = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t)) + σ_t * z
           where z ~ N(0, I) if t > 1, else z = 0

        Args:
            shape (tuple): Shape of the samples to generate (batch_size, channels, height, width).
            return_all_timesteps (bool): If True, returns all intermediate steps.

        Returns:
            torch.Tensor: Generated samples.
        """
        device = self.device
        b = shape[0]

        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []

        # Reverse process
        for i in reversed(range(0, self.scheduler.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.model(img, t)

            # Compute x_{t-1} using the reverse process formula
            alpha_t = self.scheduler.alphas[i]
            alpha_cumprod_t = self.scheduler.alphas_cumprod[i]
            beta_t = self.scheduler.betas[i]

            # Mean of reverse process
            img = (1 / torch.sqrt(alpha_t)) * (
                    img - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            )

            # Add noise for t > 0 (except for last step)
            if i > 0:
                noise = torch.randn_like(img)
                # use the posterior variance
                sigma_t = torch.sqrt(self.scheduler.posterior_variance[i])
                img += sigma_t * noise

            if return_all_timesteps:
                imgs.append(img.cpu())

        if return_all_timesteps:
            return imgs
        return img