import torch
import torch.nn.functional as F

class DDPMScheduler:
    """
    Implements the noise scheduling from Ho (2020).
    This handles the forward process q(x_t | x_{t-1})
    and related computations. The forward process is responsible
    for the noise addition in diffusion models.

    The idea is to start with a clean image and iteratively add noise
    Therefore, beta_start < beta_end, where beta controls the noise level.

    Args:
        num_timesteps (int): Number of diffusion steps.
        beta_start (float): Initial beta value. beta controls the noise level. Higher values mean more noise.
        beta_end (float): Final beta value. x
    """
    def __init__(self, num_timesteps:int=1000, beta_start:float=0.0001, beta_end:float=0.02):
        self.num_timesteps = num_timesteps

        # Linear beta schedule from Ho (2020)
        # beta_t linearly increases from beta_start to beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # Pre-compute alphas. This is part of the reparametrization trick in
        # the forward process.
        # alpha_t = 1 - beta_t
        self.alphas = 1.0 - self.betas

        # ᾱ_t = ∏(s=1 to t) α_s  (cumulative product)
        # This is key for the reparameterization trick in Eq. 4
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # ᾱ_{t-1} for reverse process computations
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Pre-compute terms for q(x_{t-1} | x_t, x_0) - Eq. 7 in paper
        # This is the "true" reverse process when x_0 is known
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev)/(1.0 - self.alphas_cumprod)

        # For the reparameterization trick: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For predicting x_0 from x_t and predicted noise
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    def q_sample(self, x_start:torch.Tensor, t:torch.Tensor, noise:torch.Tensor=None) -> torch.Tensor:
        """
        Sample from the forward process q(x_t | x_0).
        Uses the reparameterisation trick from Eq. 4:
        x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε, where ε ~ N(0,I).

        This allows us to sample any timestep directly without
        iterating.

        Ho (2020) -> Algorithm 2, Eq. 4

        Args:
            x_start (torch.Tensor): The original image tensor.
            t (torch.Tensor): The time step tensor.
            noise (torch.Tensor, optional): Optional noise to add. If None, random noise is used.

        Returns:
            torch.Tensor: Sampled noisy image at time t.
        """
        device = x_start.device
        t = t.to(self.sqrt_alphas_cumprod.device)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1).to(device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1).to(device)

        if noise is None:
            noise = torch.randn_like(x_start)
        noise = noise.to(device)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t:torch.Tensor, t:torch.Tensor, noise:torch.Tensor) -> torch.Tensor:
        """
        Predict the original image x_0 and the predicted noise ε_θ
        From reparameterization: x_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t

        Ho (2020) -> Algorithm 2, Eq. 6

        Args:
            x_t (torch.Tensor): Noisy image tensor at time t.
            t (torch.Tensor): Time step tensor.
            noise (torch.Tensor): Predicted noise tensor.

        Returns:
            torch.Tensor: Predicted original image x_0.
        """
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].reshape(-1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].reshape(-1, 1)

        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def q_posterior_mean_variance(self, x_start:torch.Tensor, x_t:torch.Tensor, t:torch.Tensor) -> tuple:
        """
        Compute mean and variance of q(x_{t-1} | x_t, x_0)
        This is the "true" reverse step when x_0 is known (Eq. 7)

        Ho (2020) -> Algorithm 2, Eq. 7

        Args:
            x_start (torch.Tensor): Original image tensor.
            x_t (torch.Tensor): Noisy image tensor at time t.
            t (torch.Tensor): Time step tensor.

        Returns:
            tuple: Posterior mean and variance tensors.
        """
        posterior_mean = (
            self.betas[t].reshape(-1, 1) * torch.sqrt(self.alphas_cumprod_prev[t]).reshape(-1, 1) / (1.0 - self.alphas_cumprod[t]).reshape(-1, 1) * x_start
            + torch.sqrt(self.alphas[t]).reshape(-1, 1) * (1.0 - self.alphas_cumprod_prev[t]).reshape(-1, 1) / (1.0 - self.alphas_cumprod[t]).reshape(-1, 1) * x_t
        )

        posterior_variance = self.posterior_variance[t].reshape(-1, 1)

        return posterior_mean, posterior_variance