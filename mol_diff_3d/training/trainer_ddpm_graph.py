"""
Trainer for Denoising Diffusion Probabilistic Models (DDPM) applied to molecular graphs
This is updated to handle the two-part loss function for both discrete features and continuous 3D positions.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

from ..models.diffusion import MolecularDiffusionModel
from ..samplers.diffusion_samplers import DDPMQSampler


class DDPMTrainer:
    """
    Trains a MolecularDiffusionModel using the DDPM algorithm.
    It orchestrates the training loop, loss calculation, and optimization.
    """

    def __init__(self, model, q_sampler, optimizer, device, config=None):
        self.model = model.to(device)
        self.q_sampler = q_sampler
        self.optimizer = optimizer
        self.device = device
        self.config = config or {}

        self.log_interval = self.config.get('log_interval', 20)
        self.losses = []
        self.epoch = 0

    def compute_loss(self, batch):
        """
        Calculates the combined loss for features and positions.
        """
        batch_size = batch.batch.max().item() + 1

        t = torch.randint(0, self.q_sampler.num_timesteps, (batch_size,), device=self.device)

        # 1. Add noise to atom features and positions
        noise_x = torch.randn_like(batch.x)
        x_noisy = self.q_sampler.q_sample_step(batch.x, t[batch.batch], noise_x)

        noise_pos = torch.randn_like(batch.pos)
        pos_noisy = self.q_sampler.q_sample_pos_step(batch.pos, t[batch.batch], noise_pos)

        # 2. Predict noise using the model
        noise_pred_x, noise_pred_pos = self.model(x_noisy, batch.edge_index, pos_noisy, batch.batch, t)

        # 3. Calculate losses
        loss_x = F.mse_loss(noise_pred_x, noise_x)
        loss_pos = F.mse_loss(noise_pred_pos, noise_pos)

        total_loss = loss_x + loss_pos
        return total_loss

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)

            loss = self.compute_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, dataloader, num_epochs=None):
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader)
            self.losses.append(avg_loss)
            self.epoch = epoch

            if epoch % self.log_interval == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")

        print("Training completed!")
        return self.losses