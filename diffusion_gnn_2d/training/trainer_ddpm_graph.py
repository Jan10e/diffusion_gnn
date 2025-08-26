import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from diffusion_gnn_2d.models.diffusion import MolecularDiffusionModel
from diffusion_gnn_2d.models.noise_scheduler import NoiseScheduler


class DDPMTrainer:
    """
    Simple trainer - minimal changes from your original.

    Key differences:
    - model + scheduler instead of monolithic ddpm
    - compute_loss() replaces ddmp.train_loss()
    - That's it!
    """

    def __init__(self, model, scheduler, optimizer, device, config=None):
        self.model = model.to(device)
        self.scheduler = scheduler.to(device)
        self.optimizer = optimizer
        self.device = device
        self.config = config or {}

        # Same as your original
        self.log_interval = config.get('log_interval', 20)
        self.vis_interval = config.get('vis_interval', 20)
        self.losses = []
        self.epoch = 0

    def compute_loss(self, batch):
        """Replaces the old self.ddpm.train_loss(data) call."""
        batch_size = batch.batch.max().item() + 1

        # Sample timesteps and noise
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device)
        noise = torch.randn_like(batch.x)

        # Add noise and predict
        x_noisy = self.scheduler.q_sample_step(batch.x, t[batch.batch], noise)
        noise_pred = self.model(x_noisy, batch.edge_index, batch.edge_attr, batch.batch, t)

        return F.mse_loss(noise_pred, noise)

    def train_epoch(self, dataloader):
        """Almost identical to your original."""
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            batch = batch.to(self.device)

            # Compute loss (this is the main change!)
            loss = self.compute_loss(batch)

            # Same backward pass as before
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, dataloader, num_epochs=None):
        """Identical to your original."""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader)
            self.losses.append(avg_loss)
            self.epoch = epoch

            # Same logging as before
            if epoch % self.log_interval == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")

            # Same visualization as before
            if epoch % self.vis_interval == 0:
                # You'll need to update this part for molecular graphs
                pass

        print("Training completed!")
        return self.losses


# Simple helper function
def create_trainer(model, scheduler, lr=1e-4):
    """Create trainer with default optimizer."""
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=lr)
    return DDPMTrainer(model, scheduler, optimizer, torch.device('cuda'))