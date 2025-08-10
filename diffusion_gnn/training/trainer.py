import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusion_gnn.evaluation.visualization import visualize_training_progress, visualize_training_progress

class DDPMTrainer:
    """
    Trainer for Denoising Diffusion Probabilistic Model (DDPM).

    Args:
        ddpm (DDPM): The DDPM model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training the DDPM.
        device (torch.device): Device to run the training on (CPU or GPU).
        config (dict, optional): Configuration parameters for training.

    """
    def __init__(self, ddpm, optimizer, device, config=None):
        self.ddpm = ddpm
        self.optimizer = optimizer
        self.device = device
        self.config = config or {}

        # Training settings
        self.log_interval = config.get('log_interval', 20)  # How often to log training progress
        self.vis_interval = config.get('vis_interval', 20)  # How often to visualize training progress

        # Tracking
        self.losses = []
        self.epoch = 0

    def train_epoch(self, dataloader):
        """Train for one epoch on the given dataloader."""
        self.ddpm.model.train()
        total_loss = 0

        for batch_idx, data in enumerate(dataloader):
            # Unpack data from TensorDataset
            if isinstance(data, (list, tuple)):
                data = data[0]  # TensorDataset returns (tensor,)

            # FIXME: debugging
            print(f"Batch {batch_idx} shape: {data.shape}")
            if batch_idx == 0:  # Just check first batch
                break


            data = data.to(self.device)

            # Compute loss
            loss = self.ddpm.train_loss(data)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, dataloader, num_epochs=None):
        """Full training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.ddpm.model.parameters()):,}")

        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader)
            self.losses.append(avg_loss)
            self.epoch = epoch

            # Logging
            if epoch % self.log_interval == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")

            # Visualization
            if epoch % self.vis_interval == 0:
                seq_len = dataloader.dataset.tensors[0].shape[-1]  # Infer seq_len
                visualize_training_progress(self.ddpm, self.device, epoch, seq_len)

        print("Training completed!")
        return self.losses
