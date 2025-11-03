import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json

from edm_3d.data.qm9_loader import prepare_qm9_data

def train_edm():
    """
    Train EDM on QM9 dataset
    """
    # Prepare data
    train_dataset, valid_dataset, test_dataset = prepare_qm9_data()

    # Initialize model
    model = EDM(
        num_atom_types=5,  # H, C, N, O, F
        hidden_dim=256,
        num_layers=9,
        num_diffusion_steps=1000,
        learning_rate=1e-4,
        batch_size=64
    )

    print("Starting training...")

    # Train model
    model.fit(
        train_dataset,
        nb_epoch=3000
    )

    return model


class EDMTrainer:
    """
    Trainer for EDM model
    """

    def __init__(
            self,
            model,
            train_data,
            val_data,
            num_epochs: int = 10,
            batch_size: int = 64,
            learning_rate: float = 1e-4,
            save_dir: str = './checkpoints',
            device: str = None
    ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def train(self):
        """
        Main training loop
        """
        # DataLoaders
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        val_loader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)

        # Save final model
        torch.save(self.model.state_dict(), self.save_dir / 'edm_final.pth')

        # Save history
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def train_epoch(self, dataloader):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc="Training"):
            # Move to device
            coords = batch['coords'].to(self.device)
            atom_types = batch['atom_types'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            batch_indices = batch['batch'].to(self.device)

            # Random timestep
            batch_size = batch_indices.max().item() + 1
            t = torch.randint(0, self.model.diffusion.num_steps, (batch_size,), device=self.device)

            # Forward diffusion (add noise)
            noisy_coords, noisy_atoms, noise_coords, noise_atoms = self.model.diffusion.forward_diffusion(
                coords, atom_types, t[batch_indices]
            )

            # Predict noise
            pred_noise_atoms, pred_noise_coords = self.model(
                noisy_atoms, noisy_coords, edge_index, t[batch_indices]
            )

            # Compute loss
            loss = self.model.loss_func(
                (pred_noise_atoms, pred_noise_coords),
                (noise_atoms, noise_coords)
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        """
        Validate model
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                coords = batch['coords'].to(self.device)
                atom_types = batch['atom_types'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                batch_indices = batch['batch'].to(self.device)

                batch_size = batch_indices.max().item() + 1
                t = torch.randint(0, self.model.diffusion.num_steps, (batch_size,), device=self.device)

                noisy_coords, noisy_atoms, noise_coords, noise_atoms = self.model.diffusion.forward_diffusion(
                    coords, atom_types, t[batch_indices]
                )

                pred_noise_atoms, pred_noise_coords = self.model(
                    noisy_atoms, noisy_coords, edge_index, t[batch_indices]
                )

                loss = self.model.loss_func(
                    (pred_noise_atoms, pred_noise_coords),
                    (noise_atoms, noise_coords)
                )

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def _collate_fn(self, batch):
        """
        Collate function for DataLoader
        Converts list of molecules to batched graph
        """
        from torch_geometric.data import Batch, Data

        data_list = []
        for item in batch:
            data = Data(
                x=item['atom_types'],
                pos=item['coords']
            )
            data_list.append(data)

        # Batch graphs
        batched = Batch.from_data_list(data_list)

        # Create fully connected edges within each graph
        edge_indices = []
        offset = 0
        for item in batch:
            num_atoms = item['num_atoms']
            # Fully connected (excluding self-loops)
            rows, cols = torch.meshgrid(
                torch.arange(num_atoms),
                torch.arange(num_atoms),
                indexing='ij'
            )
            mask = rows != cols
            edges = torch.stack([rows[mask], cols[mask]], dim=0) + offset
            edge_indices.append(edges)
            offset += num_atoms

        edge_index = torch.cat(edge_indices, dim=1)

        return {
            'coords': batched.pos,
            'atom_types': batched.x,
            'edge_index': edge_index,
            'batch': batched.batch
        }

    def _save_checkpoint(self, epoch):
        """Save training checkpoint"""
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

