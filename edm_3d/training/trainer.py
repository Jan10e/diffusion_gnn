import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


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

        # Device - use model's device
        self.device = model.device

        # Optimizer - use model's optimizer
        self.optimizer = model.optimizer

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        logger.info(f"Initialized EDMTrainer with {len(train_data)} train samples")

    def train(self):
        """
        Main training loop
        """
        # DataLoaders
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        val_loader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=0
        )

        logger.info(f"Starting training for {self.num_epochs} epochs")

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

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch)

        # Save final model
        self.model.save(self.save_dir / 'edm_final.pth')

        # Save history
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        logger.info("Training complete!")
        return self.history

    def train_epoch(self, dataloader):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            try:
                # Move to device
                coords = batch['coords'].to(self.device)
                atom_types = batch['atom_types'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                batch_indices = batch['batch'].to(self.device)

                # Compute loss using model's method
                loss = self.model.compute_loss(
                    coords=coords,
                    atom_types=atom_types,
                    edge_index=edge_index,
                    batch_indices=batch_indices
                )

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            except Exception as e:
                logger.error(f"Error in training batch: {e}")
                continue

        return total_loss / max(num_batches, 1)

    def validate(self, dataloader):
        """
        Validate model
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validating")
            for batch in pbar:
                try:
                    coords = batch['coords'].to(self.device)
                    atom_types = batch['atom_types'].to(self.device)
                    edge_index = batch['edge_index'].to(self.device)
                    batch_indices = batch['batch'].to(self.device)

                    loss = self.model.compute_loss(
                        coords=coords,
                        atom_types=atom_types,
                        edge_index=edge_index,
                        batch_indices=batch_indices
                    )

                    total_loss += loss.item()
                    num_batches += 1

                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue

        return total_loss / max(num_batches, 1)

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
                pos=item['coords'],
                num_nodes=item['num_atoms']
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
        print(f"  → Checkpoint saved: {checkpoint_path.name}")