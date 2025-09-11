"""
Improved trainer for MolDiff with joint atom-bond-position loss.
Addresses the atom-bond inconsistency problem with proper categorical diffusion.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from typing import List, Dict, Tuple

from mol_diff_3d.models.categorical_diffusion import CategoricalDiffusion, CategoricalNoiseScheduler
from mol_diff_3d.sampling.samplers import ImprovedDDPMQSampler


class ImprovedDDPMTrainer:
    """
    Trainer for joint atom-bond-position diffusion model.
    Key improvements:
    1. Joint loss for atoms, bonds, and positions
    2. Proper categorical diffusion for discrete features
    3. Different noise schedules for atoms vs bonds
    4. Bond predictor guidance loss
    """

    def __init__(self, model, bond_predictor, q_sampler, categorical_diffusion,
                 optimizer, device, config=None):
        self.model = model.to(device)
        self.bond_predictor = bond_predictor.to(device) if bond_predictor else None
        self.q_sampler = q_sampler
        self.categorical_diffusion = categorical_diffusion
        self.optimizer = optimizer
        self.device = device
        self.config = config or {}
        self.scalar = torch.cuda.amp.GradScaler(enabled=True)

        # Loss weights
        self.atom_loss_weight = self.config.get('atom_loss_weight', 1.0)
        self.pos_loss_weight = self.config.get('pos_loss_weight', 1.0)
        self.bond_loss_weight = self.config.get('bond_loss_weight', 1.0)
        self.guidance_loss_weight = self.config.get('guidance_loss_weight', 0.1)

        self.log_interval = self.config.get('log_interval', 20)
        self.losses = []
        self.detailed_losses = {'atom': [], 'pos': [], 'bond': [], 'guidance': []}
        self.epoch = 0

    def compute_joint_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute joint loss for atoms, bonds, and positions.
        """
        batch_size = batch.batch.max().item() + 1
        device = self.device

        # Sample different timesteps for atoms vs bonds (key MolDiff innovation)
        t_atoms = torch.randint(0, self.q_sampler.num_timesteps, (batch_size,), device=device)
        # Bonds get diffused earlier (faster schedule)
        t_bonds = torch.randint(0, self.q_sampler.num_timesteps // 2, (batch_size,), device=device)

        # Extend timesteps to node/edge level
        t_nodes = t_atoms[batch.batch]
        t_edges = t_bonds[batch.batch[batch.edge_index[0]]]  # Use source node's batch

        # --- Forward Diffusion ---

        # 1. Add noise to positions (continuous)
        noise_pos = torch.randn_like(batch.pos)
        pos_noisy = self.q_sampler.q_sample_pos_step(batch.pos, t_nodes, noise_pos)

        # 2. Add noise to atom types (categorical diffusion)
        x_noisy = self.categorical_diffusion.q_sample_atoms(batch.x, t_nodes)

        # 3. Add noise to bond types (categorical diffusion, faster schedule)
        edge_attr_noisy = self.categorical_diffusion.q_sample_bonds(batch.edge_attr, t_edges)

        # --- Model Prediction ---
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            atom_logits, pos_noise_pred, bond_logits = self.model(
                x_noisy, batch.edge_index, edge_attr_noisy, pos_noisy, batch.batch, t_atoms
            )

        # --- Loss Computation ---

        # 1. Position loss (MSE on noise)
        loss_pos = F.mse_loss(pos_noise_pred, noise_pos)

        # 2. Atom type loss (categorical)
        loss_atoms = self.categorical_diffusion.compute_atom_loss(
            atom_logits, batch.x, x_noisy, t_nodes
        )

        # 3. Bond type loss (categorical)
        loss_bonds = self.categorical_diffusion.compute_bond_loss(
            bond_logits, batch.edge_attr, edge_attr_noisy, t_edges
        )

        # 4. Bond predictor guidance loss (optional)
        loss_guidance = torch.tensor(0.0, device=device)
        if self.bond_predictor is not None:

            # Predict bonds from current atom types and positions
            predicted_bonds = self.bond_predictor(batch.x, batch.pos, batch.edge_index)
            loss_guidance = F.cross_entropy(
                predicted_bonds,
                batch.edge_attr.argmax(dim=-1)
            )

        # Combine losses
        total_loss = (
            self.atom_loss_weight * loss_atoms +
            self.pos_loss_weight * loss_pos +
            self.bond_loss_weight * loss_bonds +
            self.guidance_loss_weight * loss_guidance
        )

        # Return losses for logging
        loss_dict = {
            'atom': loss_atoms.item(),
            'pos': loss_pos.item(),
            'bond': loss_bonds.item(),
            'guidance': loss_guidance.item()
        }

        return total_loss, loss_dict

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with detailed loss tracking."""
        self.model.train()
        if self.bond_predictor:
            self.bond_predictor.train()

        total_losses = {'total': 0, 'atom': 0, 'pos': 0, 'bond': 0, 'guidance': 0}
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {self.epoch}"):
            batch = batch.to(self.device)

            # Compute joint loss
            total_loss, loss_dict = self.compute_joint_loss(batch)

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping for stability
            orig_grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=50.0)  # MolDiff article uses 50!
            if self.bond_predictor:
                clip_grad_norm_(self.bond_predictor.parameters(), max_norm=50.0)

            self.optimizer.step()

            # Accumulate losses
            total_losses['total'] += total_loss.item()
            for key, value in loss_dict.items():
                total_losses[key] += value
            num_batches += 1

        # Average losses over epoch
        for key in total_losses:
            total_losses[key] /= num_batches

        # Store losses for tracking
        self.losses.append(total_losses['total'])
        for key in self.detailed_losses:
            self.detailed_losses[key].append(total_losses[key])

        return total_losses

    def train(self, dataloader: DataLoader, num_epochs: int):
        """Main training loop with detailed logging."""
        print(f"Starting joint atom-bond-position training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.bond_predictor:
            print(f"Bond predictor parameters: {sum(p.numel() for p in self.bond_predictor.parameters()):,}")

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_losses = self.train_epoch(dataloader)

            if epoch % self.log_interval == 0:
                print(f"Epoch [{epoch}/{num_epochs}]:")
                print(f"  Total Loss: {epoch_losses['total']:.4f}")
                print(f"  Atom Loss: {epoch_losses['atom']:.4f}")
                print(f"  Position Loss: {epoch_losses['pos']:.4f}")
                print(f"  Bond Loss: {epoch_losses['bond']:.4f}")
                if self.bond_predictor:
                    print(f"  Guidance Loss: {epoch_losses['guidance']:.4f}")

        print("Joint training completed!")
        return self.losses, self.detailed_losses