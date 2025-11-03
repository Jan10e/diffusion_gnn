# ============================================================================
# edm_3d/models/edm.py (CORRECTED)
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
import logging

from edm_3d.core.egnn import EGNN
from edm_3d.core.diffusion_process import DiffusionProcess

logger = logging.getLogger(__name__)


class EDMCore(nn.Module):
    """
    Core EDM model (just the neural network part)
    This is a standard PyTorch nn.Module
    """

    def __init__(
            self,
            num_atom_types: int = 5,
            hidden_dim: int = 128,
            num_layers: int = 4,
            num_diffusion_steps: int = 100
    ):
        super().__init__()

        self.num_atom_types = num_atom_types
        self.hidden_dim = hidden_dim
        self.num_diffusion_steps = num_diffusion_steps

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # EGNN backbone
        self.egnn = EGNN(
            in_node_dim=num_atom_types + hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        # Diffusion process (not trainable, just utilities)
        self.diffusion = DiffusionProcess(num_steps=num_diffusion_steps)

        logger.info(f"Initialized EDMCore with {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(
            self,
            h: torch.Tensor,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: predict noise

        Args:
            h: Noisy atom features [N, num_atom_types]
            x: Noisy coordinates [N, 3]
            edge_index: Graph connectivity [2, E]
            t: Timesteps [batch_size]

        Returns:
            Predicted noise for h and x
        """
        # Time embedding
        t_emb = self.time_embedding(t.float().view(-1, 1))

        # Broadcast to all nodes
        # Assumes nodes are grouped by molecule in batch
        num_nodes = h.shape[0]
        batch_size = len(t)
        nodes_per_mol = num_nodes // batch_size

        t_emb_expanded = t_emb.repeat_interleave(nodes_per_mol, dim=0)

        # Concatenate time with node features
        h_input = torch.cat([h, t_emb_expanded], dim=-1)

        # Predict noise
        noise_h_pred, noise_x_pred = self.egnn(h_input, x, edge_index)

        return noise_h_pred, noise_x_pred


class EDM:
    """
    E(3) Equivariant Diffusion Model - Standalone wrapper

    This is a simpler wrapper that doesn't inherit from DeepChem's TorchModel
    to avoid compatibility issues. Can be used directly with PyTorch.
    """

    def __init__(
            self,
            num_atom_types: int = 5,
            hidden_dim: int = 128,
            num_layers: int = 4,
            num_diffusion_steps: int = 100,
            learning_rate: float = 1e-4,
            device: str = None
    ):
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Create core model
        self.model = EDMCore(
            num_atom_types=num_atom_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_diffusion_steps=num_diffusion_steps
        ).to(self.device)

        # Store config
        self.num_atom_types = num_atom_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_diffusion_steps = num_diffusion_steps
        self.learning_rate = learning_rate

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        logger.info(f"Initialized EDM on device: {self.device}")

    def forward(self, *args, **kwargs):
        """Forward pass through model"""
        return self.model(*args, **kwargs)

    def train(self):
        """Set model to training mode"""
        self.model.train()

    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()

    def parameters(self):
        """Get model parameters"""
        return self.model.parameters()

    def state_dict(self):
        """Get model state dict"""
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        """Load model state dict"""
        self.model.load_state_dict(state_dict)

    def to(self, device):
        """Move model to device"""
        self.device = device
        self.model = self.model.to(device)
        return self

    def compute_loss(
            self,
            coords: torch.Tensor,
            atom_types: torch.Tensor,
            edge_index: torch.Tensor,
            batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diffusion loss for a batch

        Args:
            coords: Node coordinates [N, 3]
            atom_types: Atom type indices [N]
            edge_index: Edge connectivity [2, E]
            batch_indices: Batch assignment for each node [N]

        Returns:
            Loss value
        """
        batch_size = batch_indices.max().item() + 1

        # Random timestep for each molecule in batch
        t = torch.randint(
            0,
            self.num_diffusion_steps,
            (batch_size,),
            device=self.device
        )

        # Convert atom types to one-hot
        atom_types_onehot = torch.zeros(
            len(atom_types),
            self.num_atom_types,
            device=self.device
        )
        atom_types_onehot.scatter_(1, atom_types.unsqueeze(1), 1.0)

        # Forward diffusion (add noise)
        noisy_coords, noisy_atoms, noise_coords, noise_atoms = \
            self.model.diffusion.forward_diffusion(
                coords,
                atom_types_onehot,
                t[batch_indices]
            )

        # Predict noise
        pred_noise_atoms, pred_noise_coords = self.model(
            noisy_atoms,
            noisy_coords,
            edge_index,
            t[batch_indices]
        )

        # Compute loss (MSE)
        loss_atoms = torch.mean((pred_noise_atoms - noise_atoms) ** 2)
        loss_coords = torch.mean((pred_noise_coords - noise_coords) ** 2)

        total_loss = loss_atoms + loss_coords

        return total_loss

    @torch.no_grad()
    def generate(
            self,
            num_atoms: int,
            num_molecules: int = 1,
            device: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate molecules from noise

        Args:
            num_atoms: Number of atoms per molecule
            num_molecules: Number of molecules to generate
            device: Device to use (default: self.device)

        Returns:
            coords [num_molecules, num_atoms, 3]
            atom_types [num_molecules, num_atoms]
        """
        self.eval()

        if device is None:
            device = self.device

        # Start from noise
        xt = torch.randn(num_molecules, num_atoms, 3).to(device)
        ht = torch.randn(num_molecules, num_atoms, self.num_atom_types).to(device)

        # Center coordinates
        xt = xt - xt.mean(dim=1, keepdim=True)

        # Create fully connected edges
        edge_index = self._get_fully_connected_edges(num_atoms, num_molecules).to(device)

        # Reverse diffusion
        for t in reversed(range(self.num_diffusion_steps)):
            if t % 20 == 0:
                logger.debug(f"Generation step {self.num_diffusion_steps - t}/{self.num_diffusion_steps}")

            t_tensor = torch.tensor([t] * num_molecules).to(device)

            # Predict noise
            pred_noise_h, pred_noise_x = self.model(
                ht.view(-1, self.num_atom_types),
                xt.view(-1, 3),
                edge_index,
                t_tensor
            )

            # Reshape
            pred_noise_h = pred_noise_h.view(num_molecules, num_atoms, -1)
            pred_noise_x = pred_noise_x.view(num_molecules, num_atoms, 3)

            # Denoise
            xt, ht = self.model.diffusion.reverse_diffusion_step(
                xt, ht, pred_noise_x, pred_noise_h, t
            )

        # Convert to discrete atom types
        atom_types = torch.argmax(ht, dim=-1)

        return xt, atom_types

    def _get_fully_connected_edges(
            self,
            num_atoms: int,
            num_molecules: int
    ) -> torch.Tensor:
        """Create fully connected graph edges"""
        rows, cols = torch.meshgrid(
            torch.arange(num_atoms),
            torch.arange(num_atoms),
            indexing='ij'
        )
        mask = rows != cols
        edge_index_single = torch.stack([rows[mask], cols[mask]], dim=0)

        edge_indices = []
        for i in range(num_molecules):
            offset = i * num_atoms
            edge_indices.append(edge_index_single + offset)

        return torch.cat(edge_indices, dim=1)

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'num_atom_types': self.num_atom_types,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'num_diffusion_steps': self.num_diffusion_steps,
                'learning_rate': self.learning_rate
            }
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")

    def __repr__(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"EDM(\n"
            f"  num_atom_types={self.num_atom_types},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_diffusion_steps={self.num_diffusion_steps},\n"
            f"  total_params={total_params:,},\n"
            f"  trainable_params={trainable_params:,},\n"
            f"  device={self.device}\n"
            f")"
        )