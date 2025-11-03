import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import deepchem as dc
from deepchem.models.torch_models import TorchModel

from edm_3d.core.egnn import EGNN
from edm_3d.core.diffusion_process import DiffusionProcess


class EDM(TorchModel):
    """
    E(3) Equivariant Diffusion Model for 3D Molecule Generation

    Wraps EGNN + Diffusion as a DeepChem model
    """

    def __init__(
            self,
            num_atom_types: int = 5,  # For QM9: H, C, N, O, F
            hidden_dim: int = 256,
            num_layers: int = 9,
            num_diffusion_steps: int = 1000,
            **kwargs
    ):
        # Initialize model components
        self.num_atom_types = num_atom_types
        self.hidden_dim = hidden_dim

        # Time embedding (for conditioning on diffusion timestep)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # EGNN backbone
        self.egnn = EGNN(
            in_node_dim=num_atom_types + hidden_dim,  # Atom types + time embedding
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        # Diffusion process
        self.diffusion = DiffusionProcess(num_steps=num_diffusion_steps)

        # Initialize as TorchModel
        super().__init__(self.egnn, **kwargs)

    def forward(
            self,
            h: torch.Tensor,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict noise for denoising

        Args:
            h: Noisy atom type features [N, num_atom_types]
            x: Noisy coordinates [N, 3]
            edge_index: Graph connectivity [2, E]
            t: Timestep [batch_size]

        Returns:
            Predicted noise for h and x
        """
        # Time embedding
        t_emb = self.time_embedding(t.float().view(-1, 1))  # [batch, hidden_dim]

        # Broadcast time embedding to all nodes
        # (Assuming batch processing - adapt as needed)
        t_emb_expanded = t_emb.repeat_interleave(h.shape[0] // len(t), dim=0)

        # Concatenate time embedding with node features
        h_input = torch.cat([h, t_emb_expanded], dim=-1)

        # Predict noise
        noise_h_pred, noise_x_pred = self.egnn(h_input, x, edge_index)

        return noise_h_pred, noise_x_pred

    def loss_func(
            self,
            outputs: Tuple[torch.Tensor, torch.Tensor],
            labels: Tuple[torch.Tensor, torch.Tensor],
            weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Diffusion loss (MSE between predicted and actual noise)
        """
        pred_noise_h, pred_noise_x = outputs
        true_noise_h, true_noise_x = labels

        # L2 loss on noise prediction
        loss_h = torch.mean((pred_noise_h - true_noise_h) ** 2)
        loss_x = torch.mean((pred_noise_x - true_noise_x) ** 2)

        # Combined loss (can weight differently)
        total_loss = loss_h + loss_x

        return total_loss

    @torch.no_grad()
    def generate(
            self,
            num_atoms: int,
            num_molecules: int = 1,
            device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate molecules from noise

        Args:
            num_atoms: Number of atoms per molecule
            num_molecules: Number of molecules to generate
            device: Device to run on

        Returns:
            Generated coordinates and atom types
        """
        self.eval()

        # Start from pure noise
        xt = torch.randn(num_molecules, num_atoms, 3).to(device)
        ht = torch.randn(num_molecules, num_atoms, self.num_atom_types).to(device)

        # Center coordinates
        xt = xt - xt.mean(dim=1, keepdim=True)

        # Create fully connected graph
        edge_index = self._get_fully_connected_edges(num_atoms, num_molecules).to(device)

        # Reverse diffusion
        for t in reversed(range(self.diffusion.num_steps)):
            t_tensor = torch.tensor([t] * num_molecules).to(device)

            # Predict noise
            pred_noise_h, pred_noise_x = self.forward(
                ht.view(-1, self.num_atom_types),
                xt.view(-1, 3),
                edge_index,
                t_tensor
            )

            # Reshape predictions
            pred_noise_h = pred_noise_h.view(num_molecules, num_atoms, -1)
            pred_noise_x = pred_noise_x.view(num_molecules, num_atoms, 3)

            # Denoise
            xt, ht = self.diffusion.reverse_diffusion_step(
                xt, ht, pred_noise_x, pred_noise_h, t
            )

        # Convert atom types from continuous to discrete
        atom_types = torch.argmax(ht, dim=-1)

        return xt, atom_types

    def _get_fully_connected_edges(
            self,
            num_atoms: int,
            num_molecules: int
    ) -> torch.Tensor:
        """
        Create fully connected graph (excluding self-loops)
        """
        # Create edges for one molecule
        rows, cols = torch.meshgrid(
            torch.arange(num_atoms),
            torch.arange(num_atoms),
            indexing='ij'
        )
        mask = rows != cols  # Exclude self-loops
        edge_index_single = torch.stack([rows[mask], cols[mask]], dim=0)

        # Repeat for all molecules in batch
        edge_indices = []
        for i in range(num_molecules):
            offset = i * num_atoms
            edge_indices.append(edge_index_single + offset)

        return torch.cat(edge_indices, dim=1)

