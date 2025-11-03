# ============================================================================
# edm_3d/models/edm.py
# ============================================================================

import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional

from edm_3d.core.egnn import EGNN
from edm_3d.core.diffusion_process import DiffusionProcess

logger = logging.getLogger(__name__)


class EDMCore(nn.Module):
    """Core EDM model"""

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

        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.egnn = EGNN(
            in_node_dim=num_atom_types + hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            out_node_dim=num_atom_types
        )

        self.diffusion = DiffusionProcess(num_steps=num_diffusion_steps)

        logger.info(f"Initialized EDMCore with {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(
            self,
            h: torch.Tensor,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_emb = self.time_embedding(t.float().view(-1, 1))

        num_nodes = h.shape[0]
        batch_size = len(t)
        nodes_per_mol = num_nodes // batch_size
        t_emb_expanded = t_emb.repeat_interleave(nodes_per_mol, dim=0)

        h_input = torch.cat([h, t_emb_expanded], dim=-1)

        h_out, x_out = self.egnn(h_input, x, edge_index)

        return h_out, x_out


class EDM:
    """E(3) Equivariant Diffusion Model"""

    def __init__(
            self,
            num_atom_types: int = 5,
            hidden_dim: int = 128,
            num_layers: int = 4,
            num_diffusion_steps: int = 100,
            learning_rate: float = 1e-4,
            device: str = None
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = EDMCore(
            num_atom_types=num_atom_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_diffusion_steps=num_diffusion_steps
        ).to(self.device)

        self.num_atom_types = num_atom_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_diffusion_steps = num_diffusion_steps
        self.learning_rate = learning_rate

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-12
        )

        logger.info(f"Initialized EDM on device: {self.device}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def to(self, device):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = self.model.to(self.device)
        return self

    def compute_loss(
            self,
            coords: torch.Tensor,
            atom_types: torch.Tensor,
            edge_index: torch.Tensor,
            batch_indices: torch.Tensor
    ) -> torch.Tensor:
        coords = coords.to(self.device)
        atom_types = atom_types.to(self.device)
        edge_index = edge_index.to(self.device)
        batch_indices = batch_indices.to(self.device)

        batch_size = batch_indices.max().item() + 1

        t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device)

        atom_types_onehot = torch.zeros(len(atom_types), self.num_atom_types, device=self.device)
        atom_types_onehot.scatter_(1, atom_types.unsqueeze(1), 1.0)

        noisy_coords, noisy_atoms, noise_coords, noise_atoms = \
            self.model.diffusion.forward_diffusion(coords, atom_types_onehot, t[batch_indices])

        pred_noise_atoms, pred_noise_coords = self.model(
            noisy_atoms, noisy_coords, edge_index, t[batch_indices]
        )

        loss_atoms = torch.mean((pred_noise_atoms - noise_atoms) ** 2)
        loss_coords = torch.mean((pred_noise_coords - noise_coords) ** 2)

        return loss_atoms + loss_coords

    @torch.no_grad()
    def generate(
            self,
            num_atoms: int,
            num_molecules: int = 1,
            device: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()

        if device is None:
            device = self.device

        xt = torch.randn(num_molecules, num_atoms, 3, device=device)
        ht = torch.randn(num_molecules, num_atoms, self.num_atom_types, device=device)
        xt = xt - xt.mean(dim=1, keepdim=True)

        edge_index = self._get_fully_connected_edges(num_atoms, num_molecules).to(device)

        for t in reversed(range(self.num_diffusion_steps)):
            t_tensor = torch.tensor([t] * num_molecules, device=device)

            pred_noise_h, pred_noise_x = self.model(
                ht.view(-1, self.num_atom_types),
                xt.view(-1, 3),
                edge_index,
                t_tensor
            )

            pred_noise_h = pred_noise_h.view(num_molecules, num_atoms, self.num_atom_types)
            pred_noise_x = pred_noise_x.view(num_molecules, num_atoms, 3)

            xt, ht = self.model.diffusion.reverse_diffusion_step(
                xt, ht, pred_noise_x, pred_noise_h, t
            )

        return xt, torch.argmax(ht, dim=-1)

    def _get_fully_connected_edges(self, num_atoms: int, num_molecules: int) -> torch.Tensor:
        rows, cols = torch.meshgrid(torch.arange(num_atoms), torch.arange(num_atoms), indexing='ij')
        mask = rows != cols
        edge_index_single = torch.stack([rows[mask], cols[mask]], dim=0)

        edge_indices = []
        for i in range(num_molecules):
            offset = i * num_atoms
            edge_indices.append(edge_index_single + offset)

        return torch.cat(edge_indices, dim=1)

    def save(self, path: str):
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