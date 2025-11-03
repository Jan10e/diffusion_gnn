import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
import logging

from edm_3d.core.egnn import EGNN
from edm_3d.core.diffusion_process import DiffusionProcess

logger = logging.getLogger(__name__)


class EDMCore(nn.Module):
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

        # CRITICAL FIX: Specify output dimension
        self.egnn = EGNN(
            in_node_dim=num_atom_types + hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            out_node_dim=num_atom_types  # Must output num_atom_types
        )

        self.diffusion = DiffusionProcess(num_steps=num_diffusion_steps)

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

        # h_out is now [N, num_atom_types] ✓
        # x_out is [N, 3] ✓

        return h_out, x_out