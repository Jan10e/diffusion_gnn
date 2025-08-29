"""
Generation is updated to als handel both features and 3D positions.
"""

import torch
import numpy as np
from rdkit import Chem
from typing import List, Optional, Tuple

from .samplers.diffusion_samplers import DDPMPsampler


def sample_from_model(model, p_sampler, num_molecules: int, max_atoms: int,
                      atom_dim: int, pos_dim: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample molecular features and 3D positions from the trained diffusion model.
    """
    model.eval()

    with torch.no_grad():
        total_nodes = num_molecules * max_atoms
        x = torch.randn(total_nodes, atom_dim, device=device)
        pos = torch.randn(total_nodes, pos_dim, device=device)

        batch = torch.repeat_interleave(
            torch.arange(num_molecules, device=device), max_atoms
        )

        edges = []
        for mol_idx in range(num_molecules):
            offset = mol_idx * max_atoms
            for i in range(max_atoms):
                for j in range(i + 1, max_atoms):
                    edges.extend([[offset + i, offset + j], [offset + j, offset + i]])

        if edges:
            edge_index = torch.tensor(edges, device=device).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), device=device, dtype=torch.long)

        # Iterate through the reverse process using the p_sampler
        for t in reversed(range(p_sampler.num_timesteps)):
            t_batch = torch.full((total_nodes,), t, device=device, dtype=torch.long)

            x, pos = p_sampler.p_sample_step(model, x, pos, edge_index, batch, t_batch)

    generated_x = x.reshape(num_molecules, max_atoms, -1)
    generated_pos = pos.reshape(num_molecules, max_atoms, -1)

    return generated_x, generated_pos