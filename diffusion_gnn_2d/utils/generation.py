import torch
import numpy as np
from rdkit import Chem
from typing import List, Optional, Tuple


def sample_from_model(model, scheduler, num_molecules: int, max_atoms: int,
                      atom_dim: int, bond_dim: int, device) -> torch.Tensor:
    """Sample molecular features from trained diffusion model"""
    model.eval()

    with torch.no_grad():
        # Start from pure noise
        total_nodes = num_molecules * max_atoms
        x = torch.randn(total_nodes, atom_dim, device=device)

        # Create batch assignment
        batch = torch.repeat_interleave(
            torch.arange(num_molecules, device=device), max_atoms
        )

        # Create complete graph edges (simplified but functional)
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

        edge_attr = torch.randn(edge_index.shape[1], bond_dim, device=device)

        # Reverse diffusion
        for t in reversed(range(scheduler.num_timesteps)):
            t_batch = torch.full((num_molecules,), t, device=device, dtype=torch.long)
            noise_pred = model(x, edge_index, edge_attr, batch, t_batch)

            # Simple denoising step
            if t > 0:
                beta_t = scheduler.betas[t]
                alpha_t = 1 - beta_t
                x = (x - beta_t * noise_pred) / torch.sqrt(alpha_t)
                if t > 1:
                    x += torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = x - noise_pred

    return x.reshape(num_molecules, max_atoms, -1)


def features_to_atom_types(features: torch.Tensor) -> List[str]:
    """Convert first 11 features (one-hot atomic numbers) to element symbols"""
    elements = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Other']
    atomic_indices = torch.argmax(features[:, :11], dim=1)
    return [elements[i] for i in atomic_indices.cpu().numpy()]