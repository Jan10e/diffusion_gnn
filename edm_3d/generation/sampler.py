import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import deepchem as dc
from deepchem.models.torch_models import TorchModel


class EDMSampler:
    """
    Sampler for generating molecules with EDM
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def sample(
            self,
            num_molecules: int,
            num_atoms: int,
            device: str = 'cpu',
            return_trajectory: bool = False
    ):
        """
        Generate molecules

        Args:
            num_molecules: Number of molecules to generate
            num_atoms: Number of atoms per molecule
            device: Device to run on
            return_trajectory: If True, return all intermediate steps

        Returns:
            coords, atom_types (and trajectory if requested)
        """
        self.model = self.model.to(device)

        # Start from noise
        coords = torch.randn(num_molecules, num_atoms, 3).to(device)
        atom_types = torch.randn(num_molecules, num_atoms, self.model.num_atom_types).to(device)

        # Center coordinates
        coords = coords - coords.mean(dim=1, keepdim=True)

        trajectory = [] if return_trajectory else None

        # Reverse diffusion
        for t in reversed(range(self.model.diffusion.num_steps)):
            if return_trajectory and t % 100 == 0:
                trajectory.append((coords.clone(), atom_types.clone()))

            coords, atom_types = self._denoise_step(coords, atom_types, t, device)

        # Convert to discrete atom types
        atom_types_discrete = torch.argmax(atom_types, dim=-1)

        if return_trajectory:
            return coords, atom_types_discrete, trajectory
        return coords, atom_types_discrete

    def _denoise_step(self, coords, atom_types, t, device):
        """Single denoising step"""
        batch_size = coords.shape[0]
        num_atoms = coords.shape[1]

        # Create edge index (fully connected)
        edge_index = self.model._get_fully_connected_edges(num_atoms, batch_size).to(device)

        # Timestep tensor
        t_tensor = torch.tensor([t] * batch_size).to(device)

        # Predict noise
        pred_noise_h, pred_noise_x = self.model.forward(
            atom_types.view(-1, self.model.num_atom_types),
            coords.view(-1, 3),
            edge_index,
            t_tensor
        )

        # Reshape
        pred_noise_h = pred_noise_h.view(batch_size, num_atoms, -1)
        pred_noise_x = pred_noise_x.view(batch_size, num_atoms, 3)

        # Denoise
        coords_new, atoms_new = self.model.diffusion.reverse_diffusion_step(
            coords, atom_types, pred_noise_x, pred_noise_h, t
        )

        return coords_new, atoms_new

    def sample_with_trajectory(self, num_molecules, num_atoms, save_every=100, device='cpu'):
        """Sample and return full trajectory"""
        return self.sample(
            num_molecules=num_molecules,
            num_atoms=num_atoms,
            device=device,
            return_trajectory=True
        )


# Standalone function
def generate_molecules(model, num_molecules: int = 10, device='cpu'):
    """
    Convenience function to generate molecules
    """
    print(f"Generating {num_molecules} molecules...")

    sampler = EDMSampler(model)
    coords, atom_types = sampler.sample(
        num_molecules=num_molecules,
        num_atoms=9,
        device=device
    )

    print(f"Generated coordinates shape: {coords.shape}")
    print(f"Generated atom types shape: {atom_types.shape}")

    return coords, atom_types
