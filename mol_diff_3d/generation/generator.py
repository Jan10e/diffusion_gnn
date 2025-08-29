"""
This module contains the logic for generating new molecules from a trained MolDiff model.
It now correctly uses the DDPM Sampler classes for the diffusion process.
"""

import torch
from typing import List, Optional, Tuple, Dict

# Assumes the directory structure has been updated as previously discussed
from ..sampling.samplers import DDPMPsampler
from ..utils.molecular import create_3d_molecule_from_positions, features_to_atom_types
from rdkit import Chem
import logging

logger = logging.getLogger(__name__)


def generate_molecules_from_model(
    model: torch.nn.Module,
    p_sampler: DDPMPsampler,
    num_molecules: int,
    max_atoms: int,
    atom_dim: int,
    pos_dim: int,
    device: torch.device
) -> Tuple[List[Chem.Mol], List[str]]:
    """
    Generates new molecules by running the reverse diffusion process.

    Args:
        model: The trained MolecularDiffusionModel.
        p_sampler: The DDPMPsampler instance for reverse diffusion.
        num_molecules: Number of molecules to generate.
        max_atoms: Maximum number of atoms per molecule.
        atom_dim: Dimension of atom feature vectors.
        pos_dim: Dimension of position vectors (should be 3).
        device: The device to run the generation on.

    Returns:
        A tuple of (list of valid RDKit molecules, list of their SMILES strings).
    """
    model.eval()
    logger.info(f"Starting molecule generation on {device}...")

    # Start from pure noise
    total_nodes = num_molecules * max_atoms
    x = torch.randn(total_nodes, atom_dim, device=device)
    pos = torch.randn(total_nodes, pos_dim, device=device)

    # Create a complete graph structure for the maximum number of atoms
    # This simplified approach allows the GNN to learn the relevant connections
    # from the noisy positions
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

    # 1. Reverse diffusion loop
    with torch.no_grad():
        for t in reversed(range(p_sampler.num_timesteps)):
            t_batch = torch.full((total_nodes,), t, device=device, dtype=torch.long)
            # This is the key change: we delegate the step to the sampler
            x, pos = p_sampler.p_sample_step(model, x, pos, edge_index, batch, t_batch)

    # 2. Convert final tensors to molecules
    generated_mols = []
    generated_smiles = []

    # Reshape output to (num_molecules, max_atoms, dim)
    final_x = x.reshape(num_molecules, max_atoms, -1)
    final_pos = pos.reshape(num_molecules, max_atoms, -1)

    for i in range(num_molecules):
        atom_features = final_x[i]
        positions = final_pos[i]

        # Convert features to atom types
        atom_types = features_to_atom_types(atom_features)

        # Create RDKit molecule from generated atoms and positions
        mol = create_3d_molecule_from_positions(atom_types, positions)

        if mol:
            try:
                smiles = Chem.MolToSmiles(mol)
                if smiles:
                    generated_mols.append(mol)
                    generated_smiles.append(smiles)
            except Exception as e:
                logger.warning(f"Could not convert molecule to SMILES: {e}")

    logger.info(f"Generated {len(generated_mols)} valid molecules.")

    return generated_mols, generated_smiles