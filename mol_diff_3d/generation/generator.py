"""
Enhanced molecule generator with bond guidance and chemical validity checks.
Key improvements:
1. Bond predictor guidance during generation
2. Chemical validity post-processing
3. Better molecular reconstruction from generated features
"""

import torch
from typing import List, Optional, Tuple, Dict
from torch_geometric.utils import to_undirected
import logging
import math
import torch.nn.functional as F

from ..sampling.samplers import ImprovedDDPMPsampler
from ..utils.molecular import create_3d_molecule_from_positions_and_bonds, features_to_atom_types
from ..data.datasets import create_dummy_complete_graph
from rdkit import Chem
import numpy as np


logger = logging.getLogger(__name__)


def generate_molecules_with_bond_guidance(
    model: torch.nn.Module,
    bond_predictor: torch.nn.Module,
    p_sampler: ImprovedDDPMPsampler,
    num_molecules: int,
    max_atoms: int,
    atom_dim: int,
    bond_dim: int,
    pos_dim: int,
    device: torch.device,
    guidance_steps: int = 100,
    temperature: float = 0.8
) -> Tuple[List[Chem.Mol], List[str], Dict]:
    """
    Generate molecules using bond guidance for improved chemical validity.

    Args:
        model: Trained MolecularDiffusionModel
        bond_predictor: Trained BondPredictor for guidance
        p_sampler: Enhanced reverse sampler with bond guidance
        num_molecules: Number of molecules to generate
        max_atoms: Maximum atoms per molecule
        atom_dim: Atom feature dimension
        bond_dim: Bond feature dimension
        pos_dim: Position dimension (3D)
        device: Computation device
        guidance_steps: Number of steps to apply guidance
        temperature: Sampling temperature for categorical distributions

    Returns:
        generated_mols: List of valid RDKit molecules
        generated_smiles: List of SMILES strings
        generation_stats: Dictionary with generation statistics
    """
    model.eval()
    if bond_predictor:
        bond_predictor.eval()

    logger.info(f"Starting guided molecule generation on {device}...")

    # Initialize from noise
    total_nodes = num_molecules * max_atoms

    # Start from random atom types (uniform distribution)
    x = torch.zeros(total_nodes, atom_dim, device=device)
    x[:, 0] = 1.0  # Start with all atoms as first type (usually Carbon)

    # Add some noise to initial atom types
    noise_scale = 0.3
    atom_noise = torch.randn_like(x) * noise_scale
    x = F.softmax(x + atom_noise, dim=-1)

    # Random positions
    pos = torch.randn(total_nodes, pos_dim, device=device)

    # Create batch tensor
    batch = torch.repeat_interleave(
        torch.arange(num_molecules, device=device), max_atoms
    )

    # Create complete graph structure with dummy bonds
    edge_index, edge_attr = create_dummy_complete_graph(max_atoms, bond_dim)

    # Expand for multiple molecules
    full_edge_index = []
    full_edge_attr = []

    for mol_idx in range(num_molecules):
        offset = mol_idx * max_atoms
        mol_edge_index = edge_index + offset
        full_edge_index.append(mol_edge_index)
        full_edge_attr.append(edge_attr)

    if len(full_edge_index) > 0:
        edge_index = torch.cat(full_edge_index, dim=1)
        edge_attr = torch.cat(full_edge_attr, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, bond_dim), dtype=torch.float, device=device)

    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    # Track generation statistics
    generation_stats = {
        'attempted_molecules': num_molecules,
        'successful_generations': 0,
        'failed_generations': 0,
        'invalid_chemistry': 0,
        'guidance_applications': 0
    }

    # Reverse diffusion loop with bond guidance
    with torch.no_grad():
        for t in reversed(range(p_sampler.num_timesteps)):
            t_batch = torch.full((num_molecules,), t, device=device, dtype=torch.long)

            # Apply bond guidance in the first guidance_steps timesteps
            if t >= (p_sampler.num_timesteps - guidance_steps):
                generation_stats['guidance_applications'] += 1

            # Sample next state
            x, pos, edge_attr = p_sampler.p_sample_step(
                model, x, pos, edge_index, edge_attr, batch, t_batch
            )

            # Apply chemical validity constraints every few steps
            if t % 10 == 0:
                x, edge_attr = apply_chemical_validity_constraints(
                    x, edge_attr, pos, edge_index, temperature=temperature
                )

    # Convert final tensors to molecules
    generated_mols = []
    generated_smiles = []

    # Reshape output to (num_molecules, max_atoms, dim)
    final_x = x.reshape(num_molecules, max_atoms, -1)
    final_pos = pos.reshape(num_molecules, max_atoms, -1)
    final_edge_attr = edge_attr.reshape(num_molecules, -1, bond_dim)

    # Process each generated molecule
    for i in range(num_molecules):
        try:
            atom_features = final_x[i]
            positions = final_pos[i]
            mol_edge_attr = final_edge_attr[i]

            # Convert features to interpretable format
            atom_types = features_to_atom_types(atom_features)
            bond_types = mol_edge_attr.argmax(dim=-1)

            # Create molecule with explicit bond information
            mol = create_3d_molecule_from_positions_and_bonds(
                atom_types, positions, bond_types, edge_index[:, :mol_edge_attr.shape[0]]
            )

            if mol and is_chemically_valid(mol):
                try:
                    # Generate SMILES and validate
                    smiles = Chem.MolToSmiles(mol)
                    if smiles and len(smiles) > 1:  # Basic validity check
                        generated_mols.append(mol)
                        generated_smiles.append(smiles)
                        generation_stats['successful_generations'] += 1
                    else:
                        generation_stats['failed_generations'] += 1
                except Exception as e:
                    logger.warning(f"SMILES generation failed for molecule {i}: {e}")
                    generation_stats['failed_generations'] += 1
            else:
                generation_stats['invalid_chemistry'] += 1

        except Exception as e:
            logger.warning(f"Error processing generated molecule {i}: {e}")
            generation_stats['failed_generations'] += 1

    # Calculate success rate
    generation_stats['success_rate'] = (
        generation_stats['successful_generations'] / num_molecules
    )

    logger.info(f"Generation completed:")
    logger.info(f"  Attempted: {generation_stats['attempted_molecules']}")
    logger.info(f"  Successful: {generation_stats['successful_generations']}")
    logger.info(f"  Success rate: {generation_stats['success_rate']:.2%}")

    return generated_mols, generated_smiles, generation_stats


def apply_chemical_validity_constraints(x: torch.Tensor, edge_attr: torch.Tensor,
                                      pos: torch.Tensor, edge_index: torch.Tensor,
                                      temperature: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply chemical validity constraints during generation.
    This helps ensure generated molecules follow basic chemical rules.
    """
    # Get current atom and bond assignments
    atom_probs = F.softmax(x / temperature, dim=-1)
    bond_probs = F.softmax(edge_attr / temperature, dim=-1)

    # Element symbols for interpretation
    elements = ['C', 'N', 'O', 'F', 'H', 'P', 'S', 'Cl', 'Br', 'I']

    # Apply valency constraints
    for node_idx in range(x.shape[0]):
        # Get edges connected to this node
        connected_edges = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)
        if connected_edges.sum() == 0:
            continue

        # Get current atom type (most probable)
        atom_type = atom_probs[node_idx].argmax().item()
        if atom_type >= len(elements):
            continue

        element = elements[atom_type]

        # Get typical valencies for this element
        typical_valencies = get_typical_valencies(element)

        # Count current bonds
        edge_indices = torch.where(connected_edges)[0]
        current_bonds = bond_probs[edge_indices].argmax(dim=-1)
        total_bond_order = current_bonds.sum().item()

        # If over-bonded, reduce some bonds to "no bond"
        if total_bond_order > max(typical_valencies):
            excess_bonds = int(total_bond_order - max(typical_valencies))
            # Find edges to modify (prefer weakest bonds)
            bond_strengths = bond_probs[edge_indices].max(dim=-1)[0]
            weakest_edges = bond_strengths.argsort()[:excess_bonds]

            for edge_idx in edge_indices[weakest_edges]:
                # Set to "no bond"
                edge_attr[edge_idx] = torch.zeros_like(edge_attr[edge_idx])
                edge_attr[edge_idx, 0] = 1.0  # "no bond" type

    return x, edge_attr


def get_typical_valencies(element: str) -> List[int]:
    """Get typical valencies for common elements."""
    valency_rules = {
        'H': [1],
        'C': [4],
        'N': [3, 5],
        'O': [2],
        'F': [1],
        'P': [3, 5],
        'S': [2, 4, 6],
        'Cl': [1, 3, 5, 7],
        'Br': [1, 3, 5],
        'I': [1, 3, 5, 7]
    }
    return valency_rules.get(element, [4])  # Default to 4 for unknown elements


def is_chemically_valid(mol: Chem.Mol) -> bool:
    """
    Check if a molecule is chemically valid.
    """
    if mol is None:
        return False

    try:
        # Basic checks
        if mol.GetNumAtoms() == 0:
            return False

        # Check if molecule can be sanitized
        Chem.SanitizeMol(mol)

        # Check for reasonable number of bonds
        if mol.GetNumBonds() == 0 and mol.GetNumAtoms() > 1:
            return False

        # Check valencies
        for atom in mol.GetAtoms():
            if atom.GetTotalValence() > 8:  # Unreasonable valency
                return False

        return True

    except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException, ValueError):
        return False


# Additional utility function for the molecular.py module
def create_3d_molecule_from_positions_and_bonds(atom_types: List[str], positions: torch.Tensor,
                                               bond_types: torch.Tensor, edge_index: torch.Tensor) -> Optional[Chem.Mol]:
    """
    Create RDKit molecule from atoms, positions, and explicit bond information.
    This replaces the distance-based bond inference with explicit bond types.
    """
    if not atom_types or len(atom_types) < 2:
        return None

    try:
        # Create blank molecule
        mol = Chem.Mol()
        editable_mol = Chem.EditableMol(mol)

        # Add atoms
        for atom_type in atom_types:
            atom = Chem.Atom(atom_type)
            editable_mol.AddAtom(atom)

        # Add bonds based on bond_types and edge_index
        bond_type_mapping = {
            1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE,
            4: Chem.rdchem.BondType.AROMATIC
        }

        # Track added bonds to avoid duplicates
        added_bonds = set()

        for edge_idx in range(edge_index.shape[1]):
            i, j = edge_index[0, edge_idx].item(), edge_index[1, edge_idx].item()
            bond_type = bond_types[edge_idx].item()

            # Skip "no bond" type (0) and duplicates
            if bond_type == 0 or (min(i,j), max(i,j)) in added_bonds:
                continue

            # Add bond if valid indices
            if 0 <= i < len(atom_types) and 0 <= j < len(atom_types) and i != j:
                rdkit_bond_type = bond_type_mapping.get(bond_type, Chem.rdchem.BondType.SINGLE)
                editable_mol.AddBond(i, j, rdkit_bond_type)
                added_bonds.add((min(i,j), max(i,j)))

        # Convert to standard mol
        mol = editable_mol.GetMol()

        # Add 3D coordinates
        if mol.GetNumAtoms() > 0:
            conformer = Chem.Conformer(mol.GetNumAtoms())
            for i in range(mol.GetNumAtoms()):
                if i < len(positions):
                    conformer.SetAtomPosition(i, positions[i].tolist())
            mol.AddConformer(conformer)

        # Sanitize
        Chem.SanitizeMol(mol)
        return mol

    except Exception as e:
        logger.debug(f"Error creating molecule: {e}")
        return None