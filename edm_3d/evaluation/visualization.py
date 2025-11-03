import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import deepchem as dc
from deepchem.models.torch_models import TorchModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_molecule(coords: torch.Tensor, atom_types: torch.Tensor, idx: int = 0):
    """
    Visualize a generated molecule
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("RDKit and matplotlib required for visualization")
        return

    # Extract single molecule
    mol_coords = coords[idx].cpu().numpy()
    mol_atoms = atom_types[idx].cpu().numpy()

    # Atom type mapping (for QM9)
    atom_map = {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F'}

    # 3D visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red', 'F': 'green'}

    for i, (coord, atom_type) in enumerate(zip(mol_coords, mol_atoms)):
        atom_symbol = atom_map.get(atom_type, 'H')
        ax.scatter(*coord, c=colors[atom_symbol], s=500, alpha=0.8, edgecolors='black')
        ax.text(*coord, atom_symbol, fontsize=12)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Generated Molecule {idx}')

    plt.show()


def plot_3d_molecule(coords, atom_types, ax=None, title="Molecule"):
    """
    Plot 3D molecule
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Atom colors
    atom_map = {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F'}
    colors = {'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red', 'F': 'green'}

    coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords
    atom_types_np = atom_types.cpu().numpy() if torch.is_tensor(atom_types) else atom_types

    # Plot atoms
    for coord, atom_type in zip(coords_np, atom_types_np):
        atom_symbol = atom_map.get(int(atom_type), 'C')
        color = colors[atom_symbol]
        ax.scatter(*coord, c=color, s=500, alpha=0.8, edgecolors='black', linewidths=2)
        ax.text(*coord, atom_symbol, fontsize=10, ha='center', va='center')

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title)

    return ax


def visualize_molecule(coords, atom_types, idx=0, title=None):
    """
    Visualize a single molecule
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    if title is None:
        title = f'Generated Molecule {idx}'

    plot_3d_molecule(coords[idx], atom_types[idx], ax=ax, title=title)
    plt.show()


def plot_dataset_statistics(dataset):
    """
    Plot statistics about the dataset
    """
    num_atoms_list = []
    atom_type_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for i in range(min(1000, len(dataset))):  # Sample first 1000
        item = dataset[i]
        num_atoms_list.append(item['num_atoms'])
        for atom_type in item['atom_types'].tolist():
            atom_type_counts[atom_type] = atom_type_counts.get(atom_type, 0) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Number of atoms distribution
    ax1.hist(num_atoms_list, bins=20, edgecolor='black')
    ax1.set_xlabel('Number of Atoms')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Molecule Size Distribution')
    ax1.grid(alpha=0.3)

    # Atom type distribution
    atom_names = ['H', 'C', 'N', 'O', 'F']
    counts = [atom_type_counts[i] for i in range(5)]
    ax2.bar(atom_names, counts, color=['white', 'gray', 'blue', 'red', 'green'],
            edgecolor='black', linewidth=2)
    ax2.set_ylabel('Count')
    ax2.set_title('Atom Type Distribution')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def animate_diffusion(coords_trajectory, atoms_trajectory, save_path=None):
    """
    Create animation of diffusion process
    """
    # This would use matplotlib animation
    # For now, just show key frames
    print(f"Animation with {len(coords_trajectory)} frames")
    if save_path:
        print(f"Would save to: {save_path}")