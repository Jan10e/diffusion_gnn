import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_molecule(coords, atom_types, ax=None, title="Molecule"):
    """
    Plot 3D molecule - coords and atom_types must be single molecule (not batched)

    Args:
        coords: [num_atoms, 3] array
        atom_types: [num_atoms] array
        ax: matplotlib 3D axis (optional)
        title: plot title
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Atom colors
    atom_map = {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F'}
    colors = {'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red', 'F': 'green'}

    # Convert to numpy
    coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
    atom_types_np = atom_types.cpu().numpy() if torch.is_tensor(atom_types) else np.asarray(atom_types)

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


def visualize_molecule(coords, atom_types, idx: int = 0, title: str = None, ax=None):
    """
    Robust visualize_molecule that handles both:
      - batched: coords (B, N, 3) and atom_types (B, N)
      - single: coords (N, 3) and atom_types (N)

    Args:
        coords: Coordinates tensor/array
        atom_types: Atom type indices tensor/array
        idx: Which molecule to plot if batched
        title: Plot title (optional)
        ax: matplotlib 3D axis (optional)
    """
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import AllChem, Draw  # noqa: F401
    except ImportError:
        print("RDKit required for visualization")
        pass

    # Convert to numpy if needed
    coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
    atoms_np = atom_types.cpu().numpy() if torch.is_tensor(atom_types) else np.asarray(atom_types)

    # Handle batched vs single-molecule shapes
    if coords_np.ndim == 3:
        # Batched: (B, N, 3)
        coords_sel = coords_np[idx]
    elif coords_np.ndim == 2:
        # Single molecule: (N, 3)
        coords_sel = coords_np
    else:
        raise ValueError(f"`coords` must be 2D or 3D, got shape {coords_np.shape}")

    if atoms_np.ndim == 2:
        # Batched: (B, N)
        atoms_sel = atoms_np[idx]
    elif atoms_np.ndim == 1:
        # Single molecule: (N,)
        atoms_sel = atoms_np
    else:
        raise ValueError(f"`atom_types` must be 1D or 2D, got shape {atoms_np.shape}")

    if title is None:
        title = f"Molecule {idx}"

    # Create axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Plot the molecule
    plot_3d_molecule(coords_sel, atoms_sel, ax=ax, title=title)
    plt.show()


def plot_dataset_statistics(dataset):
    """
    Plot statistics about the dataset

    Args:
        dataset: QM9Dataset object
    """
    num_atoms_list = []
    atom_type_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    sample_size = min(1000, len(dataset))
    print(f"Analyzing {sample_size} molecules for statistics...")

    for i in range(sample_size):
        if i % 100 == 0:
            print(f"  Processed {i}/{sample_size}...")

        item = dataset[i]
        num_atoms_list.append(item['num_atoms'])
        for atom_type in item['atom_types'].tolist():
            atom_type_counts[atom_type] = atom_type_counts.get(atom_type, 0) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Number of atoms distribution
    ax1.hist(num_atoms_list, bins=20, edgecolor='black', color='steelblue', alpha=0.7)
    ax1.set_xlabel('Number of Atoms', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Molecule Size Distribution', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)

    # Add statistics text
    mean_atoms = np.mean(num_atoms_list)
    ax1.axvline(mean_atoms, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_atoms:.1f}')
    ax1.legend()

    # Atom type distribution
    atom_names = ['H', 'C', 'N', 'O', 'F']
    counts = [atom_type_counts[i] for i in range(5)]
    colors_bar = ['white', 'gray', 'blue', 'red', 'green']

    bars = ax2.bar(atom_names, counts, color=colors_bar, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Atom Type Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

    print(f"\n✓ Statistics computed from {sample_size} molecules")


def visualize_diffusion_steps(coords_list, atoms_list, timesteps, title_prefix="Diffusion"):
    """
    Visualize multiple diffusion steps in a grid

    Args:
        coords_list: List of coordinate tensors
        atoms_list: List of atom type tensors
        timesteps: List of timestep values
        title_prefix: Prefix for plot titles
    """
    n_steps = len(coords_list)
    n_cols = min(3, n_steps)
    n_rows = (n_steps + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows))

    for idx, (coords, atoms, t) in enumerate(zip(coords_list, atoms_list, timesteps)):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        plot_3d_molecule(coords, atoms, ax=ax, title=f"{title_prefix} t={t}")

    plt.tight_layout()
    plt.show()


def animate_diffusion(coords_trajectory, atoms_trajectory, save_path=None):
    """
    Create animation of diffusion process

    Args:
        coords_trajectory: List of coordinate tensors at different timesteps
        atoms_trajectory: List of atom type tensors at different timesteps
        save_path: Path to save animation (optional)
    """
    print(f"Animation with {len(coords_trajectory)} frames")

    # Show key frames for now
    n_frames = len(coords_trajectory)
    key_frame_indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5), subplot_kw={'projection': '3d'})

    for idx, frame_idx in enumerate(key_frame_indices):
        coords = coords_trajectory[frame_idx]
        atoms = atoms_trajectory[frame_idx]

        # Handle batched data
        if coords.ndim == 3:
            coords = coords[0]  # Take first molecule
            atoms = atoms[0]

        plot_3d_molecule(coords, atoms, ax=axes[idx], title=f"Step {frame_idx}")

    plt.tight_layout()
    plt.show()

    if save_path:
        print(f"Would save animation to: {save_path}")
        # TODO: Implement actual animation saving with matplotlib.animation


def plot_molecule_grid(coords_batch, atoms_batch, n_cols=4, titles=None):
    """
    Plot a grid of molecules

    Args:
        coords_batch: Batched coordinates [B, N, 3]
        atoms_batch: Batched atom types [B, N]
        n_cols: Number of columns in grid
        titles: Optional list of titles
    """
    n_molecules = len(coords_batch)
    n_rows = (n_molecules + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    for idx in range(n_molecules):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

        title = titles[idx] if titles else f"Molecule {idx + 1}"
        plot_3d_molecule(coords_batch[idx], atoms_batch[idx], ax=ax, title=title)

    plt.tight_layout()
    plt.show()