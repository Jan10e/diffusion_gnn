import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

from typing import List, Dict, Optional, Tuple, Union
import logging

import io

from PIL import Image

logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


def visualize_molecule(smiles: str,
                       size: Tuple[int, int] = (400, 400),
                       highlight_atoms: List[int] = None,
                       highlight_bonds: List[int] = None,
                       title: str = None) -> None:
    """
    Visualize a molecule from SMILES string using RDKit

    Args:
        smiles: SMILES string
        size: Image size (width, height)
        highlight_atoms: List of atom indices to highlight
        highlight_bonds: List of bond indices to highlight
        title: Optional title for the plot
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            return

        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])

        # Set up highlighting
        highlight_atom_colors = {}
        highlight_bond_colors = {}

        if highlight_atoms:
            for atom_idx in highlight_atoms:
                highlight_atom_colors[atom_idx] = (1.0, 0.0, 0.0)  # Red

        if highlight_bonds:
            for bond_idx in highlight_bonds:
                highlight_bond_colors[bond_idx] = (1.0, 0.0, 0.0)  # Red

        # Draw molecule
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms or [],
                            highlightAtomColors=highlight_atom_colors,
                            highlightBonds=highlight_bonds or [],
                            highlightBondColors=highlight_bond_colors)
        drawer.FinishDrawing()

        # Get image and display
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))

        plt.figure(figsize=(size[0] / 100, size[1] / 100))
        plt.imshow(img)
        plt.axis('off')

        if title:
            plt.title(title, fontsize=12, pad=10)
        else:
            plt.title(f"SMILES: {smiles}", fontsize=10, pad=10)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"Error visualizing molecule {smiles}: {e}")
        plt.figure(figsize=(4, 4))
        plt.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center',
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title(f"Failed: {smiles[:30]}...")
        plt.show()


def visualize_molecular_graph(graph_data: Data,
                              node_labels: bool = True,
                              edge_labels: bool = False,
                              figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Visualize a molecular graph using NetworkX

    Args:
        graph_data: PyTorch Geometric Data object
        node_labels: Whether to show node labels
        edge_labels: Whether to show edge labels
        figsize: Figure size
    """
    try:
        # Convert to NetworkX
        G = to_networkx(graph_data, to_undirected=True)

        plt.figure(figsize=figsize)

        # Create layout
        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)

        # Draw nodes
        node_colors = []
        for i in range(len(G.nodes())):
            # Color by atomic number (first 11 features are one-hot atomic numbers)
            atom_features = graph_data.x[i].numpy()
            atomic_num_idx = np.argmax(atom_features[:11])
            atomic_elements = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Other']

            # Color mapping
            color_map = {
                'H': 'lightgray', 'C': 'black', 'N': 'blue', 'O': 'red',
                'F': 'lightgreen', 'P': 'orange', 'S': 'yellow',
                'Cl': 'green', 'Br': 'brown', 'I': 'purple', 'Other': 'pink'
            }

            element = atomic_elements[atomic_num_idx]
            node_colors.append(color_map.get(element, 'gray'))

        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=500, alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.6, width=2)

        # Add labels
        if node_labels:
            labels = {}
            for i in range(len(G.nodes())):
                atom_features = graph_data.x[i].numpy()
                atomic_num_idx = np.argmax(atom_features[:11])
                atomic_elements = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', '?']
                labels[i] = atomic_elements[atomic_num_idx]

            nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')

        plt.title(f"Molecular Graph\nNodes: {graph_data.x.shape[0]}, Edges: {graph_data.edge_index.shape[1] // 2}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"Error visualizing graph: {e}")
        plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, f"Error visualizing graph:\n{str(e)}",
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.show()


def plot_training_metrics(metrics: Dict[str, List[float]],
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Plot training metrics over time

    Args:
        metrics: Dictionary with metric names as keys and lists of values
        save_path: Optional path to save the plot
        figsize: Figure size
    """

    num_metrics = len(metrics)
    if num_metrics == 0:
        logger.warning("No metrics to plot")
        return

    fig, axes = plt.subplots(1, min(num_metrics, 3), figsize=figsize)
    if num_metrics == 1:
        axes = [axes]
    elif num_metrics > 3:
        # Create subplots for more than 3 metrics
        rows = (num_metrics + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
        axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, num_metrics))

    for idx, (metric_name, values) in enumerate(metrics.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Plot the metric
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, color=colors[idx], linewidth=2, marker='o', markersize=3)

        # Formatting
        ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

        # Add trend line for loss metrics
        if len(values) > 10 and 'loss' in metric_name.lower():
            z = np.polyfit(epochs, values, 1)
            p = np.poly1d(z)
            ax.plot(epochs, p(epochs), "--", alpha=0.7, color=colors[idx])

        # Add statistics text
        if len(values) > 0:
            final_value = values[-1]
            min_value = min(values)
            ax.text(0.02, 0.98, f'Final: {final_value:.4f}\nMin: {min_value:.4f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)

    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training metrics plot saved to {save_path}")

    plt.show()


def plot_molecular_properties(smiles_list: List[str],
                              max_molecules: int = 1000,
                              figsize: Tuple[int, int] = (15, 10)) -> Dict:
    """
    Analyze and plot molecular properties from SMILES

    Args:
        smiles_list: List of SMILES strings
        max_molecules: Maximum number of molecules to analyze
        figsize: Figure size

    Returns:
        Dictionary with computed statistics
    """

    properties = {
        'molecular_weight': [],
        'logp': [],
        'num_atoms': [],
        'num_bonds': [],
        'num_rings': [],
        'tpsa': [],
        'valid_smiles': []
    }

    # Analyze molecules
    for i, smiles in enumerate(smiles_list[:max_molecules]):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                properties['molecular_weight'].append(Descriptors.MolWt(mol))
                properties['logp'].append(Descriptors.MolLogP(mol))
                properties['num_atoms'].append(mol.GetNumAtoms())
                properties['num_bonds'].append(mol.GetNumBonds())
                properties['num_rings'].append(Descriptors.RingCount(mol))
                properties['tpsa'].append(Descriptors.TPSA(mol))
                properties['valid_smiles'].append(smiles)

        except Exception as e:
            logger.debug(f"Error processing SMILES {smiles}: {e}")
            continue

    if not properties['valid_smiles']:
        logger.error("No valid molecules found for analysis")
        return properties

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    plot_properties = ['molecular_weight', 'logp', 'num_atoms', 'num_bonds', 'num_rings', 'tpsa']
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_properties)))

    for idx, prop in enumerate(plot_properties):
        if prop in properties and properties[prop]:
            data = properties[prop]

            # Histogram
            axes[idx].hist(data, bins=30, alpha=0.7, color=colors[idx], edgecolor='black')

            # Add statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.8)

            # Labels and title
            axes[idx].set_xlabel(prop.replace('_', ' ').title())
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{prop.replace("_", " ").title()}\n(μ={mean_val:.2f}, σ={std_val:.2f})')
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Compute statistics
    stats = {}
    for prop in plot_properties:
        if prop in properties and properties[prop]:
            data = np.array(properties[prop])
            stats[prop] = {
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'median': np.median(data)
            }

    stats['num_valid'] = len(properties['valid_smiles'])
    stats['num_total'] = min(len(smiles_list), max_molecules)
    stats['validity_rate'] = stats['num_valid'] / stats['num_total'] if stats['num_total'] > 0 else 0.0

    # Print summary
    print(f"\nMolecular Analysis Summary:")
    print(f"Total molecules analyzed: {stats['num_total']}")
    print(f"Valid molecules: {stats['num_valid']}")
    print(f"Validity rate: {stats['validity_rate']:.2%}")

    return stats


def plot_diffusion_process(original_data: torch.Tensor,
                           noisy_samples: List[torch.Tensor],
                           timesteps: List[int],
                           figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Visualize the diffusion process at different timesteps

    Args:
        original_data: Original clean data
        noisy_samples: List of noisy samples at different timesteps
        timesteps: List of timestep values
        figsize: Figure size
    """

    num_samples = len(noisy_samples)
    fig, axes = plt.subplots(1, num_samples + 1, figsize=figsize)

    # Plot original
    axes[0].imshow(original_data.squeeze(), cmap='viridis', aspect='auto')
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Plot noisy samples
    for i, (sample, t) in enumerate(zip(noisy_samples, timesteps)):
        axes[i + 1].imshow(sample.squeeze(), cmap='viridis', aspect='auto')
        axes[i + 1].set_title(f't = {t}')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


def calculate_molecular_diversity(smiles_list: List[str],
                                  fingerprint_type: str = 'morgan') -> Dict:
    """
    Calculate diversity metrics for a set of molecules

    Args:
        smiles_list: List of SMILES strings
        fingerprint_type: Type of fingerprint ('morgan', 'topological')

    Returns:
        Dictionary with diversity metrics
    """
    from rdkit.Chem import rdMolDescriptors
    from rdkit import DataStructs

    valid_mols = []
    fingerprints = []

    # Generate fingerprints
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mols.append(mol)

                if fingerprint_type == 'morgan':
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                elif fingerprint_type == 'topological':
                    fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)
                else:
                    raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")

                fingerprints.append(fp)

        except Exception as e:
            logger.debug(f"Error processing molecule {smiles}: {e}")
            continue

    if len(fingerprints) < 2:
        return {'error': 'Not enough valid molecules for diversity calculation'}

    # Calculate pairwise similarities
    similarities = []
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarities.append(sim)

    similarities = np.array(similarities)

    # Calculate diversity metrics
    diversity_metrics = {
        'num_molecules': len(valid_mols),
        'mean_similarity': similarities.mean(),
        'std_similarity': similarities.std(),
        'min_similarity': similarities.min(),
        'max_similarity': similarities.max(),
        'mean_diversity': 1.0 - similarities.mean(),
        'unique_molecules': len(set(smiles_list)),
        'uniqueness_rate': len(set(smiles_list)) / len(smiles_list) if len(smiles_list) > 0 else 0.0
    }

    return diversity_metrics