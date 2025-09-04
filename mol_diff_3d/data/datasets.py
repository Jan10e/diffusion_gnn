"""
Enhanced QM9 dataset loader that extracts explicit bond features.
Key improvement: Proper bond type extraction and one-hot encoding.
"""
import torch
from torch_geometric.data import Data, DataLoader
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from typing import List, Tuple, Optional, Dict
import logging
import numpy as np

# Required for 3D molecular operations
from rdkit.Chem import AllChem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedQm9MolecularDataset:
    """
    Enhanced QM9 dataset loader that extracts explicit bond information.
    This addresses the atom-bond inconsistency problem by providing proper bond features.
    """

    def __init__(self, max_atoms=25, cache_graphs=True):
        self.max_atoms = max_atoms
        self.cache_graphs = cache_graphs
        self.dataset = self._load_dataset()
        self._graph_cache = {} if cache_graphs else None

        # Bond type mapping: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic
        self.bond_types = {
            Chem.rdchem.BondType.SINGLE: 1,
            Chem.rdchem.BondType.DOUBLE: 2,
            Chem.rdchem.BondType.TRIPLE: 3,
            Chem.rdchem.BondType.AROMATIC: 4
        }

        logger.info(f"Loaded {type(self.dataset).__name__} with {len(self.dataset)} molecules")

    def _load_dataset(self):
        """Load the QM9 dataset from DeepChem."""
        logger.info("Loading DeepChem dataset: QM9")
        tasks, datasets, transformers = dc.molnet.load_qm9(featurizer='Raw')
        train, valid, test = datasets
        return train

    def _get_bond_features(self, bond: Chem.Bond) -> List[float]:
        """
        Extract bond features for explicit bond modeling.
        """
        features = []

        # Bond type (one-hot encoded)
        bond_type = self.bond_types.get(bond.GetBondType(), 0)
        bond_type_onehot = [0.0] * 5  # 5 bond types including "no bond"
        if bond_type < len(bond_type_onehot):
            bond_type_onehot[bond_type] = 1.0
        features.extend(bond_type_onehot)

        # Bond properties
        features.append(float(bond.GetIsConjugated()))
        features.append(float(bond.IsInRing()))

        return features

    def _get_bond_feature_dim(self) -> int:
        """Get dimension of bond features (5 types + 2 properties)"""
        return 7

    def _create_complete_graph_with_bonds(self, mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a complete graph but with proper bond type labels.
        This is essential for learning chemical connectivity.
        """
        num_atoms = mol.GetNumAtoms()

        # Create all possible edges (complete graph)
        edge_indices = []
        edge_features = []

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):  # Avoid self-loops and duplicates
                # Add both directions for undirected graph
                edge_indices.extend([[i, j], [j, i]])

                # Check if there's an actual bond
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    # Real bond - extract features
                    bond_feat = self._get_bond_features(bond)
                else:
                    # No bond - use "no bond" features
                    bond_feat = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # "no bond" is type 0

                # Add same features for both directions
                edge_features.extend([bond_feat, bond_feat])

        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, self._get_bond_feature_dim()), dtype=torch.float)

        return edge_index, edge_attr

    def mol_to_graph(self, mol: Chem.Mol) -> Optional[Data]:
        """
        Convert RDKit Mol to PyTorch Geometric Data with explicit bond features.
        """
        try:
            if mol is None:
                return None

            # Check cache first
            if self._graph_cache is not None and mol.HasProp('_Name') and mol.GetProp('_Name') in self._graph_cache:
                return self._graph_cache[mol.GetProp('_Name')]

            if mol.GetNumAtoms() > self.max_atoms:
                logger.debug(f"Molecule too large: {mol.GetNumAtoms()} atoms (max: {self.max_atoms})")
                return None

            # Get atom features
            atom_features = [self._get_atom_features(atom) for atom in mol.GetAtoms()]

            # Get 3D positions
            if mol.GetNumConformers() == 0:
                logger.warning("Molecule has no conformer, skipping.")
                return None

            conformer = mol.GetConformer(0)
            positions = conformer.GetPositions()

            # Get bond information (complete graph with bond type labels)
            edge_index, edge_attr = self._create_complete_graph_with_bonds(mol)

            # Convert to tensors
            x = torch.tensor(atom_features, dtype=torch.float)
            pos = torch.tensor(positions, dtype=torch.float)

            # Create PyTorch Geometric data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
            data.num_atoms = mol.GetNumAtoms()

            # Add molecular properties for evaluation
            data.num_bonds = sum(1 for bond in mol.GetBonds())
            data.smiles = Chem.MolToSmiles(mol)

            # Cache the result
            if self.cache_graphs and mol.HasProp('_Name'):
                self._graph_cache[mol.GetProp('_Name')] = data

            return data

        except Exception as e:
            logger.error(f"Error processing molecule: {e}")
            return None

    def _get_atom_features(self, atom: Chem.Atom) -> List[float]:
        """
        Extract atom features with better chemical representation.
        """
        features = []

        # Atomic number one-hot encoded (expanded set)
        atomic_nums = [6, 7, 8, 9, 1, 17, 35, 53, 15, 16]  # C, N, O, F, H, Cl, Br, I, P, S
        features.extend([1.0 if atom.GetAtomicNum() == num else 0.0 for num in atomic_nums])

        # Chemical properties
        features.append(float(atom.GetTotalNumHs()))
        features.append(float(atom.GetDegree()))
        features.append(float(atom.GetFormalCharge()))
        features.append(float(atom.GetHybridization().real if atom.GetHybridization() else 0))
        features.append(float(atom.GetIsAromatic()))

        return features

    def _get_atom_feature_dim(self) -> int:
        """Get dimension of atom features (10 elements + 5 properties)"""
        return 15

    def create_dataloader(self, batch_size=32, shuffle=True, num_workers=0, max_samples=None):
        """Create DataLoader with bond features."""
        graphs = []
        failed_conversions = 0

        total_samples = min(len(self.dataset), max_samples) if max_samples else len(self.dataset)

        logger.info(f"Converting {total_samples} molecules to graphs with bond features...")

        for i in range(total_samples):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{total_samples} molecules...")

            mol = self.dataset.X[i]
            graph = self.mol_to_graph(mol)

            if graph is not None:
                graphs.append(graph)
            else:
                failed_conversions += 1

        success_rate = len(graphs) / total_samples * 100
        logger.info(f"Successfully converted {len(graphs)}/{total_samples} molecules ({success_rate:.1f}%)")
        logger.info(f"Failed conversions: {failed_conversions}")

        if len(graphs) == 0:
            raise ValueError("No valid graphs created from dataset")

        # Log dataset statistics
        self._log_dataset_statistics(graphs)

        return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def _log_dataset_statistics(self, graphs: List[Data]):
        """Log useful statistics about the processed dataset."""
        num_atoms = [g.num_atoms for g in graphs]
        num_bonds = [g.num_bonds for g in graphs]

        logger.info(f"Dataset statistics:")
        logger.info(f"  Average atoms per molecule: {np.mean(num_atoms):.1f}")
        logger.info(f"  Average bonds per molecule: {np.mean(num_bonds):.1f}")
        logger.info(f"  Max atoms: {max(num_atoms)}")
        logger.info(f"  Max bonds: {max(num_bonds)}")

        # Bond type distribution
        bond_types = []
        for g in graphs[:100]:  # Sample first 100 for efficiency
            bond_types.extend(g.edge_attr.argmax(dim=1).tolist())

        bond_type_counts = np.bincount(bond_types)
        logger.info(f"  Bond type distribution: {dict(enumerate(bond_type_counts))}")

    def get_dataset_info(self):
        """Get comprehensive dataset information."""
        return {
            'name': 'qm9_with_bonds',
            'size': len(self.dataset),
            'max_atoms': self.max_atoms,
            'atom_feature_dim': self._get_atom_feature_dim(),
            'bond_feature_dim': self._get_bond_feature_dim(),
            'num_atom_types': 10,
            'num_bond_types': 5,
        }


def create_dummy_complete_graph(num_atoms: int, bond_feature_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to create a complete graph with dummy bond features.
    Useful for generation when starting from noise.
    """
    # Create complete graph edges
    edge_indices = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            edge_indices.extend([[i, j], [j, i]])

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        # Initialize with "no bond" type (index 0)
        edge_attr = torch.zeros((edge_index.shape[1], bond_feature_dim))
        edge_attr[:, 0] = 1.0  # Set "no bond" to 1
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, bond_feature_dim))

    return edge_index, edge_attr