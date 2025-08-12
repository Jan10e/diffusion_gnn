"""
Data processing utilities for molecular diffusion models.
This module handles DeepChem dataset loading and conversion to PyTorch Geometric format.
"""

import torch
from torch_geometric.data import Data, DataLoader
import deepchem as dc
import numpy as np
from rdkit import Chem
from typing import List, Tuple, Optional, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepChemMolecularDataset:
    """Convert DeepChem datasets to PyTorch Geometric format for GNN diffusion"""

    def __init__(self, dataset_name='tox21', max_atoms=50, cache_graphs=True):
        self.max_atoms = max_atoms
        self.dataset_name = dataset_name
        self.cache_graphs = cache_graphs
        self.dataset = self._load_dataset()
        self._graph_cache = {} if cache_graphs else None

        logger.info(f"Loaded {dataset_name} dataset with {len(self.dataset)} molecules")

    def _load_dataset(self):
        """Load DeepChem dataset"""
        logger.info(f"Loading DeepChem dataset: {self.dataset_name}")

        if self.dataset_name == 'tox21':
            tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='Raw')
        elif self.dataset_name == 'zinc15':
            tasks, datasets, transformers = dc.molnet.load_zinc15(featurizer='Raw')
        elif self.dataset_name == 'qm9':
            tasks, datasets, transformers = dc.molnet.load_qm9(featurizer='Raw')
        elif self.dataset_name == 'esol':
            tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='Raw')
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")

        train, valid, test = datasets
        return train

    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES string to PyTorch Geometric Data object"""

        # Check cache first to see whether graph already exists
        if self._graph_cache is not None and smiles in self._graph_cache:
            return self._graph_cache[smiles]

        try:
            mol = Chem.MolFromSmiles(smiles)  # Parse SMILES to RDKit molecule
            if mol is None:
                logger.warning(f"Could not parse SMILES: {smiles}")
                return None

            if mol.GetNumAtoms() > self.max_atoms:
                logger.debug(f"Molecule too large: {mol.GetNumAtoms()} atoms (max: {self.max_atoms})")
                return None

            # Get atom features. Extract 37 features for each atom
            atom_features = []
            for atom in mol.GetAtoms():
                features = self._get_atom_features(atom)
                atom_features.append(features)

            # Get bond indices and features
            edge_indices = []
            edge_features = []

            # For each bond, extract 10 features
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                bond_feat = self._get_bond_features(bond)

                # Add both directions for undirected graph
                edge_indices.extend([[i, j], [j, i]])
                edge_features.extend([bond_feat, bond_feat])

            # Convert to tensors
            x = torch.tensor(atom_features, dtype=torch.float)

            if len(edge_indices) > 0:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_features, dtype=torch.float)
            else:
                # Handle single atom molecules
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, self._get_bond_feature_dim()), dtype=torch.float)

            # Create PyTorch Geometric data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data.smiles = smiles
            data.num_atoms = mol.GetNumAtoms()

            # Cache the result
            if self._graph_cache is not None:
                self._graph_cache[smiles] = data

            return data

        except Exception as e:
            logger.error(f"Error processing SMILES {smiles}: {e}")
            return None

    def _get_atom_features(self, atom) -> List[float]:
        """Extract comprehensive atom features for GNN"""
        features = []

        # Atomic number (one-hot encoded for common elements)
        atomic_nums = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I
        atomic_num = atom.GetAtomicNum()
        features.extend([1.0 if atomic_num == num else 0.0 for num in atomic_nums])
        features.append(1.0 if atomic_num not in atomic_nums else 0.0)  # Other

        # Degree (0-6)
        degree = atom.GetDegree()
        for i in range(7):
            features.append(1.0 if degree == i else 0.0)

        # Formal charge (-2 to +2)
        formal_charge = atom.GetFormalCharge()
        for charge in [-2, -1, 0, 1, 2]:
            features.append(1.0 if formal_charge == charge else 0.0)
        features.append(1.0 if formal_charge not in [-2, -1, 0, 1, 2] else 0.0)

        # Hybridization
        hyb_types = [Chem.rdchem.HybridizationType.S,
                     Chem.rdchem.HybridizationType.SP,
                     Chem.rdchem.HybridizationType.SP2,
                     Chem.rdchem.HybridizationType.SP3]
        hyb = atom.GetHybridization()
        features.extend([1.0 if hyb == h else 0.0 for h in hyb_types])
        features.append(1.0 if hyb not in hyb_types else 0.0)  # Other

        # Aromaticity
        features.append(1.0 if atom.GetIsAromatic() else 0.0)

        # Number of hydrogens (0-4)
        num_hs = atom.GetTotalNumHs()
        for i in range(5):
            features.append(1.0 if num_hs == i else 0.0)
        features.append(1.0 if num_hs >= 5 else 0.0)

        # Chirality
        features.append(1.0 if atom.HasProp('_ChiralityPossible') else 0.0)

        return features

    def _get_bond_features(self, bond) -> List[float]:
        """Extract comprehensive bond features"""
        features = []

        # Bond type
        bond_types = [Chem.rdchem.BondType.SINGLE,
                      Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE,
                      Chem.rdchem.BondType.AROMATIC]
        bond_type = bond.GetBondType()
        features.extend([1.0 if bond_type == bt else 0.0 for bt in bond_types])

        # Conjugation
        features.append(1.0 if bond.GetIsConjugated() else 0.0)

        # Ring membership
        features.append(1.0 if bond.IsInRing() else 0.0)

        # Stereo configuration
        stereo_types = [Chem.rdchem.BondStereo.STEREONONE,
                        Chem.rdchem.BondStereo.STEREOANY,
                        Chem.rdchem.BondStereo.STEREOZ,
                        Chem.rdchem.BondStereo.STEREOE]
        stereo = bond.GetStereo()
        features.extend([1.0 if stereo == st else 0.0 for st in stereo_types])

        return features

    def _get_atom_feature_dim(self) -> int:
        """Get dimension of atom features"""
        # atomic_num(11) + degree(7) + formal_charge(6) + hybridization(5) + aromatic(1) + num_hs(6) + chirality(1)
        return 11 + 7 + 6 + 5 + 1 + 6 + 1

    def _get_bond_feature_dim(self) -> int:
        """Get dimension of bond features"""
        # bond_type(4) + conjugated(1) + in_ring(1) + stereo(4)
        return 4 + 1 + 1 + 4

    def create_dataloader(self, batch_size=32, shuffle=True, num_workers=0, max_samples=None):
        """Create DataLoader for the dataset"""
        graphs = []
        failed_conversions = 0

        # Determine how many samples to process
        total_samples = min(len(self.dataset), max_samples) if max_samples else len(self.dataset)

        logger.info(f"Converting {total_samples} molecules to graphs...")

        # Process SMILES from DeepChem dataset
        for i in range(total_samples):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{total_samples} molecules...")

            smiles = self.dataset.ids[i]
            graph = self.smiles_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)
            else:
                failed_conversions += 1

        success_rate = len(graphs) / total_samples * 100
        logger.info(f"Successfully converted {len(graphs)}/{total_samples} molecules ({success_rate:.1f}%)")
        logger.info(f"Failed conversions: {failed_conversions}")

        if len(graphs) == 0:
            raise ValueError("No valid graphs created from dataset")

        return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def get_sample_smiles(self, num_samples=10):
        """Get sample SMILES strings for inspection"""
        return [self.dataset.ids[i] for i in range(min(num_samples, len(self.dataset)))]

    def get_dataset_info(self):
        """Get information about the dataset"""
        return {
            'name': self.dataset_name,
            'size': len(self.dataset),
            'max_atoms': self.max_atoms,
            'atom_feature_dim': self._get_atom_feature_dim(),
            'bond_feature_dim': self._get_bond_feature_dim()
        }


def create_molecular_datasets(dataset_names=['tox21'], max_atoms=50, test_split=0.1):
    """Create multiple molecular datasets for training and testing"""
    datasets = {}

    for name in dataset_names:
        logger.info(f"Creating dataset: {name}")
        dataset = DeepChemMolecularDataset(name, max_atoms)
        datasets[name] = dataset

    return datasets


def analyze_dataset_statistics(dataset: DeepChemMolecularDataset, num_samples=1000):
    """Analyze dataset statistics for quality assessment"""
    from rdkit.Chem import Descriptors

    stats = {
        'num_atoms': [],
        'num_bonds': [],
        'molecular_weight': [],
        'valid_molecules': 0,
        'invalid_molecules': 0
    }

    sample_size = min(num_samples, len(dataset.dataset))

    for i in range(sample_size):
        smiles = dataset.dataset.ids[i]
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            stats['num_atoms'].append(mol.GetNumAtoms())
            stats['num_bonds'].append(mol.GetNumBonds())
            stats['molecular_weight'].append(Descriptors.MolWt(mol))
            stats['valid_molecules'] += 1
        else:
            stats['invalid_molecules'] += 1

    # Convert lists to numpy arrays for easier statistics
    for key in ['num_atoms', 'num_bonds', 'molecular_weight']:
        if stats[key]:
            array = np.array(stats[key])
            stats[f'{key}_mean'] = array.mean()
            stats[f'{key}_std'] = array.std()
            stats[f'{key}_min'] = array.min()
            stats[f'{key}_max'] = array.max()

    return stats