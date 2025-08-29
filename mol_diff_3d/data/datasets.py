import torch
from torch_geometric.data import Data, DataLoader
import deepchem as dc
from rdkit import Chem
from typing import List, Tuple, Optional, Dict
import logging

# Required for 3D molecular operations
from rdkit.Chem import AllChem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Qm9MolecularDataset:
    """
    Loads the QM9 dataset from DeepChem and converts it to
    PyTorch Geometric format, including 3D coordinates.
    """

    def __init__(self, max_atoms=25, cache_graphs=True):
        self.max_atoms = max_atoms
        self.cache_graphs = cache_graphs
        self.dataset = self._load_dataset()
        self._graph_cache = {} if cache_graphs else None
        
        logger.info(f"Loaded QM9 dataset with {len(self.dataset)} molecules")

    def _load_dataset(self):
        """Load the QM9 dataset from DeepChem."""
        logger.info("Loading DeepChem dataset: QM9")
        # The 'Raw' featurizer preserves the original RDKit Mol objects, which contain 3D coordinates.
        tasks, datasets, transformers = dc.molnet.load_qm9(featurizer='Raw')
        train, valid, test = datasets
        return train

    def mol_to_graph(self, mol: Chem.Mol) -> Optional[Data]:
        """
        Convert an RDKit Mol object (with 3D conformer) to a
        PyTorch Geometric Data object.
        """
        try:
            # Check cache first
            if self._graph_cache is not None and mol.GetProp('_Name') in self._graph_cache:
                return self._graph_cache[mol.GetProp('_Name')]

            if mol.GetNumAtoms() > self.max_atoms:
                logger.debug(f"Molecule too large: {mol.GetNumAtoms()} atoms (max: {self.max_atoms})")
                return None

            # Get atom features
            atom_features = [self._get_atom_features(atom) for atom in mol.GetAtoms()]

            # Get 3D positions from the first conformer
            if mol.GetNumConformers() == 0:
                logger.warning("Molecule has no conformer, skipping.")
                return None
            
            conformer = mol.GetConformer(0)
            positions = conformer.GetPositions()

            # Get bond indices for connectivity
            edge_indices = []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_indices.extend([[i, j], [j, i]])

            # Convert to tensors
            x = torch.tensor(atom_features, dtype=torch.float)
            pos = torch.tensor(positions, dtype=torch.float)

            if len(edge_indices) > 0:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            # Create PyTorch Geometric data object
            # Note: We do not include edge_attr as the E3GNN does not use it.
            data = Data(x=x, edge_index=edge_index, pos=pos)
            data.num_atoms = mol.GetNumAtoms()

            # Cache the result
            if self.cache_graphs:
                self._graph_cache[mol.GetProp('_Name')] = data

            return data

        except Exception as e:
            logger.error(f"Error processing molecule: {e}")
            return None

    def _get_atom_features(self, atom: Chem.Atom) -> List[float]:
        """
        Extract simplified atom features for the diffusion model.
        These features should align with your model's input dimension.
        """
        features = []
        # Atomic number one-hot encoded (up to 10 elements)
        atomic_nums = [6, 7, 8, 9, 1, 17, 35, 53, 15, 16] # C, N, O, F, H, Cl, Br, I, P, S
        features.extend([1.0 if atom.GetAtomicNum() == num else 0.0 for num in atomic_nums])
        
        # Total number of hydrogens (simplified to a single value)
        features.append(float(atom.GetTotalNumHs()))
        
        return features

    def _get_atom_feature_dim(self) -> int:
        """Get dimension of atom features (10 elements + 1 for H count)"""
        return 11

    def create_dataloader(self, batch_size=32, shuffle=True, num_workers=0, max_samples=None):
        """Create DataLoader for the dataset, converting Mol objects to graphs."""
        graphs = []
        failed_conversions = 0

        total_samples = min(len(self.dataset), max_samples) if max_samples else len(self.dataset)

        logger.info(f"Converting {total_samples} molecules to graphs...")

        for i in range(total_samples):
            if i % 5000 == 0:
                logger.info(f"Processed {i}/{total_samples} molecules...")

            # The DeepChem dataset already contains an RDKit Mol object
            mol = self.dataset.mols[i]
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

        return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def get_dataset_info(self):
        """Get information about the dataset."""
        return {
            'name': 'qm9',
            'size': len(self.dataset),
            'max_atoms': self.max_atoms,
            'atom_feature_dim': self._get_atom_feature_dim(),
        }