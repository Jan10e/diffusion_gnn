import torch
from torch.utils.data import Dataset
import deepchem as dc
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_qm9_data(max_samples: Optional[int] = None, test_mode: bool = False):
    """
    Load and prepare QM9 dataset for EDM training

    Args:
        max_samples: Limit dataset size for faster training (None for full dataset)
        test_mode: If True, use very small subset for quick testing

    Returns:
        train_dataset, valid_dataset, test_dataset (QM9Dataset objects)
    """
    logger.info("Loading QM9 dataset from DeepChem...")

    # Load QM9 - using 'Raw' featurizer to get SMILES strings
    # We'll generate 3D coordinates ourselves using RDKit
    tasks, datasets, transformers = dc.molnet.load_qm9(
        featurizer='Raw',
        splitter='random',
        reload=False  # Set to True if you want to force reload
    )

    train_data, valid_data, test_data = datasets

    # Apply sample limit if specified
    if test_mode:
        max_samples = 100
        logger.info("TEST MODE: Using only 100 samples")

    if max_samples is not None:
        train_size = min(max_samples, len(train_data))
        valid_size = min(max_samples // 5, len(valid_data))
        test_size = min(max_samples // 5, len(test_data))

        # Create subset indices
        train_indices = list(range(train_size))
        valid_indices = list(range(valid_size))
        test_indices = list(range(test_size))

        # Select subsets
        train_data = train_data.select(train_indices)
        valid_data = valid_data.select(valid_indices)
        test_data = test_data.select(test_indices)

    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(valid_data)}")
    logger.info(f"Test samples: {len(test_data)}")

    # Wrap in custom PyTorch datasets
    train_dataset = QM9Dataset(train_data, split='train')
    valid_dataset = QM9Dataset(valid_data, split='valid')
    test_dataset = QM9Dataset(test_data, split='test')

    return train_dataset, valid_dataset, test_dataset


class QM9Dataset(Dataset):
    """
    PyTorch Dataset wrapper for QM9 molecules
    Converts SMILES to 3D coordinates using RDKit
    """

    def __init__(self, deepchem_dataset, split='train'):
        """
        Args:
            deepchem_dataset: DeepChem dataset object
            split: 'train', 'valid', or 'test'
        """
        self.dataset = deepchem_dataset
        self.split = split

        # Atom type mapping for QM9 (H, C, N, O, F)
        self.atom_types_map = {
            'H': 0,
            'C': 1,
            'N': 2,
            'O': 3,
            'F': 4
        }

        # Reverse mapping
        self.idx_to_atom = {v: k for k, v in self.atom_types_map.items()}

        # Cache for processed molecules
        self._cache = {}

        logger.info(f"Initialized QM9Dataset ({split}) with {len(self)} molecules")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with:
                - coords: [num_atoms, 3] tensor of 3D coordinates
                - atom_types: [num_atoms] tensor of atom type indices
                - num_atoms: int
                - smiles: str (for reference)
        """
        # Check cache
        if idx in self._cache:
            return self._cache[idx]

        # Get SMILES from DeepChem dataset
        smiles = self.dataset.ids[idx]

        # Convert SMILES to 3D molecule
        mol_data = self._smiles_to_3d(smiles)

        if mol_data is None:
            # If conversion fails, try next molecule
            logger.warning(f"Failed to process molecule {idx}: {smiles}")
            # Return a simple fallback (single carbon atom)
            return self._get_fallback_molecule()

        # Cache the result
        self._cache[idx] = mol_data

        return mol_data

    def _smiles_to_3d(self, smiles: str) -> Optional[dict]:
        """
        Convert SMILES string to 3D coordinates using RDKit

        Args:
            smiles: SMILES string

        Returns:
            Dictionary with coords, atom_types, num_atoms, smiles
        """
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Add hydrogens (important for QM9)
            mol = Chem.AddHs(mol)

            # Check atom types are valid (only H, C, N, O, F for QM9)
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in self.atom_types_map:
                    logger.debug(f"Invalid atom type {atom.GetSymbol()} in {smiles}")
                    return None

            # Generate 3D coordinates using ETKDG
            success = AllChem.EmbedMolecule(mol, randomSeed=42)
            if success == -1:
                # Try with different parameters
                success = AllChem.EmbedMolecule(
                    mol,
                    randomSeed=42,
                    useRandomCoords=True,
                    maxAttempts=10
                )
                if success == -1:
                    logger.debug(f"Could not generate 3D coords for {smiles}")
                    return None

            # Optimize geometry (optional but recommended)
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except:
                pass  # Optimization failed, but we still have coordinates

            # Extract coordinates and atom types
            conf = mol.GetConformer()
            coords = []
            atom_types = []

            for atom in mol.GetAtoms():
                # Get 3D position
                pos = conf.GetAtomPosition(atom.GetIdx())
                coords.append([pos.x, pos.y, pos.z])

                # Get atom type index
                symbol = atom.GetSymbol()
                atom_type = self.atom_types_map[symbol]
                atom_types.append(atom_type)

            # Convert to tensors
            coords = torch.tensor(coords, dtype=torch.float32)
            atom_types = torch.tensor(atom_types, dtype=torch.long)

            # Center coordinates (important for E(3) equivariance)
            coords = coords - coords.mean(dim=0, keepdim=True)

            return {
                'coords': coords,
                'atom_types': atom_types,
                'num_atoms': len(atom_types),
                'smiles': smiles
            }

        except Exception as e:
            logger.error(f"Error processing SMILES {smiles}: {e}")
            return None

    def _get_fallback_molecule(self):
        """
        Return a simple fallback molecule (methane CH4)
        Used when molecule processing fails
        """
        # Methane: 1 carbon + 4 hydrogens
        coords = torch.tensor([
            [0.0, 0.0, 0.0],  # C
            [1.09, 0.0, 0.0],  # H
            [-0.36, 1.03, 0.0],  # H
            [-0.36, -0.51, 0.89],  # H
            [-0.36, -0.51, -0.89]  # H
        ], dtype=torch.float32)

        atom_types = torch.tensor([1, 0, 0, 0, 0], dtype=torch.long)  # C, H, H, H, H

        # Center
        coords = coords - coords.mean(dim=0, keepdim=True)

        return {
            'coords': coords,
            'atom_types': atom_types,
            'num_atoms': 5,
            'smiles': 'C'
        }

    def get_atom_decoder(self):
        """Return mapping from indices to atom symbols"""
        return self.idx_to_atom

    def get_statistics(self):
        """
        Compute dataset statistics
        Returns dict with mean/std/min/max for num_atoms
        """
        num_atoms_list = []

        sample_size = min(1000, len(self))
        for i in range(sample_size):
            item = self[i]
            num_atoms_list.append(item['num_atoms'])

        num_atoms = np.array(num_atoms_list)

        return {
            'num_atoms_mean': num_atoms.mean(),
            'num_atoms_std': num_atoms.std(),
            'num_atoms_min': num_atoms.min(),
            'num_atoms_max': num_atoms.max(),
            'num_samples': len(self)
        }


