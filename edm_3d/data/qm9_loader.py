import torch
from torch.utils.data import Dataset
import deepchem as dc
import numpy as np


def prepare_qm9_data(max_samples: int = None):
    """
    Load and prepare QM9 dataset for EDM training

    Args:
        max_samples: Limit dataset size for faster training (None for full dataset)

    Returns:
        train_dataset, valid_dataset, test_dataset
    """
    print("Loading QM9 dataset from DeepChem...")

    # Load QM9 with ConformerFeaturizer to get 3D coordinates
    tasks, datasets, transformers = dc.molnet.load_qm9(
        featurizer=dc.feat.ConformerFeaturizer(),
        splitter='random'
    )
    train_data, valid_data, test_data = datasets

    # Limit dataset size if specified
    if max_samples is not None:
        train_data = train_data.select(range(min(max_samples, len(train_data))))
        valid_data = valid_data.select(range(min(max_samples // 5, len(valid_data))))
        test_data = test_data.select(range(min(max_samples // 5, len(test_data))))

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")
    print(f"Test samples: {len(test_data)}")

    # Wrap in custom dataset
    train_dataset = QM9Dataset(train_data)
    valid_dataset = QM9Dataset(valid_data)
    test_dataset = QM9Dataset(test_data)

    return train_dataset, valid_dataset, test_dataset


class QM9Dataset(Dataset):
    """
    PyTorch Dataset wrapper for QM9 molecules
    """

    def __init__(self, deepchem_dataset):
        self.dataset = deepchem_dataset
        self.atom_types_map = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with:
                - coords: [num_atoms, 3] tensor
                - atom_types: [num_atoms] tensor (indices)
                - num_atoms: int
        """
        # Get molecule from DeepChem dataset
        mol_data = self.dataset.X[idx]

        # Extract coordinates and atom types from RDKit molecule
        from rdkit import Chem
        mol = mol_data

        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)

        # Get 3D coordinates
        conf = mol.GetConformer()
        coords = []
        atom_types = []

        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])

            symbol = atom.GetSymbol()
            atom_type = self.atom_types_map.get(symbol, 0)
            atom_types.append(atom_type)

        coords = torch.tensor(coords, dtype=torch.float32)
        atom_types = torch.tensor(atom_types, dtype=torch.long)

        # Center coordinates
        coords = coords - coords.mean(dim=0, keepdim=True)

        return {
            'coords': coords,
            'atom_types': atom_types,
            'num_atoms': len(atom_types)
        }
