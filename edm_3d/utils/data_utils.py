import torch
from torch.utils.data import Dataset
import deepchem as dc
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
from typing import Optional, Tuple

from edm_3d.data.qm9_loader import prepare_qm9_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def test_qm9_loading():
    """
    Test function to verify QM9 loading works correctly
    """
    logger.info("Testing QM9 data loading...")

    # Load small subset
    train_data, val_data, test_data = prepare_qm9_data(max_samples=10)

    # Test first sample
    sample = train_data[0]

    logger.info(f"Sample molecule:")
    logger.info(f"  SMILES: {sample['smiles']}")
    logger.info(f"  Num atoms: {sample['num_atoms']}")
    logger.info(f"  Coords shape: {sample['coords'].shape}")
    logger.info(f"  Atom types shape: {sample['atom_types'].shape}")
    logger.info(f"  Atom types: {sample['atom_types'].tolist()}")

    # Get statistics
    stats = train_data.get_statistics()
    logger.info(f"\nDataset statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    logger.info("\n✓ QM9 loading test passed!")

    return train_data, val_data, test_data


if __name__ == "__main__":
    # Run test
    test_qm9_loading()