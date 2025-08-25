import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import List, Dict, Tuple
import pandas as pd


def calculate_validity_metrics(smiles_list: List[str]) -> Dict:
    """Calculate molecular validity using RDKit"""
    if not smiles_list:
        return {'validity': 0.0, 'valid_count': 0, 'total_count': 0}

    valid_count = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_count += 1

    return {
        'validity': valid_count / len(smiles_list),
        'valid_count': valid_count,
        'total_count': len(smiles_list)
    }


def calculate_novelty_uniqueness(generated_smiles: List[str],
                                 training_smiles: List[str]) -> Dict:
    """Calculate novelty and uniqueness metrics"""
    if not generated_smiles:
        return {'uniqueness': 0.0, 'novelty': 0.0, 'unique_count': 0}

    training_set = set(training_smiles)
    generated_set = set(generated_smiles)

    uniqueness = len(generated_set) / len(generated_smiles)
    novel_molecules = generated_set - training_set
    novelty = len(novel_molecules) / len(generated_set) if generated_set else 0

    return {
        'uniqueness': uniqueness,
        'novelty': novelty,
        'unique_count': len(generated_set),
        'novel_count': len(novel_molecules)
    }


def extract_molecular_properties(smiles_list: List[str]) -> pd.DataFrame:
    """Extract molecular properties using RDKit"""
    properties = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            properties.append({
                'smiles': smiles,
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds()
            })

    return pd.DataFrame(properties)