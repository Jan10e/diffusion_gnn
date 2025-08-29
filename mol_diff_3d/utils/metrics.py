from typing import List, Dict
from rdkit import Chem
from collections import Counter
import numpy as np
from rdkit.Chem import AllChem

def validate_smiles(smiles: str) -> bool:
    """Check if SMILES is valid using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def calculate_validity_rate(smiles_list: List[str]) -> float:
    """Calculate the fraction of valid molecules in a generated list of SMILES."""
    if not smiles_list:
        return 0.0
    valid_count = sum(1 for smi in smiles_list if validate_smiles(smi))
    return valid_count / len(smiles_list)

def calculate_uniqueness_rate(smiles_list: List[str]) -> float:
    """Calculate the fraction of unique molecules in a generated list of SMILES."""
    if not smiles_list:
        return 0.0
    unique_smiles = set(smiles_list)
    return len(unique_smiles) / len(smiles_list)

def calculate_novelty_rate(generated_smiles: List[str], training_smiles: List[str]) -> float:
    """Calculate the fraction of generated molecules that are novel (not in the training set)."""
    if not generated_smiles:
        return 0.0
    training_set = set(training_smiles)
    generated_set = set(generated_smiles)
    novel_molecules = generated_set - training_set
    return len(novel_molecules) / len(generated_set)

def calculate_diversity_rate(smiles_list: List[str]) -> float:
    """
    Calculate the diversity of a list of generated SMILES strings.
    This is a simplified approach using a set of canonical smiles.
    """
    if not smiles_list:
        return 0.0
    canonical_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True) for smi in smiles_list if validate_smiles(smi)]
    if not canonical_smiles:
        return 0.0
    unique_smiles = set(canonical_smiles)
    return len(unique_smiles) / len(canonical_smiles)