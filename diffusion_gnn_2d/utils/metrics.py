from typing import List, Dict
from collections import Counter
import numpy as np
from difflib import SequenceMatcher   # Simple diversity based on string edit distance


def calculate_validity_rate(smiles_list: List[str]) -> float:
    """Calculate fraction of valid SMILES"""
    from .molecule_conversion import validate_smiles
    valid_count = sum(1 for smi in smiles_list if validate_smiles(smi))
    return valid_count / len(smiles_list) if smiles_list else 0.0


def calculate_uniqueness_rate(smiles_list: List[str]) -> float:
    """Calculate fraction of unique molecules"""
    unique_smiles = set(smiles_list)
    return len(unique_smiles) / len(smiles_list) if smiles_list else 0.0


def calculate_novelty_rate(generated_smiles: List[str], training_smiles: List[str]) -> float:
    """Calculate fraction of novel molecules"""
    training_set = set(training_smiles)
    generated_set = set(generated_smiles)
    novel_molecules = generated_set - training_set
    return len(novel_molecules) / len(generated_set) if generated_set else 0.0


def calculate_diversity_metrics(smiles_list: List[str]) -> Dict:
    """Calculate molecular diversity metrics"""
    if not smiles_list:
        return {'diversity': 0.0, 'scaffold_diversity': 0.0}

    total_similarity = 0
    comparisons = 0

    for i, smi1 in enumerate(smiles_list):
        for j, smi2 in enumerate(smiles_list[i + 1:], i + 1):
            similarity = SequenceMatcher(None, smi1, smi2).ratio()
            total_similarity += similarity
            comparisons += 1

    diversity = 1 - (total_similarity / comparisons) if comparisons > 0 else 0.0

    return {
        'diversity': diversity,
        'scaffold_diversity': diversity  # Simplified
    }