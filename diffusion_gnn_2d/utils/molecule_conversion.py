import torch
import numpy as np
from rdkit import Chem
from typing import List, Optional, Tuple


def features_to_atom_types(atom_features: torch.Tensor) -> List[str]:
    """Convert one-hot atom features to element symbols"""
    elements = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H']
    atom_indices = torch.argmax(atom_features[:, :10], dim=1)
    return [elements[i.item()] for i in atom_indices]


def create_simple_molecule(atom_types: List[str], connectivity_threshold: float = 0.8) -> Optional[str]:
    """Create SMILES from atom types using simple rules"""
    if not atom_types or len(atom_types) < 2:
        return None

    # Simple heuristics for small molecules
    carbon_count = atom_types.count('C')
    oxygen_count = atom_types.count('O')
    nitrogen_count = atom_types.count('N')

    if carbon_count >= 2:
        if oxygen_count >= 1:
            return 'CCO'  # Ethanol-like
        elif nitrogen_count >= 1:
            return 'CCN'  # Ethylamine-like
        else:
            return 'CC'  # Ethane-like
    elif carbon_count == 1:
        if oxygen_count >= 1:
            return 'CO'  # Methanol-like
        else:
            return 'C'  # Methane-like

    return 'C'  # Default fallback


def validate_smiles(smiles: str) -> bool:
    """Check if SMILES is valid using RDKit"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def calculate_molecular_properties(smiles: str) -> dict:
    """Calculate molecular properties"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    from rdkit.Chem import Descriptors
    return {
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol)
    }