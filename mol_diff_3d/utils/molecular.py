import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Optional, Tuple, Dict
import numpy as np

# A dictionary of typical covalent radii (in Angstroms)
# Used to infer bond lengths from 3D coordinates
# Source: MolDiff paper or standard chemistry tables
COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39
}


def features_to_atom_types(atom_features: torch.Tensor) -> List[str]:
    """Convert one-hot atom features to element symbols."""
    elements = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H']
    atom_indices = torch.argmax(atom_features, dim=1)
    return [elements[i.item()] for i in atom_indices]


def create_3d_molecule_from_positions(
        atom_types: List[str],
        positions: torch.Tensor,
        bond_length_tolerance: float = 0.5
) -> Optional[Chem.Mol]:
    """
    Creates an RDKit Mol object from atom types and 3D coordinates.
    This is the core new function.
    MolDiff Paper (Section 3.2): The model generates atom positions (p) and types (x).
    We must infer connectivity (A) from these positions.
    """
    mol = Chem.Mol()
    # 1. Add atoms to the molecule
    for atom_type in atom_types:
        atom = Chem.Atom(atom_type)
        mol.AddAtom(atom)

    # 2. Infer bonds based on inter-atomic distances
    # A bond exists if the distance between two atoms is close to the sum of their covalent radii.
    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            # Calculate Euclidean distance between atom i and atom j
            dist = torch.norm(positions[i] - positions[j]).item()

            # Sum of covalent radii for atoms i and j
            covalent_radii_sum = COVALENT_RADII[atom_types[i]] + COVALENT_RADII[atom_types[j]]

            # Check if distance is within tolerance
            if abs(dist - covalent_radii_sum) < bond_length_tolerance:
                # Add a bond if the distance is plausible
                mol.AddBond(i, j, Chem.rdchem.BondType.SINGLE)

    # 3. Create 3D conformer and assign positions
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(i, positions[i].tolist())
    mol.AddConformer(conformer)

    # 4. Sanitize and validate the molecule
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Chem.rdkit.Chem.rdchem.KekulizeException:
        return None


def validate_molecule(mol: Optional[Chem.Mol]) -> bool:
    """Check if an RDKit Mol object is valid."""
    return mol is not None and mol.GetNumAtoms() > 0 and len(mol.GetBonds()) > 0


def calculate_molecular_properties(mol: Chem.Mol) -> dict:
    """Calculate molecular properties from a valid RDKit Mol object."""
    from rdkit.Chem import Descriptors

    if not validate_molecule(mol):
        return {}

    return {
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol)
    }