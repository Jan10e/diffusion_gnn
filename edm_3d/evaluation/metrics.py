import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def compute_validity(coords, atom_types):
    """
    Compute percentage of valid molecules
    """
    valid_count = 0
    total = len(coords)

    for i in range(total):
        if is_valid_molecule(coords[i], atom_types[i]):
            valid_count += 1

    return valid_count / total


def compute_uniqueness(coords, atom_types):
    """
    Compute percentage of unique molecules
    """
    unique_smiles = set()
    total = len(coords)

    for i in range(total):
        smiles = coords_to_smiles(coords[i], atom_types[i])
        if smiles:
            unique_smiles.add(smiles)

    return len(unique_smiles) / total if total > 0 else 0.0


def compute_novelty(coords, atom_types, training_set):
    """
    Compute percentage of novel molecules (not in training set)
    """
    training_smiles = set()
    for item in training_set:
        smiles = coords_to_smiles(item['coords'], item['atom_types'])
        if smiles:
            training_smiles.add(smiles)

    novel_count = 0
    total = len(coords)

    for i in range(total):
        smiles = coords_to_smiles(coords[i], atom_types[i])
        if smiles and smiles not in training_smiles:
            novel_count += 1

    return novel_count / total if total > 0 else 0.0


def compute_stability(coords, atom_types):
    """
    Compute percentage of geometrically stable molecules
    """
    stable_count = 0
    total = len(coords)

    for i in range(total):
        if is_stable_geometry(coords[i], atom_types[i]):
            stable_count += 1

    return stable_count / total


def is_valid_molecule(coords, atom_types):
    """Check if molecule is chemically valid"""
    try:
        mol = coords_to_mol(coords, atom_types)
        if mol is None:
            return False
        Chem.SanitizeMol(mol)
        return True
    except:
        return False


def is_stable_geometry(coords, atom_types):
    """Check if molecular geometry is stable (no atomic clashes)"""
    coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords

    # Check for atomic clashes (atoms too close)
    for i in range(len(coords_np)):
        for j in range(i + 1, len(coords_np)):
            dist = np.linalg.norm(coords_np[i] - coords_np[j])
            if dist < 0.5:  # Too close (less than 0.5 Angstroms)
                return False

    return True


def coords_to_smiles(coords, atom_types):
    """Convert coordinates and atom types to SMILES string"""
    try:
        mol = coords_to_mol(coords, atom_types)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except:
        return None


def coords_to_mol(coords, atom_types):
    """Convert coordinates and atom types to RDKit molecule"""
    try:
        atom_map = {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F'}

        coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords
        atom_types_np = atom_types.cpu().numpy() if torch.is_tensor(atom_types) else atom_types

        mol = Chem.RWMol()

        # Add atoms
        for atom_type in atom_types_np:
            atom_symbol = atom_map.get(int(atom_type), 'C')
            atom = Chem.Atom(atom_symbol)
            mol.AddAtom(atom)

        # Add bonds based on distance
        for i in range(len(coords_np)):
            for j in range(i + 1, len(coords_np)):
                dist = np.linalg.norm(coords_np[i] - coords_np[j])
                if 0.8 < dist < 1.8:  # Typical bond length range
                    mol.AddBond(i, j, Chem.BondType.SINGLE)

        return mol.GetMol()
    except:
        return None


def extract_molecular_properties(dataset):
    """Extract properties from training dataset"""
    properties = {
        'num_atoms': [],
        'num_bonds': [],
        'avg_bond_length': [],
        'molecular_weight': [],
        'radius_of_gyration': [],
    }

    for i in range(min(1000, len(dataset))):
        mol_data = dataset[i]
        coords = mol_data['coords'].cpu().numpy()
        atom_types = mol_data['atom_types'].cpu().numpy()

        properties['num_atoms'].append(len(coords))

        # Compute bonds and bond lengths
        bonds = []
        bond_lengths = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                if 0.8 < dist < 1.8:
                    bonds.append((i, j))
                    bond_lengths.append(dist)

        properties['num_bonds'].append(len(bonds))
        properties['avg_bond_length'].append(np.mean(bond_lengths) if bond_lengths else 0)

        # Radius of gyration
        centroid = coords.mean(axis=0)
        rog = np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1)))
        properties['radius_of_gyration'].append(rog)

        # Molecular weight (approximate)
        weights = {0: 1.0, 1: 12.0, 2: 14.0, 3: 16.0, 4: 19.0}
        mw = sum(weights.get(int(at), 12.0) for at in atom_types)
        properties['molecular_weight'].append(mw)

    return properties


def extract_molecular_properties_from_tensors(coords, atom_types):
    """Extract properties from generated molecules"""
    properties = {
        'num_atoms': [],
        'num_bonds': [],
        'avg_bond_length': [],
        'molecular_weight': [],
        'radius_of_gyration': [],
    }

    coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords
    atoms_np = atom_types.cpu().numpy() if torch.is_tensor(atom_types) else atom_types

    for mol_coords, mol_atoms in zip(coords_np, atoms_np):
        properties['num_atoms'].append(len(mol_coords))

        bonds = []
        bond_lengths = []
        for i in range(len(mol_coords)):
            for j in range(i + 1, len(mol_coords)):
                dist = np.linalg.norm(mol_coords[i] - mol_coords[j])
                if 0.8 < dist < 1.8:
                    bonds.append((i, j))
                    bond_lengths.append(dist)

        properties['num_bonds'].append(len(bonds))
        properties['avg_bond_length'].append(np.mean(bond_lengths) if bond_lengths else 0)

        centroid = mol_coords.mean(axis=0)
        rog = np.sqrt(np.mean(np.sum((mol_coords - centroid) ** 2, axis=1)))
        properties['radius_of_gyration'].append(rog)

        weights = {0: 1.0, 1: 12.0, 2: 14.0, 3: 16.0, 4: 19.0}
        mw = sum(weights.get(int(at), 12.0) for at in mol_atoms)
        properties['molecular_weight'].append(mw)

    return properties
