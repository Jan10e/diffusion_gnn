"""
Updated molecular utilities with explicit bond handling.
Key improvements for the MolDiff implementation.
"""
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
from typing import List, Optional, Tuple, Dict
import numpy as np

# Extended covalent radii for more elements
COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39
}


def features_to_atom_types(atom_features: torch.Tensor) -> List[str]:
    """Convert atom feature vectors to element symbols."""
    elements = ['C', 'N', 'O', 'F', 'H', 'P', 'S', 'Cl', 'Br', 'I']

    if atom_features.dim() == 2:
        # One-hot or soft assignments
        atom_indices = torch.argmax(atom_features[:, :len(elements)], dim=1)
    else:
        # Single atom
        atom_indices = torch.argmax(atom_features[:len(elements)])
        atom_indices = atom_indices.unsqueeze(0)

    return [elements[i.item()] if i.item() < len(elements) else 'C'
            for i in atom_indices]


def create_3d_molecule_from_positions_and_bonds(
        atom_types: List[str],
        positions: torch.Tensor,
        bond_types: torch.Tensor,
        edge_index: torch.Tensor,
        validate: bool = True
) -> Optional[Chem.Mol]:
    """
    Create RDKit molecule from atoms, positions, and explicit bond information.
    This is the key improvement over distance-based bond inference.
    """
    if not atom_types or len(atom_types) < 1:
        return None

    try:
        # Create blank molecule
        mol = Chem.Mol()
        editable_mol = Chem.EditableMol(mol)

        # Add atoms
        for atom_type in atom_types:
            if atom_type in ['C', 'N', 'O', 'F', 'H', 'P', 'S', 'Cl', 'Br', 'I']:
                atom = Chem.Atom(atom_type)
            else:
                atom = Chem.Atom('C')  # Default to carbon for unknown types
            editable_mol.AddAtom(atom)

        # Bond type mapping
        bond_type_mapping = {
            0: None,  # No bond
            1: rdchem.BondType.SINGLE,
            2: rdchem.BondType.DOUBLE,
            3: rdchem.BondType.TRIPLE,
            4: rdchem.BondType.AROMATIC
        }

        # Add bonds based on explicit bond types
        added_bonds = set()

        for edge_idx in range(edge_index.shape[1]):
            i, j = edge_index[0, edge_idx].item(), edge_index[1, edge_idx].item()

            # Skip invalid indices
            if i >= len(atom_types) or j >= len(atom_types) or i == j:
                continue

            # Get bond type
            if edge_idx < len(bond_types):
                bond_type = bond_types[edge_idx].item()
            else:
                bond_type = 0  # No bond

            # Skip no-bond type and already added bonds
            if bond_type == 0 or (min(i, j), max(i, j)) in added_bonds:
                continue

            # Add bond if valid
            rdkit_bond_type = bond_type_mapping.get(bond_type)
            if rdkit_bond_type is not None:
                try:
                    editable_mol.AddBond(i, j, rdkit_bond_type)
                    added_bonds.add((min(i, j), max(i, j)))
                except Exception:
                    # Skip invalid bonds
                    continue

        # Convert to standard molecule
        mol = editable_mol.GetMol()

        if mol is None:
            return None

        # Add 3D conformer
        if mol.GetNumAtoms() > 0:
            conformer = Chem.Conformer(mol.GetNumAtoms())
            for i in range(mol.GetNumAtoms()):
                if i < len(positions):
                    pos = positions[i].detach().cpu().numpy() if hasattr(positions[i], 'detach') else positions[
                        i].numpy()
                    conformer.SetAtomPosition(i, pos.tolist())
            mol.AddConformer(conformer)

        # Sanitize molecule
        if validate:
            try:
                Chem.SanitizeMol(mol)
            except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException):
                return None

        return mol

    except Exception as e:
        return None


def create_3d_molecule_from_positions(
        atom_types: List[str],
        positions: torch.Tensor,
        bond_length_tolerance: float = 0.5
) -> Optional[Chem.Mol]:
    """
    Legacy function - creates molecule using distance-based bond inference.
    Kept for compatibility but create_3d_molecule_from_positions_and_bonds is preferred.
    """
    if not atom_types or len(atom_types) < 2:
        return None

    try:
        # Create a blank RDKit molecule
        mol = Chem.Mol()
        editable_mol = Chem.EditableMol(mol)

        # Add atoms
        for atom_type in atom_types:
            atom = Chem.Atom(atom_type)
            editable_mol.AddAtom(atom)

        # Infer bonds based on distances
        for i in range(len(atom_types)):
            for j in range(i + 1, len(atom_types)):
                # Calculate distance
                pos_i = positions[i].detach().cpu().numpy() if hasattr(positions[i], 'detach') else positions[i].numpy()
                pos_j = positions[j].detach().cpu().numpy() if hasattr(positions[j], 'detach') else positions[j].numpy()
                dist = np.linalg.norm(pos_i - pos_j)

                # Check bond criteria
                radius_i = COVALENT_RADII.get(atom_types[i], 0.7)
                radius_j = COVALENT_RADII.get(atom_types[j], 0.7)
                covalent_radii_sum = radius_i + radius_j

                # Add bond if within tolerance
                if abs(dist - covalent_radii_sum) < bond_length_tolerance:
                    editable_mol.AddBond(i, j, rdchem.BondType.SINGLE)

        # Convert to standard molecule
        mol = editable_mol.GetMol()

        # Add 3D conformer
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            pos = positions[i].detach().cpu().numpy() if hasattr(positions[i], 'detach') else positions[i].numpy()
            conformer.SetAtomPosition(i, pos.tolist())
        mol.AddConformer(conformer)

        # Sanitize and validate
        try:
            Chem.SanitizeMol(mol)
            return mol
        except Chem.rdchem.KekulizeException:
            return None

    except Exception:
        return None


def validate_molecule(mol: Optional[Chem.Mol]) -> bool:
    """Enhanced molecule validation with chemical rules."""
    if mol is None:
        return False

    try:
        # Basic checks
        if mol.GetNumAtoms() == 0:
            return False

        # Check sanitization
        Chem.SanitizeMol(mol)

        # Check connectivity for multi-atom molecules
        if mol.GetNumAtoms() > 1 and mol.GetNumBonds() == 0:
            return False

        # Check valencies
        for atom in mol.GetAtoms():
            valence = atom.GetTotalValence()
            # Check reasonable valence limits
            max_valences = {'H': 1, 'C': 4, 'N': 5, 'O': 2, 'F': 1, 'P': 5, 'S': 6}
            max_val = max_valences.get(atom.GetSymbol(), 8)
            if valence > max_val:
                return False

        return True

    except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException, ValueError):
        return False


def calculate_molecular_properties(mol: Chem.Mol) -> Dict:
    """Calculate comprehensive molecular properties."""
    from rdkit.Chem import Descriptors, rdMolDescriptors

    if not validate_molecule(mol):
        return {}

    properties = {
        # Basic properties
        'molecular_weight': Descriptors.MolWt(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'num_rings': rdMolDescriptors.CalcNumRings(mol),

        # Chemical properties
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'hbd': Descriptors.NumHDonors(mol),  # Hydrogen bond donors
        'hba': Descriptors.NumHAcceptors(mol),  # Hydrogen bond acceptors

        # Drug-likeness indicators
        'lipinski_violations': 0,
        'qed': None,  # Quantitative Estimate of Drug-likeness
    }

    # Calculate Lipinski violations
    violations = 0
    if properties['molecular_weight'] > 500: violations += 1
    if properties['logp'] > 5: violations += 1
    if properties['hbd'] > 5: violations += 1
    if properties['hba'] > 10: violations += 1
    properties['lipinski_violations'] = violations

    # Calculate QED if possible
    try:
        from rdkit.Chem import QED
        properties['qed'] = QED.qed(mol)
    except ImportError:
        pass

    return properties


def analyze_bond_distribution(molecules: List[Chem.Mol]) -> Dict:
    """Analyze bond type distribution in a set of molecules."""
    bond_counts = {'single': 0, 'double': 0, 'triple': 0, 'aromatic': 0}
    total_bonds = 0

    for mol in molecules:
        if mol is None:
            continue

        for bond in mol.GetBonds():
            total_bonds += 1
            bond_type = bond.GetBondType()

            if bond_type == rdchem.BondType.SINGLE:
                bond_counts['single'] += 1
            elif bond_type == rdchem.BondType.DOUBLE:
                bond_counts['double'] += 1
            elif bond_type == rdchem.BondType.TRIPLE:
                bond_counts['triple'] += 1
            elif bond_type == rdchem.BondType.AROMATIC:
                bond_counts['aromatic'] += 1

    # Convert to percentages
    if total_bonds > 0:
        bond_percentages = {k: (v / total_bonds) * 100 for k, v in bond_counts.items()}
    else:
        bond_percentages = {k: 0.0 for k in bond_counts}

    return {
        'counts': bond_counts,
        'percentages': bond_percentages,
        'total_bonds': total_bonds,
        'molecules_analyzed': len([m for m in molecules if m is not None])
    }


def compare_molecular_distributions(generated_mols: List[Chem.Mol],
                                    reference_mols: List[Chem.Mol]) -> Dict:
    """Compare distributions between generated and reference molecules."""

    def get_property_distributions(mols):
        properties = []
        for mol in mols:
            if mol is not None:
                props = calculate_molecular_properties(mol)
                if props:  # Only add if valid
                    properties.append(props)
        return properties

    gen_props = get_property_distributions(generated_mols)
    ref_props = get_property_distributions(reference_mols)

    if not gen_props or not ref_props:
        return {'error': 'Insufficient valid molecules for comparison'}

    # Calculate means for comparison
    comparison = {}

    for prop_name in ['molecular_weight', 'num_atoms', 'num_bonds', 'logp', 'tpsa']:
        gen_values = [p[prop_name] for p in gen_props if prop_name in p]
        ref_values = [p[prop_name] for p in ref_props if prop_name in p]

        if gen_values and ref_values:
            comparison[prop_name] = {
                'generated_mean': np.mean(gen_values),
                'reference_mean': np.mean(ref_values),
                'difference': np.mean(gen_values) - np.mean(ref_values),
                'generated_std': np.std(gen_values),
                'reference_std': np.std(ref_values)
            }

    return comparison


def convert_bonds_to_adjacency_matrix(mol: Chem.Mol) -> torch.Tensor:
    """Convert RDKit molecule bonds to adjacency matrix representation."""
    if mol is None:
        return torch.zeros(0, 0)

    n_atoms = mol.GetNumAtoms()
    adj_matrix = torch.zeros(n_atoms, n_atoms, dtype=torch.long)

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()

        # Map bond types to integers
        if bond_type == rdchem.BondType.SINGLE:
            bond_val = 1
        elif bond_type == rdchem.BondType.DOUBLE:
            bond_val = 2
        elif bond_type == rdchem.BondType.TRIPLE:
            bond_val = 3
        elif bond_type == rdchem.BondType.AROMATIC:
            bond_val = 4
        else:
            bond_val = 1  # Default to single

        adj_matrix[i, j] = bond_val
        adj_matrix[j, i] = bond_val  # Symmetric for undirected

    return adj_matrix


def create_molecule_from_adjacency(atom_types: List[str],
                                   positions: torch.Tensor,
                                   adj_matrix: torch.Tensor) -> Optional[Chem.Mol]:
    """Create RDKit molecule from adjacency matrix representation."""
    if not atom_types or adj_matrix.shape[0] != len(atom_types):
        return None

    try:
        mol = Chem.Mol()
        editable_mol = Chem.EditableMol(mol)

        # Add atoms
        for atom_type in atom_types:
            atom = Chem.Atom(atom_type)
            editable_mol.AddAtom(atom)

        # Add bonds from adjacency matrix
        bond_type_mapping = {
            1: rdchem.BondType.SINGLE,
            2: rdchem.BondType.DOUBLE,
            3: rdchem.BondType.TRIPLE,
            4: rdchem.BondType.AROMATIC
        }

        for i in range(adj_matrix.shape[0]):
            for j in range(i + 1, adj_matrix.shape[1]):
                bond_val = adj_matrix[i, j].item()
                if bond_val > 0:
                    bond_type = bond_type_mapping.get(bond_val, rdchem.BondType.SINGLE)
                    editable_mol.AddBond(i, j, bond_type)

        mol = editable_mol.GetMol()

        # Add conformer
        if mol and mol.GetNumAtoms() > 0:
            conformer = Chem.Conformer(mol.GetNumAtoms())
            for i in range(mol.GetNumAtoms()):
                if i < len(positions):
                    pos = positions[i].detach().cpu().numpy() if hasattr(positions[i], 'detach') else positions[
                        i].numpy()
                    conformer.SetAtomPosition(i, pos.tolist())
            mol.AddConformer(conformer)

        # Sanitize
        Chem.SanitizeMol(mol)
        return mol

    except Exception:
        return None