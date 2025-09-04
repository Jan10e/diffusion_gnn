"""
Improved samplers with bond guidance and proper categorical diffusion.
Key improvements:
1. Bond predictor guidance during sampling
2. Proper categorical diffusion for discrete features
3. Chemical validity constraints
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from mol_diff_3d.models.categorical_diffusion import CategoricalDiffusion


class ImprovedDDPMQSampler:
    """
    Enhanced forward diffusion process with categorical support.
    """

    def __init__(self, scheduler_params: Dict, categorical_diffusion: CategoricalDiffusion):
        self.alphas_cumprod = scheduler_params['alphas_cumprod']
        self.sqrt_alphas_cumprod = scheduler_params['sqrt_alphas_cumprod']
        self.sqrt_one_minus_alphas_cumprod = scheduler_params['sqrt_one_minus_alphas_cumprod']
        self.num_timesteps = len(self.alphas_cumprod)
        self.categorical_diffusion = categorical_diffusion

    def q_sample_step(self, x_start: torch.Tensor, t: torch.Tensor,
                      noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion for atom types using categorical diffusion."""
        return self.categorical_diffusion.q_sample_atoms(x_start, t)

    def q_sample_pos_step(self, pos_start: torch.Tensor, t: torch.Tensor,
                          noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion for continuous positions (unchanged)."""
        if noise is None:
            noise = torch.randn_like(pos_start)

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        while sqrt_alpha_cumprod_t.dim() < pos_start.dim():
            sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)

        return sqrt_alpha_cumprod_t * pos_start + sqrt_one_minus_alpha_cumprod_t * noise

    def q_sample_bonds_step(self, edge_attr_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward diffusion for bond types using categorical diffusion."""
        return self.categorical_diffusion.q_sample_bonds(edge_attr_start, t)


class ImprovedDDPMPsampler:
    """
    Enhanced reverse diffusion process with bond guidance and chemical constraints.
    """

    def __init__(self, scheduler_params: Dict, categorical_diffusion: CategoricalDiffusion,
                 bond_predictor=None, guidance_scale: float = 1.0):
        self.betas = scheduler_params['betas']
        self.alphas_cumprod = scheduler_params['alphas_cumprod']
        self.alphas_cumprod_prev = scheduler_params['alphas_cumprod_prev']
        self.posterior_variance = scheduler_params['posterior_variance']
        self.num_timesteps = len(self.betas)

        self.categorical_diffusion = categorical_diffusion
        self.bond_predictor = bond_predictor
        self.guidance_scale = guidance_scale

        # Chemical constraints
        self.bond_length_constraints = {
            ('C', 'C'): (1.2, 1.8),  # (min, max) bond lengths in Angstroms
            ('C', 'N'): (1.1, 1.7),
            ('C', 'O'): (1.0, 1.6),
            ('N', 'N'): (1.0, 1.6),
            ('N', 'O'): (1.0, 1.5),
            ('O', 'O'): (1.2, 1.8),
        }

    def apply_bond_guidance(self, atom_logits: torch.Tensor, pos_t: torch.Tensor,
                            edge_index: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Apply bond predictor guidance to improve chemical validity.
        This is a key MolDiff innovation for addressing atom-bond inconsistency.
        """
        if self.bond_predictor is None:
            return atom_logits

        # Get current atom type probabilities
        atom_probs = F.softmax(atom_logits, dim=-1)

        # Predict bond probabilities given current state
        with torch.no_grad():
            bond_logits = self.bond_predictor(atom_probs, pos_t, edge_index)
            bond_probs = F.softmax(bond_logits, dim=-1)

        # Compute guidance gradient (encourage chemically valid atom-bond combinations)
        guidance_grad = torch.zeros_like(atom_logits)

        for edge_idx in range(edge_index.shape[1]):
            i, j = edge_index[0, edge_idx], edge_index[1, edge_idx]

            # Get positions and distance
            pos_i, pos_j = pos_t[i], pos_t[j]
            distance = torch.norm(pos_i - pos_j)

            # Apply distance-based guidance
            for atom_type_i in range(atom_logits.shape[1]):
                for atom_type_j in range(atom_logits.shape[1]):
                    # Check if this atom pair can form bonds at this distance
                    if self._is_valid_bond_distance(atom_type_i, atom_type_j, distance):
                        # Encourage this atom type combination
                        guidance_grad[i, atom_type_i] += self.guidance_scale * 0.1
                        guidance_grad[j, atom_type_j] += self.guidance_scale * 0.1

        # Apply guidance by modifying logits
        guided_logits = atom_logits + guidance_grad

        return guided_logits

    def _is_valid_bond_distance(self, atom_type_i: int, atom_type_j: int, distance: float) -> bool:
        """Check if distance is valid for potential bond between atom types."""
        # Convert atom type indices to element symbols (simplified)
        elements = ['C', 'N', 'O', 'F', 'H', 'P', 'S', 'Cl', 'Br', 'I']
        if atom_type_i >= len(elements) or atom_type_j >= len(elements):
            return False

        elem_i, elem_j = elements[atom_type_i], elements[atom_type_j]

        # Check bond length constraints
        pair_key = tuple(sorted([elem_i, elem_j]))
        if pair_key in self.bond_length_constraints:
            min_len, max_len = self.bond_length_constraints[pair_key]
            return min_len <= distance.item() <= max_len

        # Default constraint for unknown pairs
        return 1.0 <= distance.item() <= 2.0

    def p_sample_step(self, model, x_t: torch.Tensor, pos_t: torch.Tensor,
                      edge_index: torch.Tensor, edge_attr_t: torch.Tensor,
                      batch: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reverse sampling step with bond guidance and chemical constraints.
        """
        # Get model predictions
        atom_logits, pos_noise_pred, bond_logits = model(
            x_t, edge_index, edge_attr_t, pos_t, batch, t
        )

        # Apply bond guidance to atom predictions
        guided_atom_logits = self.apply_bond_guidance(atom_logits, pos_t, edge_index, t)

        # Expand timesteps to match tensor dimensions
        t_expanded = t[batch]  # For nodes
        t_edges = t[batch[edge_index[0]]]  # For edges

        # --- Sample Positions (Continuous) ---
        mean_pos = (
                1 / (1 - self.betas[t_expanded]).sqrt().unsqueeze(-1)
                * (pos_t - self.betas[t_expanded].unsqueeze(-1) /
                   (1 - self.alphas_cumprod[t_expanded]).sqrt().unsqueeze(-1) * pos_noise_pred)
        )

        variance_pos = self.posterior_variance[t_expanded].unsqueeze(-1)

        if t[0] > 0:
            pos_t_minus_1 = mean_pos + variance_pos.sqrt() * torch.randn_like(pos_t)
        else:
            pos_t_minus_1 = mean_pos

        # --- Sample Atom Types (Categorical) ---
        x_t_minus_1 = self.categorical_diffusion.p_sample_atoms(
            guided_atom_logits, x_t, t_expanded, temperature=0.8
        )

        # --- Sample Bond Types (Categorical) ---
        edge_attr_t_minus_1 = self.categorical_diffusion.p_sample_bonds(
            bond_logits, edge_attr_t, t_edges, temperature=0.8
        )

        # Apply chemical validity post-processing
        x_t_minus_1, edge_attr_t_minus_1 = self._apply_chemical_constraints(
            x_t_minus_1, edge_attr_t_minus_1, pos_t_minus_1, edge_index
        )

        return x_t_minus_1, pos_t_minus_1, edge_attr_t_minus_1

    def _apply_chemical_constraints(self, x: torch.Tensor, edge_attr: torch.Tensor,
                                    pos: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply chemical validity constraints as post-processing.
        """
        # Convert to hard assignments for constraint checking
        atom_types = x.argmax(dim=-1)
        bond_types = edge_attr.argmax(dim=-1)

        # Check and fix invalid bonds
        elements = ['C', 'N', 'O', 'F', 'H', 'P', 'S', 'Cl', 'Br', 'I']

        for edge_idx in range(edge_index.shape[1]):
            i, j = edge_index[0, edge_idx], edge_index[1, edge_idx]

            # Get distance
            distance = torch.norm(pos[i] - pos[j])

            # Get atom types
            atom_i = atom_types[i].item()
            atom_j = atom_types[j].item()

            if atom_i < len(elements) and atom_j < len(elements):
                elem_i, elem_j = elements[atom_i], elements[atom_j]

                # Check if current bond is valid
                if not self._is_valid_bond_distance(atom_i, atom_j, distance):
                    # Set bond to "no bond" (type 0)
                    bond_types[edge_idx] = 0

        # Convert back to one-hot
        x_constrained = F.one_hot(atom_types, x.shape[-1]).float()
        edge_attr_constrained = F.one_hot(bond_types, edge_attr.shape[-1]).float()

        return x_constrained, edge_attr_constrained