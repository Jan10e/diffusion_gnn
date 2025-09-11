"""
Categorical diffusion with corrected q_sample_atoms method.
The original had unused variables and incorrect tensor handling.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class CategoricalNoiseScheduler:
    """
    Noise scheduler for categorical (discrete) variables like atom types and bond types.
    Uses a different schedule than continuous variables to address atom-bond inconsistency.
    """

    def __init__(self, num_timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 schedule_type: str = 'linear'):
        self.num_timesteps = num_timesteps

        # Create different noise schedules for atoms and bonds
        if schedule_type == 'linear':
            # Standard linear schedule for atoms
            self.atom_betas = torch.linspace(beta_start, beta_end, num_timesteps)
            # Faster schedule for bonds (diffuse bonds earlier)
            self.bond_betas = torch.linspace(beta_start, beta_end * 2, num_timesteps // 2)
            self.bond_betas = torch.cat([
                self.bond_betas,
                torch.full((num_timesteps - len(self.bond_betas),), beta_end * 2)
            ])

        # Compute cumulative products for categorical diffusion
        self.atom_alphas = 1.0 - self.atom_betas
        self.atom_alphas_cumprod = torch.cumprod(self.atom_alphas, dim=0)

        self.bond_alphas = 1.0 - self.bond_betas
        self.bond_alphas_cumprod = torch.cumprod(self.bond_alphas, dim=0)

    def to(self, device):
        self.atom_betas = self.atom_betas.to(device)
        self.bond_betas = self.bond_betas.to(device)
        self.atom_alphas = self.atom_alphas.to(device)
        self.bond_alphas = self.bond_alphas.to(device)
        self.atom_alphas_cumprod = self.atom_alphas_cumprod.to(device)
        self.bond_alphas_cumprod = self.bond_alphas_cumprod.to(device)
        return self


class CategoricalDiffusion:
    """
    Handles categorical diffusion for discrete molecular features.
    Based on Austin et al. "Structured Denoising Diffusion Models in Discrete State-Spaces"
    """

    def __init__(self, num_atom_types: int, num_bond_types: int, scheduler: CategoricalNoiseScheduler):
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.scheduler = scheduler

        # Precompute transition matrices for efficiency
        self.atom_transitions = self._compute_transition_matrices(
            num_atom_types, scheduler.atom_alphas_cumprod
        )
        self.bond_transitions = self._compute_transition_matrices(
            num_bond_types, scheduler.bond_alphas_cumprod
        )

    def _compute_transition_matrices(self, num_classes: int, alphas_cumprod: torch.Tensor):
        """Compute transition matrices Q_t for categorical diffusion."""
        device = alphas_cumprod.device
        transitions = []

        for alpha_t in alphas_cumprod:
            # Transition matrix: Q_t = alpha_t * I + (1 - alpha_t) / K * 1
            # where I is identity and 1 is matrix of ones
            alpha_t = alpha_t.item()
            uniform_prob = (1 - alpha_t) / num_classes

            Q_t = torch.full((num_classes, num_classes), uniform_prob, device=device)
            Q_t.fill_diagonal_(alpha_t + uniform_prob)

            transitions.append(Q_t)

        return torch.stack(transitions)

    def q_sample_atoms(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        FIXED: Forward diffusion for atom types.
        x_start: one-hot encoded atom types [N, num_atom_types]
        t: timesteps [N]
        """
        device = x_start.device

        # Handle different timestep formats
        if t.dim() == 0:
            # Single timestep - expand to match batch size
            t = t.expand(x_start.shape[0])

        # Ensure timesteps are valid indices
        t = torch.clamp(t, 0, len(self.atom_transitions) - 1)

        # Get transition matrices for the given timesteps
        Q_t = self.atom_transitions[t].to(device)  # [N, num_atom_types, num_atom_types]

        # Apply transition: x_t = x_start @ Q_t^T (batch matrix multiply)
        x_t_probs = torch.bmm(x_start.unsqueeze(1), Q_t.transpose(-2, -1)).squeeze(1)

        # Ensure probabilities are valid (non-negative, sum to 1)
        x_t_probs = torch.clamp(x_t_probs, min=1e-8)
        x_t_probs = x_t_probs / x_t_probs.sum(dim=-1, keepdim=True)

        # Sample from categorical distribution
        x_t = F.gumbel_softmax(x_t_probs.log(), tau=1.0, hard=True)

        return x_t

    def q_sample_bonds(self, edge_attr_start: torch.Tensor, t_edges: torch.Tensor) -> torch.Tensor:
        """
        FIXED: Forward diffusion for bond types.
        edge_attr_start: one-hot encoded bond types [E, num_bond_types]
        t_edges: timesteps for edges [E]
        """
        device = edge_attr_start.device

        # Handle empty edge case
        if edge_attr_start.shape[0] == 0:
            return edge_attr_start

        # Handle different timestep formats
        if t_edges.dim() == 0:
            t_edges = t_edges.expand(edge_attr_start.shape[0])

        # Ensure timesteps are valid indices
        t_edges = torch.clamp(t_edges, 0, len(self.bond_transitions) - 1)

        # Get transition matrices
        Q_t = self.bond_transitions[t_edges].to(device)  # [E, num_bond_types, num_bond_types]

        # Apply transition
        edge_t_probs = torch.bmm(
            edge_attr_start.unsqueeze(1),
            Q_t.transpose(-2, -1)
        ).squeeze(1)

        # Ensure probabilities are valid
        edge_t_probs = torch.clamp(edge_t_probs, min=1e-8)
        edge_t_probs = edge_t_probs / edge_t_probs.sum(dim=-1, keepdim=True)

        # Sample from categorical distribution
        edge_t = F.gumbel_softmax(edge_t_probs.log(), tau=1.0, hard=True)

        return edge_t

    def p_sample_atoms(self, model_logits: torch.Tensor, x_t: torch.Tensor,
                      t: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Reverse sampling for atom types using model predictions.
        """
        # Convert logits to probabilities
        model_probs = F.softmax(model_logits / temperature, dim=-1)

        # Check if we're at the final timestep
        is_final_step = (t == 0).all() if t.dim() > 0 else (t == 0)

        if not is_final_step:
            x_t_minus_1 = F.gumbel_softmax(model_probs.log(), tau=temperature, hard=True)
        else:
            # At t=0, take the most likely class
            x_t_minus_1 = F.one_hot(model_probs.argmax(dim=-1), self.num_atom_types).float()

        return x_t_minus_1

    def p_sample_bonds(self, model_logits: torch.Tensor, edge_t: torch.Tensor,
                      t: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Reverse sampling for bond types using model predictions.
        """
        # Handle empty edge case
        if model_logits.shape[0] == 0:
            return edge_t

        # Convert logits to probabilities
        model_probs = F.softmax(model_logits / temperature, dim=-1)

        # Check if we're at the final timestep
        is_final_step = (t == 0).all() if t.dim() > 0 else (t == 0)

        if not is_final_step:
            edge_t_minus_1 = F.gumbel_softmax(model_probs.log(), tau=temperature, hard=True)
        else:
            # At t=0, take the most likely class
            edge_t_minus_1 = F.one_hot(model_probs.argmax(dim=-1), self.num_bond_types).float()

        return edge_t_minus_1

    def compute_atom_loss(self, model_logits: torch.Tensor, x_start: torch.Tensor,
                         x_t: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        """
        Compute cross-entropy loss for atom type prediction.
        Uses true atom type classes as targets for the predicted logits.
        """
        # True class indices
        true_classes = x_start.argmax(dim=-1)

        # Cross-entropy loss
        loss = F.cross_entropy(model_logits, true_classes)

        return loss

    def compute_bond_loss(self, model_logits: torch.Tensor, edge_attr_start: torch.Tensor,
                         edge_attr_t: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        """
        Compute cross-entropy loss for bond type prediction.
        Uses true bond type classes as targets for the predicted logits.
        """
        # Handle empty edge case
        if model_logits.shape[0] == 0:
            return torch.tensor(0.0, device=model_logits.device, requires_grad=True)

        # True bond class indices
        true_bond_classes = edge_attr_start.argmax(dim=-1)

        # Cross-entropy loss
        loss = F.cross_entropy(model_logits, true_bond_classes)

        return loss


def create_categorical_features(atom_types: list, bond_types: list,
                              num_atom_classes: int = 10, num_bond_classes: int = 5):
    """
    Convert atom and bond type lists to one-hot encoded tensors.

    Args:
        atom_types: List of atom type strings ['C', 'N', 'O', ...]
        bond_types: List of bond type integers [0, 1, 2, 3, 4] for [none, single, double, triple, aromatic]
        num_atom_classes: Number of atom type classes
        num_bond_classes: Number of bond type classes (including "no bond")

    Returns:
        atom_features: One-hot encoded atom types
        bond_features: One-hot encoded bond types
    """
    # Define atom type mapping
    atom_mapping = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'H': 4,
                   'P': 5, 'S': 6, 'Cl': 7, 'Br': 8, 'I': 9}

    # Convert atom types to indices
    atom_indices = [atom_mapping.get(atom, 0) for atom in atom_types]  # Default to C
    atom_features = F.one_hot(torch.tensor(atom_indices), num_atom_classes).float()

    # Convert bond types to one-hot
    bond_features = F.one_hot(torch.tensor(bond_types), num_bond_classes).float()

    return atom_features, bond_features
