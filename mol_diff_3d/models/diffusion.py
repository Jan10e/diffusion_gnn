"""
Diffusion model for 3D molecular graphs with explicit bond modelling.
This addresses the core atom-bond inconsistency problem from the MolDiff paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .gnn import E3GNN
from .time_embedding import SinusoidalTimeEmbedding


class MolecularDiffusionModel(nn.Module):
    """
    Full MolDiff model that jointly models atoms, positions, and bonds.
    Explicit bond diffusion to solve atom-bond inconsistency.
    """

    def __init__(self,
                 atom_dim: int,
                 bond_dim: int,
                 pos_dim: int = 3,
                 hidden_dim: int = 128,
                 time_dim: int = 128,
                 num_gnn_layers: int = 4,
                 max_atoms: int = 25):
        super().__init__()

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.pos_dim = pos_dim
        self.max_atoms = max_atoms

        # E(3)-equivariant GNN that handles both atoms and bonds
        self.gnn = E3GNN(
            in_feat_dim=atom_dim,
            bond_feat_dim=bond_dim,
            pos_dim=pos_dim,
            hidden_dim=hidden_dim,
            out_feat_dim=hidden_dim,
            num_layers=num_gnn_layers
        )

        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)

        # Separate predictors for atoms, positions, and bonds
        self.atom_predictor = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, atom_dim)  # Predict atom type logits
        )

        self.pos_predictor = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pos_dim)  # Predict position noise
        )

        # Bond predictor for joint atom-bond modelling
        self.bond_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, bond_dim)  # Predict bond type logits
        )

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,  # NEW: bond features
                pos: torch.Tensor,
                batch: torch.Tensor,
                t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Joint prediction of atom, position, and bond noise.

        Returns:
            atom_logits: Predicted atom type probabilities (for categorical diffusion)
            pos_noise: Predicted position noise
            bond_logits: Predicted bond type probabilities
        """
        # 1. Embed timestep
        time_emb = self.time_embedding(t.float())
        time_emb = time_emb[batch]  # Broadcast to per-node

        # 2. Process molecular graph with bond-aware E(3)-equivariant GNN
        node_features, edge_features, pos_pred = self.gnn(
            x, pos, edge_index, edge_attr, batch
        )

        # 3. Predict atom types (categorical diffusion)
        node_time_features = torch.cat([node_features, time_emb], dim=-1)
        atom_logits = self.atom_predictor(node_time_features)

        # 4. Predict position noise
        pos_noise = self.pos_predictor(node_time_features)

        # 5. Predict bond types (categorical diffusion)
        # Create edge-level time embeddings
        time_emb_edges = time_emb[edge_index[0]]  # Use source node's batch

        # Combine source and target node features for edge prediction
        source_features = node_features[edge_index[0]]
        target_features = node_features[edge_index[1]]
        edge_combined = torch.cat([source_features, target_features, time_emb_edges], dim=-1)

        bond_logits = self.bond_predictor(edge_combined)

        return atom_logits, pos_noise, bond_logits


class BondPredictor(nn.Module):
    """
    Separate bond predictor for guidance during sampling.
    This is used to guide atom generation to be more chemically valid.
    """

    def __init__(self, atom_dim: int, pos_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.atom_embedding = nn.Linear(atom_dim, hidden_dim)
        self.pos_embedding = nn.Linear(pos_dim, hidden_dim)

        self.bond_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),  # +1 for distance
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4)  # 4 bond types: none, single, double, triple
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict bond probabilities given atom types and positions.
        Used for guidance during generation.
        """
        # Embed atoms and positions
        atom_emb = self.atom_embedding(x)
        pos_emb = self.pos_embedding(pos)
        node_emb = atom_emb + pos_emb

        # Get edge features
        source_emb = node_emb[edge_index[0]]
        target_emb = node_emb[edge_index[1]]

        # Calculate distances
        source_pos = pos[edge_index[0]]
        target_pos = pos[edge_index[1]]
        distances = torch.norm(source_pos - target_pos, dim=-1, keepdim=True)

        # Predict bond probabilities
        edge_input = torch.cat([source_emb, target_emb, distances], dim=-1)
        bond_logits = self.bond_net(edge_input)

        return bond_logits