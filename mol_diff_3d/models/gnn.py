"""
E(3)-Equivariant Graph Neural Network for Molecular Generation.
Different than in the 2D Diffusion+GNN, we can't use the GAT, GIN, or GCN layers, as these are not
able to handle 3D coordinates and maintain equivariance.
Therefore, we need to replace the standard GNN layers with E(3)-equivariant layers.
The EGNN architecture is described in Satorras et al, in which the key equation is the message passing update:
x'_i = x_i + Σ_j (m_ij * (x_i - x_j))  -> a simplified form of Eqn 2 in the paper.
This eqn ensures that the change in a node's position is a weighted sum of vectors pointing to its neighbors,
maintaining the overall geometric configuration.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class EquivariantMessagePassing(nn.Module):
    """
    Core component of an Equivariant Graph Neural Network (EGNN).
    This module handles message passing for both features and 3D coordinates.
    """

    def __init__(self, in_feat_dim: int, hidden_dim: int, out_feat_dim: int):
        super().__init__()
        # Message passing network for node features
        # phi_x computes the scalar messages for node feature updates
        self.phi_x = nn.Sequential(
            nn.Linear(in_feat_dim * 2 + 1, hidden_dim),  # +1 for distance
            nn.SiLU(),
            nn.Linear(hidden_dim, out_feat_dim)
        )
        # Network for updating coordinates
        # phi_pos computes scalar weights that scale the relative position vectors.
        self.phi_pos = nn.Sequential(
            nn.Linear(in_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        row, col = edge_index
        rel_pos = pos[row] - pos[col]

        # Squared Euclidean distance
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)

        # Concatenate features of source and target nodes with distance
        combined_features = torch.cat([x[row], x[col], dist_sq], dim=-1)

        # Message for features
        msg_x = self.phi_x(combined_features)

        # Aggregation of feature messages
        aggregated_x = torch.zeros_like(x).scatter_add_(0, col.unsqueeze(-1).expand_as(msg_x), msg_x)

        # Weight for position updates
        weight_pos = self.phi_pos(combined_features)

        # Position update, analogous to Eq. 2 from the EGNN paper.
        pos_update = weight_pos * rel_pos

        # Aggregation of position updates
        aggregated_pos = torch.zeros_like(pos).scatter_add_(0, col.unsqueeze(-1).expand_as(pos_update), pos_update)

        return aggregated_x, aggregated_pos


class E3GNN(nn.Module):
    """
    An E(3)-equivariant Graph Neural Network for molecular generation.
    This model explicitly handles 3D positions and is invariant to translations and rotations.
    """

    def __init__(self, in_feat_dim: int, pos_dim: int, hidden_dim: int, out_feat_dim: int, num_layers: int = 4):
        super().__init__()

        self.in_feat_dim = in_feat_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim

        self.feat_proj = nn.Linear(in_feat_dim, hidden_dim)

        self.layers = nn.ModuleList([
            EquivariantMessagePassing(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.out_feat_proj = nn.Linear(hidden_dim, out_feat_dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        x_proj = self.feat_proj(x)

        for layer in self.layers:
            delta_x, delta_pos = layer(x_proj, pos, edge_index)
            x_proj = x_proj + delta_x
            pos = pos + delta_pos

        return self.out_feat_proj(x_proj), pos