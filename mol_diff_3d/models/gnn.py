"""
Bond-aware E(3)-Equivariant Graph Neural Network for MolDiff.
Explicit bond feature handling in message passing.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class BondAwareMessagePassing(nn.Module):
    """
    E(3)-equivariant message passing that explicitly handles bond features.
    This addresses the atom-bond inconsistency problem.
    """

    def __init__(self, in_feat_dim: int, bond_feat_dim: int, hidden_dim: int, out_feat_dim: int):
        super().__init__()

        # Node feature update network (phi_x)
        # Input: [source_node, target_node, edge_features, distance]
        self.phi_x = nn.Sequential(
            nn.Linear(in_feat_dim * 2 + bond_feat_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_feat_dim)
        )

        # Position update network (phi_pos)
        self.phi_pos = nn.Sequential(
            nn.Linear(in_feat_dim * 2 + bond_feat_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Edge feature update network (phi_edge)
        self.phi_edge = nn.Sequential(
            nn.Linear(in_feat_dim * 2 + bond_feat_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, bond_feat_dim)
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row, col = edge_index

        # Calculate relative positions and distances
        rel_pos = pos[row] - pos[col]
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)

        # Combine all features for message passing
        combined_features = torch.cat([
            x[row],         # source node features
            x[col],         # target node features
            edge_attr,      # bond features
            dist_sq         # squared distance
        ], dim=-1)

        # Update node features
        msg_x = self.phi_x(combined_features)
        aggregated_x = torch.zeros_like(x).scatter_add_(
            0, col.unsqueeze(-1).expand_as(msg_x), msg_x
        )

        # Update positions
        weight_pos = self.phi_pos(combined_features)
        pos_update = weight_pos * rel_pos
        aggregated_pos = torch.zeros_like(pos).scatter_add_(
            0, col.unsqueeze(-1).expand_as(pos_update), pos_update
        )

        # Update edge features
        edge_update = self.phi_edge(combined_features)

        return aggregated_x, aggregated_pos, edge_update


class E3GNN(nn.Module):
    """
    Bond-aware E(3)-equivariant Graph Neural Network.
    Processes atoms, positions, and bonds jointly.
    """

    def __init__(self, in_feat_dim: int, bond_feat_dim: int, pos_dim: int,
                 hidden_dim: int, out_feat_dim: int, num_layers: int = 4):
        super().__init__()

        self.in_feat_dim = in_feat_dim
        self.bond_feat_dim = bond_feat_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim

        # Project input features to hidden dimension
        self.feat_proj = nn.Linear(in_feat_dim, hidden_dim)
        self.bond_proj = nn.Linear(bond_feat_dim, bond_feat_dim)

        # Bond-aware message passing layers
        self.layers = nn.ModuleList([
            BondAwareMessagePassing(hidden_dim, bond_feat_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Output projections
        self.out_feat_proj = nn.Linear(hidden_dim, out_feat_dim)
        self.out_bond_proj = nn.Linear(bond_feat_dim, bond_feat_dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor, edge_attr: torch.Tensor,
                batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Project input features
        x_proj = self.feat_proj(x)
        edge_attr_proj = self.bond_proj(edge_attr)

        # Apply message passing layers
        for layer in self.layers:
            delta_x, delta_pos, edge_update = layer(x_proj, pos, edge_index, edge_attr_proj)
            x_proj = x_proj + delta_x
            pos = pos + delta_pos
            edge_attr_proj = edge_update  # Direct update for edge features

        # Final projections
        final_x = self.out_feat_proj(x_proj)
        final_edge_attr = self.out_bond_proj(edge_attr_proj)

        return final_x, final_edge_attr, pos