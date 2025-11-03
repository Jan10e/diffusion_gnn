# ============================================================================
# edm_3d/core/egnn.py
# ============================================================================

import torch
import torch.nn as nn
from typing import Tuple, Optional


class EGNNLayer(nn.Module):
    """E(n) Equivariant Graph Neural Network Layer"""

    def __init__(self, hidden_dim: int, edge_feat_dim: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(
            self,
            h: torch.Tensor,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index

        rel_pos = x[row] - x[col]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)

        edge_feat = torch.cat([h[row], h[col], dist], dim=-1)
        if edge_attr is not None:
            edge_feat = torch.cat([edge_feat, edge_attr], dim=-1)

        edge_feat = self.edge_mlp(edge_feat)

        coord_weights = self.coord_mlp(edge_feat)
        coord_diff = rel_pos * coord_weights

        x_new = x.clone()
        x_new.index_add_(0, row, coord_diff)

        h_agg = torch.zeros_like(h)
        h_agg.index_add_(0, row, edge_feat)

        h_input = torch.cat([h, h_agg], dim=-1)
        h_new = h + self.node_mlp(h_input)

        return h_new, x_new


class EGNN(nn.Module):
    """Multi-layer E(n) Equivariant Graph Neural Network"""

    def __init__(
            self,
            in_node_dim: int,
            hidden_dim: int = 256,
            num_layers: int = 9,
            edge_feat_dim: int = 0,
            out_node_dim: int = None
    ):
        super().__init__()

        if out_node_dim is None:
            out_node_dim = in_node_dim

        self.node_embedding = nn.Linear(in_node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_feat_dim)
            for _ in range(num_layers)
        ])
        self.node_output = nn.Linear(hidden_dim, out_node_dim)

    def forward(
            self,
            h: torch.Tensor,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.node_embedding(h)

        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)

        h = self.node_output(h)

        return h, x