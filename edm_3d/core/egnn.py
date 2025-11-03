import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import deepchem as dc
from deepchem.models.torch_models import TorchModel


class EGNNLayer(nn.Module):
    """
    E(n) Equivariant Graph Neural Network Layer

    Processes node features and coordinates while preserving E(3) equivariance
    (rotation, translation, and reflection symmetry)
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Edge model: processes edge features based on distances
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Node model: updates node features
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Coordinate model: updates coordinates equivariantly
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(
            self,
            h: torch.Tensor,  # Node features [N, hidden_dim]
            x: torch.Tensor,  # Node coordinates [N, 3]
            edge_index: torch.Tensor,  # Edge connectivity [2, E]
            edge_attr: Optional[torch.Tensor] = None  # Edge features [E, edge_feat_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass maintaining E(3) equivariance

        Returns:
            Updated node features and coordinates
        """
        row, col = edge_index  # Source and target nodes

        # Compute relative positions and distances
        rel_pos = x[row] - x[col]  # [E, 3]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [E, 1]

        # Edge features
        edge_feat = torch.cat([h[row], h[col], dist], dim=-1)
        if edge_attr is not None:
            edge_feat = torch.cat([edge_feat, edge_attr], dim=-1)

        # Process edges
        edge_feat = self.edge_mlp(edge_feat)  # [E, hidden_dim]

        # Update coordinates (equivariant operation)
        coord_weights = self.coord_mlp(edge_feat)  # [E, 1]
        coord_diff = rel_pos * coord_weights  # [E, 3]

        # Aggregate coordinate updates
        x_new = x.clone()
        x_new.index_add_(0, row, coord_diff)

        # Update node features (invariant operation)
        h_agg = torch.zeros_like(h)
        h_agg.index_add_(0, row, edge_feat)

        h_input = torch.cat([h, h_agg], dim=-1)
        h_new = h + self.node_mlp(h_input)  # Residual connection

        return h_new, x_new


class EGNN(nn.Module):
    """
    Multi-layer E(n) Equivariant Graph Neural Network
    """

    def __init__(
            self,
            in_node_dim: int,
            hidden_dim: int = 256,
            num_layers: int = 9,
            edge_feat_dim: int = 0
    ):
        super().__init__()

        # Initial node embedding
        self.node_embedding = nn.Linear(in_node_dim, hidden_dim)

        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_feat_dim)
            for _ in range(num_layers)
        ])

        # Output projections
        self.node_output = nn.Linear(hidden_dim, in_node_dim)

    def forward(
            self,
            h: torch.Tensor,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all EGNN layers

        Args:
            h: Node features (atom types) [N, in_node_dim]
            x: Node coordinates [N, 3]
            edge_index: Graph connectivity [2, E]
            edge_attr: Optional edge features [E, edge_feat_dim]

        Returns:
            Updated node features and coordinates
        """
        # Embed node features
        h = self.node_embedding(h)

        # Apply EGNN layers
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)

        # Project to output space
        h = self.node_output(h)

        return h, x


