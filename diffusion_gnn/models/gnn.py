import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from typing import Optional


class GNNLayerFactory:
    """Factory for creating different GNN layer types."""

    @staticmethod
    def create_gat_layer(in_dim: int, out_dim: int, heads: int = 4,
                         edge_dim: Optional[int] = None, dropout: float = 0.1) -> GATConv:
        head_dim = out_dim // heads
        return GATConv(in_dim, head_dim, heads=heads, concat=True,
                       edge_dim=edge_dim, dropout=dropout)

    @staticmethod
    def create_gcn_layer(in_dim: int, out_dim: int) -> GCNConv:
        return GCNConv(in_dim, out_dim)

    @staticmethod
    def create_gin_layer(in_dim: int, out_dim: int, dropout: float = 0.1) -> nn.Module:
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
        return GINConv(mlp)


class MolecularGNN(nn.Module):
    """
    Pure GNN for molecular graphs.
    Single responsibility: graph convolution and feature extraction.
    No diffusion, no time embedding, no training logic.
    """

    def __init__(self,
                 atom_dim: int,
                 bond_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 gnn_type: str = 'gat',
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bond_dim = bond_dim

        # Input projections
        self.atom_proj = nn.Linear(atom_dim, hidden_dim)
        self.bond_proj = nn.Linear(bond_dim, hidden_dim) if bond_dim > 0 else None

        # GNN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if gnn_type == 'gat':
                layer = GNNLayerFactory.create_gat_layer(
                    hidden_dim, hidden_dim, edge_dim=hidden_dim if self.bond_proj else None
                )
            elif gnn_type == 'gcn':
                layer = GNNLayerFactory.create_gcn_layer(hidden_dim, hidden_dim)
            elif gnn_type == 'gin':
                layer = GNNLayerFactory.create_gin_layer(hidden_dim, hidden_dim, dropout)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.layers.append(layer)
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.gnn_type = gnn_type

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.atom_proj(x)
        edge_features = self.bond_proj(edge_attr) if edge_attr is not None and self.bond_proj else None

        for layer, norm in zip(self.layers, self.norms):
            h_in = h

            # Apply layer based on type
            if self.gnn_type == 'gat' and edge_features is not None:
                h = layer(h, edge_index, edge_attr=edge_features)
            else:
                h = layer(h, edge_index)

            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)

            # Residual connection
            if h.shape == h_in.shape:
                h = h + h_in

        return h
