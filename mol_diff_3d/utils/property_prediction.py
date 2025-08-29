import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from diffusion_gnn_2d.models.gnn import MolecularGNN


class MolecularPropertyPredictor(nn.Module):
    """
    Property prediction using the same GNN backbone.
    Demonstrates component reusability.
    """

    def __init__(self, gnn: MolecularGNN, num_tasks: int = 1):
        super().__init__()
        self.gnn = gnn  # Reuse existing GNN
        self.predictor = nn.Sequential(
            nn.Linear(gnn.hidden_dim, gnn.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gnn.hidden_dim, num_tasks)
        )

        self.pool = global_mean_pool

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        node_features = self.gnn(x, edge_index, edge_attr)
        graph_features = self.pool(node_features, batch)
        return self.predictor(graph_features)