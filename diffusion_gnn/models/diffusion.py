import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from .gnn import MolecularGNN
from .time_embedding import SinusoidalTimeEmbedding


class NoisePredictor(nn.Module):
    """
    Simple noise prediction network.
    Single responsibility: predict noise from features.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeConditioner(nn.Module):
    """
    Combines time embeddings with molecular features.
    Single responsibility: time conditioning logic.
    """

    def __init__(self, time_dim: int, mol_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(mol_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, time_emb: torch.Tensor, graph_features: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """Fuse time and graph information"""
        t_features = self.time_mlp(time_emb)
        g_features = self.graph_proj(graph_features)

        # Combine and broadcast to nodes
        combined = t_features + g_features
        return combined[batch]  # Broadcast to node level


class MolecularDiffusionModel(nn.Module):
    """
    Main diffusion model that composes all components.
    Single responsibility: coordinate components for noise prediction.
    """

    def __init__(self,
                 atom_dim: int,
                 bond_dim: int,
                 hidden_dim: int = 128,
                 time_dim: int = 128,
                 **gnn_kwargs):
        super().__init__()

        # Store dimensions
        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim

        # Components (composition, not inheritance)
        self.gnn = MolecularGNN(atom_dim, bond_dim, hidden_dim, **gnn_kwargs)
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_conditioner = TimeConditioner(time_dim, hidden_dim, hidden_dim)
        self.noise_predictor = NoisePredictor(hidden_dim * 2, atom_dim, hidden_dim)

        # Graph pooling
        self.pool = global_mean_pool

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """Predict noise for given noisy input and timestep"""

        # 1. Embed timestep
        time_emb = self.time_embedding(t.float())

        # 2. Process molecular graph
        node_features = self.gnn(x, edge_index, edge_attr)

        # 3. Pool to graph level
        graph_features = self.pool(node_features, batch)

        # 4. Combine time and molecular info
        time_features = self.time_conditioner(time_emb, graph_features, batch)

        # 5. Predict noise
        combined = torch.cat([node_features, time_features], dim=-1)
        return self.noise_predictor(combined)
