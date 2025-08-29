"""
The diffusion model for 3D molecular graphs.
The diffusion model now needs to handle both discrete (atom and bond features) and continuous (3D positions) data.
Therefore, the forward pass must now accept noisy 3D coordinates in addition to noisy features.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .gnn import E3GNN
from .time_embedding import SinusoidalTimeEmbedding


class MolecularDiffusionModel(nn.Module):
    """
    The full diffusion model for 3D molecular graphs.
    It takes noisy features and positions and predicts the noise for both.
    """

    def __init__(self,
                 atom_dim: int,
                 pos_dim: int = 3,
                 hidden_dim: int = 128,
                 time_dim: int = 128,
                 num_gnn_layers: int = 4):
        super().__init__()

        # GNN for feature and position processing
        self.gnn = E3GNN(atom_dim, pos_dim, hidden_dim, hidden_dim, num_layers=num_gnn_layers)

        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)

        # Noise predictor
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            # Predicts noise for features and positions
            nn.Linear(hidden_dim, atom_dim + pos_dim)
        )

        self.atom_dim = atom_dim
        self.pos_dim = pos_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                pos: torch.Tensor, batch: torch.Tensor,
                t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts noise for given noisy input and timestep.

        The time embedding is now combined with the graph features before
        the final noise prediction, simplifying the flow.
        """
        # 1. Embed timestep
        time_emb = self.time_embedding(t.float())
        time_emb = time_emb[batch]  # Broadcast to per-node

        # 2. Process molecular graph with E(3)-equivariant GNN
        node_features, pos_pred = self.gnn(x, pos, edge_index)

        # 3. Combine graph features with time embedding
        combined_features = torch.cat([node_features, time_emb], dim=-1)

        # 4. Predict noise for both features and positions
        output = self.noise_predictor(combined_features)

        # Use ellipsis to flexibly slice the output tensor
        noise_pred_x = output[..., :self.atom_dim]
        noise_pred_pos = output[..., self.atom_dim:]

        return noise_pred_x, noise_pred_pos