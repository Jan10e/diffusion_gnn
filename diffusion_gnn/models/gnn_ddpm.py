"""
Model architectures for molecular diffusion with GNNs.
This module contains the core GNN and diffusion model implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch
import math
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class MolecularGNN(nn.Module):
    """Graph Neural Network for molecular representation learning"""

    def __init__(self,
                 atom_dim: int,
                 bond_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 gnn_type: str = 'gat',
                 dropout: float = 0.1,
                 use_batch_norm: bool = True):
        super().__init__()

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type.lower()
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Input projections
        self.atom_embedding = nn.Linear(atom_dim, hidden_dim)
        self.bond_embedding = nn.Linear(bond_dim, hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        for i in range(num_layers):
            if self.gnn_type == 'gat':
                conv = GATConv(
                    hidden_dim, hidden_dim // 4,
                    heads=4, concat=True,
                    edge_dim=hidden_dim,
                    dropout=dropout
                )
            elif self.gnn_type == 'gcn':
                conv = GCNConv(hidden_dim, hidden_dim)
            elif self.gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                conv = GINConv(mlp)
            else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")

            self.convs.append(conv)

            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Input embeddings
        h = self.atom_embedding(x)

        if edge_attr is not None and self.gnn_type == 'gat':
            edge_attr = self.bond_embedding(edge_attr)

        # Apply GNN layers with residual connections
        for i, conv in enumerate(self.convs):
            h_in = h

            if self.gnn_type == 'gat':
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)

            if self.use_batch_norm:
                h = self.batch_norms[i](h)

            h = F.relu(h)
            h = self.dropout_layer(h)

            # Residual connection
            if h.shape == h_in.shape:
                h = h + h_in

        return self.output_proj(h)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion models"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class MolecularDiffusionGNN(nn.Module):
    """GNN-based molecular diffusion model"""

    def __init__(self,
                 atom_dim: int,
                 bond_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 max_atoms: int = 50,
                 gnn_type: str = 'gat',
                 time_embedding_dim: int = 128):
        super().__init__()

        self.max_atoms = max_atoms
        self.hidden_dim = hidden_dim
        self.atom_dim = atom_dim

        # GNN backbone
        self.gnn = MolecularGNN(
            atom_dim, bond_dim, hidden_dim, num_layers, gnn_type
        )

        # Time embedding
        self.time_embedding = TimeEmbedding(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Noise prediction network
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, atom_dim)
        )

        # Graph-level processing
        self.graph_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch, t):
        batch_size = batch.max().item() + 1

        # Time embedding
        t_emb = self.time_embedding(t.float())
        t_emb = self.time_mlp(t_emb)  # [batch_size, hidden_dim]

        # GNN forward pass
        node_features = self.gnn(x, edge_index, edge_attr, batch)  # [num_nodes, hidden_dim]

        # Graph-level features
        graph_features = global_mean_pool(node_features, batch)  # [batch_size, hidden_dim]
        graph_features = self.graph_pooling(graph_features)

        # Combine time and graph information
        time_graph_features = t_emb + graph_features  # [batch_size, hidden_dim]

        # Broadcast to nodes
        node_time_features = time_graph_features[batch]  # [num_nodes, hidden_dim]

        # Combine node and time features
        combined_features = torch.cat([node_features, node_time_features], dim=-1)

        # Predict noise
        noise_pred = self.noise_predictor(combined_features)

        return noise_pred


class MolecularDDPM:
    """Denoising Diffusion Probabilistic Model for molecular generation"""

    def __init__(self,
                 model: MolecularDiffusionGNN,
                 num_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 device: str = 'cuda'):

        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device

        # DDPM noise schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance',
                             betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor.to(self.device))

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process (add noise)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Expand dimensions to match x_start
        while sqrt_alphas_cumprod_t.dim() < x_start.dim():
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return (sqrt_alphas_cumprod_t * x_start +
                sqrt_one_minus_alphas_cumprod_t * noise)

    def p_sample(self, x, t, batch):
        """Reverse diffusion process (remove noise)"""
        batch_size = batch.max().item() + 1

        # Model prediction
        with torch.no_grad():
            noise_pred = self.model(x, batch.edge_index, batch.edge_attr, batch.batch, t)

        # Compute coefficients
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Expand dimensions
        while sqrt_recip_alphas_t.dim() < x.dim():
            sqrt_recip_alphas_t = sqrt_recip_alphas_t.unsqueeze(-1)
            betas_t = betas_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        # Compute mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )

        # Add noise (except for t=0)
        if t.min() > 0:
            posterior_variance_t = self.posterior_variance[t]
            while posterior_variance_t.dim() < x.dim():
                posterior_variance_t = posterior_variance_t.unsqueeze(-1)

            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return model_mean

    def training_loss(self, batch):
        """Compute training loss for DDPM"""
        batch_size = batch.batch.max().item() + 1

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

        # Sample noise
        noise = torch.randn_like(batch.x)

        # Forward diffusion (add noise)
        x_noisy = self.q_sample(batch.x, t[batch.batch], noise)

        # Predict noise
        noise_pred = self.model(x_noisy, batch.edge_index, batch.edge_attr, batch.batch, t)

        # Compute loss
        loss = F.mse_loss(noise_pred, noise, reduction='mean')

        return loss

    @torch.no_grad()
    def sample(self, shape, batch_template=None):
        """Generate samples using reverse diffusion"""
        # This is a simplified version - full implementation would need 
        # proper graph structure generation
        device = self.device

        # Start from random noise
        x = torch.randn(shape, device=device)

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((1,), i, device=device, dtype=torch.long)

            if batch_template is not None:
                x = self.p_sample(x, t, batch_template)

        return x


class MolecularPropertyPredictor(nn.Module):
    """Property prediction model for molecular graphs (for validation)"""

    def __init__(self,
                 atom_dim: int,
                 bond_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_tasks: int = 1,
                 pooling: str = 'mean'):
        super().__init__()

        self.gnn = MolecularGNN(atom_dim, bond_dim, hidden_dim, num_layers)
        self.pooling = pooling

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_tasks)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Get node features
        node_features = self.gnn(x, edge_index, edge_attr, batch)

        # Pool to graph level
        if self.pooling == 'mean':
            graph_features = global_mean_pool(node_features, batch)
        elif self.pooling == 'max':
            graph_features = global_max_pool(node_features, batch)
        else:
            # Combine mean and max pooling
            mean_pool = global_mean_pool(node_features, batch)
            max_pool = global_max_pool(node_features, batch)
            graph_features = torch.cat([mean_pool, max_pool], dim=-1)

        # Predict properties
        predictions = self.predictor(graph_features)

        return predictions


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_model(atom_dim: int,
                     bond_dim: int,
                     config: Dict = None) -> MolecularDiffusionGNN:
    """Initialize a molecular diffusion model with given configuration"""

    default_config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'max_atoms': 50,
        'gnn_type': 'gat',
        'time_embedding_dim': 128
    }

    if config is not None:
        default_config.update(config)

    model = MolecularDiffusionGNN(
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        **default_config
    )

    logger.info(f"Initialized model with {count_parameters(model):,} parameters")

    return model