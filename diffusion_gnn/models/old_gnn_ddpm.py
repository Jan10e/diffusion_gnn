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
    """
    Graph Neural Network for molecular representation learning.

    This module implements various GNN architectures for processing molecular graphs:

    - **GAT (Graph Attention Network)**: Uses attention mechanisms to weight
      neighbor contributions. Good for molecules where atom importance varies.
      Best for: Complex molecules with diverse atom types and bonding patterns.

    - **GCN (Graph Convolutional Network)**: Simple message passing with
      spectral graph convolution. Computationally efficient.
      Best for: Large datasets, simple molecular patterns, baseline comparisons.

    - **GIN (Graph Isomorphism Network)**: Theoretically most expressive for
      graph structure learning. Uses MLPs for powerful feature transformation.
      Best for: When molecular graph structure is crucial, small-medium datasets.

    Args:
        atom_dim (int): Dimension of input atom features
        bond_dim (int): Dimension of input bond/edge features
        hidden_dim (int): Hidden layer dimension for all GNN layers
        num_layers (int): Number of GNN layers to stack
        gnn_type (str): Type of GNN ('gat', 'gcn', 'gin')
        dropout (float): Dropout probability for regularization
        use_batch_norm (bool): Whether to use batch normalization

    Note:
        - GAT outputs may have different dimensions due to multi-head attention
        - All GNN types are made compatible through careful dimension handling
        - Residual connections are used when input/output dimensions match
    """

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
                # GAT with muliple heads
                heads = 4
                head_dim = hidden_dim // heads
                conv = GATConv(
                    hidden_dim, head_dim, heads=heads, concat=True,
                    edge_dim=hidden_dim, dropout=dropout
                )

            elif self.gnn_type == 'gcn':
                # Simple GCN layer; no edge features supported
                conv = GCNConv(hidden_dim, hidden_dim)

            elif self.gnn_type == 'gin':
                # Graph Isomorphism Network (GIN) with MLP for maximum expressiveness
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout), #TODO: test the effect of dropout
                    nn.Linear(hidden_dim, hidden_dim)
                )
                conv = GINConv(mlp)
            else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}. "
                                 f"Choose from 'gat', 'gcn', 'gin'")

            self.convs.append(conv)

            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass through the GNN.

        Args:
            x: Node features [num_nodes, atom_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, bond_dim] (optional)
            batch: Batch assignment for nodes (optional)

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        # Input embeddings
        h = self.atom_embedding(x)

        # Prepare edge features for GAT
        if edge_attr is not None and self.gnn_type == 'gat':
            edge_attr = self.bond_embedding(edge_attr)

        # Apply GNN layers with residual connections
        for i, conv in enumerate(self.convs):
            h_in = h

            # Apply GNN layer based on type
            if self.gnn_type == 'gat':
                h = conv(h, edge_index, edge_attr=edge_attr)
            elif self.gnn_type == 'gcn':
                h = conv(h, edge_index)  # GCN doesn't use edge features
            elif self.gnn_type == 'gin':
                h = conv(h, edge_index)  # GIN doesn't use edge features

            # Batch normalization
            if self.use_batch_norm and self.batch_norms is not None:
                h = self.batch_norms[i](h)

            # Activation and dropout
            h = F.relu(h)
            h = self.dropout_layer(h)

            # Residual connection (only if dimensions match)
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
    """
    Complete diffusion model for molecular generation using GNNs.

    This model combines a molecular GNN backbone with DDPM diffusion process
    in a single, cohesive architecture. It handles both the neural network
    prediction and the diffusion scheduling internally.

    Key Components:
    1. **GNN Backbone**: Processes molecular graphs using GAT/GCN/GIN
    2. **Time Conditioning**: Embeds diffusion timesteps for noise level awareness
    3. **DDPM Scheduling**: Manages forward/reverse diffusion processes
    4. **Noise Prediction**: Predicts noise to be removed at each timestep

    Args:
        atom_dim (int): Dimension of atom features (from molecular featurization)
        bond_dim (int): Dimension of bond features (from edge featurization)
        hidden_dim (int): Hidden dimension for all neural network layers
        num_layers (int): Number of GNN layers for message passing
        max_atoms (int): Maximum atoms per molecule (for memory management)
        gnn_type (str): GNN architecture ('gat', 'gcn', 'gin')
        time_embedding_dim (int): Dimension of time embeddings
        num_timesteps (int): Number of diffusion steps (T in DDPM papers)
        beta_start (float): Initial noise level (small values, e.g., 1e-4)
        beta_end (float): Final noise level (larger values, e.g., 0.02)

    Training:
        model = MolecularDiffusionGNN(atom_dim=39, bond_dim=10)
        loss = model.training_loss(batch)
        loss.backward()

    Generation:
        samples = model.sample(num_molecules=10, max_atoms=30)

    Note:
        This unified design eliminates the confusion between MolecularDiffusionGNN
        and MolecularDDPM by combining both functionalities in one class.
    """

    def __init__(self,
                 atom_dim: int,
                 bond_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 max_atoms: int = 50,
                 gnn_type: str = 'gat',
                 time_embedding_dim: int = 128,
                 num_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02):
        super().__init__()

        self.max_atoms = max_atoms
        self.hidden_dim = hidden_dim
        self.atom_dim = atom_dim
        self.num_timesteps = num_timesteps

        # === GNN COMPONENTS ===
        self.gnn = MolecularGNN(
            atom_dim, bond_dim, hidden_dim, num_layers, gnn_type
        )

        # === TIME CONDITIONING ===
        self.time_embedding = TimeEmbedding(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # === NOISE PREDICTION NETWORK ===
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, atom_dim)
        )

        # === GRAPH-LEVEL PROCESSING ===
        self.graph_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # === DDPM SCHEDULING  ===
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (automatically moved to device with model)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance',
                             betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def forward(self, x, edge_index, edge_attr, batch, t):
        """
        Forward pass: predict noise for denoising diffusion.

        This is the core neural network prediction used during both training
        and sampling. Given noisy molecular features and a timestep, predicts
        the noise that should be removed to get closer to clean data.

        Args:
            x: Noisy node features [num_nodes, atom_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, bond_dim]
            batch: Batch assignment [num_nodes]
            t: Timesteps [batch_size]

        Returns:
            Predicted noise [num_nodes, atom_dim]

        Process:
            1. Embed timestep information
            2. Process molecular graph with GNN
            3. Combine time and molecular information
            4. Predict noise to be removed
        """
        batch_size = batch.max().item() + 1

        # Time conditioning: convert timestep to rich representation
        t_emb = self.time_embedding(t.float())
        t_emb = self.time_mlp(t_emb)

        # Molecular representation: GNN processes graph structure
        node_features = self.gnn(x, edge_index, edge_attr, batch)

        # Global context: pool node features to graph level
        graph_features = global_mean_pool(node_features, batch)
        graph_features = self.graph_pooling(graph_features)

        # Information fusion: combine time and molecular understanding
        time_graph_features = t_emb + graph_features
        node_time_features = time_graph_features[batch]  # Broadcast to nodes

        # Final prediction: estimate noise based on combined information
        combined_features = torch.cat([node_features, node_time_features], dim=-1)
        noise_pred = self.noise_predictor(combined_features)

        return noise_pred

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: add noise to clean data.

        Implements the forward process q(x_t | x_0) that gradually corrupts
        clean molecular data by adding Gaussian noise.

        Args:
            x_start: Clean molecular features [num_nodes, atom_dim]
            t: Timesteps [batch_size] or [num_nodes] if per-node
            noise: Optional pre-sampled noise (defaults to random)

        Returns:
            Noisy features at timestep t

        Formula:
            x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
            where ᾱ_t = ∏(1 - β_s) for s=1 to t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Handle broadcasting for different tensor shapes
        while sqrt_alphas_cumprod_t.dim() < x_start.dim():
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return (sqrt_alphas_cumprod_t * x_start +
                sqrt_one_minus_alphas_cumprod_t * noise)

    def training_loss(self, batch):
        """
        Compute DDPM training loss.

        Implements the simplified loss from Ho et al. (2020):
        L = E[||ε - ε_θ(√(ᾱ_t)x_0 + √(1-ᾱ_t)ε, t)||²]

        This trains the model to predict the noise ε that was added
        to create the noisy version of the molecular data.

        Args:
            batch: PyTorch Geometric batch with molecular graphs

        Returns:
            MSE loss between predicted and actual noise

        Training Process:
            1. Sample random timesteps for each molecule in batch
            2. Sample noise and create noisy versions of clean data
            3. Predict noise using current model
            4. Compute MSE loss between predicted and actual noise
        """
        batch_size = batch.batch.max().item() + 1
        device = batch.x.device

        # Sample random timesteps for each molecule
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # Sample noise to add
        noise = torch.randn_like(batch.x)

        # Create noisy version: forward diffusion
        x_noisy = self.q_sample(batch.x, t[batch.batch], noise)

        # Predict the noise that was added
        noise_pred = self.forward(x_noisy, batch.edge_index, batch.edge_attr, batch.batch, t)

        # Loss: how well can we predict the noise?
        loss = F.mse_loss(noise_pred, noise, reduction='mean')

        return loss

    @torch.no_grad()
    def sample(self, num_molecules, max_atoms=None, device=None):
        """
        Generate new molecules using reverse diffusion.

        Starts from pure noise and iteratively removes noise using the
        trained model to generate new molecular representations.

        Args:
            num_molecules: Number of molecules to generate
            max_atoms: Max atoms per molecule (defaults to self.max_atoms)
            device: Device for computation (defaults to model device)

        Returns:
            Generated molecular features [num_molecules * max_atoms, atom_dim]

        Note:
            This is a simplified implementation. A complete version would need:
            - Proper graph structure generation (edges, connectivity)
            - Conversion back to SMILES strings
            - Validity checking and filtering
            - Variable molecule sizes within batch
        """
        if max_atoms is None:
            max_atoms = self.max_atoms
        if device is None:
            device = next(self.parameters()).device

        # Start from pure noise
        total_nodes = num_molecules * max_atoms
        x = torch.randn(total_nodes, self.atom_dim, device=device)

        # Create dummy batch structure (simplified)
        batch = torch.repeat_interleave(torch.arange(num_molecules, device=device), max_atoms)

        # Create dummy edges (complete graphs - simplified)
        # In practice, you'd want more sophisticated graph generation
        edge_index = self._create_dummy_edges(num_molecules, max_atoms, device)
        edge_attr = torch.randn(edge_index.shape[1], self.gnn.bond_dim, device=device)

        # Reverse diffusion: iteratively denoise
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((num_molecules,), i, device=device, dtype=torch.long)
            x = self._p_sample_step(x, t, batch, edge_index, edge_attr)

        return x

    def _p_sample_step(self, x, t, batch, edge_index, edge_attr):
        """Single reverse diffusion step"""
        # Predict noise
        noise_pred = self.forward(x, edge_index, edge_attr, batch, t)

        # Compute denoising coefficients
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Broadcast to node level
        sqrt_recip_alphas_t = sqrt_recip_alphas_t[batch].unsqueeze(-1)
        betas_t = betas_t[batch].unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[batch].unsqueeze(-1)

        # Compute denoised mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )

        # Add noise (except final step)
        if t[0] > 0:
            posterior_variance_t = self.posterior_variance[t]
            posterior_variance_t = posterior_variance_t[batch].unsqueeze(-1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return model_mean

    def _create_dummy_edges(self, num_molecules, max_atoms, device):
        """Create dummy complete graph edges (simplified for demo)"""
        edges = []
        for mol_idx in range(num_molecules):
            offset = mol_idx * max_atoms
            # Create complete graph within each molecule
            for i in range(max_atoms):
                for j in range(i + 1, max_atoms):
                    edges.append([offset + i, offset + j])
                    edges.append([offset + j, offset + i])  # Undirected

        if edges:
            return torch.tensor(edges, device=device).t().contiguous()
        else:
            return torch.empty((2, 0), device=device, dtype=torch.long)




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