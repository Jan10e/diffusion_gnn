import torch.nn as nn
from .gnn import MolecularGNN, GNNLayerFactory
from .time_embedding import SinusoidalTimeEmbedding
from .noise_scheduler import NoiseScheduler
from .diffusion import MolecularDiffusionModel, NoisePredictor, TimeConditioner
from .property_prediction import MolecularPropertyPredictor

# Factory functions (simple functions, not classes)
def create_molecular_gnn(atom_dim: int, bond_dim: int, **kwargs) -> MolecularGNN:
    """Factory function for GNN creation"""
    return MolecularGNN(atom_dim, bond_dim, **kwargs)

def create_diffusion_model(atom_dim: int, bond_dim: int, **kwargs) -> MolecularDiffusionModel:
    """Factory function for diffusion model creation"""
    return MolecularDiffusionModel(atom_dim, bond_dim, **kwargs)

def create_noise_scheduler(**kwargs) -> NoiseScheduler:
    """Factory function for noise scheduler creation"""
    return NoiseScheduler(**kwargs)

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Clean public API
__all__ = [
    # Core components
    'MolecularGNN', 'GNNLayerFactory',
    'SinusoidalTimeEmbedding',
    'NoiseScheduler',
    'NoisePredictor', 'TimeConditioner',
    'MolecularDiffusionModel',
    'MolecularPropertyPredictor',
    # Factory functions
    'create_molecular_gnn',
    'create_diffusion_model',
    'create_noise_scheduler',
    'count_parameters'
]