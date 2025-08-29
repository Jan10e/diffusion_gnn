"""
Utility functions for molecular diffusion models.
This module contains visualization, logging, and analysis utilities.
"""

import torch

import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Dict, Optional, Tuple, Union
import logging


logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")





def log_model_info(model: torch.nn.Module, logger: logging.Logger = None) -> None:
    """
    Log detailed information about a model

    Args:
        model: PyTorch model
        logger: Logger instance (uses module logger if None)
    """
    if logger is None:
        logger = globals()['logger']

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")

    # Model size estimation
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_size = (param_size + buffer_size) / 1024 / 1024  # Convert to MB
    logger.info(f"Model size: {model_size:.2f} MB")


def save_training_checkpoint(model: torch.nn.Module,
                             optimizer: torch.optim.Optimizer,
                             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                             epoch: int,
                             loss: float,
                             metrics: Dict,
                             filepath: str) -> None:
    """
    Save a training checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch
        loss: Current loss
        metrics: Training metrics dictionary
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_training_checkpoint(filepath: str,
                             model: torch.nn.Module,
                             optimizer: Optional[torch.optim.Optimizer] = None,
                             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict:
    """
    Load a training checkpoint

    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)

    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    logger.info(f"Checkpoint loaded from {filepath}")
    logger.info(f"Resumed from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")

    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'metrics': checkpoint.get('metrics', {})
    }