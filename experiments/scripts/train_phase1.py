# experiments/scripts/train_phase1.py
import sys
import os
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
from torch.utils.data import DataLoader, TensorDataset

from diffusion_gnn.core.ddpm import DDPM
from diffusion_gnn.core.noise_scheduler import DDPMScheduler
from diffusion_gnn.models.unet import SimpleUNet
from diffusion_gnn.data.synthetic import create_toy_dataset
from diffusion_gnn.training.trainer_ddpm import DDPMTrainer
from diffusion_gnn.evaluation.visualization import visualize_final_samples, plot_loss_curve


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path=None):
    # Load configuration
    if config_path is None:
        config_path = '../configs/phase1_1d_ddpm.yaml'

    config = load_config(config_path)

    # Setup device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset
    seq_len = config['data']['seq_len']
    dataset = create_toy_dataset(
        n_samples=config['data']['n_samples'],
        seq_len=seq_len
    )
    print(f"Dataset shape after creation: {dataset.shape}")

    dataloader = DataLoader(
        TensorDataset(dataset),
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    # Initialize components
    scheduler = DDPMScheduler(
        num_timesteps=config['scheduler']['num_timesteps'],
        beta_start=config['scheduler']['beta_start'],
        beta_end=config['scheduler']['beta_end']
    )

    model = SimpleUNet(
        dim=config['model']['dim'],
        channels=config['model']['channels']
    ).to(device)

    ddpm = DDPM(model, scheduler, device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config['training']['learning_rate'])
    )

    # Create trainer
    trainer = DDPMTrainer(
        ddpm,
        optimizer,
        device,
        config=config.get('training', {})
    )

    # Train
    losses = trainer.train(dataloader, config['training']['num_epochs'])

    # Final visualization
    print("\nGenerating final samples...")
    samples = visualize_final_samples(ddpm, device, seq_len=seq_len)

    # Plot training curve
    plot_loss_curve(losses, "Phase 1: 1D DDPM Training Loss")

    # Save results
    results_dir = '../results/phase1'
    os.makedirs(results_dir, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'scheduler_config': config['scheduler'],
        'model_config': config['model'],
        'losses': losses,
        'samples': samples
    }, f'{results_dir}/phase1_checkpoint.pth')

    print(f"Results saved to {results_dir}/")

    return ddpm, model, scheduler, losses


if __name__ == "__main__":
    ddpm, model, scheduler, losses = main()