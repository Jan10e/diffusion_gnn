import matplotlib.pyplot as plt
import torch

def visualize_training_progress(ddpm, device, epoch, seq_len=64):
    """Visualize samples during training"""
    ddpm.model.eval()
    with torch.no_grad():
        # Sample a few examples
        samples = ddpm.sample((4, 1, seq_len))

        plt.figure(figsize=(12, 3))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.plot(samples[i, 0].cpu().numpy())
            plt.title(f'Sample {i + 1}')
            plt.ylim(-2, 2)
        plt.suptitle(f'Generated Samples - Epoch {epoch}')
        plt.tight_layout()
        plt.show()


def visualize_final_samples(ddpm, device, num_samples=8, seq_len=64):
    """Visualize final generated samples"""
    ddpm.model.eval()
    with torch.no_grad():
        samples = ddpm.sample((num_samples, 1, seq_len))

        plt.figure(figsize=(15, 6))
        for i in range(num_samples):
            plt.subplot(2, 4, i + 1)
            plt.plot(samples[i, 0].cpu().numpy())
            plt.title(f'Final Sample {i + 1}')
            plt.ylim(-2, 2)
        plt.suptitle('Final Generated Samples')
        plt.tight_layout()
        plt.show()

    return samples


def plot_loss_curve(losses, title="Training Loss"):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()