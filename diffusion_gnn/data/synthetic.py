import torch
import numpy as np

def create_toy_dataset(n_samples=1000, seq_len=64):
    """
    Create simple 1D toy data for testing
    Mix of sine waves and gaussian bumps
    """
    data = []

    for _ in range(n_samples):
        x = torch.linspace(0, 4 * np.pi, seq_len)

        if torch.rand(1) < 0.5:
            # Sine wave with random frequency and phase
            freq = torch.rand(1) * 2 + 0.5
            phase = torch.rand(1) * 2 * np.pi
            y = torch.sin(freq * x + phase)
        else:
            # Gaussian bump
            center = torch.rand(1) * seq_len
            width = torch.rand(1) * 10 + 5
            y = torch.exp(-((torch.arange(seq_len) - center) / width) ** 2)

        data.append(y)

    return torch.stack(data).unsqueeze(1)  # Add channel dimension
