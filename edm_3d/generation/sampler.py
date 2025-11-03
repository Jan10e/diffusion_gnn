import torch
import logging

logger = logging.getLogger(__name__)


class EDMSampler:
    """
    Sampler for generating molecules with EDM
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()
        logger.info(f"Initialized EDMSampler on device: {model.device}")

    @torch.no_grad()
    def sample(
            self,
            num_molecules: int,
            num_atoms: int,
            device: str = None,
            return_trajectory: bool = False
    ):
        """
        Generate molecules

        Args:
            num_molecules: Number of molecules to generate
            num_atoms: Number of atoms per molecule
            device: Device to run on (if None, uses model's device)
            return_trajectory: If True, return all intermediate steps

        Returns:
            coords, atom_types (and trajectory if requested)
        """
        # Use model's device if not specified
        if device is None:
            device = self.model.device
        else:
            device = torch.device(device)

        logger.info(f"Generating {num_molecules} molecules with {num_atoms} atoms on {device}")

        # Ensure model is on correct device
        self.model.to(device)

        # Start from noise
        coords = torch.randn(num_molecules, num_atoms, 3, device=device)
        atom_types = torch.randn(num_molecules, num_atoms, self.model.num_atom_types, device=device)

        # Center coordinates
        coords = coords - coords.mean(dim=1, keepdim=True)

        trajectory = [] if return_trajectory else None

        # Reverse diffusion
        num_steps = self.model.num_diffusion_steps
        for t in reversed(range(num_steps)):
            if t % 20 == 0:
                logger.debug(f"Generation step {num_steps - t}/{num_steps}")

            if return_trajectory and t % 10 == 0:
                trajectory.append((coords.clone().cpu(), atom_types.clone().cpu()))

            coords, atom_types = self._denoise_step(coords, atom_types, t, device)

        # Convert to discrete atom types
        atom_types_discrete = torch.argmax(atom_types, dim=-1)

        logger.info(f"✓ Generation complete")

        if return_trajectory:
            return coords.cpu(), atom_types_discrete.cpu(), trajectory
        return coords.cpu(), atom_types_discrete.cpu()

    def _denoise_step(self, coords, atom_types, t, device):
        """Single denoising step"""
        batch_size = coords.shape[0]
        num_atoms = coords.shape[1]

        # Create edge index (fully connected)
        edge_index = self.model._get_fully_connected_edges(num_atoms, batch_size).to(device)

        # Timestep tensor
        t_tensor = torch.tensor([t] * batch_size, device=device)

        # Predict noise using model's forward method
        pred_noise_h, pred_noise_x = self.model.model(
            atom_types.view(-1, self.model.num_atom_types),
            coords.view(-1, 3),
            edge_index,
            t_tensor
        )

        # Reshape
        pred_noise_h = pred_noise_h.view(batch_size, num_atoms, -1)
        pred_noise_x = pred_noise_x.view(batch_size, num_atoms, 3)

        # Denoise using model's diffusion process
        coords_new, atoms_new = self.model.model.diffusion.reverse_diffusion_step(
            coords, atom_types, pred_noise_x, pred_noise_h, t
        )

        return coords_new, atoms_new

    def sample_with_trajectory(self, num_molecules, num_atoms, save_every=10, device=None):
        """Sample and return full trajectory"""
        return self.sample(
            num_molecules=num_molecules,
            num_atoms=num_atoms,
            device=device,
            return_trajectory=True
        )


def generate_molecules(model, num_molecules: int = 10, num_atoms: int = 9, device=None):
    """
    Convenience function to generate molecules

    Args:
        model: EDM model
        num_molecules: Number of molecules to generate
        num_atoms: Number of atoms per molecule
        device: Device to use (None = use model's device)
    """
    print(f"Generating {num_molecules} molecules...")

    sampler = EDMSampler(model)
    coords, atom_types = sampler.sample(
        num_molecules=num_molecules,
        num_atoms=num_atoms,
        device=device
    )

    print(f"Generated coordinates shape: {coords.shape}")
    print(f"Generated atom types shape: {atom_types.shape}")

    return coords, atom_types