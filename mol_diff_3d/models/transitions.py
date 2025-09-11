"""
GeneralCategoricalTransition implementation from MolDiff repository.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# Helper functions (from MolDiff's diffusion.py)
def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def extract(coef, t, batch, ndim=2):
    out = coef[t][batch]
    if ndim == 1:
        return out
    elif ndim == 2:
        return out.unsqueeze(-1)
    elif ndim == 3:
        return out.unsqueeze(-1).unsqueeze(-1)
    else:
        raise NotImplementedError('ndim > 3')


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    return sample_index


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=-1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=-1)


# Beta schedule functions (from MolDiff's diffusion.py)
def get_beta_schedule(beta_schedule, num_timesteps, **kwargs):
    """Get beta schedule for diffusion process."""

    if beta_schedule == "linear":
        betas = np.linspace(
            kwargs['beta_start'], kwargs['beta_end'], num_timesteps, dtype=np.float64
        )
    elif beta_schedule == "cosine":
        s = kwargs.get('s', 0.008)
        steps = num_timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0, 0.999)
    elif beta_schedule == "advance":
        scale_start = kwargs.get('scale_start', 0.999)
        scale_end = kwargs.get('scale_end', 0.001)
        width = kwargs.get('width', 2)

        k = width
        A0 = scale_end
        A1 = scale_start

        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)

        a = (A0 - A1) / (sigmoid(-k) - sigmoid(k))
        b = 0.5 * (A0 + A1 - a)

        x = np.linspace(-1, 1, num_timesteps)
        y = a * sigmoid(- k * x) + b

        alphas_cumprod = y
        alphas = np.zeros_like(alphas_cumprod)
        alphas[0] = alphas_cumprod[0]
        alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = 1 - alphas
        betas = np.clip(betas, 0, 1)
    elif beta_schedule == "segment":
        time_segment = kwargs['time_segment']
        segment_diff = kwargs['segment_diff']

        assert np.sum(time_segment) == num_timesteps
        alphas_cumprod = []
        for i in range(len(time_segment)):
            time_this = time_segment[i] + 1
            params = segment_diff[i]
            _, alphas_this = advance_schedule(time_this, **params, return_alphas_bar=True)
            alphas_cumprod.extend(alphas_this[1:])
        alphas_cumprod = np.array(alphas_cumprod)

        alphas = np.zeros_like(alphas_cumprod)
        alphas[0] = alphas_cumprod[0]
        alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = 1 - alphas
        betas = np.clip(betas, 0, 1)
    else:
        raise NotImplementedError(beta_schedule)

    assert betas.shape == (num_timesteps,)
    return betas


def advance_schedule(timesteps, scale_start, scale_end, width, return_alphas_bar=False):
    """Advance schedule helper for segment schedule."""
    k = width
    A0 = scale_end
    A1 = scale_start

    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    a = (A0 - A1) / (sigmoid(-k) - sigmoid(k))
    b = 0.5 * (A0 + A1 - a)

    x = np.linspace(-1, 1, timesteps)
    y = a * sigmoid(- k * x) + b

    alphas_cumprod = y
    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)

    if not return_alphas_bar:
        return betas
    else:
        return betas, alphas_cumprod


class GeneralCategoricalTransition(nn.Module):
    """
    MolDiff's categorical diffusion implementation.
    This is the exact class from the MolDiff repository.
    """

    def __init__(self, betas, num_classes, init_prob=None):
        super().__init__()
        self.eps = 1e-30
        self.num_classes = num_classes

        # Handle different init_prob options
        if init_prob is None:
            self.init_prob = np.ones(num_classes) / num_classes
        elif init_prob == 'absorb':  # absorb all states into the first one
            init_prob = 0.01 * np.ones(num_classes)
            init_prob[0] = 1
            self.init_prob = init_prob / np.sum(init_prob)
        elif init_prob == 'tomask':  # absorb all states into the mask type (last one)
            init_prob = 0.001 * np.ones(num_classes)
            init_prob[-1] = 1.
            self.init_prob = init_prob / np.sum(init_prob)
        elif init_prob == 'uniform':
            self.init_prob = np.ones(num_classes) / num_classes
        else:
            self.init_prob = init_prob / np.sum(init_prob)

        self.betas = betas
        self.num_timesteps = len(betas)

        # Construct transition matrices for q(x_t | x_{t-1})
        q_one_step_mats = [self._get_transition_mat(t) for t in range(0, self.num_timesteps)]
        q_one_step_mats = np.stack(q_one_step_mats, axis=0)  # (T, K, K)

        # Construct transition matrices for q(x_t | x_0)
        q_mat_t = q_one_step_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):
            q_mat_t = np.tensordot(q_mat_t, q_one_step_mats[t], axes=[[1], [0]])
            q_mats.append(q_mat_t)
        q_mats = np.stack(q_mats, axis=0)

        transpose_q_onestep_mats = np.transpose(q_one_step_mats, axes=[0, 2, 1])

        # Register as buffers (not parameters)
        self.register_buffer('q_mats', torch.from_numpy(q_mats).float())
        self.register_buffer('transpose_q_onestep_mats', torch.from_numpy(transpose_q_onestep_mats).float())

    def _get_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1})."""
        beta_t = self.betas[t]
        mat = np.repeat(np.expand_dims(self.init_prob, 0), self.num_classes, axis=0)
        mat = beta_t * mat
        mat_diag = np.eye(self.num_classes) * (1. - beta_t)
        mat = mat + mat_diag
        return mat

    def add_noise(self, v, time_step, batch):
        """
        Add noise to categorical variables.
        Returns 3 values: (v_perturbed, log_vt, log_v0) - this is the key difference!
        """
        log_node_v0 = index_to_log_onehot(v, self.num_classes)
        v_perturbed, log_node_vt = self.q_vt_sample(log_node_v0, time_step, batch)
        v_perturbed = F.one_hot(v_perturbed, self.num_classes).float()
        return v_perturbed, log_node_vt, log_node_v0

    def onehot_encode(self, v):
        return F.one_hot(v, self.num_classes).float()

    def q_vt_sample(self, log_v0, t, batch):
        """Sample from q(vt | v0)"""
        log_q_vt_v0 = self.q_vt_pred(log_v0, t, batch)
        sample_class = log_sample_categorical(log_q_vt_v0)
        log_sample = index_to_log_onehot(sample_class, self.num_classes)
        return sample_class, log_sample

    def q_vt_pred(self, log_v0, t, batch):
        """Compute q(vt | v0) using precomputed transition matrices."""
        qt_mat = extract(self.q_mats, t, batch, ndim=1)
        q_vt = torch.einsum('...i,...ij->...j', log_v0.exp(), qt_mat)
        return torch.log(q_vt + self.eps).clamp_min(-32.)

    def q_v_posterior(self, log_v0, log_vt, t, batch, v0_prob=True):
        """
        Compute posterior q(v_{t-1} | v_t, v_0).
        This is the key method for KL divergence loss computation.
        """
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)

        fact1 = extract(self.transpose_q_onestep_mats, t, batch, ndim=1)
        fact1 = torch.einsum('bj,bjk->bk', torch.exp(log_vt), fact1)

        if v0_prob:
            fact2 = extract(self.q_mats, t_minus_1, batch, ndim=1)
            fact2 = torch.einsum('bj,bjk->bk', torch.exp(log_v0), fact2)
        else:
            fact2 = extract(self.q_mats, t_minus_1, batch, ndim=1)
            class_v0 = log_v0.argmax(dim=-1)
            fact2 = fact2[torch.arange(len(class_v0)), class_v0]

        ndim = log_v0.ndim
        if ndim == 2:
            t_expand = t[batch].unsqueeze(-1)
        elif ndim == 3:
            t_expand = t[batch].unsqueeze(-1).unsqueeze(-1)
        else:
            raise NotImplementedError('ndim not supported')

        out = torch.log(fact1 + self.eps).clamp_min(-32.) + torch.log(fact2 + self.eps).clamp_min(-32.)
        out = out - torch.logsumexp(out, dim=-1, keepdim=True)
        out_t0 = log_v0
        out = torch.where(t_expand == 0, out_t0, out)
        return out

    def compute_v_Lt(self, log_v_post_true, log_v_post_pred, log_v0, t, batch):
        """
        Compute KL divergence loss term.
        This is what MolDiff uses instead of cross-entropy.
        """
        kl_v = categorical_kl(log_v_post_true, log_v_post_pred)
        decoder_nll_v = -log_categorical(log_v0, log_v_post_pred)

        ndim = log_v_post_true.ndim
        if ndim == 2:
            mask = (t == 0).float()[batch]
        elif ndim == 3:
            mask = (t == 0).float()[batch].unsqueeze(-1)
        else:
            raise NotImplementedError('ndim not supported')

        loss_v = mask * decoder_nll_v + (1 - mask) * kl_v
        return loss_v

    def sample_init(self, n):
        """Sample initial state for generation."""
        init_log_atom_vt = torch.log(
            torch.from_numpy(self.init_prob) + self.eps).clamp_min(-32.).to(self.q_mats.device)
        init_log_atom_vt = init_log_atom_vt.unsqueeze(0).repeat(n, 1)
        init_types = log_sample_categorical(init_log_atom_vt)
        init_onehot = self.onehot_encode(init_types)
        log_vt = index_to_log_onehot(init_types, self.num_classes)
        return init_types, init_onehot, log_vt


# For continuous variables (positions)
class ContinuousTransition(nn.Module):
    """Continuous diffusion for positions (from MolDiff)."""

    def __init__(self, betas, num_classes=None, scaling=1.):
        super().__init__()
        self.num_classes = num_classes
        self.scaling = scaling
        alphas = 1. - betas
        alphas_bar = np.cumprod(alphas, axis=0)
        alphas_bar_prev = np.concatenate([[1.], alphas_bar[:-1]])

        self.betas = to_torch_const(betas)
        self.alphas = to_torch_const(alphas)
        self.alphas_bar = to_torch_const(alphas_bar)
        self.alphas_bar_prev = to_torch_const(alphas_bar_prev)

        # for q(x_{t-1}|x_0, x_t)
        self.coef_x0 = to_torch_const(np.sqrt(alphas_bar_prev) * betas / (1 - alphas_bar))
        self.coef_xt = to_torch_const(np.sqrt(alphas) * (1 - alphas_bar_prev) / (1 - alphas_bar))
        self.std = to_torch_const(np.sqrt((1 - alphas_bar_prev) * betas / (1 - alphas_bar)))

    def add_noise(self, x, time_step, batch):
        """Add Gaussian noise to continuous variables."""
        if self.num_classes is not None:  # categorical values using continuous noise
            x = F.one_hot(x, self.num_classes).float()
        x = x / self.scaling

        a_bar = self.alphas_bar.index_select(0, time_step)
        a_bar = a_bar.index_select(0, batch).unsqueeze(-1)
        noise = torch.zeros_like(x).to(x)
        noise.normal_()
        pert = a_bar.sqrt() * x + (1 - a_bar).sqrt() * noise

        if self.num_classes is None:  # continuous values
            return pert
        else:
            return pert, x