# MolDiff Architecture: Compartmentalized Pseudocode

## Overview
MolDiff addresses the **atom-bond inconsistency problem** in 3D molecule generation through a **two-stage diffusion process** with explicit bond modeling and guidance.

---

## 1. Data Representation

```python
class MolecularGraph:
    def __init__(self):
        self.x = atom_coordinates      # Shape: [N, 3] - continuous 3D positions
        self.v = atom_types           # Shape: [N, K_atom] - discrete categorical
        self.e = bond_types           # Shape: [N, N, K_bond] - discrete categorical
        # Where N = number of atoms, K_atom = atom type classes, K_bond = bond type classes
```

---

## 2. Two-Stage Diffusion Process

### Stage 1: Bond-Focused Diffusion
```python
def stage_1_forward_process(x0, v0, e0, t):
    """
    Bond types get MORE noise than atoms
    Addresses atom-bond inconsistency
    """
    # Bond noise schedule (higher noise)
    β_bond_t = get_bond_noise_schedule(t)  # Custom aggressive schedule
    
    # Atom noise schedule (lower noise)  
    β_atom_t = get_atom_noise_schedule(t)  # Standard schedule
    
    # Forward noising
    x_t = add_gaussian_noise(x0, β_atom_t)     # Light position noise
    v_t = add_categorical_noise(v0, β_atom_t)  # Light atom type noise
    e_t = add_categorical_noise(e0, β_bond_t)  # HEAVY bond type noise
    
    return x_t, v_t, e_t
```

### Stage 2: Complete Diffusion
```python
def stage_2_forward_process(x_stage1, v_stage1, t):
    """
    Standard diffusion for atoms and coordinates
    Bonds already at absorbing state from Stage 1
    """
    x_t = add_gaussian_noise(x_stage1, β_t)
    v_t = add_categorical_noise(v_stage1, β_t)
    e_t = absorbing_state  # Bonds already fully noised
    
    return x_t, v_t, e_t
```

---

## 3. Equivariant Graph Neural Network (EGNN) Architecture

```python
class MolDiffEGNN:
    def __init__(self):
        # E(3)-equivariant message passing
        self.node_mlp = MLP([d_node, d_hidden, d_node])
        self.edge_mlp = MLP([d_edge, d_hidden, d_edge])
        self.coord_mlp = MLP([d_hidden, d_hidden, 1])
        
    def forward(self, x_t, v_t, e_t, t):
        """
        Key Innovation: Explicit edge (bond) message passing
        Unlike EDM which infers bonds from coordinates
        """
        # Embed timestep
        t_emb = sinusoidal_embedding(t)
        
        # Message passing with BOTH nodes AND edges
        for layer in range(self.num_layers):
            # Node-to-node messages
            m_ij = self.compute_node_messages(v_t, x_t, e_t)
            
            # Edge-to-edge messages (KEY INNOVATION)
            m_edge = self.compute_edge_messages(e_t, v_t, x_t)
            
            # Update node features (atom types)
            v_t = v_t + self.node_mlp(m_ij + t_emb)
            
            # Update edge features (bond types)  
            e_t = e_t + self.edge_mlp(m_edge + t_emb)
            
            # Update coordinates (equivariant)
            x_t = x_t + self.update_coordinates(m_ij, x_t)
            
        return x_t, v_t, e_t
```

---

## 4. Loss Functions

### Main Denoising Loss
```python
def moldiff_loss(model, x0, v0, e0):
    """
    Multi-modal loss with different noise schedules
    """
    t = sample_timestep()
    
    # Forward diffusion with different noise levels
    x_t, v_t, e_t = two_stage_forward(x0, v0, e0, t)
    
    # Predict original values
    x_pred, v_pred, e_pred = model(x_t, v_t, e_t, t)
    
    # Multi-component loss
    L_pos = mse_loss(x_pred, x0)                    # L2 for positions
    L_atom = kl_divergence(v_pred, v0)              # KL for atom types  
    L_bond = kl_divergence(e_pred, e0)              # KL for bond types
    
    # Weighted combination
    return λ_pos * L_pos + λ_atom * L_atom + λ_bond * L_bond
```

### Bond Guidance Loss
```python
def bond_guidance_loss(bond_predictor, x_t):
    """
    Additional guidance to ensure atom positions support bond formation
    """
    # Predict bonds from current atom positions
    bonds_pred = bond_predictor(x_t)
    
    # Encourage realistic bond lengths and geometries
    bond_length_loss = mse_loss(compute_distances(x_t), ideal_bond_lengths)
    bond_angle_loss = mse_loss(compute_angles(x_t), ideal_bond_angles)
    
    return bond_length_loss + bond_angle_loss
```

---

## 5. Noise Scheduling

```python
class MolDiffNoiseScheduler:
    def __init__(self):
        # Different schedules for different modalities
        self.β_coord = linear_schedule(β_start=1e-4, β_end=0.02, T=1000)
        self.β_atom = linear_schedule(β_start=1e-4, β_end=0.02, T=1000)  
        self.β_bond = aggressive_schedule(β_start=1e-3, β_end=0.1, T=500)  # Faster/stronger
        
    def get_noise_level(self, t, modality):
        if modality == "coordinates":
            return self.β_coord[t]
        elif modality == "atoms":
            return self.β_atom[t] 
        elif modality == "bonds":
            return self.β_bond[t]  # Higher noise for bonds
```

---

## 6. Sampling Algorithm

```python
def moldiff_sampling(model, bond_predictor, num_atoms, T=1000):
    """
    Reverse diffusion with bond guidance
    """
    # Initialize from noise
    x_T = torch.randn(num_atoms, 3)
    v_T = random_categorical(num_atoms, K_atom)
    e_T = absorbing_bonds(num_atoms, num_atoms)
    
    x_t, v_t, e_t = x_T, v_T, e_T
    
    # Reverse diffusion
    for t in reversed(range(T)):
        # Predict noise/original values
        x_pred, v_pred, e_pred = model(x_t, v_t, e_t, t)
        
        # Bond guidance (key innovation)
        if t > guidance_start_step:
            grad_x = compute_bond_guidance_gradient(bond_predictor, x_t)
            x_pred = x_pred + guidance_scale * grad_x
            
        # Denoise step with different schedules
        x_t = denoise_step(x_t, x_pred, self.β_coord[t])
        v_t = denoise_categorical(v_t, v_pred, self.β_atom[t])
        e_t = denoise_categorical(e_t, e_pred, self.β_bond[t])
        
        # Add noise (except final step)
        if t > 0:
            x_t += noise_injection(t)
            
    return x_t, v_t, e_t
```

---

## 7. Key Equations Captured

### Equivariance Constraint
- **E(3) Equivariance**: `f(Rx + t) = Rf(x) + t` for rotations R and translations t
- Implemented through coordinate updates that preserve rotational/translational symmetry

### Loss Components
- **Position Loss**: `L_pos = ||x_pred - x_0||²` (L2 norm)
- **Atom Type Loss**: `L_atom = KL(softmax(v_pred), v_0)` (KL divergence)  
- **Bond Type Loss**: `L_bond = KL(softmax(e_pred), e_0)` (KL divergence)

### Noise Scheduling Innovation
- **Standard**: `β_t = β_start + (β_end - β_start) * t/T`
- **Bond-specific**: More aggressive noise schedule for bonds to address inconsistency

### Bond Guidance
- **Gradient**: `∇_x log p(bonds|x) ≈ ∇_x bond_predictor(x)`
- Applied during sampling to ensure chemically valid bond formation

---

## 8. Training Algorithm

```python
def train_moldiff():
    for epoch in epochs:
        for batch in dataloader:
            x0, v0, e0 = batch
            
            # Sample timestep and add noise
            t = sample_timestep()
            x_t, v_t, e_t = two_stage_forward(x0, v0, e0, t)
            
            # Forward pass
            x_pred, v_pred, e_pred = model(x_t, v_t, e_t, t)
            
            # Compute multi-modal loss
            loss = moldiff_loss(model, x0, v0, e0)
            
            # Add bond guidance loss
            if use_bond_guidance:
                loss += bond_guidance_loss(bond_predictor, x_t)
            
            # Backprop
            loss.backward()
            optimizer.step()
```

## Key Innovations Summary

1. **Two-stage diffusion** with different noise schedules for bonds vs atoms
2. **Explicit bond modeling** in EGNN message passing  
3. **Bond guidance** during sampling for chemical validity
4. **E(3)-equivariant** architecture preserving molecular symmetries
5. **Multi-modal loss** with appropriate metrics for each component type