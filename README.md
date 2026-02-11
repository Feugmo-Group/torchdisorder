# TorchDisorder

**Differentiable Amorphous Structure Generation from Scattering Data**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Mathematical Background](#mathematical-background)
5. [Package Organization](#package-organization)
6. [Workflow](#workflow)
7. [Constraint Generation](#constraint-generation)
8. [Configuration Files](#configuration-files)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)
11. [Citation](#citation)

---

## Overview

### What is TorchDisorder?

TorchDisorder is a PyTorch-based framework for generating atomistic models of amorphous materials that match experimental scattering data while satisfying local structural constraints. It combines:

- **Reverse Monte Carlo (RMC) principles**: Optimize atomic positions to match diffraction data
- **Differentiable programming**: Use gradient descent instead of random moves
- **Constrained optimization**: Enforce physically meaningful local environments
- **Cooper's method**: Handle inequality constraints via augmented Lagrangian

### Why TorchDisorder?

Traditional RMC methods suffer from:
- Slow convergence (random moves)
- Unphysical local structures (no chemical constraints)
- Difficulty with complex constraints

TorchDisorder addresses these by:
- Using gradient information for faster convergence
- Enforcing coordination numbers, bond angles, and order parameters
- Supporting environment-specific constraints (e.g., PS₄³⁻ tetrahedra)

### Supported Materials

| Material Class | Examples | Scattering Type |
|----------------|----------|-----------------|
| Oxide glasses | SiO₂, GeO₂, B₂O₃ | Neutron, X-ray |
| Sulfide glasses | Li₂S-P₂S₅, Li₂S-SiS₂ | X-ray |
| Metallic glasses | Pd-Ni-P, Cu-Zr | X-ray |
| Chalcogenides | As₂S₃, GeSe₂ | Neutron, X-ray |

---

## Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA (optional, for GPU acceleration)

### Install from Source

```bash
# Clone repository
git clone https://github.com/your-repo/torchdisorder.git
cd torchdisorder

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
ase>=3.22.0           # Atomic Simulation Environment
hydra-core>=1.3.0     # Configuration management
omegaconf>=2.3.0
wandb>=0.15.0         # Experiment tracking (optional)
plotly>=5.14.0        # Interactive plots
matplotlib>=3.7.0
```

---

## Quick Start

### 1. Prepare Your Data

Create CSV files with your experimental data:

**S(Q) or F(Q) data** (`S_of_Q.csv`):
```csv
Q,S,dS
0.5,0.92,0.05
0.6,0.88,0.05
...
```

**G(r) or g(r) data** (`g_of_r.csv`):
```csv
r,G,dG
0.5,-0.12,0.02
0.6,-0.08,0.02
...
```

### 2. Run Optimization

```bash
# Basic run with default settings
python scripts/train.py data=LiPS structure=LiPS

# Specify target function
python scripts/train.py data=LiPS structure=LiPS target=G_r

# Adjust optimization parameters
python scripts/train.py data=LiPS max_steps=10000 trainer.primal_lr=0.002
```

### 3. View Results

Results are saved to `outputs/<material>_<date>/<time>/`:
- `plots/` - Spectrum comparisons, constraint evolution
- `checkpoints/` - Model snapshots
- `final_results/` - Optimized structure (CIF), final spectra

---

## Mathematical Background

### Scattering Theory

#### Structure Factor S(Q)

The total structure factor for a multi-component system is computed using the **Faber-Ziman formalism**:

```
S(Q) = 1 + ρ Σ_α Σ_β w_αβ(Q) ∫₀^rmax 4πr² [g_αβ(r) - 1] sin(Qr)/(Qr) dr
```

where:
- `Q` is the scattering vector magnitude (Å⁻¹)
- `ρ` is the number density (atoms/Å³)
- `g_αβ(r)` is the partial pair distribution function
- `w_αβ(Q)` are the weighting factors

#### Weighting Factors

**For neutron scattering:**
```
w_αβ = (c_α c_β b_α b_β) / <b>²
```

where `b_α` are coherent scattering lengths and `c_α` are concentrations.

**For X-ray scattering:**
```
w_αβ(Q) = (c_α c_β f_α(Q) f_β(Q)) / <f(Q)>²
```

where the atomic form factors are computed using **Cromer-Mann coefficients**:
```
f(Q) = Σᵢ aᵢ exp(-bᵢ (Q/4π)²) + c
```

#### Related Functions

| Function | Definition | Asymptotic Behavior |
|----------|------------|---------------------|
| S(Q) | Structure factor | → 1 as Q → ∞ |
| F(Q) | Reduced structure factor: `F(Q) = Q[S(Q) - 1]` | → 0 as Q → ∞ |
| g(r) | Pair distribution function | → 1 as r → ∞ |
| G(r) | Reduced PDF: `G(r) = 4πρr[g(r) - 1]` | → 0 as r → ∞ |
| T(r) | Total correlation: `T(r) = G(r) + 4πρr` | → 4πρr as r → ∞ |

#### Fourier Transform Relationships

```
Q-space                              r-space
───────                              ───────
                    FT
S(Q)  ─────────────────────────────→  g(r)
  │                                     │
  │ ×Q                              ×4πρr│
  ↓                                     ↓
F(Q)  ─────────────────────────────→  G(r)
                    FT
```

**Forward transform (Q → r):**
```
G(r) = (2/π) ∫₀^Qmax F(Q) sin(Qr) dQ
```

**Inverse transform (r → Q):**
```
F(Q) = ∫₀^rmax G(r) sin(Qr) dr
```

### Partial RDF Calculation

The partial pair distribution function `g_αβ(r)` is computed using **Gaussian kernel density estimation**:

```
g_αβ(r) = V/(4πr² N_α N_β) Σᵢ∈α Σⱼ∈β (1/√(2π)σ) exp[-(r - rᵢⱼ)²/(2σ²)]
```

where:
- `V` is the cell volume
- `N_α, N_β` are atom counts
- `rᵢⱼ` is the interatomic distance
- `σ` is the kernel width (smoothing parameter)

### Order Parameters

TorchDisorder supports several order parameters to characterize local structure:

#### Coordination Number (CN)
```
CN_i = Σⱼ S(rᵢⱼ)
```
where `S(r)` is a smooth switching function.

#### Tetrahedral Order Parameter (q_tet)
```
q_tet = 1 - (3/8) Σⱼ₌₁³ Σₖ₌ⱼ₊₁⁴ (cos(ψⱼₖ) + 1/3)²
```
- Perfect tetrahedron: `q_tet = 1`
- Random arrangement: `q_tet ≈ 0`

#### Steinhardt Order Parameters (q_l)
```
q_l = √((4π/(2l+1)) Σₘ₌₋ₗˡ |q_lm|²)
```
where `q_lm = (1/N_b) Σⱼ Y_lm(θᵢⱼ, φᵢⱼ)`

Common values:
- `q₂`: Linear arrangements
- `q₄`: Cubic/tetrahedral symmetry
- `q₆`: Icosahedral symmetry

### Constrained Optimization

TorchDisorder uses **Cooper's augmented Lagrangian method** for constrained optimization.

#### Problem Formulation

Minimize the χ² loss subject to structural constraints:

```
min_r L(r) = χ²(r)   subject to   G_k(r) ≤ 0
```

where:
```
χ² = Σᵢ (yᵢᶜᵃˡᶜ - yᵢᵉˣᵖ)² / σᵢ²
```

#### Augmented Lagrangian

```
L_AL(r, λ) = χ²(r) + Σₖ [λₖ Gₖ(r) + (ρₖ/2) Gₖ(r)²]
```

#### Gradient Descent Ascent (GDA)

The optimization proceeds via alternating updates:

**Primal update** (atomic positions):
```
r^(t+1) ← r^(t) - η_r ∇_r L_AL
```

**Dual update** (Lagrange multipliers):
```
λₖ^(t+1) ← [λₖ^(t) + η_λ Gₖ(r^(t))]₊
```

where `[·]₊` projects to non-negative values (for inequality constraints).

---

## Package Organization

```
torchdisorder/
│
├── model/                      # Core computational modules
│   ├── scattering.py          # Unified S(Q), F(Q), g(r), G(r), T(r) calculator
│   ├── xrd.py                 # High-level XRD model wrapper
│   ├── loss.py                # χ² loss functions (CooperLoss)
│   ├── rdf.py                 # Legacy RDF calculations
│   └── generator.py           # Structure generation utilities
│
├── engine/                     # Optimization engine
│   ├── constrained_optimizer.py   # Main optimizer with constraints
│   ├── order_params.py        # Order parameter calculations (CN, q_tet, q_l)
│   ├── callbacks.py           # Training callbacks (logging, checkpointing)
│   └── optimizer.py           # Base optimizer classes
│
├── common/                     # Shared utilities
│   ├── target_rdf.py          # TargetRDFData class for experimental data
│   ├── neighbors.py           # Neighbor list computations
│   └── utils.py               # General utilities
│
├── constraints/                # Constraint generators for specific materials
│   ├── lps_generator.py       # Li-P-S glass constraints
│   ├── sio2_generator.py      # SiO₂ glass constraints
│   └── geo2_generator.py      # GeO₂ glass constraints
│
├── viz/                        # Visualization
│   └── plotting.py            # Spectrum plots, constraint evolution
│
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main configuration
│   ├── data/                  # Data configurations (LiPS, SiO2, GeO2)
│   ├── structure/             # Structure configurations
│   ├── trainer/               # Optimizer configurations
│   └── wandb/                 # Logging configurations
│
├── scripts/
│   └── train.py               # Main training script
│
└── data/                       # Example data files
    ├── crystal-structures/    # Starting CIF files
    ├── xrd_measurements/      # Experimental scattering data
    ├── json/                  # Constraint JSON files
    └── environment/           # Environment analysis files
```

### Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `UnifiedSpectrumCalculator` | `model/scattering.py` | Compute all scattering functions |
| `XRDModel` | `model/xrd.py` | High-level model for training |
| `CooperLoss` | `model/loss.py` | χ² loss with constraint support |
| `TargetRDFData` | `common/target_rdf.py` | Load and manage experimental data |
| `StructureFactorCMPWithConstraints` | `engine/optimizer.py` | Main constrained optimizer |
| `TorchSimOrderParameters` | `engine/order_params.py` | Calculate local order parameters |

---

## Workflow

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TorchDisorder Workflow                          │
└─────────────────────────────────────────────────────────────────────────┘

Step 1: PREPARE INPUT DATA
─────────────────────────────
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  Crystal     │     │ Experimental │     │  Constraint  │
    │  Structure   │     │  Scattering  │     │    JSON      │
    │   (.cif)     │     │   (.csv)     │     │   (.json)    │
    └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
           │                    │                    │
           ▼                    ▼                    ▼
Step 2: GENERATE CONSTRAINTS (if needed)
─────────────────────────────────────────
    ┌─────────────────────────────────────────────────────────┐
    │  python -m torchdisorder.constraints.lps_generator      │
    │    --input Li7P3S11.cif                                 │
    │    --supercell 5,8,5                                    │
    │    --output glass_67Li2S                                │
    └─────────────────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────┐     ┌──────────────┐
    │   Supercell  │     │  Constraint  │
    │    (.cif)    │     │   (.json)    │
    └──────┬───────┘     └──────┬───────┘
           │                    │
           ▼                    ▼
Step 3: CONFIGURE AND RUN
─────────────────────────
    ┌─────────────────────────────────────────────────────────┐
    │  python scripts/train.py                                │
    │    data=LiPS                                            │
    │    structure=LiPS                                       │
    │    target=G_r                                           │
    │    max_steps=5000                                       │
    └─────────────────────────────────────────────────────────┘
           │
           ▼
Step 4: OPTIMIZATION LOOP
─────────────────────────
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   for step in range(max_steps):                        │
    │                                                         │
    │     1. Forward pass:                                    │
    │        - Compute g(r) from positions                    │
    │        - Fourier transform → S(Q)                       │
    │        - Compute order parameters                       │
    │                                                         │
    │     2. Loss calculation:                                │
    │        - χ² = Σ (calc - exp)² / σ²                     │
    │        - Constraint violations                          │
    │        - Augmented Lagrangian                           │
    │                                                         │
    │     3. Backward pass:                                   │
    │        - Compute gradients ∂L/∂r                        │
    │                                                         │
    │     4. Update:                                          │
    │        - Positions: r ← r - η·∇L                        │
    │        - Multipliers: λ ← [λ + η·G(r)]₊                 │
    │        - Penalties: ρ ← adapt(ρ, violations)            │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
           │
           ▼
Step 5: OUTPUT
──────────────
    outputs/<material>_<date>/<time>/
    ├── plots/
    │   ├── spectrum_step_*.png      # Spectrum comparison at each checkpoint
    │   ├── constraints_evolution.png # Constraint satisfaction over time
    │   └── final_spectrum_evolution.html  # Interactive animation
    ├── checkpoints/
    │   └── step_*/                  # Model snapshots
    └── final_results/
        ├── optimized_structure.cif  # Final atomic structure
        ├── final_spectrum.csv       # Computed vs experimental
        └── constraint_analysis.json # Final constraint values
```

### Step-by-Step Guide

#### Step 1: Prepare Experimental Data

Your experimental scattering data should be in CSV format:

```python
# Example: Load and inspect data
import pandas as pd

# Q-space data (structure factor)
df_sq = pd.read_csv('data/xrd_measurements/Li3PS4/S_of_Q.csv')
print(df_sq.head())
#      Q       S      dS
# 0  0.46  0.921  0.050
# 1  0.48  0.892  0.050
# ...

# r-space data (pair distribution)
df_gr = pd.read_csv('data/xrd_measurements/Li3PS4/g_of_r.csv')
print(df_gr.head())
#      r       G      dG
# 0  0.01  -0.02  0.020
# 1  0.02  -0.03  0.020
# ...
```

**Important columns:**
- `Q` or `q`: Scattering vector (Å⁻¹)
- `r` or `R`: Distance (Å)
- `S`, `SQ`, `S(Q)`: Structure factor
- `F`, `FQ`, `F(Q)`: Reduced structure factor
- `g`, `gr`, `g(r)`: Pair distribution function
- `G`, `Gr`, `G(r)`: Reduced PDF
- `dS`, `dF`, `dG`, `error`, `uncertainty`: Uncertainties

#### Step 2: Prepare Starting Structure

You need a starting atomic configuration in CIF format:

```python
from ase.io import read, write
from ase.build import make_supercell
import numpy as np

# Load crystal structure
atoms = read('data/crystal-structures/Li7P3S11.cif')

# Create supercell
supercell = make_supercell(atoms, [[5,0,0], [0,8,0], [0,0,5]])

# Optional: Add random displacement to break symmetry
supercell.positions += np.random.randn(*supercell.positions.shape) * 0.1

# Save
write('glass_structure.cif', supercell)
```

#### Step 3: Generate Constraints

Use the constraint generators to create environment-based constraints:

```bash
# For Li-P-S glass
python -m torchdisorder.constraints.lps_generator \
    --input glass_structure.cif \
    --output data/json/my_constraints

# For SiO₂ glass  
python -m torchdisorder.constraints.sio2_generator \
    --input silica_structure.cif \
    --cutoff 2.2 \
    --output data/json/sio2_constraints

# For GeO₂ glass
python -m torchdisorder.constraints.geo2_generator \
    --input germania_structure.cif \
    --cutoff 2.4 \
    --output data/json/geo2_constraints
```

#### Step 4: Configure and Run

```bash
# Run with default settings
python scripts/train.py data=LiPS structure=LiPS

# Customize via command line
python scripts/train.py \
    data=LiPS \
    structure=LiPS \
    target=G_r \
    max_steps=10000 \
    trainer.primal_lr=0.002 \
    trainer.constraint_warmup_steps=1000 \
    data.q_min=0.5 \
    data.q_max=20.0
```

---

## Constraint Generation

### Overview

Constraints encode prior knowledge about local atomic environments. TorchDisorder uses **environment-based grouping** where atoms with similar local chemistry share constraint settings.

### Constraint JSON Format

```json
{
  "metadata": {
    "version": "v6",
    "structure_type": "li_p_s_glass",
    "source_file": "Li7P3S11.cif",
    "central_element": "P",
    "neighbor_element": "S"
  },
  "cutoff": 3.5,
  "element_filter": [15, 16],
  "atom_constraints": {
    "0": {
      "atom_index": 0,
      "element": "P",
      "environment": "P4",
      "environment_label": "PS4^3- (isolated tetrahedron)",
      "order_parameters": {
        "cn": {
          "target": 4.0,
          "tolerance": 0.5,
          "weight": 1.5
        },
        "tet": {
          "target": 0.85,
          "min": 0.7,
          "max": 1.0,
          "weight": 2.0
        },
        "q4": {
          "target": 0.6,
          "min": 0.4,
          "max": 0.8,
          "weight": 0.5
        }
      }
    },
    "5": {
      "atom_index": 5,
      "element": "P",
      "environment": "Pa",
      "environment_label": "P2S7^4- (corner-sharing dimer)",
      "order_parameters": {
        "cn": {"target": 4.0, "tolerance": 0.5},
        "tet": {"target": 0.65, "min": 0.4, "max": 0.85}
      }
    }
  },
  "environment_priorities": {
    "P4": 2.0,
    "Pa": 1.5,
    "P2": 1.5,
    "P3": 1.2
  },
  "global_constraints": {
    "density": {
      "target": 1.85,
      "tolerance": 0.1,
      "unit": "g/cm³"
    }
  }
}
```

### Li-P-S Glass Environments

| Code | Structure | Description | Typical CN | q_tet |
|------|-----------|-------------|------------|-------|
| `P4` | PS₄³⁻ | Isolated tetrahedron | 4 | 0.8-1.0 |
| `Pa` | P₂S₇⁴⁻ | Corner-sharing dimer | 4 | 0.5-0.8 |
| `P2` | P₂S₆⁴⁻ | Edge-sharing (P-P bond) | 4 | 0.4-0.7 |
| `P3` | PS₃⁻ | Pyramidal (3-coordinate) | 3 | N/A |

### SiO₂ Glass Environments

| Code | Structure | Description | Typical CN | q_tet |
|------|-----------|-------------|------------|-------|
| `Si4` | SiO₄ | Tetrahedral (normal) | 4 | 0.8-1.0 |
| `Si3` | SiO₃ | Under-coordinated (defect) | 3 | N/A |
| `Si5` | SiO₅ | Over-coordinated (rare) | 5 | N/A |
| `Si6` | SiO₆ | Octahedral (high-P phase) | 6 | N/A |

### GeO₂ Glass Environments

| Code | Structure | Description | Typical CN | q_tet |
|------|-----------|-------------|------------|-------|
| `Ge4` | GeO₄ | Tetrahedral (quartz-like) | 4 | 0.7-1.0 |
| `Ge5` | GeO₅ | Five-coordinate | 5 | N/A |
| `Ge6` | GeO₆ | Octahedral (rutile-like) | 6 | N/A |

---

## Configuration Files

### Main Configuration (`configs/config.yaml`)

```yaml
defaults:
  - data: LiPS
  - structure: LiPS
  - trainer: cooper
  - wandb: enabled

# Target function: S_Q, F_Q, T_r, g_r, G_r
target: G_r

# Device
accelerator: cuda  # or 'cpu', 'mps'

# Optimization
max_steps: 5000
seed: 42

# Output
run_name: ${now:%Y%m%d_%H%M%S}
output_dir: outputs/${data.name}_${now:%Y-%m-%d}/${now:%H-%M-%S}
```

### Data Configuration (`configs/data/LiPS.yaml`)

```yaml
root_dir: ${oc.env:PROJECT_ROOT,./}/data

# Scattering type
scattering_type: xray  # or 'neutron'

# Data paths
data:
  s_of_q_path: ${data.root_dir}/xrd_measurements/Li3PS4/S_of_Q.csv
  g_of_r_path: ${data.root_dir}/xrd_measurements/Li3PS4/g_of_r.csv
  t_of_r_path: null
  
  # Format specification
  input_is_F_Q: false  # true=F(Q), false=S(Q), null=auto-detect
  
  # Subsampling
  stride_q: 4
  stride_r: 1

# Q-space range (Å⁻¹)
q_min: 0.5
q_max: 17.5

# r-space range (Å)
r_min: 0.01
r_max: 50.0
n_r_bins: 500

# Spectrum calculation
kernel_width: 0.03
number_density: 0.0352  # atoms/Å³

# Scattering parameters
neutron_scattering_lengths:
  Li: -1.90
  P: 5.13
  S: 2.847

xray_form_factor_params:
  Li:
    a: [1.1282, 0.7508, 0.6175, 0.4653]
    b: [3.9546, 1.0524, 85.3905, 168.261]
    c: [0.0377]
  P:
    a: [6.4345, 4.1791, 1.78, 1.4908]
    b: [1.9067, 27.157, 0.526, 68.1645]
    c: [1.1149]
  S:
    a: [6.9053, 5.2034, 1.4379, 1.5863]
    b: [1.4679, 22.2151, 0.2536, 56.172]
    c: [0.8669]

# Order parameters
central: 'P'
neighbour: 'S'
cutoff: 5.5

# Constraints
json_path: ${data.root_dir}/json/glass_67Li2S_constraints.json
```

### Trainer Configuration (`configs/trainer/cooper.yaml`)

```yaml
# Learning rates
primal_lr: 0.001      # Position update rate
dual_lr: 0.01         # Lagrange multiplier update rate

# Gradient safety
grad_clip_norm: 1.0   # Maximum gradient norm
max_displacement: 0.1  # Maximum position change per step (Å)

# Constraint handling
constraint_warmup_steps: 500  # Steps before enabling constraints
use_adaptive_penalty: true

# Penalty coefficients
penalty:
  init: 10.0
  growth_rate: 1.5
  decay_rate: 0.95
  max_penalty: 1000.0
  min_penalty: 1.0
  patience: 10

# Logging
plot_interval: 200
checkpoint_interval: 200
log_interval: 100
```

---

## Advanced Usage

### Custom Target Functions

```python
from torchdisorder.model.loss import CooperLoss

# Use G(r) as target
loss_fn = CooperLoss(
    target_data=rdf_data,
    target_type='G_r',  # Options: 'S_Q', 'F_Q', 'T_r', 'g_r', 'G_r'
    device='cuda'
)
```

### Manual Spectrum Calculation

```python
from torchdisorder.model.scattering import UnifiedSpectrumCalculator, ScatteringConfig
import torch

# Configure calculator
config = ScatteringConfig(
    scattering_type='xray',
    xray_form_factor_params={
        'Si': {'a': [...], 'b': [...], 'c': [...]},
        'O': {'a': [...], 'b': [...], 'c': [...]}
    },
    kernel_width=0.05
)
calc = UnifiedSpectrumCalculator(config)

# Define structure
symbols = ['Si', 'O', 'O', 'Si', ...]
positions = torch.tensor([[0, 0, 0], [1.6, 0, 0], ...])
cell = torch.eye(3) * 20.0

# Define bins
r_bins = torch.linspace(0.01, 10.0, 500)
q_bins = torch.linspace(0.5, 25.0, 500)

# Compute all functions
results = calc.compute_all(symbols, positions, cell, r_bins, q_bins)
print(results.keys())  # ['S_Q', 'F_Q', 'g_r', 'G_r', 'T_r']
```

### Environment Analysis

```python
from torchdisorder.engine.order_params import TorchSimOrderParameters
from ase.io import read

# Load structure
atoms = read('structure.cif')

# Initialize calculator
op_calc = TorchSimOrderParameters(cutoff=3.5, device='cuda')

# Compute for all P atoms
positions = torch.tensor(atoms.positions, device='cuda')
cell = torch.tensor(atoms.cell.array, device='cuda')
symbols = atoms.get_chemical_symbols()

p_indices = [i for i, s in enumerate(symbols) if s == 'P']

for i in p_indices:
    cn = op_calc.coordination_number(positions, cell, i, neighbor_element='S')
    q_tet = op_calc.tetrahedral_order(positions, cell, i)
    print(f"Atom {i}: CN={cn:.2f}, q_tet={q_tet:.3f}")
```

### Batch Processing

```bash
# Run multiple configurations
for target in S_Q F_Q G_r; do
    for lr in 0.001 0.002 0.005; do
        python scripts/train.py \
            target=$target \
            trainer.primal_lr=$lr \
            run_name="${target}_lr${lr}" &
    done
done
wait
```

---

## Troubleshooting

### Common Issues

#### 1. Loss is Always Zero

**Symptom:** `Loss=0.000000` throughout optimization

**Cause:** Uncertainty values are all zeros (marked as placeholders)

**Solution:** Check your CSV file for an `error` or `uncertainty` column with zeros. Either:
- Remove the column (default 0.05 will be used)
- Fill with realistic uncertainties
- Use v11+ which auto-detects this issue

#### 2. NaN in Loss

**Symptom:** `Loss=nan` after some steps

**Cause:** Numerical instability, often from atoms getting too close

**Solution:**
```yaml
# Reduce learning rate
trainer:
  primal_lr: 0.0005
  grad_clip_norm: 0.5
  max_displacement: 0.05
```

#### 3. Constraints Not Satisfied

**Symptom:** Violations remain high

**Solution:**
```yaml
# Increase constraint weight
trainer:
  penalty:
    init: 50.0
    growth_rate: 2.0
  constraint_warmup_steps: 200
```

#### 4. Slow Convergence

**Symptom:** Loss decreases very slowly

**Solution:**
- Increase `stride_q` to reduce Q-points
- Use coarser `kernel_width`
- Reduce `r_max` and `q_max` to focus on relevant range

#### 5. Memory Issues

**Symptom:** CUDA out of memory

**Solution:**
```yaml
# Reduce data points
data:
  stride_q: 8
  stride_r: 2
  q_max: 15.0
  r_max: 20.0
```

### Debug Mode

```bash
# Run with verbose output
python scripts/train.py hydra.verbose=true

# Check configuration
python scripts/train.py --cfg job
```

---

## Citation

If you use TorchDisorder in your research, please cite:

```bibtex
@software{torchdisorder2024,
  title={TorchDisorder: Differentiable Amorphous Structure Generation from Scattering Data},
  author={Advait Gore, Xander Gouws, Tetsassi Feugmoand  Conrard Giresse},
  year={2024},
  institution={University of Waterloo},
  url={https://github.com/feugmo-group/torchdisorder}
}
```



## License

MIT License

Copyright (c) 2024 Tetsassi Feugmo Research Group, University of Waterloo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
