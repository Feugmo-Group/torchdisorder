# TorchDisorder v6

Differentiable structure optimization from scattering data with environment-based constraints.

## What's New in v6

### 1. Unified Scattering Module (`model/scattering.py`)

A single, unified interface replaces the separate `rdf.py` and `rdf2.py` files from v5.

**Features:**
- Computes all scattering functions through one interface: `S(Q)`, `F(Q)`, `g(r)`, `G(r)`, `T(r)`
- Proper normalization for both Faber-Ziman and other conventions
- Supports both neutron and X-ray scattering
- Direct Debye formula computation for efficiency
- Full gradient flow through PyTorch autograd

**Usage:**
```python
from torchdisorder.model.scattering import UnifiedSpectrumCalculator, ScatteringConfig

config = ScatteringConfig(
    neutron_scattering_lengths={'Li': -1.90, 'P': 5.13, 'S': 2.847},
    xray_form_factor_params={...},
    kernel_width=0.1,
)
calc = UnifiedSpectrumCalculator(config)

# Compute specific output
S_Q = calc.compute(symbols, positions, cell, q_bins=q, output='S_Q')

# Compute all outputs efficiently
results = calc.compute_all(symbols, positions, cell, r_bins=r, q_bins=q)
# results = {'g_r': ..., 'G_r': ..., 'T_r': ..., 'S_Q': ..., 'F_Q': ...}
```

### 2. Environment-Based Constraint Grouping (`engine/constrained_optimizer.py`)

**v5 Approach:** Group constraints by order parameter type (cn, tet, q4, etc.)
- One Cooper constraint for all coordination number constraints
- One Cooper constraint for all tetrahedral constraints
- etc.

**v6 Approach:** Group constraints by local atomic environment
- One Cooper constraint for PS₄³⁻ tetrahedral units
- One Cooper constraint for P₂S₇⁴⁻ bridging units  
- One Cooper constraint for PS₃⁻ terminal units
- etc.

**Benefits:**
1. **Better physics:** Atoms in the same environment have correlated structural parameters
2. **Smarter adaptation:** Penalties can be tuned per-environment
3. **Easier interpretation:** Constraint satisfaction maps directly to structural motifs
4. **Priority weighting:** Critical environments (e.g., PS₄) can have higher priority

**Usage:**
```python
from torchdisorder.engine import EnvironmentConstrainedOptimizer

cmp = EnvironmentConstrainedOptimizer(
    model=xrd_model,
    base_state=sim_state,
    target=rdf_data,
    constraints_file='constraints.json',  # Now includes environment info
    device='cuda',
)
```

**Constraint JSON Format (v6):**

The lps_generator.py now outputs v6-compatible format with `"environment"` key:

```json
{
  "metadata": {
    "version": "v6",
    "structure_type": "li_p_s_glass",
    ...
  },
  "cutoff": 3.5,
  "element_filter": [15, 16],
  "atom_constraints": {
    "0": {
      "atom_index": 0,
      "element": "P",
      "environment": "P4",
      "environment_label": "PS4^3-",
      "order_parameters": {
        "cn": {"target": 4.0, "tolerance": 0.5, "weight": 1.5},
        "tet": {"target": 0.85, "min": 0.7, "max": 1.0, "weight": 2.0},
        "q4": {"target": 0.6, "min": 0.4, "max": 0.8, "weight": 0.5}
      }
    },
    "5": {
      "atom_index": 5,
      "element": "P",
      "environment": "Pa",
      "environment_label": "P2S7^4- (dimer)",
      "order_parameters": {
        "cn": {"target": 4.0, "tolerance": 0.5, "weight": 1.0},
        "tet": {"target": 0.6, "min": 0.4, "max": 0.8, "weight": 1.5}
      }
    }
  },
  "environment_priorities": {
    "P4": 2.0,
    "Pa": 1.5,
    "P2": 1.5,
    "P3": 1.2
  },
  "global_constraints": {...}
}
```

**Key changes from v5:**
- Uses `"environment"` key (not `"environment_type"`)
- Added `"environment_priorities"` section for adaptive weighting
- Added `"version": "v6"` in metadata

### 3. Adaptive Penalty Coefficients

**v5:** Fixed penalty coefficient ρ for all constraints

**v6:** Per-constraint adaptive penalties that:
- **Grow** exponentially when a constraint is persistently violated
- **Decay** when a constraint is consistently satisfied
- **Cap** at maximum to prevent numerical instability

**Algorithm:**
```
For each constraint group:
    if violation > threshold for patience steps:
        ρ *= growth_rate  (e.g., 1.5)
    if satisfied for patience steps:
        ρ *= decay_rate   (e.g., 0.95)
```

**Configuration:**
```python
penalty_config = {
    'init': 10.0,           # Starting penalty
    'growth_rate': 1.5,     # Factor for increasing
    'decay_rate': 0.95,     # Factor for decreasing
    'max_penalty': 1000.0,  # Upper cap
    'min_penalty': 1.0,     # Lower cap
    'patience': 10,         # Steps before adapting
}
```

## Installation

```bash
# Clone repository
git clone <repo_url>
cd torchdisorder_v6

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

### Target Selection

TorchDisorder v6 supports multiple target functions. Choose via config:

```bash
# Structure factor (default)
python scripts/train.py target=S_Q

# Reduced structure factor F(Q) = Q[S(Q)-1]
python scripts/train.py target=F_Q

# Total correlation function
python scripts/train.py target=T_r

# Pair distribution function
python scripts/train.py target=g_r
```

**Which target to use?**
- `S_Q`: Best for high-Q data, oscillates around 1
- `F_Q`: Better for low-Q features, oscillates around 0
- `T_r`: Real-space, emphasizes short-range structure
- `g_r`: Real-space, normalized to 1 at large r

### Training Examples

```bash
# Default: optimize to match S(Q)
python scripts/train.py

# Optimize to match F(Q) with higher initial penalty
python scripts/train.py target=F_Q penalty.init=20.0

# Use adaptive penalties
python scripts/train.py constraints.use_adaptive=true

# Customize penalty behavior
python scripts/train.py penalty.init=20.0 penalty.growth_rate=2.0

# Disable WandB logging
python scripts/train.py wandb=disabled

# Multiple options
python scripts/train.py target=F_Q max_steps=50000 constraints.warmup_steps=200
```

## Module Structure

```
torchdisorder/
├── __init__.py              # Package exports
├── model/
│   ├── scattering.py        # Unified scattering calculations
│   ├── xrd.py               # XRD/neutron diffraction model
│   ├── loss.py              # Loss functions
│   └── generator.py         # Structure generators
├── engine/
│   ├── constrained_optimizer.py  # Environment-based optimization
│   ├── order_params.py      # Order parameter calculations
│   └── callbacks.py         # Training callbacks
├── common/
│   ├── target_rdf.py        # Target data structures
│   ├── neighbors.py         # Neighbor list utilities
│   └── utils.py             # General utilities
├── constraints/
│   ├── lps_generator.py     # Li-P-S constraint generator
│   ├── sio2_generator.py    # SiO₂ constraint generator
│   └── geo2_generator.py    # GeO₂ constraint generator
└── viz/
    └── plotting.py          # Visualization utilities
```

## Scattering Function Relationships

```
Reciprocal Space (Q-space):
    S(Q) = 1 + (4πρ/Q) ∫ r·G(r)·sin(Qr) dr     [Structure factor]
    F(Q) = Q[S(Q) - 1]                          [Reduced structure factor]

Real Space (r-space):
    g(r) = 1 + G(r)/(4πρr)                      [Pair distribution function]
    G(r) = 4πρr[g(r) - 1]                       [Reduced PDF]
    T(r) = 4πρr·g(r) = G(r) + 4πρr             [Total correlation]

Fourier Transform:
    S(Q) ↔ g(r) via sine transform
```

## Environment Types for Li-P-S Glass

| Environment | Description | Typical Constraints | Priority |
|-------------|-------------|---------------------|----------|
| P4 | PS₄³⁻ tetrahedral (isolated) | cn=4, tet∈[0.7,1.0], q4∈[0.4,0.8] | 2.0 |
| Pa | P₂S₇⁴⁻ dimer (bridging S) | cn=4, tet∈[0.4,0.8] | 1.5 |
| P2 | P₂S₆⁴⁻ dumbbell (P-P bond) | cn=4, tet∈[0.4,0.85] | 1.5 |
| P3 | PS₃⁻ pyramidal (3-coord) | cn=3, q2∈[0.3,0.7] | 1.2 |

**Generating constraints:**
```bash
# Generate constraints from Li7P3S11 structure for 67Li2S-33P2S5 glass
python -m torchdisorder.constraints.lps_generator \
    --input Li7P3S11.cif \
    --target 67Li2S-33P2S5 \
    --supercell 5,8,5 \
    --output glass_67Li2S

# Output files:
#   glass_67Li2S.cif              - Structure file
#   glass_67Li2S_constraints.json - v6-format constraints
#   glass_67Li2S_P_environments.txt/json - Environment analysis
```

## Environment Types for SiO2 Glass

| Environment | Description | Typical Constraints | Priority |
|-------------|-------------|---------------------|----------|
| Si4 | SiO₄ tetrahedral | cn=4, tet∈[0.7,1.0] | 2.0 |
| Si3 | SiO₃ undercoordinated (defect) | cn=3 | 1.0 |
| Si5 | SiO₅ overcoordinated (defect) | cn=5 | 1.0 |
| Si6 | SiO₆ octahedral (high pressure) | cn=6 | 1.5 |

**Generating constraints:**
```bash
# Generate constraints from crystalline SiO2
python -m torchdisorder.constraints.sio2_generator \
    --input c-SiO2.cif \
    --cutoff 2.2 \
    --output sio2_glass

# Output files:
#   sio2_glass_constraints.json    - v6-format constraints
#   sio2_glass_Si_environments.json - Environment data
#   sio2_glass_Si_environments.txt - Human-readable summary
```

## Environment Types for GeO2 Glass

| Environment | Description | Typical Constraints | Priority |
|-------------|-------------|---------------------|----------|
| Ge4 | GeO₄ tetrahedral (quartz-like) | cn=4, tet∈[0.65,1.0] | 2.0 |
| Ge3 | GeO₃ undercoordinated (defect) | cn=3 | 1.0 |
| Ge5 | GeO₅ five-coordinate | cn=5 | 1.2 |
| Ge6 | GeO₆ octahedral (rutile-like) | cn=6 | 1.5 |

**Note:** GeO2 has longer Ge-O bonds (~1.73-1.88 Å) than Si-O (~1.60-1.62 Å), so:
- Default cutoff is 2.4 Å (vs 2.2 Å for SiO2)
- Tetrahedral order parameter targets are slightly lower (0.80 vs 0.85)
- Ge shows higher tendency for 5/6-fold coordination under pressure

**Generating constraints:**
```bash
# Generate constraints from crystalline GeO2
python -m torchdisorder.constraints.geo2_generator \
    --input c-GeO2.cif \
    --cutoff 2.4 \
    --output geo2_glass

# Output files:
#   geo2_glass_constraints.json    - v6-format constraints
#   geo2_glass_Ge_environments.json - Environment data
#   geo2_glass_Ge_environments.txt - Human-readable summary
```

## Citation

If you use this code, please cite:

```bibtex
@software{torchdisorder2024,
  title={TorchDisorder: Differentiable Structure Optimization from Scattering Data},
  author={Tetsassi Feugmo Research Group},
  year={2024},
  url={https://github.com/...}
}
```

## License

MIT License
