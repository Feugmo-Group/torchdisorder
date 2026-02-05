# TorchDisorder

**Differentiable Generation of Amorphous Atomic Structures from Diffraction Data**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

TorchDisorder is a PyTorch-based framework for generating physically realistic amorphous atomic structures by fitting experimental diffraction data while enforcing local coordination constraints.

## Key Features

- **Gradient-based optimization** using automatic differentiation
- **Constrained optimization** via Cooper library (augmented Lagrangian method)
- **Order parameter constraints**: tetrahedral, octahedral, bond-orientational
- **Multiple target types**: S(Q), F(Q), T(r), G(r), g(r)
- **Integration** with torch-sim, MACE, and Pymatgen

## Installation

```bash
# Using Poetry (recommended)
poetry install

# Using pip
pip install -e .
```

## Quick Start

```python
from torchdisorder import (
    StructureFactorCMPWithConstraints,
    XRDModel,
    SpectrumCalculator,
    TargetRDFData,
    generate_atoms_from_config
)
import cooper
import torch

# Load target data
rdf_data = TargetRDFData.from_dict(config, device='cuda')

# Create spectrum calculator and XRD model
spec_calc = SpectrumCalculator.from_config_dict(config)
xrd_model = XRDModel(spec_calc, rdf_data, dtype=torch.float32, device='cuda')

# Create constrained minimization problem
cmp = StructureFactorCMPWithConstraints(
    model=xrd_model,
    base_state=state,
    target_vec=rdf_data,
    constraints_file='constraints.json',
    loss_fn=loss_fn,
    device='cuda'
)

# Setup optimizers
primal_optimizer = torch.optim.Adam([state.positions], lr=1e-3)
dual_optimizer = torch.optim.SGD(cmp.dual_parameters(), lr=1e-2, maximize=True)

# Create Cooper simultaneous optimizer
cooper_opt = cooper.optim.SimultaneousOptimizer(
    cmp=cmp,
    primal_optimizers=primal_optimizer,
    dual_optimizers=dual_optimizer
)

# Optimization loop
for step in range(max_steps):
    roll_out = cooper_opt.roll(
        compute_cmp_state_kwargs={
            "positions": state.positions,
            "cell": state.cell,
            "step": step
        }
    )
    loss = roll_out.cmp_state.loss
```

## Configuration

TorchDisorder uses Hydra for configuration management:

```bash
# Run with default SiO2 config
python scripts/train.py

# Override parameters
python scripts/train.py data=SiO2 max_steps=100000 accelerator=cuda

# Use different glass system
python scripts/train.py data=GeO2 structure=germania
```

## Project Structure

```
torchdisorder/
├── torchdisorder/
│   ├── common/           # Utilities, data loading
│   │   ├── utils.py
│   │   └── target_rdf.py
│   ├── engine/           # Optimization engine
│   │   ├── optimizer.py  # Cooper CMP implementation
│   │   ├── order_params.py
│   │   └── callbacks.py
│   ├── model/            # Structure factor models
│   │   ├── xrd.py
│   │   ├── rdf.py
│   │   ├── loss.py
│   │   └── generator.py
│   ├── viz/              # Visualization
│   │   └── plotting.py
│   └── constraints/      # Constraint generators
│       ├── sio2_generator.py
│       └── lps_generator.py
├── configs/              # Hydra configuration
│   ├── config.yaml
│   ├── data/
│   ├── structure/
│   └── trainer/
├── scripts/
│   └── train.py
├── data/                 # Experimental data
│   ├── xrd_measurements/
│   ├── crystal-structures/
│   └── json/
└── tests/
```

## Constraint JSON Format

```json
{
  "cutoff": 2.3,
  "element_filter": [8, 14],
  "atom_constraints": {
    "0": {
      "atom_index": 0,
      "element": "Si",
      "environment_type": "Si4",
      "order_parameters": {
        "tet": {
          "target": 0.7,
          "min": 0.6,
          "max": 1.0,
          "weight": 2.0
        }
      }
    }
  }
}
```

## Supported Glass Systems

| System | Order Parameters | Description |
|--------|-----------------|-------------|
| SiO₂ | q_tet | Vitreous silica, tetrahedral Si |
| GeO₂ | q_tet | Germania glass, tetrahedral Ge |
| Li₂S-P₂S₅ | q_tet | Lithium thiophosphate, P₄/Pₐ/P₂ environments |

## Citation

```bibtex
@article{torchdisorder2026,
  title={TorchDisorder: A Differentiable Framework for Generating Amorphous Atomic Structures from Diffraction Data},
  author={Gore, A. and Feugmo, C. G.},
  journal={},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Cooper](https://github.com/cooper-org/cooper) - Constrained optimization library
- [torch-sim](https://github.com/atomistic-ml/torch-sim) - Atomistic simulation in PyTorch
- [MACE](https://github.com/ACEsuit/mace) - Machine learning interatomic potentials
- [Pymatgen](https://pymatgen.org/) - Materials analysis library
