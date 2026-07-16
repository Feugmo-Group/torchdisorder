# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TorchDisorder v6** — differentiable amorphous structure optimization from scattering data. The library optimizes atomic positions to match experimental neutron/X-ray scattering functions (S(Q), F(Q), g(r), G(r), T(r)) while satisfying structural constraints derived from local atomic environments, using augmented Lagrangian optimization via the `cooper` library.

Primary materials systems: Li-P-S glass, SiO₂ glass, GeO₂ glass.

## Commands

```bash
# Install (requires Python 3.12 exactly)
poetry install

# Run tests
pytest                              # all tests
pytest tests/test_foo.py::test_bar  # single test
pytest --cov=torchdisorder          # with coverage

# Lint / format
ruff check .                        # lint
ruff format .                       # format
codespell .                         # spell check
mypy torchdisorder/                 # type check

# Train (Hydra-based)
python scripts/train.py                          # default (LiPS, S_Q)
python scripts/train.py target=F_Q              # change target function
python scripts/train.py data=SiO2 structure=silica
python scripts/train.py accelerator=cuda max_steps=10000

# Generate constraints from CIF files
python -m torchdisorder.constraints.lps_generator --input Li7P3S11.cif --output glass_67Li2S
python -m torchdisorder.constraints.sio2_generator --input c-SiO2.cif --cutoff 2.2 --output sio2_glass
python -m torchdisorder.constraints.geo2_generator --input c-GeO2.cif --cutoff 2.4 --output geo2_glass
```

## Architecture

### Data flow

```
CIF file → constraint generator → constraints.json
                                          ↓
experimental data (.gr/.sq file) → TargetRDFData
                                          ↓
                     XRDModel (scattering.py + xrd.py)
                          ↓ predicted S(Q)/g(r)/…
                     CooperLoss / ChiSquaredObjective
                          ↓
               EnvironmentConstrainedOptimizer (cooper lib)
                    ↑ TorchSimOrderParameters (cn, tet, q4…)
                          ↓
                   optimized atomic positions
```

### Module responsibilities

- **`model/scattering.py`** — `UnifiedSpectrumCalculator` / `ScatteringConfig`: single differentiable entry point for all scattering functions. Configured via `ScatteringType` (neutron/xray) and `OutputType` (S_Q, F_Q, g_r, G_r, T_r). This is the primary model module; `model/rdf.py` is a v5 legacy alias.

- **`model/xrd.py`** — `XRDModel` / `XRDModelConfig`: wraps the scattering calculator, handles atom symbols, r/q grids, and provides the forward pass that `EnvironmentConstrainedOptimizer` calls.

- **`model/loss.py`** — `CooperLoss` (augmented Lagrangian), `ChiSquaredObjective`, `chi_squared`, `r_squared`, `rmse`. `CooperLoss` feeds into `cooper` for constrained optimization.

- **`engine/constrained_optimizer.py`** — `EnvironmentConstrainedOptimizer` (v6 primary): groups Cooper constraints by local atomic environment (e.g., PS₄³⁻, P₂S₇⁴⁻) rather than order-parameter type. `AdaptivePenalty` grows/decays per environment based on violation history.

- **`engine/optimizer.py`** — `StructureFactorCMPWithConstraints`: v5 legacy optimizer kept for backward compatibility. Groups constraints by OP type (cn, tet, q4…).

- **`engine/order_params.py`** — `TorchSimOrderParameters` / `PyTorchOrderParameters`: compute coordination number (cn), tetrahedral order (tet), bond-angle order (q4, q2) from atomic positions. Supports `warp` GPU backend when available, PyTorch fallback.

- **`engine/callbacks.py`** — `EarlyStoppingCallback`, `CheckpointCallback`, `PlateauDetector`.

- **`common/target_rdf.py`** — `TargetRDFData`: loads and normalizes experimental scattering data files.

- **`common/neighbors.py`** — neighbor list utilities (wraps `vesin` / `vesin-torch`).

- **`constraints/`** — one generator module per material system; each reads a crystalline CIF, identifies local environments, and writes a v6-format `constraints.json` with `"environment"` and `"environment_priorities"` keys.

- **`viz/plotting.py`** — Plotly/Dash visualization helpers.

### Configuration (Hydra + OmegaConf)

Config lives in `configs/`. The root `config.yaml` selects `data`, `structure`, `trainer`, and `wandb` sub-configs. Override any key on the CLI:
```bash
python scripts/train.py target=G_r penalty.init=20.0 constraints.use_adaptive=true
```
Outputs land in `outputs/<experiment_name>_<date>/<time>/` by default.

### Key external dependencies

| Package | Role |
|---------|------|
| `torch` | autodiff / GPU compute |
| `torch-sim-atomistic` (`torch_sim`) | simulation state, FIRE relaxation, melt-quench |
| `cooper-optim` | augmented Lagrangian / CMP solver |
| `mace-torch` | MACE interatomic potential (used in melt-quench) |
| `vesin` / `vesin-torch` | neighbor lists |
| `warp-lang` | optional GPU order-parameter backend |
| `ase` / `pymatgen` | CIF I/O, structure manipulation |
| `hydra-core` | config management |
| `wandb` | experiment tracking |

### Constraint JSON format (v6)

The `"environment"` key (not `"environment_type"`) groups atoms into structural motifs. `"environment_priorities"` sets per-group penalty weights. `"metadata.version": "v6"` distinguishes from v5 files. See README for full schema.

### v5 → v6 migration notes

- Use `EnvironmentConstrainedOptimizer` instead of `StructureFactorCMPWithConstraints`.
- Use `UnifiedSpectrumCalculator` / `ScatteringConfig` instead of legacy `SpectrumCalculator` from `model/rdf.py`.
- Regenerate `constraints.json` with the current generators to get v6 keys.
- v5 imports still work via backward-compat re-exports in `__init__.py`.
