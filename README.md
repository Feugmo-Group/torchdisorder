# TorchDisorder v6

Differentiable structure optimization from scattering data with environment-based constraints.

---

## Table of Contents

1. [Installation](#installation)
2. [Overview](#overview)
3. [Workflow: from Crystal to Optimized Glass](#workflow-from-crystal-to-optimized-glass)
4. [Step 1 — Generate CIF and JSON Constraints](#step-1--generate-cif-and-json-constraints)
5. [Building a Glass Seed by Melt-Quench](#building-a-glass-seed-by-melt-quench)
6. [Step 2 — Configure the Structure](#step-2--configure-the-structure)
7. [Step 3 — Configure the Experimental Data](#step-3--configure-the-experimental-data)
8. [Step 4 — Run Training](#step-4--run-training)
9. [All Training Config Options](#all-training-config-options)
10. [Structure Initialization Modes](#structure-initialization-modes)
11. [Constraint JSON Format](#constraint-json-format)
12. [Available Structures and Run Scripts](#available-structures-and-run-scripts)
13. [Scattering Functions](#scattering-functions)
14. [Environment Types](#environment-types)
15. [Local Inversion Symmetry — F_IS](#local-inversion-symmetry--f_is)
16. [MLIP Energy Regularizer (MACE)](#mlip-energy-regularizer-mace)
17. [Module Structure](#module-structure)

---

## Installation

```bash
git clone <repo_url>
cd torchdisorder
pip install -r requirements.txt
pip install -e .
export PROJECT_ROOT=$(pwd)
```

---

## Overview

TorchDisorder optimizes atomic positions to match experimental X-ray or neutron scattering data (S(Q), F(Q), g(r), G(r), T(r)) using the Augmented Lagrangian method. Local structural constraints (coordination number, tetrahedral order, bond angles) are enforced simultaneously via Cooper's constrained minimization framework.

```
Base crystal CIF
       │
       ▼
  lps_generator.py  ──►  glass CIF  (disordered supercell, density-scaled)
       │                 constraints JSON  (per-atom P environment targets)
       ▼
  train.py  ──►  reads CIF, loads JSON, minimizes χ²(S_pred, S_exp)
       │
       ▼
  optimized glass structure  (trajectory.xyz, checkpoints)
```

---

## Step 1 — Generate CIF and JSON Constraints

The generator builds a disordered glass supercell from a base crystal and writes both the starting structure (CIF) and the per-atom constraint targets (JSON) in a single step. The two files are always generated together and must stay in sync.

### Li-P-S Glass

```bash
cd data/cif-generation

# 67Li2S-33P2S5 — P+S only (Li removed)
python lps_generator.py \
    --input  Li7P3S11.cif \
    --target 67Li2S-33P2S5 \
    --supercell 4,2,2 \
    --output glass_67Li2S_small_noLi \
    --seed 42

# Same composition — keep Li atoms
python lps_generator.py \
    --input  Li7P3S11.cif \
    --target 67Li2S-33P2S5 \
    --supercell 4,2,2 \
    --output glass_67Li2S_small_withLi \
    --seed 42 \
    --keep-li
```

**Generator arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input` | yes | — | Base crystal CIF (e.g. `Li7P3S11.cif`, `Li3PS4_beta.cif`) |
| `--target` | yes | — | Glass composition: `67Li2S-33P2S5`, `70Li2S-30P2S5`, or `75Li2S-25P2S5` |
| `--supercell` | no | `5,8,5` | Supercell expansion `nx,ny,nz` — choose near-cubic to avoid MIC artifacts |
| `--output` | no | `glass_structure` | File prefix for all output files |
| `--disorder` | no | `0.3` | Random displacement magnitude in Å applied after supercell creation |
| `--seed` | no | `42` | Random seed for reproducibility |
| `--keep-li` | no | off | Retain Li atoms in the CIF and add Li non-overlap constraints to the JSON |

**Output files (four per run):**

| File | Content |
|---|---|
| `<output>.cif` | Disordered glass supercell (density not yet scaled — scaling happens at training time) |
| `<output>_constraints.json` | Per-atom P environment targets + global fractions |
| `<output>_P_environments.txt` | Human-readable environment summary |
| `<output>_P_environments.json` | Machine-readable environment data |

**Supercell size guidance:**

The MIC (minimum image convention) cutoff for g(r) integration is `L/2` where `L` is the shortest box dimension. If the supercell is elongated (one axis much shorter than the others), the dead zone in g(r) inflates S(Q) by orders of magnitude. Always prefer near-cubic supercells.

| Composition | Base crystal | Recommended small | Recommended full |
|---|---|---|---|
| 67% / 70% | Li7P3S11 | `4,2,2` (~20 Å) | `5,8,5` (~60 Å) |
| 75% | Li3PS4_β | `2,2,3` (~16 Å) | `5,6,9` (~50 Å) |

### Batch generation (all 6 small structures)

```bash
cd data/cif-generation
bash generate_lips_small_structures.sh
# Outputs copied automatically to:
#   data/crystal-structures/  (CIF files)
#   data/json/                (constraint JSON files)
```

### SiO2 and GeO2

```bash
python -m torchdisorder.constraints.sio2_generator \
    --input c-SiO2.cif --cutoff 2.2 --output data/json/sio2_glass

python -m torchdisorder.constraints.geo2_generator \
    --input c-GeO2.cif --cutoff 2.4 --output data/json/geo2_glass
```

---

## Building a Glass Seed by Melt-Quench

Refinement cannot repair a broken seed — it inherits whatever you give it, and the
trainer now refuses to start from a structure with overlapping atoms. The supported
route from a crystal to a physically plausible glass is an MLIP melt-quench, which
has been validated against published models for both SiO₂ and GeO₂.

### Reproducing the validated SiO₂ and GeO₂ glasses

```bash
# SiO2: 1125 atoms, 30 ps melt at 4000 K, 30 ps quench, 5 ps anneal
poetry run python scripts/build_glass_melt_quench.py \
    --input data/crystal-structures/c-SiO2.cif \
    --central Si --neighbour O --cutoff 2.2 --expected-cn 4 \
    --density 2.20 --melt-temp 4000 --gamma 1.0 \
    --melt-steps 30000 --quench-steps 30000 --anneal-steps 5000 \
    --device cuda --output data/crystal-structures/SiO2_mq_hot.cif

# GeO2: same route, Ge-O cutoff 2.4
poetry run python scripts/build_glass_melt_quench.py \
    --input data/crystal-structures/c-GeO2.cif \
    --central Ge --neighbour O --cutoff 2.4 --expected-cn 4 \
    --density 3.65 --melt-temp 4000 --gamma 1.0 \
    --melt-steps 30000 --quench-steps 30000 --anneal-steps 5000 \
    --device cuda --output data/crystal-structures/GeO2_mq.cif
```

**`--gamma 1.0` is what makes this work, and it is worth understanding why.**
Velocities initialise at the melt temperature and equipartition immediately halves
it, so the thermostat supplies the rest. A Langevin damping of 0.1 ps⁻¹ has a 10 ps
relaxation time, so a 20 ps melt never arrives: the original SiO₂ run reached only
3730 K of its 4000 K setpoint and came out partly crystalline (q₄ = 0.297 against
the published 0.142). That was read as a failure of the potential for months. It was
not. **Check the printed temperature trace before interpreting any melt output** — if
T has not reached setpoint within the first ~10 % of the melt, the run is invalid
regardless of what the health gate says.

Melt temperature is chemistry-specific. 4000 K suits the oxides; Li₂S–P₂S₅ melts near
900–1100 K and is run at 1500 K, because 4000 K would dissociate the sulfide.

### Verifying the result

The health gate built into the build script only proves the structure is not broken —
a hot crystal passes it too. To show you have a *glass*, score it against a published
model on quantities the route never fitted:

```bash
# numbers
poetry run python scripts/compare_order_params.py \
    --reference data/crystal-structures/sio2_glass_gap.cif \
    --test data/crystal-structures/SiO2_mq_hot.cif \
    --labels melt-quench --central Si --neighbour O --cutoff 2.2

# the distributions behind those numbers, both oxides in one figure
poetry run python scripts/plot_order_param_distributions.py
```

The second command writes `outputs/melt_quench_validation.png`. Every order-parameter
shift should sit inside the reference's own standard deviation:

| | cn | tet | q₄ | q₆ | F_IS |
|---|---|---|---|---|---|
| SiO₂ Δ | −0.0006 | +0.0046 | +0.0034 | +0.0005 | +0.0011 |
| GeO₂ Δ | −0.0480 | −0.0015 | +0.0084 | +0.0054 | +0.0053 |

**Judge by q₄, not by the PASS verdict.** q₄ is the sensitive discriminator between a
glass and a hot crystal; F_IS barely moves across the transition, because both phases
are built from near-ideal tetrahedra and F_IS is dominated by that local geometry.
The default figure draws the failed γ = 0.1 run alongside for exactly this reason —
its q₄ forms a distinct second population near 0.38 that has no counterpart in the
published model.

---

## Step 2 — Configure the Structure

Each run uses a structure config YAML from `configs/structure/`. It tells the trainer how to build the starting atomic structure.

### Example: `configs/structure/LiPS_67_noLi_small.yaml`

```yaml
root_dir: ${oc.env:PROJECT_ROOT,./}/data

# Initialization mode: from_cif | random_icp | pymatgen
init: from_cif

# Path to the glass CIF generated in Step 1
cif_path: ${structure.root_dir}/crystal-structures/glass_67Li2S_small_noLi.cif

# Target density in g/cm³ — the CIF is rescaled to this density at load time
target_density: 1.85

# Used only when init: random_icp (ignored for from_cif)
species: [P, S]
stoichiometry: [48, 176]
box_length: 19.5

# Order parameter settings
central_element: P
neighbor_elements: [P, S]
order_param_cutoff: 5.5
order_params: [tet, cn, q4]
```

### Structure config fields

| Field | Used by | Description |
|---|---|---|
| `init` | always | Initialization mode (see below) |
| `cif_path` | `from_cif`, `pymatgen` | Path to the glass CIF file |
| `target_density` | `from_cif`, `pymatgen` | Density in g/cm³ to scale the cell to |
| `species` | `random_icp` | List of element symbols |
| `stoichiometry` | `random_icp` | Number of atoms per species |
| `box_length` | `random_icp` | Cubic box side length in Å |
| `central_element` | order params | The atom type whose environments are constrained (usually P or Si) |
| `neighbor_elements` | order params | Elements included in neighbor search |
| `order_param_cutoff` | order params | Neighbor cutoff in Å for order parameter calculation |
| `order_params` | order params | Which order parameters to compute: `tet`, `cn`, `q4`, `fis` |

---

## Step 3 — Configure the Experimental Data

Data configs live in `configs/data/` and specify the experimental scattering data, scattering factors, and the path to the constraint JSON.

### Example: `configs/data/LiPS_67.yaml`

```yaml
root_dir: ${oc.env:PROJECT_ROOT,./}/data

scattering_type: xray   # xray or neutron

data:
  s_of_q_path: ${data.root_dir}/xrd_measurements/Li3PS4/S_of_Q.csv
  g_of_r_path: ${data.root_dir}/xrd_measurements/Li3PS4/g_of_r.csv
  t_of_r_path: null
  input_is_F_Q: false
  stride_q: 4    # downsample Q grid by this factor
  stride_r: 1

q_min: 0.5
q_max: 17.5
r_min: 0.01
r_max: 50.0
n_r_bins: 500
kernel_width: 0.03

number_density: 0.0244   # atoms/Å³ — for documentation only, not used in S(Q) computation

neutron_scattering_lengths:
  Li: -1.90
  P: 5.13
  S: 2.847

xray_form_factor_params:
  P:
    a: [6.4345, 4.1791, 1.78, 1.4908]
    b: [1.9067, 27.157, 0.526, 68.1645]
    c: [1.1149]
  S:
    a: [6.9053, 5.2034, 1.4379, 1.5863]
    b: [1.4679, 22.2151, 0.2536, 56.172]
    c: [0.8669]

central: 'P'
neighbour: 'S'
cutoff: 5.5

# Path to the constraint JSON generated in Step 1
json_path: ${data.root_dir}/json/glass_67Li2S_small_noLi_constraints.json
```

### Data config fields

| Field | Description |
|---|---|
| `scattering_type` | `xray` or `neutron` |
| `data.s_of_q_path` | CSV with columns Q, S(Q) |
| `data.g_of_r_path` | CSV with columns r, g(r) |
| `data.input_is_F_Q` | Set true if S(Q) file actually contains F(Q) |
| `data.stride_q` / `stride_r` | Downsample factor to reduce compute cost |
| `q_min` / `q_max` | Q range used during optimization (Å⁻¹) |
| `r_min` / `r_max` | r range for real-space targets (Å) |
| `n_r_bins` | Number of r bins |
| `kernel_width` | Gaussian broadening of computed peaks (Å⁻¹) |
| `number_density` | Atoms/Å³ — documentation only |
| `json_path` | Constraint JSON from Step 1 — set `null` to run unconstrained |

---

## Step 4 — Run Training

Pre-built run scripts handle all config wiring. Just execute the one matching your system:

> **The `run_lips_*` scripts below have been retired to `scripts/retired/`.** Every
> LiPS glass they seed from contains overlapping atoms (down to 0.066 Å), so the runs
> they produce are invalid — see `data/crystal-structures/retired/README.md`. Use
> `scripts/slurm_lips_glass.sh`, which regenerates the glass by melt-quench first.
> The commands here are kept as a record of what those runs were.

```bash
# Small systems (fast, for testing and development)
bash scripts/run_lips_67_noLi_small.sh
bash scripts/run_lips_67_withLi_small.sh
bash scripts/run_lips_70_noLi_small.sh
bash scripts/run_lips_70_withLi_small.sh
bash scripts/run_lips_75_noLi_small.sh
bash scripts/run_lips_75_withLi_small.sh

# Full systems (production)
bash scripts/run_lips_67_noLi.sh
bash scripts/run_lips_67_withLi.sh
bash scripts/run_lips_70_noLi.sh
bash scripts/run_lips_70_withLi.sh

# Other materials
bash scripts/run_sio2.sh
bash scripts/run_geo2.sh
```

Each script accepts `--steps N` to override the step count:

```bash
bash scripts/run_lips_67_noLi_small.sh --steps 10000
```

For cluster submission, use the matching `slurm_*.sh` script in the same directory.

---

## All Training Config Options

All options can be overridden on the command line as `key=value` (Hydra syntax):

```bash
python scripts/train.py data=LiPS_67 structure=LiPS_67_noLi_small target=S_Q max_steps=5000
```

### Core

| Key | Default | Description |
|---|---|---|
| `experiment_name` | `SiO2` | Name used for output directory and W&B run |
| `data` | `SiO2` | Data config to load from `configs/data/` |
| `structure` | `silica` | Structure config to load from `configs/structure/` |
| `target` | `F_Q` | Optimization target: `S_Q`, `F_Q`, `T_r`, `g_r`, `G_r` |
| `max_steps` | `5000` | Total number of optimization steps |
| `accelerator` | `cpu` | Device: `cpu`, `cuda`, or `mps` |
| `optimize_cell` | `false` | Also optimize cell vectors (not just positions) |

### Learning Rates

| Key | Default | Description |
|---|---|---|
| `lr.primal` | `1e-3` | Learning rate for atomic positions |
| `lr.dual` | `1e-2` | Learning rate for Lagrange multipliers |
| `lr.primal_reduced` | `1e-4` | Reduced primal LR after convergence detection |
| `lr.dual_reduced` | `1e-3` | Reduced dual LR after convergence detection |

### Constraints

| Key | Default | Description |
|---|---|---|
| `constraints.enabled` | `true` | Enable/disable all constraints |
| `constraints.use_types` | `all` | Filter by order parameter type: `all` or list like `[tet, cn]` |
| `constraints.regenerate_json` | `false` | Re-run lps_generator before training (requires `constraints.generator`) |
| `constraints.generator.script` | — | Path to `lps_generator.py` |
| `constraints.generator.input_cif` | — | Base crystal CIF (not the glass CIF) |
| `constraints.generator.target` | — | Composition: `67Li2S-33P2S5`, `70Li2S-30P2S5`, `75Li2S-25P2S5` |
| `constraints.generator.supercell` | — | Supercell string e.g. `"4,2,2"` |
| `constraints.generator.keep_li` | `false` | Keep Li atoms |
| `constraints.generator.disorder` | `0.3` | Displacement magnitude in Å |
| `constraints.generator.seed` | `42` | Random seed |

### Adaptive Penalty

| Key | Default | Description |
|---|---|---|
| `penalty.init` | `10.0` | Initial penalty coefficient ρ for each constraint group |
| `penalty.growth_rate` | `1.5` | Factor by which ρ grows when violated persistently |
| `penalty.decay_rate` | `0.95` | Factor by which ρ shrinks when consistently satisfied |
| `penalty.max_penalty` | `1000.0` | Upper cap on ρ |
| `penalty.min_penalty` | `1.0` | Lower floor on ρ |
| `penalty.patience` | `10` | Steps before adapting ρ |

### Stability

| Key | Default | Description |
|---|---|---|
| `stability.grad_clip_norm` | `1.0` | Maximum gradient norm (clipped above this) |
| `stability.max_displacement` | `0.1` | Maximum allowed displacement per step in Å |
| `stability.constraint_warmup_steps` | `500` | Linearly ramp constraint strength from 0 over this many steps |

### Output

| Key | Default | Description |
|---|---|---|
| `output.write_trajectory` | `true` | Save atomic positions periodically to `trajectory.xyz` |
| `output.trajectory_interval` | `1000` | Steps between trajectory frames |
| `output.trajectory_path` | `<run_dir>/trajectory` | Directory for trajectory files |
| `output.plot_interval` | `100` | Steps between spectrum plots logged to W&B |
| `output.save_plots` | `true` | Save plots to disk as well |
| `output.plots_dir` | `<run_dir>/plots` | Directory for plot files |
| `checkpoint_interval` | `10000` | Steps between full checkpoint saves |

### W&B

| Key | Default | Description |
|---|---|---|
| `wandb` | `enabled` | W&B logging: `enabled` or `disabled` |

---

## Structure Initialization Modes

The `init` field in the structure config selects how the starting structure is built.

### `from_cif` (recommended)

Reads the glass CIF generated by `lps_generator.py`, applies a small random rattle (σ = 0.05 Å), and rescales the cell to `target_density`.

```yaml
init: from_cif
cif_path: data/crystal-structures/glass_67Li2S_small_noLi.cif
target_density: 1.85
```

- `species`, `stoichiometry`, and `box_length` are **ignored**
- Atom ordering in the CIF matches the per-atom indices in the constraint JSON — this is the mode that uses per-atom environment assignments
- The CIF and JSON **must be generated together** by the same `lps_generator.py` run; they cannot be mixed

### `random_icp`

Places atoms at uniformly random positions inside a cubic box. No CIF is needed.

```yaml
init: random_icp
species: [P, S]
stoichiometry: [48, 176]
box_length: 19.5
```

- `cif_path` and `target_density` are **ignored**
- When constraints are loaded, `EnvironmentConstrainedOptimizer` detects that the JSON atom indices no longer correspond to P atoms in the random structure, and automatically switches to **fraction-based redistribution**: the global P environment fractions from the JSON are used to assign environments to P atoms proportionally, without relying on per-atom indices
- Good for a quick sanity check without generating a CIF, but `from_cif` gives a better starting point

### `pymatgen`

Like `from_cif` but parses the CIF via pymatgen (handles more CIF dialects). Same fields as `from_cif`.

---

## Constraint JSON Format

The JSON is produced by `lps_generator.py` and consumed by `EnvironmentConstrainedOptimizer`. Never edit it by hand — always regenerate it alongside the CIF.

```json
{
  "metadata": {
    "version": "v6",
    "structure_type": "li_p_s_glass",
    "composition": "67Li2S-33P2S5",
    "supercell": [4, 2, 2],
    "n_p_atoms": 48,
    "order_parameter_types": ["cn", "tet", "q4"]
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
        "cn":  {"target": 4.0, "tolerance": 0.5, "weight": 1.5},
        "tet": {"target": 0.85, "min": 0.7, "max": 1.0, "weight": 2.0},
        "q4":  {"target": 0.6,  "min": 0.4, "max": 0.8, "weight": 0.5}
      }
    },
    "5": {
      "atom_index": 5,
      "element": "P",
      "environment": "Pa",
      "environment_label": "P2S7^4- (dimer)",
      "order_parameters": {
        "cn":  {"target": 4.0, "tolerance": 0.5, "weight": 1.0},
        "tet": {"target": 0.6,  "min": 0.4, "max": 0.8, "weight": 1.5}
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
    "p_environment_distribution": {
      "target_fractions": {
        "PS4^3-":              73.6,
        "P2S7^4- (dimer)":    12.6,
        "P2S6^4- (dumbbell)":  4.2,
        "PS3^-":               9.6
      }
    }
  }
}
```

**Key sections:**

| Section | Description |
|---|---|
| `atom_constraints` | Per-atom targets keyed by atom index in the CIF — only valid with `from_cif` init |
| `environment_priorities` | Weight multiplier per environment type applied on top of per-OP weights |
| `global_constraints.p_environment_distribution` | Global target fractions (%) — used by the auto-detection logic in `random_icp` mode |
| `cutoff` | Neighbor cutoff used when the JSON was generated (should match `order_param_cutoff` in structure config) |

**Order parameter constraint formats:**

```json
{"target": 4.0, "tolerance": 0.5, "weight": 1.5}   // soft: penalise |val - target| > tolerance
{"min": 0.7, "max": 1.0, "weight": 2.0}             // box: penalise if outside [min, max]
```

### Regenerating the JSON on the fly

If you need to regenerate CIF + JSON without leaving the training workflow:

```bash
python scripts/train.py \
    data=LiPS_67 structure=LiPS_67_noLi_small \
    constraints.regenerate_json=true \
    'constraints.generator.script=data/cif-generation/lps_generator.py' \
    'constraints.generator.input_cif=data/cif-generation/Li7P3S11.cif' \
    'constraints.generator.target=67Li2S-33P2S5' \
    'constraints.generator.supercell=4,2,2'
```

This regenerates the CIF and JSON before loading them, then continues with training as normal. The `output_prefix` is inferred from `json_path` when not specified.

---

## Available Structures and Run Scripts

> **Historical.** Everything in the two Li-P-S tables below now lives in
> `scripts/retired/`, and the structures they refer to in
> `data/crystal-structures/retired/`. They are listed for provenance — so a past
> `run_lips_*` result can be traced — not as things to run. There is currently **no
> validated LiPS glass seed**; the melt-quench regeneration is still in flight.

### Li-P-S glass — small supercells (development / fast iteration)

| Script | Structure config | Atoms | Box (Å) | MIC cut |
|---|---|---|---|---|
| `run_lips_67_noLi_small.sh` | `LiPS_67_noLi_small` | 224 (48P+176S) | ~19.5 | ~9.5 Å |
| `run_lips_67_withLi_small.sh` | `LiPS_67_withLi_small` | 336 (112Li+48P+176S) | ~19.5 | ~9.5 Å |
| `run_lips_70_noLi_small.sh` | `LiPS_70_noLi_small` | 224 (48P+176S) | ~19.5 | ~9.5 Å |
| `run_lips_70_withLi_small.sh` | `LiPS_70_withLi_small` | 336 (112Li+48P+176S) | ~20.0 | ~9.9 Å |
| `run_lips_75_noLi_small.sh` | `LiPS_75_noLi_small` | 240 (48P+192S) | ~16.1 | ~8.0 Å |
| `run_lips_75_withLi_small.sh` | `LiPS_75_withLi_small` | 384 (144Li+48P+192S) | ~16.1 | ~8.0 Å |

### Li-P-S glass — full supercells (production)

| Script | Structure config | Atoms | Box (Å) |
|---|---|---|---|
| `run_lips_67_noLi.sh` | `LiPS_67_noLi` | ~4824 (1200P+3624S) | ~58 |
| `run_lips_67_withLi.sh` | `LiPS_67_withLi` | ~7224 (2800Li+...) | ~58 |
| `run_lips_70_noLi.sh` | `LiPS_70_noLi` | similar | ~60 |
| `run_lips_70_withLi.sh` | `LiPS_70_withLi` | similar | ~60 |

### Other materials

| Script | Material | Notes |
|---|---|---|
| `run_sio2.sh` | Amorphous SiO₂ | Neutron scattering |
| `run_geo2.sh` | Amorphous GeO₂ | Neutron scattering |

---

## Scattering Functions

TorchDisorder can optimize against any of these targets. All are computed from the same atomic positions via the unified scattering module.

```
Reciprocal space:
  S(Q)  — structure factor, oscillates around 1 as Q → ∞
  F(Q)  — reduced structure factor F(Q) = Q[S(Q) − 1], oscillates around 0

Real space:
  g(r)  — pair distribution function, → 1 as r → ∞
  G(r)  — reduced PDF G(r) = 4πρr[g(r) − 1], → 0 as r → ∞  (PDFgetX3 output)
  T(r)  — total correlation T(r) = 4πρr·g(r)

Fourier relationship:
  S(Q) ←→ g(r) via sine transform
```

**Which target to choose:**

| Target | Best for |
|---|---|
| `S_Q` | High-Q data, symmetric oscillations around 1 |
| `F_Q` | Low-Q sensitivity, better dynamic range at small Q |
| `G_r` | PDFgetX3 output (most common X-ray PDF format) |
| `T_r` | Emphasizes short-range structure |
| `g_r` | Normalized PDF, useful for comparison across compositions |

---

## Environment Types

### Li-P-S glass

| Code | Label | Description | cn | tet range | Priority |
|---|---|---|---|---|---|
| `P4` | PS₄³⁻ | Isolated tetrahedron | 4 | 0.70–1.00 | 2.0 |
| `Pa` | P₂S₇⁴⁻ (dimer) | Two tetrahedra sharing one S | 4 | 0.40–0.80 | 1.5 |
| `P2` | P₂S₆⁴⁻ (dumbbell) | Two P atoms directly bonded | 4 | 0.40–0.85 | 1.5 |
| `P3` | PS₃⁻ | Pyramidal, 3-coordinate P | 3 | — | 1.2 |

### SiO2 / GeO2 glass

| Code | Description | cn | Priority |
|---|---|---|---|
| `Si4` / `Ge4` | Perfect tetrahedron | 4 | 2.0 |
| `Si3` / `Ge3` | Under-coordinated (defect) | 3 | 1.0 |
| `Si5` / `Ge5` | Five-coordinate | 5 | 1.2 |
| `Si6` / `Ge6` | Octahedral | 6 | 1.5 |

---

## Local Inversion Symmetry — F_IS

### What is F_IS?

F_IS (local inversion symmetry order parameter) was introduced by Milkus and Zaccone (PRB 2016) to quantify how centrosymmetric the local atomic environment of each atom is. Where Bond-Orientational Order (BOO) parameters such as q4 and q6 capture the angular arrangement of neighbors, F_IS measures whether those neighbors are paired into antiparallel ("inversion-symmetric") directions. This makes F_IS sensitive to the type of structural disorder rather than just its degree.

Empirically, F_IS correlates more strongly with vibrational and mechanical properties than q4 or q6. In particular, the fraction of low-F_IS environments tracks closely with the density of soft (quasi-localized) vibrational modes and with shear modulus heterogeneity, making it a useful structural fingerprint for glasses.

### Mathematics

For atom *i* with neighbors *j*, F_IS is computed from the bond-unit vectors **n**ij = **r**ij / |**r**ij| and scalar weights w_ij:

```
                  Σ_(μ,ν)  | Σ_j  w_ij  n̂^μ_ij  n̂^ν_ij  n̂_ij |²
F_IS_i  =  1  -   ────────────────────────────────────────────────
                  Σ_(μ,ν)  Σ_j  [ w_ij  n̂^μ_ij  n̂^ν_ij ]²
```

Both sums run over the shear planes (μ, ν) ∈ {(x,y), (x,z), (y,z)}. A value close to 1 means the environment is centrosymmetric; a value close to 0 means it is not.

Note that the planes are combined **at the numerator/denominator level, not by averaging the three per-plane ratios**. This is a correctness requirement, not a convention. A shear plane in which n̂^μ n̂^ν vanishes for every bond is degenerate and carries no information; averaging ratios forces it to contribute a spurious 0. An environment rotated about a *single* axis — i.e. any structure taken from a CIF with a symmetry axis along a cell vector — has two such degenerate planes, so averaging returns (1+0+0)/3 = **+1/3 for a perfectly centrosymmetric octahedron** instead of +1. Summing first lets the empty planes drop out of both sums. The two schemes agree for generic orientations and for the ideal tetrahedron. (Combination scheme suggested by A. Zaccone; verified by `scripts/validate_fis_tetrahedron.py`.)

An environment degenerate in *all three* planes — e.g. an octahedron sitting exactly on the Cartesian axes, a measure-zero set — has no well-defined F_IS and is reported as 0.

**Analytical limits:**

| Configuration | F_IS |
|---|---|
| Two antiparallel bonds (perfectly centrosymmetric) | 1 |
| Octahedron (Oh — has an inversion center) | 1 |
| Single bond (no inversion partner) | 0 |
| Perfect SiO₄ tetrahedron (Td — no inversion center) | −1/3 |

The SiO₄ result (F_IS = −1/3) is notable: a regular tetrahedron has *lower* inversion symmetry than a random arrangement, so F_IS can be negative. The numerator is a squared vector *sum* (with cross terms) while the denominator is a sum of squares, so F_IS is **not** bounded to [0, 1].

### Computation modes

Two weighting schemes are available via the `mode` argument:

| Mode | Weights w_ij | Notes |
|---|---|---|
| `variable_R` | w_ij = |**r**ij| (bond length) | **Recommended.** Down-weights distant neighbors naturally; produces smoother distributions. |
| `milkus2016` | w_ij = 1 (uniform) | Matches the original Milkus & Zaccone (2016) definition exactly. |

### Usage

Add `'fis'` to `order_params` in the structure config:

```yaml
# configs/structure/LiPS_67_noLi_small.yaml
order_params: [tet, cn, fis]
```

Compute F_IS directly via `TorchSimOrderParameters`:

```python
from torchdisorder.engine.order_params import TorchSimOrderParameters

op = TorchSimOrderParameters(
    cutoff=5.5,
    element_filter=[15, 16],   # atomic numbers of P and S
    mode="variable_R",         # recommended
)

# positions: (N, 3) tensor of fractional or Cartesian coords
# cell:      (3, 3) cell matrix
fis_values = op.compute_fis(positions, cell)
# fis_values: (N,) tensor, one value per central atom
```

### Comparing F_IS with BOO

```python
import torch
from torchdisorder.engine.order_params import TorchSimOrderParameters

op = TorchSimOrderParameters(
    cutoff=5.5,
    element_filter=[15, 16],
    mode="variable_R",
)

fis = op.compute_fis(positions, cell)         # local inversion symmetry
q4  = op.compute_steinhardt(positions, cell, l=4)  # BOO l=4
q6  = op.compute_steinhardt(positions, cell, l=6)  # BOO l=6

print(f"F_IS  — mean: {fis.mean():.3f}  std: {fis.std():.3f}")
print(f"q4    — mean: {q4.mean():.3f}  std: {q4.std():.3f}")
print(f"q6    — mean: {q6.mean():.3f}  std: {q6.std():.3f}")
```

### F_IS as an optimization objective (CooperLoss)

F_IS is fully differentiable, so it can be used directly inside the
scattering optimization loop as an additional loss term alongside the
chi-squared scattering fit:

```
L_total = χ²(F_Q)  +  w × (mean_F_IS − F_IS_target)²
```

This steers the optimizer toward a target mean F_IS derived from a
reference structure (e.g. a prior TorchDisorder run or a melt-quench MD)
without requiring any additional experimental measurement.

**Reference value — c-SiO₂ (α-quartz, 375 Si, cutoff 2.2 Å):**

| Structure | F_IS mean | q4 mean | tet mean |
|---|---|---|---|
| c-SiO₂ (crystal) | −0.331 | +0.250 | +0.997 |

The crystal value is pure local tetrahedral geometry: an SiO₄ unit isolated
from the structure gives −0.3309 ± 0.0015 on its own, against −1/3 for an
ideal tetrahedron. The +0.0025 offset is the genuine α-quartz distortion.

> ⚠️ **Withdrawn:** earlier revisions of this README quoted a glass value
> (F_IS = +0.005, Δ = +0.336) and the claim that F_IS is "more than three
> times as sensitive as q4". The a-SiO₂ structure those numbers came from is
> unphysical — ⟨CN⟩ = 2.98 with 240 Si–O contacts below 1.4 Å (minimum
> 0.342 Å) and no first-shell peak. It reproduces the experimental F(Q) via
> the classic underdetermined-fit failure mode. The Δ therefore conflated
> tetrahedral distortion with a 4→3 coordination change and atom overlap.
> Do not quote it.
>
> **Root cause — the seed, not the refinement.** MLIP regularization *was*
> enabled in that run. The problem is that the seed structure
> `sio2_glass.cif` (now `data/crystal-structures/retired/`) is itself broken before any
> optimization (⟨CN⟩ = 3.02, min Si–O 0.185 Å, 298 contacts < 1.4 Å), and the
> refinement never repairs it — checkpoints sit at ⟨CN⟩ ≈ 3.0 from step 200 to
> step 2000. A `force_weight` of 1e-5 applied every 50 steps cannot undo a
> 0.2 Å overlap, and `max_force_clip: 5.0` discards the very forces that would.
> Use `scripts/build_sio2_glass.py` to generate a physical seed by MACE
> melt-quench, and check ⟨CN⟩ = 4.000 over a cutoff plateau with no Si–O
> contacts below ~1.5 Å before refining;
> `scripts/validate_fis_tetrahedron.py` test [9] runs this health check.
>
> Note also that α-quartz is *itself* non-centrosymmetric, so a crystal→glass
> change in F_IS should not be described as "loss of inversion symmetry" —
> both phases lack an inversion centre. What changes is the coherent
> tetrahedral addition in the affine force field.

**Run SiO₂ optimization with F_IS regularization:**

```bash
python scripts/train.py data=SiO2 structure=silica target=F_Q \
  fis.target=0.005 fis.weight=5.0 fis.cutoff=2.2 \
  fis.central_z=14 fis.neighbor_z=8
```

All `fis.*` keys are optional Hydra overrides. Omitting them disables the
F_IS term entirely (backward compatible). Available parameters:

| Key | Default | Description |
|---|---|---|
| `fis.target` | — | Target mean F_IS (enables the term when set) |
| `fis.weight` | `1.0` | Weight relative to scattering χ² |
| `fis.cutoff` | `2.2` | Neighbor cutoff in Å (first coordination shell) |
| `fis.central_z` | `14` | Atomic number of central atoms (Si=14, Ge=32, P=15) |
| `fis.neighbor_z` | `None` | Atomic number of neighbor filter (O=8); `None` = all |
| `fis.mode` | `variable_R` | Weighting scheme (`variable_R` or `milkus2016`) |

**Python API (direct use in custom training loops):**

```python
from torchdisorder.model.loss import CooperLoss

loss = CooperLoss(
    target_data=rdf_data,
    target_type='F_Q',
    device='cpu',
    fis_target=0.005,    # target mean F_IS for a-SiO₂
    fis_weight=5.0,
    fis_cutoff=2.2,
    fis_central_z=14,    # Si
    fis_neighbor_z=8,    # O only
)

# In your training loop:
results = xrd_model(state)          # scattering spectra
loss_dict = loss(results, state)    # or loss(results) — state auto-extracted when called from optimizer

chi2   = loss_dict['chi2_loss']
fis_l  = loss_dict.get('fis_loss')   # present only when fis_target is set
fis_mu = loss_dict.get('fis_mean')   # current mean F_IS (detached, for logging)
total  = loss_dict['total_loss']     # chi2 + fis_loss
```

### Target F_IS values for common materials

Use these as starting points for `fis.target`. Values are approximate; amorphous structures will deviate from their crystalline references.

| System | central_z | neighbor_z | Crystalline F_IS | Amorphous target | Notes |
|---|---|---|---|---|---|
| a-SiO₂ | 14 (Si) | 8 (O) | −0.331 | ~0.005 | Tetrahedra disorder significantly |
| a-GeO₂ | 32 (Ge) | 8 (O) | −0.331 | ~0.005 | Same topology as SiO₂ |
| a-Fe₂O₃ | 26 (Fe) | 8 (O) | +1.0 | ~0.4–0.7 | Octahedral Fe sites |
| Li₂HfCl₆ (ordered) | 72 (Hf) | 17 (Cl) | +1.0 | ~0.6–0.9 | Hf octahedra; drops with F⁻ substitution |
| Li₂HfCl₆₋ₓFₓ | 72 (Hf) | 17 (Cl) | +1.0 | <0.9, decreasing with x | Mixed Cl/F coordination |
| LiPON (PO₄) | 15 (P) | 8 (O) | −0.333 | ~−0.1 | Tetrahedral P; slight disorder |

**CLI examples:**

```bash
# SiO₂ — drive toward amorphous target
python scripts/train.py data=SiO2 structure=silica target=F_Q \
  fis.target=0.005 fis.weight=5.0 fis.cutoff=2.2 \
  fis.central_z=14 fis.neighbor_z=8

# Fe₂O₃ — octahedral iron
python scripts/train.py data=Fe2O3 structure=Fe2O3 target=F_Q \
  fis.target=0.5 fis.weight=0.5 fis.cutoff=2.2 \
  fis.central_z=26 fis.neighbor_z=8

# Li₂HfCl₆ — halide SSE
python scripts/train.py data=LiHfCl6 structure=LiHfCl6 target=S_Q \
  fis.target=0.7 fis.weight=1.0 fis.cutoff=2.7 \
  fis.central_z=72 fis.neighbor_z=17
```

### Dynamic F_IS feedback

`fis_feedback` dynamically increases the F_IS penalty for environments that are violating the target most severely, focusing optimization resources where they matter most.

| Key | Default | Description |
|---|---|---|
| `fis_feedback.enabled` | `false` | Enable dynamic per-environment weight adjustment |
| `fis_feedback.update_interval` | `200` | Steps between weight updates |
| `fis_feedback.feedback_strength` | `2.0` | Multiplier applied to the worst-violating environments |
| `fis_feedback.warmup_steps` | `500` | Steps before feedback begins (let scattering converge first) |

```bash
python scripts/train.py data=Fe2O3 structure=Fe2O3 target=F_Q \
  fis.target=0.5 fis.weight=0.5 fis.central_z=26 fis.neighbor_z=8 \
  fis_feedback.enabled=true \
  fis_feedback.update_interval=200 \
  fis_feedback.feedback_strength=2.0 \
  fis_feedback.warmup_steps=500
```

### Reference

> A. Milkus and A. Zaccone, "Local inversion-symmetry breaking controls the boson peak in glasses and crystals," *Phys. Rev. B* **93**, 094204 (2016). https://doi.org/10.1103/PhysRevB.93.094204

---

## MLIP Energy Regularizer (MACE)

`MLIPRegularizer` (`engine/optimizer.py`) applies a machine-learning interatomic potential (MACE by default) as a soft regularizer during scattering optimization. It nudges atoms toward physically reasonable geometries without dominating the scattering fit.

**Why it is needed:** Scattering optimization alone is insensitive to unphysical short contacts, stretched bonds, and extreme local geometries. MACE forces penalize these without requiring a full MD trajectory.

**How it works:** Every `apply_every` steps (default 50), MACE computes forces on the current structure. A proxy loss `L_MLIP = −λ · Σ(F_clip · Δr)` is added, where forces are clipped to `max_force_clip` eV/Å per component to prevent runaway in highly disordered structures. The chi-squared scattering loss always dominates; MACE is a small correction.

### Activation

```bash
python scripts/train.py data=SiO2 structure=silica mlip.enabled=true
```

MACE is **disabled by default**. Enabling it requires `mace-torch` to be installed.

### Configuration parameters

All keys live under `mlip:` in `configs/config.yaml` or as CLI overrides.

| Key | Default | Description |
|---|---|---|
| `mlip.enabled` | `false` | Must be `true` to load and use MACE |
| `mlip.backend` | `mace` | Potential backend: `mace`, `sevennet`, `chgnet`, `m3gnet`, `orb`, `mattersim` |
| `mlip.model` | `small` | Backend-specific model name (e.g. `small`, `medium`, `large` for MACE) |
| `mlip.force_weight` | `1e-5` | λ — initial regularization weight; keep `\|penalty\|/chi²` < 1% |
| `mlip.apply_every` | `50` | Apply the force penalty every N steps |
| `mlip.warmup_steps` | `500` | Steps before MACE activates; let scattering converge first |
| `mlip.max_force_clip` | `5.0` | Per-component force clip in eV/Å — prevents runaway on strained bonds |
| `mlip.adaptive_weight` | `false` | Dynamically tune `force_weight` to track `target_ratio` |
| `mlip.target_ratio` | `0.005` | Target `\|penalty\|/chi²` ratio (0.5%) when `adaptive_weight=true` |
| `mlip.adapt_rate` | `0.15` | Multiplicative step per MACE fire (shrinks 15% if ratio too high) |
| `mlip.force_weight_min` | `1e-7` | Lower bound for the adaptive weight |
| `mlip.force_weight_max` | `1e-3` | Upper bound for the adaptive weight |

### Recommended settings

Amorphous structures start far from the MACE energy minimum so forces are large (F_rms ~ 100–10000 eV/Å). The defaults are tuned to be conservative:

```bash
# Fixed weight (safe starting point)
python scripts/train.py data=SiO2 structure=silica mlip.enabled=true \
  mlip.force_weight=1e-5 mlip.apply_every=50 \
  mlip.warmup_steps=500 mlip.max_force_clip=5.0

# Adaptive weight (preferred — self-tunes λ to stay at 0.5% of chi²)
python scripts/train.py data=SiO2 structure=silica mlip.enabled=true \
  mlip.adaptive_weight=true mlip.warmup_steps=500
```

### Diagnostic output

Every 100 steps (and logged to wandb under `mlip/`) a diagnostic line is printed:

```
[MACE] F_rms=3.21 eV/Å  F_max=12.4  clipped=3  penalty=1.2e-04  λ=1.00e-05  ratio=0.0003 (keep < 0.01)
```

| Field | Meaning |
|---|---|
| `F_rms` | RMS force magnitude over all atoms (eV/Å) |
| `F_max` | Maximum per-component force before clipping |
| `clipped` | Number of force components that were clipped |
| `penalty` | The MACE proxy loss value added to chi² |
| `λ` | Current force weight (changes when `adaptive_weight=true`) |
| `ratio` | `\|penalty\|/chi²` — keep below 0.01 to prevent MACE dominating |

With `adaptive_weight=true`, `λ` is printed each step and logged to wandb as `mlip/force_weight`.

---

## Module Structure

```
torchdisorder/
├── model/
│   ├── scattering.py        # S(Q), F(Q), g(r), G(r), T(r) — all targets
│   ├── xrd.py               # XRD/neutron model wrapping scattering.py
│   ├── loss.py              # χ² and Cooper loss functions
│   └── generator.py         # from_cif / random_icp / pymatgen init modes
├── engine/
│   ├── constrained_optimizer.py  # Augmented Lagrangian, env-based grouping
│   ├── order_params.py      # Coordination number, tet, q4, q2 order parameters
│   └── callbacks.py         # Training callbacks
├── common/
│   ├── target_rdf.py        # Loads and preprocesses experimental data
│   ├── neighbors.py         # Neighbor list utilities
│   └── utils.py             # General utilities
├── constraints/
│   ├── lps_generator.py     # Li-P-S constraint generator (package version)
│   ├── sio2_generator.py    # SiO₂ constraint generator
│   └── geo2_generator.py    # GeO₂ constraint generator
└── viz/
    └── plotting.py          # Visualization utilities

configs/
├── config.yaml              # Master defaults (all keys documented above)
├── data/
│   ├── LiPS_67.yaml         # 67Li2S-33P2S5 experimental data + json_path
│   ├── LiPS_70.yaml         # 70Li2S-30P2S5
│   ├── LiPS_75.yaml         # 75Li2S-25P2S5
│   ├── SiO2.yaml
│   └── GeO2.yaml
└── structure/
    ├── LiPS_67_noLi_small.yaml    # small supercell, no Li
    ├── LiPS_67_withLi_small.yaml  # small supercell, with Li
    ├── LiPS_70_noLi_small.yaml
    ├── LiPS_70_withLi_small.yaml
    ├── LiPS_75_noLi_small.yaml
    ├── LiPS_75_withLi_small.yaml
    ├── LiPS_67_noLi.yaml          # full supercell
    ├── LiPS_67_withLi.yaml
    ├── LiPS_70_noLi.yaml
    ├── LiPS_70_withLi.yaml
    ├── silica.yaml
    └── GeO2.yaml

data/
├── cif-generation/
│   ├── lps_generator.py           # Main generator script
│   ├── generate_lips_small_structures.sh  # Batch generation for all 6 small systems
│   ├── Li7P3S11.cif               # Base crystal for 67% / 70%
│   └── Li3PS4_beta.cif            # Base crystal for 75%
├── crystal-structures/            # Generated glass CIF files
├── json/                          # Generated constraint JSON files
└── xrd_measurements/              # Experimental S(Q) and g(r) data

scripts/
├── train.py                       # Main training entry point
├── run_lips_*_small.sh            # Local run scripts (small systems)
├── run_lips_*.sh                  # Local run scripts (full systems)
├── slurm_*.sh                     # Cluster submission scripts
└── calc_fz_weights.py             # Faber-Ziman weight calculation
```

---

## Citation

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
