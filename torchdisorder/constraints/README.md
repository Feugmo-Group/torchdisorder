# TorchDisorder Constraint Generators

Each generator reads a crystalline CIF file, identifies local atomic environments,
and writes a **v6-format `constraints.json`** for use with `EnvironmentConstrainedOptimizer`.

---

## Quick reference

| System | Central atom | Environments | Module |
|--------|-------------|--------------|--------|
| Li-P-S glass | P | PS₄, P₂S₇ (dimer), P₂S₆ (dumbbell) | `lps_generator` |
| SiO₂ glass | Si | Si4, Si3, Si5, Si6 | `sio2_generator` |
| GeO₂ glass | Ge | Ge4, Ge3, Ge5, Ge6 | `geo2_generator` |
| ε-Fe₂O₃ | Fe | Fe4 (tet), Fe6 (oct), Fe5 (defect) | `fe2o3_generator` |
| Amorphous Si | Si | Si4, Si3, Si5 | `si_generator` |
| LiPON | P | PO4, PO3N, PO2N | `lipon_generator` |
| NaTaCl₆ | Ta (+ Na) | Ta6, Ta5, Ta7, Na6, Na8 | `natacl6_generator` |
| Li₂HfCl₆₋ₓFₓ | Hf | Hf6, Hf6F0–Hf6F6, Hf5, Hf7 | `li2hfcl6_generator` |

---

## Output files (all generators)

Every generator produces the same three file types:

```
{output}_constraints.json       ← feed this to the training script
{output}_{Atom}_environments.json   ← machine-readable per-site data
{output}_{Atom}_environments.txt    ← human-readable summary
```

For generators that also write the supercell CIF:

```
{output}.cif                    ← pymatgen supercell
```

---

## Li-P-S glass (`lps_generator`)

Classifies P environments by bridging vs. terminal sulfur atoms:

- **P4 / PS₄³⁻** — 4 terminal S (CN=4)
- **Pa / P₂S₇⁴⁻ dimer** — 3 terminal S + 1 bridging S
- **P2 / P₂S₆⁴⁻ dumbbell** — 3 terminal S + 1 P–P bond

Cutoff: P–S 3.0 Å.

```bash
# 67Li₂S·33P₂S₅  from Li₇P₃S₁₁
python -m torchdisorder.constraints.lps_generator \
    --input Li7P3S11.cif \
    --target 67Li2S-33P2S5 \
    --supercell 5,8,5 \
    --output glass_67Li2S

# 70Li₂S·30P₂S₅
python -m torchdisorder.constraints.lps_generator \
    --input Li7P3S11.cif \
    --target 70Li2S-30P2S5 \
    --supercell 5,8,5 \
    --output glass_70Li2S

# 75Li₂S·25P₂S₅  from β-Li₃PS₄
python -m torchdisorder.constraints.lps_generator \
    --input Li3PS4_beta.cif \
    --target 75Li2S-25P2S5 \
    --supercell 5,6,9 \
    --output glass_75Li2S
```

---

## SiO₂ glass (`sio2_generator`)

Classifies Si by total O coordination:

- **Si4** — SiO₄ tetrahedral (dominant, CN=4)
- **Si3** — SiO₃ three-coordinate (defect)
- **Si5** — SiO₅ five-coordinate (over-coordinated)
- **Si6** — SiO₆ octahedral (high-pressure polymorph)

Cutoff: Si–O 2.0 Å.

```bash
# From CIF, no supercell
python -m torchdisorder.constraints.sio2_generator \
    --input c-SiO2.cif --output sio2_glass

# Target ~1000-atom supercell
python -m torchdisorder.constraints.sio2_generator \
    --input c-SiO2.cif --supercell 1000 --output sio2_glass

# Manual replication
python -m torchdisorder.constraints.sio2_generator \
    --input c-SiO2.cif --replicate 3 3 3 --output sio2_glass
```

---

## GeO₂ glass (`geo2_generator`)

Same environment scheme as SiO₂ but for Ge–O (longer bonds):

- **Ge4** — GeO₄ tetrahedral (CN=4)
- **Ge3/Ge5/Ge6** — defect / over-coordinated

Cutoff: Ge–O 2.4 Å (Ge–O bonds are ~1.73–1.88 Å).

```bash
python -m torchdisorder.constraints.geo2_generator \
    --input c-GeO2.cif --cutoff 2.4 --output geo2_glass

python -m torchdisorder.constraints.geo2_generator \
    --input c-GeO2.cif --supercell 800 --output geo2_sc800
```

---

## ε-Fe₂O₃ (`fe2o3_generator`)

Handles the mixed tetrahedral + octahedral iron sites unique to ε-Fe₂O₃:

- **Fe4** — FeO₄ tetrahedral (CN=4, priority=3.0) — unique to ε polymorph
- **Fe6** — FeO₆ octahedral (CN=6) — dominant in all polymorphs
- **Fe5** — FeO₅ five-coordinate (defect)

Fe–O cutoff: 2.3 Å (tetrahedral Fe1 bonds ~1.87 Å; octahedral Fe2–4 bonds ~1.97–2.10 Å).

```bash
python -m torchdisorder.constraints.fe2o3_generator \
    --input epsilon_Fe2O3.cif --output fe2o3_glass

# Only octahedral sites (for α or γ polymorphs)
python -m torchdisorder.constraints.fe2o3_generator \
    --input my_fe2o3.cif --cutoff 2.3 --environments Fe6 --output fe2o3_oct

# Supercell ~800 atoms
python -m torchdisorder.constraints.fe2o3_generator \
    --input epsilon_Fe2O3.cif --supercell 800 --output fe2o3_sc800
```

---

## Amorphous Si (`si_generator`)

Pure elemental system; only Si–Si bonds:

- **Si4** — tetrahedral (CN=4, dominant)
- **Si3** — three-coordinate (dangling-bond defect)
- **Si5** — five-coordinate (over-coordinated)

Cutoff: Si–Si 2.85 Å (bond length ~2.35 Å; clear gap at 3.84 Å).

```bash
python -m torchdisorder.constraints.si_generator \
    --input c-Si.cif --output si_glass

python -m torchdisorder.constraints.si_generator \
    --input c-Si.cif --supercell 1000 --output si_sc1000
```

**Use with training:**
```bash
python scripts/train.py data=Si structure=Si \
    fis.target=-0.33 fis.weight=2.0 fis.central_z=14
```

---

## LiPON glass (`lipon_generator`)

Classifies P environments by the number of O and N neighbours within a
single P–X cutoff (1.85 Å default), distinguishing the PO₃N environment
that gives LiPON its ionic conductivity:

- **PO4** — PO₄ tetrahedral, no N (standard phosphate)
- **PO3N** — PO₃N tetrahedra with one N (key LiPON feature; highest priority)
- **PO2N** — PO₂N+1 defect (CN=4 target)
- **P_other** — catch-all for unclassified environments

```bash
python -m torchdisorder.constraints.lipon_generator \
    --input LiPON_defected.cif --output lipon_glass

python -m torchdisorder.constraints.lipon_generator \
    --input LiPON_defected.cif --supercell 800 --output lipon_sc800
```

**Use with training:**
```bash
python scripts/train.py data=LiPON structure=LiPON target=F_Q
```

---

## NaTaCl₆ (`natacl6_generator`)

Halide perovskite / radiation detector. Constrains the TaCl₆²⁻ octahedron:

- **Ta6** — TaCl₆ octahedral (CN=6, priority=2.5, q4 target=0.764)
- **Ta5** — TaCl₅ defect (CN=5)
- **Ta7** — TaCl₇ over-coordinated (CN=7)
- **Na6** — NaCl₆ octahedral (optional)
- **Na8** — NaCl₈ cubic (optional)

Ta–Cl cutoff: 2.6 Å; Na–Cl cutoff: 3.2 Å.

```bash
python -m torchdisorder.constraints.natacl6_generator \
    --input NaTaCl6.cif --output natacl6_glass

# Ta sites only
python -m torchdisorder.constraints.natacl6_generator \
    --input NaTaCl6.cif --central Ta --output natacl6_ta_only

# Supercell
python -m torchdisorder.constraints.natacl6_generator \
    --input NaTaCl6.cif --supercell 800 --output natacl6_sc800
```

---

## Li₂HfCl₆₋ₓFₓ (`li2hfcl6_generator`)

Designed for Syed Jawad Hussain's mixed-anion halide electrolyte series.
Handles both the pure chloride baseline and fluorine-substituted variants.

### Environment types

**Pure chloride (x=0) — `pure_cl` mode (default when no F in structure):**

| Environment | Description | CN | Priority |
|-------------|-------------|-----|----------|
| `Hf6` | HfCl₆ octahedral | 6 | 2.5 |
| `Hf5` | HfCl₅ defect | 5 | 1.0 |
| `Hf7` | HfCl₇ over-coord | 7 | 1.0 |

**Mixed-anion (x>0) — `mixed_anion` mode, classified by n_F:**

| Environment | Description | CN | Priority |
|-------------|-------------|-----|----------|
| `Hf6F0` | HfCl₆ (0 F) | 6 | 2.5 |
| `Hf6F1` | HfCl₅F | 6 | 2.0 |
| `Hf6F2` | HfCl₄F₂ | 6 | 1.96 |
| `Hf6F3` | HfCl₃F₃ (mer/fac) | 6 | 1.92 |
| `Hf6F4` | HfCl₂F₄ | 6 | 1.88 |
| `Hf6F5` | HfClF₅ | 6 | 1.84 |
| `Hf6F6` | HfF₆ | 6 | 2.5 |
| `Hf5` / `Hf7` | Defect sites | 5/7 | 1.0 |

Cutoffs: Hf–Cl 2.70 Å (bonds ~2.48–2.56 Å); Hf–F 2.20 Å (bonds ~1.96–2.04 Å).

### Usage

```bash
# Li₂HfCl₆ baseline (pure chloride)
python -m torchdisorder.constraints.li2hfcl6_generator \
    --input Li2HfCl6.cif --output li2hfcl6_glass

# Li₂HfCl₄F₂  (x=2, mixed-anion — subclassifies by n_F)
python -m torchdisorder.constraints.li2hfcl6_generator \
    --input Li2HfCl4F2.cif --mode mixed_anion --output li2hfcl4f2_glass

# Li₂HfF₆  (fully fluorinated)
python -m torchdisorder.constraints.li2hfcl6_generator \
    --input Li2HfF6.cif --mode mixed_anion --output li2hff6_glass

# Supercell ~800 atoms
python -m torchdisorder.constraints.li2hfcl6_generator \
    --input Li2HfCl6.cif --supercell 800 --output li2hfcl6_sc800

# Wider cutoff for distorted amorphous glass
python -m torchdisorder.constraints.li2hfcl6_generator \
    --input Li2HfCl6.cif --cutoff 2.80 --f_cutoff 2.25 --output li2hfcl6_loose

# Only well-defined octahedral sites
python -m torchdisorder.constraints.li2hfcl6_generator \
    --input Li2HfCl6.cif --environments Hf6 --output li2hfcl6_oct_only
```

**Use with training:**
```bash
python scripts/train.py data=Li2HfCl6 structure=Li2HfCl6 target=F_Q \
    fis.target=1.0 fis.central_z=72 fis.neighbor_z=17
```

**Post-training F_IS analysis (Syed's workflow):**
```bash
# Step 1 — standard analysis
python scripts/analyze.py \
    --run_dir outputs/Li2HfCl6_2026-07-16/<time>/ \
    --central Hf --neighbor Cl --cutoff 2.7

# Step 2 — F_IS-derived properties (F_IS by CN, distortion, SRO, autocorrelation)
python scripts/analyze_fis_properties.py \
    --run_dir outputs/Li2HfCl6_2026-07-16/<time>/ \
    --system Li2HfCl6 \
    --central Hf --neighbor Cl --cutoff 2.7 \
    --central_z 72 --neighbor_z 17

# For mixed-anion x>0 — also include F neighbours
python scripts/analyze_fis_properties.py \
    --run_dir outputs/Li2HfCl4F2_2026-07-16/<time>/ \
    --system Li2HfCl4F2 \
    --central Hf --neighbor Cl --cutoff 2.7 \
    --central_z 72 --neighbor_z "17,9"
```

### Scientific background

Each Cl→F substitution shortens one Hf–X bond (~0.55 Å shorter) and breaks
the O_h inversion symmetry of the octahedron:

- Pure HfCl₆: F_IS ≈ +1.0 (centrosymmetric)
- HfCl₅F: F_IS drops below +1.0 (one short Hf–F breaks inversion)
- More F → F_IS decreases monotonically
- Warren-Cowley α(Hf,F) < 0 → ordered F substitution; ≈ 0 → random mixing

This F_IS vs x_F trajectory is the primary structural characterisation of
the Li₂HfCl₆₋ₓFₓ series and feeds directly into ionic conductivity models.

---

## Common options

All generators accept:

| Flag | Description |
|------|-------------|
| `--supercell N` | Auto-replicate to ~N atoms |
| `--replicate NA NB NC` | Explicit 3×3 replication |
| `--cutoff Å` | Primary first-shell cutoff |
| `--environments E1 E2` | Restrict to specific environment types |

---

## F_IS — Local Inversion Symmetry order parameter

F_IS (Milkus & Zaccone, PRB 2016) measures how centrosymmetric a local
coordination shell is.  It is computed from the vibrational density of states
of the cage but approximated here from atomic positions:

- **Ideal octahedron** (O_h): F_IS ≈ +1.0
- **Ideal tetrahedron** (T_d): F_IS ≈ −1/3
- **Mixed/distorted**: between −1/3 and +1.0

### Typical target values by material

| System | Central | Target F_IS | Note |
|--------|---------|-------------|------|
| SiO₂, GeO₂ | Si, Ge | −0.33 | tetrahedral |
| Fe₂O₃ (oct Fe) | Fe | +1.0 | octahedral |
| Li₂HfCl₆ (pure) | Hf | +1.0 | octahedral HfCl₆ |
| Li₂HfCl₆₋ₓFₓ | Hf | <+1.0, decreasing with x | F substitution breaks inversion |
| Amorphous Si | Si | −0.33 | tetrahedral |
| LiPON (PO₄) | P | −0.33 | tetrahedral |

### Activating F_IS during training

F_IS is controlled by the `fis:` block in the data config YAML:

```yaml
# configs/data/MySystem.yaml
fis:
  target: -0.33       # target F_IS value (see table above)
  weight: 1.0         # loss weight; start at 0.5–2.0
  cutoff: 2.2         # same as constraint cutoff (Å)
  central_z: 14       # atomic number of central atom (Si=14, Fe=26, Hf=72, …)
  neighbor_z: 8       # atomic number of neighbour (O=8, Cl=17, S=16, …)
  mode: variable_R    # variable_R (recommended) or fixed_R
```

Or override on the CLI without editing the YAML:

```bash
# SiO2 — tetrahedral Si (F_IS = -1/3)
python scripts/train.py data=SiO2 structure=silica \
    fis.target=-0.33 fis.weight=1.0 fis.central_z=14 fis.neighbor_z=8

# Fe2O3 — octahedral Fe (F_IS = +1)
python scripts/train.py data=Fe2O3 structure=Fe2O3 \
    fis.target=1.0 fis.weight=0.5 fis.central_z=26 fis.neighbor_z=8

# Li2HfCl6 — octahedral Hf (F_IS = +1)
python scripts/train.py data=Li2HfCl6 structure=Li2HfCl6 \
    fis.target=1.0 fis.weight=1.0 fis.central_z=72 fis.neighbor_z=17
```

### Dynamic F_IS feedback (advanced)

F_IS feedback dynamically raises the constraint penalty for environments that
deviate most from the F_IS target, focusing the optimizer on the worst sites:

```bash
python scripts/train.py data=Fe2O3 structure=Fe2O3 \
    fis.target=1.0 fis.weight=0.5 fis.central_z=26 \
    fis_feedback.enabled=true \
    fis_feedback.update_interval=200 \
    fis_feedback.feedback_strength=2.0 \
    fis_feedback.warmup_steps=500
```

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fis_feedback.enabled` | `false` | Must be `true` to activate |
| `fis_feedback.update_interval` | 200 | Steps between priority updates |
| `fis_feedback.feedback_strength` | 2.0 | `priority = 1 + strength × |F_IS − target|` |
| `fis_feedback.warmup_steps` | 500 | Steps before feedback activates |
| `fis_feedback.min_scale` | 0.5 | Minimum priority multiplier |
| `fis_feedback.max_scale` | 5.0 | Maximum priority multiplier |

### Post-training analysis

```bash
python scripts/analyze_fis_properties.py \
    --checkpoint outputs/MyRun/checkpoints/step_5000 \
    --constraints data/json/my_constraints.json \
    --central_z 26 --neighbor_z 8 --cutoff 2.2
```

---

## Adding a new material system

1. Copy the closest existing generator as a template.
2. Set the correct central element, cutoff, and environment names.
3. Adjust `_make_env_op_specs()` for the expected CN and q4 (if octahedral: 0.764; tetrahedral: varies).
4. Add a corresponding `configs/data/<System>.yaml` and `configs/structure/<System>.yaml`.
5. Run the generator, then test with `python scripts/train.py data=<System>`.
