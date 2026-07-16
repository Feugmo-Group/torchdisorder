"""
Tutorial: F_IS vs BOO on TorchDisorder SiO2 Glass + MACE MD
============================================================

What this tutorial covers
--------------------------
1. Load a TorchDisorder-generated SiO2 glass and the crystalline reference.
2. Compute F_IS (local inversion symmetry) alongside BOO (q4, q6, tet) on both.
3. Plot histograms comparing the two metrics — is F_IS better than BOO?
4. Use MACE-MP-0 via torch_sim to compute energies and FIRE-relax the glass.
5. Run a short NVT Langevin MD trajectory and track F_IS along it.

Is F_IS better than BOO?
-------------------------
They measure different things:
  - BOO (q4, q6): how ORDERED / tetrahedral the coordination shell is.
  - F_IS: how CENTROSYMMETRIC the local elastic environment is.

F_IS correlates more strongly with vibrational (boson-peak) and mechanical
(shear modulus) properties of glasses.  A glass and a crystal can have
nearly identical q4 yet very different F_IS distributions, because F_IS is
sensitive to asymmetric bond-length distortions that BOO misses.

References
----------
A. Milkus & A. Zaccone, Phys. Rev. B 93, 094204 (2016).
https://doi.org/10.1103/PhysRevB.93.094204
"""

# ============================================================================
# Imports
# ============================================================================

import math
from pathlib import Path
import numpy as np
import torch
import torch_sim as ts
from torch_sim.io import atoms_to_state, state_to_atoms
from torch_sim.models.mace import MaceModel

import matplotlib
matplotlib.use("Agg")          # headless — saves to PNG
import matplotlib.pyplot as plt

from pymatgen.core import Structure

from torchdisorder.engine.order_params import TorchSimOrderParameters

try:
    from mace.calculators.foundations_models import mace_mp
    MACE_AVAILABLE = True
except Exception:
    MACE_AVAILABLE = False
    print("MACE not available — MD and energy sections will be skipped.")

# ============================================================================
# Paths & constants
# ============================================================================

REPO_ROOT  = Path(__file__).parent.parent
DATA       = REPO_ROOT / "data" / "crystal-structures"
PLOTS_DIR  = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

CRYSTAL_CIF = DATA / "c-SiO2.cif"
GLASS_CIF   = DATA / "sio2_glass.cif"

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE   = torch.float32
SI_Z, O_Z = 14, 8
KB_EV   = 8.617333e-5   # eV / K

print(f"Device: {DEVICE}")

# ============================================================================
# Step 1 — Load structures
# ============================================================================

print("\n" + "=" * 60)
print("Step 1 — Load SiO2 structures")
print("=" * 60)

crystal = Structure.from_file(str(CRYSTAL_CIF))
print(f"Crystal: {len(crystal)} atoms, a = {crystal.lattice.a:.2f} Å")

if GLASS_CIF.exists():
    glass = Structure.from_file(str(GLASS_CIF))
    print(f"Glass  : {len(glass)} atoms  (TorchDisorder-optimized)")
    has_glass = True
else:
    import copy as _copy
    glass = _copy.deepcopy(crystal)
    rng = np.random.default_rng(42)
    for site in glass:
        site.coords[:] += rng.normal(0, 0.15, 3)
    print("Glass CIF not found — using rattled crystal as stand-in.")
    has_glass = False


def pmg_to_state(struct: Structure) -> ts.SimState:
    pos   = torch.tensor(struct.cart_coords,    dtype=DTYPE, device=DEVICE)
    cell  = torch.tensor(struct.lattice.matrix, dtype=DTYPE, device=DEVICE)
    znums = torch.tensor([s.specie.Z for s in struct], dtype=torch.long, device=DEVICE)
    return ts.SimState(
        positions=pos, cell=cell.unsqueeze(0),
        atomic_numbers=znums, pbc=torch.tensor([True, True, True]),
        masses=znums.float(),
    )


state_crystal = pmg_to_state(crystal)
state_glass   = pmg_to_state(glass)

# ============================================================================
# Step 2 — Order-parameter calculator
# ============================================================================

print("\n" + "=" * 60)
print("Step 2 — Compute F_IS and BOO on crystal vs glass")
print("=" * 60)

op_calc = TorchSimOrderParameters(
    cutoff=2.2,            # first Si–O shell only
    device=DEVICE,
    max_neighbors=8,
    fis_mode="variable_R",
)

si_c = torch.where(state_crystal.atomic_numbers == SI_Z)[0]
si_g = torch.where(state_glass.atomic_numbers   == SI_Z)[0]

OPS = ["fis", "q4", "q6", "tet"]
ops_c = op_calc(state_crystal, si_c, OPS, element_filter=[O_Z])
ops_g = op_calc(state_glass,   si_g, OPS, element_filter=[O_Z])

print(f"\n{'':>6}  {'crystal mean':>14}  {'crystal std':>11}  "
      f"{'glass mean':>10}  {'glass std':>9}  {'delta':>8}")
for key in OPS:
    mc, sc = ops_c[key].mean().item(), ops_c[key].std().item()
    mg, sg = ops_g[key].mean().item(), ops_g[key].std().item()
    print(f"  {key:4s}  {mc:+14.4f}  {sc:11.4f}  {mg:+10.4f}  {sg:9.4f}  {mg-mc:+8.4f}")

# ============================================================================
# Step 3 — Comparison plots
# ============================================================================

print("\n" + "=" * 60)
print("Step 3 — Plot distributions")
print("=" * 60)

labels_map = {
    "fis": "F_IS  (local inversion symmetry)",
    "q4":  "q₄   (Steinhardt BOO l=4)",
    "q6":  "q₆   (Steinhardt BOO l=6)",
    "tet": "Tetrahedral order  ψ",
}

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("SiO₂: order-parameter distributions — crystal vs glass", fontsize=13)

for ax, key in zip(axes.flat, OPS):
    vc = ops_c[key].detach().cpu().numpy()
    vg = ops_g[key].detach().cpu().numpy()
    lo = min(vc.min(), vg.min()) - 0.02
    hi = max(vc.max(), vg.max()) + 0.02
    bins = np.linspace(lo, hi, 50)
    ax.hist(vc, bins=bins, alpha=0.6, label="crystal", color="steelblue",  density=True)
    ax.hist(vg, bins=bins, alpha=0.6, label="glass",   color="darkorange", density=True)
    ax.set_xlabel(labels_map[key], fontsize=10)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_title(f"μ_crystal={vc.mean():.3f}  μ_glass={vg.mean():.3f}", fontsize=9)

plt.tight_layout()
out1 = PLOTS_DIR / "sio2_op_distributions.png"
plt.savefig(out1, dpi=150);  plt.close()
print(f"Saved → {out1}")

# Scatter: F_IS vs q4 (shows they carry independent information)
fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.scatter(ops_c["q4"].detach().cpu(), ops_c["fis"].detach().cpu(),
            s=12, alpha=0.5, label="crystal", c="steelblue")
ax2.scatter(ops_g["q4"].detach().cpu(), ops_g["fis"].detach().cpu(),
            s=12, alpha=0.5, label="glass",   c="darkorange")
ax2.set_xlabel("q₄  (BOO)", fontsize=11)
ax2.set_ylabel("F_IS", fontsize=11)
ax2.set_title("F_IS vs q₄ — SiO₂ (independent axes of information)", fontsize=11)
ax2.legend()
plt.tight_layout()
out2 = PLOTS_DIR / "sio2_fis_vs_q4.png"
plt.savefig(out2, dpi=150);  plt.close()
print(f"Saved → {out2}")

# ============================================================================
# Step 4 — MACE: energy and forces
# ============================================================================

print("\n" + "=" * 60)
print("Step 4 — MACE-MP-0 energy + forces")
print("=" * 60)

if not MACE_AVAILABLE:
    print("Skipped — install mace-torch to enable.")
else:
    print("Loading MACE-MP-0 (small)…")
    ase_calc = mace_mp(model="small", device=DEVICE, default_dtype="float32")

    ase_glass   = glass.to_ase_atoms();   ase_glass.calc   = ase_calc
    ase_crystal = crystal.to_ase_atoms(); ase_crystal.calc = ase_calc

    e_glass   = ase_glass.get_potential_energy()
    e_crystal = ase_crystal.get_potential_energy()
    f_glass   = ase_glass.get_forces()

    print(f"Crystal energy/atom : {e_crystal / len(ase_crystal):.4f} eV/atom")
    print(f"Glass   energy/atom : {e_glass   / len(ase_glass):.4f} eV/atom")
    print(f"Glass   RMS force   : {np.sqrt((f_glass**2).mean()):.4f} eV/Å")
    print(f"Glass   max |F|     : {np.abs(f_glass).max():.4f} eV/Å")

# ============================================================================
# Step 5 — FIRE relaxation via torch_sim
# ============================================================================

print("\n" + "=" * 60)
print("Step 5 — FIRE relaxation (torch_sim)")
print("=" * 60)

if not MACE_AVAILABLE:
    print("Skipped.")
    fire_state = None
else:
    ts_mace = MaceModel(
        model=ase_calc.models[0],
        device=DEVICE,
        dtype=DTYPE,
        compute_forces=True,
        compute_stress=False,
    )

    state_relax = atoms_to_state(ase_glass, device=DEVICE, dtype=DTYPE)
    fire_state  = ts.fire_init(state_relax, ts_mace)

    n_fire = 300
    for step in range(n_fire):
        fire_state = ts.fire_step(fire_state, ts_mace, dt_max=1.0, max_step=0.2)
        fmax = ts.system_wise_max_force(fire_state).item()
        if step % 50 == 0:
            print(f"  FIRE {step:3d}  |F|_max = {fmax:.4f} eV/Å")
        if fmax < 0.05:
            print(f"  Converged  step={step}  |F|_max = {fmax:.4f} eV/Å")
            break

    si_r = torch.where(fire_state.atomic_numbers == SI_Z)[0]
    ops_r = op_calc(fire_state, si_r, ["fis", "q4", "tet"], element_filter=[O_Z])
    print("\nAfter FIRE relaxation:")
    for key, vals in ops_r.items():
        print(f"  {key:4s}  mean={vals.mean().item():+.4f}  std={vals.std().item():.4f}")

# ============================================================================
# Step 6 — NVT Langevin MD via torch_sim
# ============================================================================

print("\n" + "=" * 60)
print("Step 6 — NVT Langevin MD (torch_sim)")
print("=" * 60)

if not MACE_AVAILABLE:
    print("Skipped.")
else:
    T_K   = 300.0
    kT    = T_K * KB_EV
    dt_fs = 0.5    # femtoseconds — conservative for Si-O stretches ~30 fs period
    n_md  = 500
    log_n = 50
    gamma = 0.1    # ps^-1 Langevin damping — strong enough to thermostat quickly

    print(f"NVT Langevin  T={T_K:.0f} K  dt={dt_fs} fs  steps={n_md}  gamma={gamma}")

    md_start = fire_state if fire_state is not None else atoms_to_state(
        ase_glass, device=DEVICE, dtype=DTYPE)

    md_state = ts.nvt_langevin_init(md_start, ts_mace, kT=kT, seed=42)

    fis_traj, q4_traj, tet_traj, steps_logged = [], [], [], []
    TEMP_CEILING = T_K * 100   # abort if temperature explodes

    for step in range(n_md):
        md_state = ts.nvt_langevin_step(md_state, ts_mace, dt=dt_fs, kT=kT, gamma=gamma)

        if step % log_n == 0:
            si_md  = torch.where(md_state.atomic_numbers == SI_Z)[0]
            ops_md = op_calc(md_state, si_md, ["fis", "q4", "tet"], element_filter=[O_Z])
            fis_traj.append(ops_md["fis"].mean().item())
            q4_traj.append(ops_md["q4"].mean().item())
            tet_traj.append(ops_md["tet"].mean().item())
            steps_logged.append(step)
            T_inst = ts.calc_temperature(
                masses=md_state.masses, momenta=md_state.momenta
            ).item()
            print(f"  step {step:4d}  T_inst={T_inst:.0f} K  "
                  f"F_IS={fis_traj[-1]:+.4f}  q4={q4_traj[-1]:.4f}")
            if T_inst > TEMP_CEILING:
                print("  *** Temperature explosion — stopping MD. Try smaller dt. ***")
                break

    # MD trajectory plot
    fig3, axes3 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig3.suptitle(f"NVT Langevin MD @ {T_K:.0f} K — order parameters vs time", fontsize=12)
    for ax, data, lbl, col in zip(
        axes3,
        [fis_traj, q4_traj, tet_traj],
        ["F_IS", "q₄", "tet"],
        ["darkorange", "steelblue", "seagreen"],
    ):
        ax.plot(steps_logged, data, "o-", color=col, ms=5)
        ax.set_ylabel(lbl, fontsize=10)
        ax.axhline(np.mean(data), ls="--", color="grey", lw=0.8,
                   label=f"mean = {np.mean(data):.3f}")
        ax.legend(fontsize=8, loc="upper right")
    axes3[-1].set_xlabel("MD step", fontsize=10)
    plt.tight_layout()
    out3 = PLOTS_DIR / "sio2_md_trajectory.png"
    plt.savefig(out3, dpi=150);  plt.close()
    print(f"\nSaved → {out3}")

# ============================================================================
# Step 7 — Summary: is F_IS better than BOO?
# ============================================================================

print("\n" + "=" * 60)
print("Step 7 — Summary: F_IS vs BOO")
print("=" * 60)

print("""
Metric comparison
-----------------
q4 / q6 (Steinhardt BOO):
  + Standard, widely compared in the literature.
  + q4 ≈ 0.51 for a perfect isolated tetrahedron; easy to interpret.
  - Blind to inversion asymmetry at a given order level.
  - Cannot distinguish glassy SiO4 from crystalline SiO4 when both
    have the same angular arrangement.

F_IS (Milkus & Zaccone PRB 2016):
  + Sensitive to CENTROSYMMETRY — independent axis vs q4/q6.
  + Correlates with vibrational soft modes (boson peak) and shear modulus.
  + Negative for perfect Td tetrahedra (−1/3) — directly encodes the
    absence of an inversion centre.
  - Less intuitive; negative values surprise newcomers.
  - Sensitive to the cutoff choice.

Verdict: F_IS is COMPLEMENTARY to BOO, not a replacement.
  Use both: q4/q6 for angular arrangement, F_IS for local elastic
  environment.  The scatter plot (sio2_fis_vs_q4.png) shows they carry
  largely independent structural information.

TorchDisorder integration:
  Add 'fis' to the structure config order_params to constrain F_IS
  during scattering optimisation:
    order_params: [tet, cn, q4, fis]
""")
