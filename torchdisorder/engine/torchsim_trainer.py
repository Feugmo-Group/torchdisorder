
# This trainer
#   • loads a pre‑trained MACE model via torch_sim,
#   • couples it with your Augmented‑Lagrangian GBSD loss in a single Module,
#   • optimises atomic positions & cell using torch_sim.optimizers.fire.

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch_sim as ts
from torch_sim.models.mace import MaceModel, MaceUrls
from torch_sim.optimizers import fire
from ase.io import read, Trajectory

from torchdisorder.common.target_rdf import TargetRDFData
from torchdisorder.model.rdf import SpectrumCalculator
from torchdisorder.model.loss import AugLagLoss, AugLagHyper


__all__ = ["run_torchsim_fire"]

# # -----------------------------------------------------------------------------
# # 1.  Wrapper module that merges MACE energy with GBSD + AL loss
# # -----------------------------------------------------------------------------
#
# class MaceGBSDModule(torch.nn.Module):
#     def __init__(
#         self,
#         mace_model: MaceModel,
#         gbsd_loss: AugLagLoss,
#     ) -> None:
#         super().__init__()
#         self.mace = mace_model
#         self.gbsd = gbsd_loss
#
#     def forward(self, state: ts.State) -> dict[str, torch.Tensor]:
#         energies, forces, stresses = [], [], []
#         atoms_batch = ts.io.state_to_atoms(state, detach=False)
#
#         for atoms in atoms_batch:
#             # --- MACE energy & forces from torch_sim model ---
#             mace_out = self.mace.single_forward(atoms)  # dict with energy/force/stress
#             e_mace = mace_out["energy"]
#             f_mace = mace_out["force"]
#             s_mace = mace_out["stress"]
#
#             # --- GBSD Aug‑Lag contribution ---
#             symbols = atoms.get_chemical_symbols()
#             pos = torch.tensor(atoms.positions, device=state.device, dtype=state.dtype, requires_grad=True)
#             cell = torch.tensor(atoms.cell.array, device=state.device, dtype=state.dtype, requires_grad=True)
#             e_gbsd = self.gbsd(pos, cell, symbols)
#
#             total_E = e_mace + e_gbsd
#             energies.append(total_E)
#
#             # Combine forces: MACE already returns them, but GBSD only via autograd
#             # to keep things simple we re‑compute total forces via autograd.
#             f_total = -torch.autograd.grad(total_E, pos, create_graph=True)[0]
#             forces.append(f_total)
#             stresses.append(s_mace)  # stress dominated by MACE; GBSD small
#
#         return {
#             "energy": torch.stack(energies),
#             "force": torch.nested.nested_tensor(forces),
#             "stress": torch.stack(stresses),
#         }
#
# # -----------------------------------------------------------------------------
# # 2.  Main optimisation entry
# # -----------------------------------------------------------------------------
#
# def run_torchsim_fire(
#     cif_path: str | Path,
#     scattering_yaml: str | Path,
#     rdf_yaml: str | Path,
#     loss_yaml: str | Path,
#     *,
#     device: str = "cuda",
#     n_steps: int = 1000,
#     traj_out: str | None = None,
# ) -> None:
#     """Optimise a structure with MACE+GBSD using Torch‑Sim FIRE.
#
#     Parameters
#     ----------
#     cif_path : path to initial CIF/XYZ structure (single cell)
#     scattering_yaml : YAML containing neutron/x‑ray scattering params + kernel_width
#     rdf_yaml : YAML pointing to experimental T(r) and F(Q) CSV paths
#     loss_yaml : YAML with AugLag hyper‑params
#     device : "cuda" or "cpu"
#     n_steps : FIRE iterations
#     traj_out : optional trajectory file to write every 10 steps
#     """
#     device = torch.device(device)
#
#     # ------------------------------------------------------------------
#     # Load configs & build helper objects
#     # ------------------------------------------------------------------
#     scat_cfg = ScatteringConfig.from_yaml(scattering_yaml)
#     spec_calc = SpectrumCalculator(scat_cfg)
#
#     rdf_data = TargetRDFData.from_yaml(rdf_yaml, device=device)
#     spec_calc.r_bins = rdf_data.r_bins
#     spec_calc.q_bins = rdf_data.q_bins
#     spec_calc.T_r_target, spec_calc.F_q_target = rdf_data.T_r_target, rdf_data.F_q_target
#     spec_calc.F_q_uncert = rdf_data.F_q_uncert
#
#     hyper = AugLagHyper.from_yaml(loss_yaml)
#     gbsd_loss = AugLagLoss(spec_calc, hyper=hyper, device=device)
#
#     # ------------------------------------------------------------------
#     # Build MACE model via torch_sim – medium MPA model
#     # ------------------------------------------------------------------
#     mace_raw = MaceUrls.mace_mpa_medium
#     mace_model = MaceModel(model=mace_raw, device=device, compute_forces=True, compute_stress=True)
#
#     # Combined Module
#     model = MaceGBSDModule(mace_model, gbsd_loss).to(device)
#
#     # ------------------------------------------------------------------
#     # Build Torch‑Sim State from CIF (single structure)
#     # ------------------------------------------------------------------
#     atoms = read(str(cif_path))
#     state = ts.io.atoms_to_state([atoms], device=device)
#
#     # ------------------------------------------------------------------
#     # FIRE optimiser
#     # ------------------------------------------------------------------
#     init_fn, update_fn = fire(model=model, timestep=1e-3)
#     state = init_fn(state)
#
#     # Optional trajectory writer
#     traj = Trajectory(traj_out, "w") if traj_out else None
#
#     print("— Torch‑Sim FIRE optimisation —")
#     for step in range(n_steps):
#         state = update_fn(state)
#         if step % 50 == 0:
#             E = state.energy[0].item()
#             print(f"step {step:4d}  E = {E:.6f} eV")
#             if traj:
#                 traj.write(ts.io.state_to_atoms(state)[0])
#
#     if traj:
#         traj.close()
#     print("Final energy:", state.energy[0].item(), "eV")






# -----------------------------------------------------------------------------

import torch
import torch_sim as ts

from typing import Callable
from ase import Atoms
from ase.io import write

def run_torchsim_fire(
    atoms: Atoms,
    loss_fn: Callable[[torch.Tensor, torch.Tensor, list[str]], torch.Tensor],
    device: str | torch.device = "cpu",
    steps: int = 500,
    dtype: torch.dtype = torch.float32,
    log_interval: int = 20,
    out_path: str = "optimized.xyz",
):
    """
    Run torch_sim FIRE optimization on an ASE Atoms object using a custom loss.

    Parameters
    ----------
    atoms : ASE Atoms
        Initial structure.
    loss_fn : callable
        Function taking (positions, cell, symbols) and returning a scalar loss.
    device : str or torch.device
        Compute device.
    steps : int
        Number of FIRE steps.
    dtype : torch.dtype
        Tensor type.
    log_interval : int
        Log progress every n steps.
    out_path : str
        Output filename for final XYZ structure.
    """
    # Convert Atoms to torch_sim state
    state = ts.io.atoms_to_state([atoms], device=device, dtype=dtype)

    # Enable gradients
    state.positions.requires_grad_(True)
    state.cell.requires_grad_(True)

    # Prepare optimizer (FIRE)
    init_fn, update_fn = ts.optimizers.fire(lambda s: loss_fn(s.positions[0], s.cell[0], s.symbols[0]))
    state = init_fn(state)

    print("\n[torch_sim] Starting optimization:")
    for step in range(steps):
        if step % log_interval == 0:
            loss_val = loss_fn(state.positions[0], state.cell[0], state.symbols[0]).item()
            print(f"Step {step:4d} | Loss: {loss_val:.6f}")

        state = update_fn(state)

    print("\n[torch_sim] Optimization complete.")
    write(out_path, ts.io.state_to_atoms(state)[0])
    print(f"Final structure written to: {out_path}")


