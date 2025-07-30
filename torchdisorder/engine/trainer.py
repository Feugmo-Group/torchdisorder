# # import wandb
# # import torch
# # from torchdisorder.model.rdf import compute_rdf
# # from torchdisorder.data.target_rdf import load_target_rdf
# # from torchdisorder.model import generator
#
# # class Trainer:
# #     def __init__(self, cfg):
# #         self.cfg = cfg
# #         self.coords = generator.init_coords(cfg).to(cfg.device).requires_grad_()
# #         self.optimizer = instantiate(cfg.optimizer, [self.coords])
# #         self.target_r, self.target_gr = load_target_rdf(cfg.experiment.target_rdf_path)
# #         wandb.init(**cfg.wandb)
#
# #     def step(self, step_idx):
# #         self.optimizer.zero_grad()
# #         r, g_r = compute_rdf(self.coords, self.cfg.system.box_length, self.cfg.experiment.rdf_bins)
# #         loss = ((g_r - self.target_gr) ** 2).mean()
# #         loss.backward()
# #         self.optimizer.step()
#
# #         wandb.log({"loss": loss.item()}, step=step_idx)
# #         if step_idx % self.cfg.wandb.log_rdf_every == 0:
# #             wandb.log({"g_r": wandb.plot.line_series(xs=r.cpu(),
# #                                                      ys=[self.target_gr.cpu(), g_r.detach().cpu()],
# #                                                      keys=["target", "model"],
# #                                                      title=f"RDF @ step {step_idx}")})
#
#
#
# # # engine/trainer.py
# # # ------------------------------------------------------------------------------
# # # Uses ASE FIRE optimizer and hooks into our Augmented Lagrangian loss wrapper.
#
# # import time
# # from typing import Tuple
# # from ase.optimize import FIRE
# # import numpy as np
# # import torch
# # from ase import Atoms
# # import wandb
#
#
# # def run_optimization(
# #     atoms: Atoms,
# #     max_steps: int,
# #     loss_fn,
# #     calc,
# #     use_wandb: bool = True
# # ) -> Tuple[list[float], list[float], dict]:
# #     """
# #     Run ASE FIRE optimizer with mixed calculator and augmented Lagrangian loss.
#
# #     Parameters
# #     ----------
# #     atoms : ASE Atoms
# #         Atomic configuration with MixedCalculator already attached.
# #     max_steps : int
# #         Number of steps for FIRE optimizer.
# #     loss_fn : callable
# #         Instance of AugLagLoss.
# #     calc : ASE calculator
# #         MixedCalculator (e.g. CHGNet + GBSD).
# #     use_wandb : bool
# #         If True, logs metrics to Weights & Biases.
#
# #     Returns
# #     -------
# #     gbsd_losses : list of float
# #     chgnet_energies : list of float
# #     loss_state : dict (final lambda, rho, time)
# #     """
# #     optimizer = FIRE(atoms)
#
# #     gbsd_losses = []
# #     chgnet_energies = []
# #     start_time = time.perf_counter()
#
# #     for step in optimizer.irun(steps=max_steps):
# #         energy, loss_val = calc.get_energy_contributions(atoms)
# #         gbsd_losses.append(loss_val)
# #         chgnet_energies.append(energy)
#
# #         if use_wandb:
# #             wandb.log({
# #                 "step": step,
# #                 "GBSD Loss": loss_val,
# #                 "CHGNet Energy (rel)": energy - np.min(chgnet_energies),
# #                 "Lambda_corr": loss_fn.lambda_corr.item(),
# #                 "Penalty_rho": loss_fn.rho.item(),
# #             })
#
# #     optimization_time = time.perf_counter() - start_time
#
# #     # Collect stats
# #     lag_stats = loss_fn.state_dict()
# #     lag_stats["total_time_s"] = optimization_time
# #     lag_stats["percent_loss_time"] = 100 * lag_stats["time_spent_s"] / optimization_time
#
# #     print("\n--- Optimization Summary ---")
# #     print(f"TOTAL TIME           : {optimization_time:.2f} s")
# #     print(f"LOSS TIME (GBSD)     : {lag_stats['time_spent_s']:.2f} s = {lag_stats['percent_loss_time']:.2f}%")
# #     print(f"CHGNet TIME (approx) : {getattr(calc, 'calculation_time', 0):.2f} s")
# #     print(f"Final Lagrange λ     : {lag_stats['lambda_corr']:.3e}, ρ = {lag_stats['rho']:.3e}")
#
# #     return gbsd_losses, chgnet_energies, lag_stats
#
#
# # engine/trainer.py
# # ------------------------------------------------------------------------------
# # Uses **torch‑sim**'s GPU‑friendly FIRE optimiser instead of the ASE version
# # and streams metrics to Weights & Biases.
#
# import time
# from typing import Tuple
#
# import numpy as np
# import torch
# from ase import Atoms
# import wandb
#
# import os
#
# import numpy as np
# import torch
# from ase.build import bulk
# from mace.calculators.foundations_models import mace_mp
#
# import torch_sim as ts
# from torch_sim.models.mace import MaceModel, MaceUrls
# from torch_sim.optimizers import fire
#
#
# def run_optimization(
#     atoms: Atoms,
#     max_steps: int,
#     loss_fn,
#     calc,
#     use_wandb: bool = True,
# ) -> Tuple[list[float], list[float], dict]:
#     """Minimise the system with torch‑sim’s FIRE and augmented‑Lagrangian loss.
#
#     Parameters
#     ----------
#     atoms : ASE Atoms
#         System with a *MixedCalculator* attached.
#     max_steps : int
#         FIRE iteration budget.
#     loss_fn : AugLagLoss
#         Constraint/loss callable used by the GBSD part of the calculator.
#     calc : MixedCalculator
#         Should expose ``get_energy_contributions(atoms)`` that returns
#         (chgnet_energy, gbsd_loss).
#     use_wandb : bool, default = True
#         Log curves to Weights & Biases.
#
#     Returns
#     -------
#     gbsd_losses : list[float]
#     chgnet_energies : list[float]
#     lag_stats : dict
#     """
#
#     # torch‑sim’s FIRE works directly on the ASE *Atoms* object (since v0.5)
#     optimizer = FIRE(atoms, max_steps=max_steps)
#
#     gbsd_losses: list[float] = []
#     chgnet_energies: list[float] = []
#
#     t0 = time.perf_counter()
#     for step in optimizer:
#         # MixedCalculator internally calls *calc2* (GBSD) which uses loss_fn
#
#
#
#         energy, loss_val = calc.get_energy_contributions(atoms)
#         gbsd_losses.append(loss_val)
#         chgnet_energies.append(energy)
#
#         if use_wandb:
#             wandb.log(
#                 {
#                     "step": step,
#                     "GBSD Loss": loss_val,
#                     "CHGNet Energy (rel)": energy - chgnet_energies[0],
#                     "Lambda_corr": loss_fn.lambda_corr.item(),
#                     "Penalty_rho": loss_fn.rho.item(),
#                 }
#             )
#
#     total_time = time.perf_counter() - t0
#
#     # Summarise augmented‑Lagrangian stats
#     lag_stats = loss_fn.state_dict()
#     lag_stats.update(
#         {
#             "total_time_s": total_time,
#             "percent_loss_time": 100 * lag_stats["time_spent_s"] / total_time,
#         }
#     )
#
#     print("\n--- Optimization Summary (torch‑sim FIRE) ---")
#     print(f"TOTAL TIME           : {total_time:.2f} s")
#     print(
#         f"LOSS TIME (GBSD)     : {lag_stats['time_spent_s']:.2f} s = {lag_stats['percent_loss_time']:.2f}%"
#     )
#     print(f"Final λ              : {lag_stats['lambda_corr']:.3e} | ρ = {lag_stats['rho']:.3e}")
#
#     return gbsd_losses, chgnet_energies, lag_stats


from __future__ import annotations

import time
from typing import Dict, List, Union

from omegaconf import DictConfig

import time
from typing import Dict, List, Union

import torch
from torch import nn
import torch_sim as ts
from torch_sim.state import DeformGradMixin, SimState
from torch_sim.optimizers import fire
from torch_sim.models.mace import MaceModel, MaceUrls
from mace.calculators.foundations_models import mace_mp
from ase import Atoms
from torch_sim.state import DeformGradMixin, SimState
try:
    import wandb  # soft‑import
except ModuleNotFoundError:  # pragma: no cover
    wandb = None  # type: ignore


import logging

try:
    import wandb  # soft‑import
except ModuleNotFoundError:  # pragma: no cover
    wandb = None  # type: ignore
from torchdisorder.model.calculator import DescriptorModel
from torchdisorder.model.rdf import SpectrumCalculator
from torchdisorder.model.loss import AugLagHyper, TargetRDFData, AugLagLoss

from torch_sim.state import DeformGradMixin, SimState
logger = logging.getLogger(__name__)
__all__ = [
    "run_optimization",
]

# ------------------------------------------------------------
# 3) Optimiser wrapper with W&B logging + interleaved FIRE steps
# ------------------------------------------------------------
#
# def run_optimization(
#     cfg: DictConfig,
#     atoms: Atoms,
#     spectrum_calc: SpectrumCalculator,
#     rdf_data: TargetRDFData,
#     hyper: AugLagHyper,
#     *,
#     device: Union[str, torch.device] = "cpu",
#     steps: int = 500,
#     # --- new MACE‑relaxation controls ---
#     mace_relax_every: int = 5,     # perform a MACE block every N outer iterations
#     mace_relax_steps: int = 10,    # number of FIRE updates inside that block
#     use_wandb: bool = False,
#     log_every: int = 25,
#     dtype: torch.dtype = torch.float32,
# ) -> Atoms:
#     device = torch.device(device)
#
#     desc_model = DescriptorModel(spectrum_calc, rdf_data, central=cfg.data.central, neighbour=cfg.data.neighbour, cutoff=cfg.data.cutoff,  device=device).to(device)
#
#
#
#
#     loss_mod = AugLagLoss(rdf_data, hyper, device=device).to(device)
#
#     # Convert atoms to state
#     atoms_list=[atoms]
#     state = ts.io.atoms_to_state(atoms_list, device=device, dtype=dtype)
#
#
#     class EnergyModel(nn.Module):
#         def __init__(self, desc: DescriptorModel, loss_m: AugLagLoss,device='cpu',dtype=torch.float32):
#             super().__init__()
#             self.desc = desc
#             self.loss_m = loss_m
#             self.device = device
#             self.dtype = dtype
#         def forward(self, s: ts.State):
#             atoms_list = ts.io.state_to_atoms(s)
#             d = self.desc(atoms_list[0])
#             out = self.loss_m(d)
#             s.energy = torch.tensor([out["loss"]], device=self.device)
#             s.forces = torch.tensor([out["loss"]], device=self.device)
#             s._aux = out  # keep for logging
#             return {"energy": s.energy, "forces": s.forces}
#
#     model = EnergyModel(desc_model, loss_mod)
#
#     # --- MACE relaxation model (forces only) --------------------
#     loaded_model = mace_mp(model=MaceUrls.mace_mpa_medium, return_raw_model=True, default_dtype=dtype, device=device)
#     mace_wrapper = MaceModel(model=loaded_model, device=device, compute_forces=True, compute_stress=True, dtype=dtype, enable_cueq=False)
#     mace_init, mace_update = fire(model=mace_wrapper)
#     init_fn, update_fn = fire(model=model)
#
#     state = mace_init(state)  # init MACE FIRE
#     state = init_fn(state)  # init AL model
#
#
#     logger.info("— Optimising with AugLag + FIRE —")
#     for step in range(steps):
#
#         # --- Aug‑Lag FIRE update --------------------------------
#         state = update_fn(state)
#         aux = state._aux  # type: ignore[attr-defined]
#
#         # -------------------------------------------
#         # Optional fast MACE relaxation every K steps
#         # -------------------------------------------
#         if mace_relax_every > 0 and step % mace_relax_every == 0:
#             logger.info(f"[MACE] Relax block at outer step {step}")
#             state = ts.io.atoms_to_state(atoms_list, device=device, dtype=dtype)
#             state = mace_init(state)  # init MACE FIRE
#             for sub in range(mace_relax_steps):
#                 state = mace_update(state)
#                 if sub % 3 == 0:  # lightweight print
#                     e = state.energy[0].item()
#                     logger.info(f"    MACE sub‑step {sub:02d} | E = {e:.6f} eV")
#
#
#
#         if step % log_every == 0:
#             msg = (
#                 f"step {step:4d} | E = {aux['loss'].item():.6f} "
#                 f"| χ²_corr={aux['chi2_corr'].item():.3e} "
#                 f"| χ²_scatt={aux['chi2_scatt'].item():.3e} "
#                 f"| q_loss={aux['q_loss'].item():.3e}"
#             )
#             logger.info(msg)
#             if use_wandb and wandb is not None:
#                 wandb.log({
#                     "step": step,
#                     "loss": aux["loss"].item(),
#                     "chi2_corr": aux["chi2_corr"].item(),
#                     "chi2_scatt": aux["chi2_scatt"].item(),
#                     "q_loss": aux["q_loss"].item(),
#                 })
#
#     # write back
#     final_atoms = ts.io.state_to_atoms(state)[0]
#     atoms.set_positions(final_atoms.positions)
#     atoms.set_cell(final_atoms.cell)
#     return atoms


#
# class AugLagEnergyModel(nn.Module):
#     """Turn Descriptor → AugLagLoss into per‑atom energy & forces for FIRE."""
#
#     def __init__(self, desc: DescriptorModel, auglag: AugLagLoss, device: torch.device, dtype: torch.float32) -> None:
#         super().__init__()
#         self.desc = desc
#         self.auglag = auglag
#         self.device = device
#         self.dtype = dtype
#     def forward(self, state: ts.state.SimState) -> Dict[str, torch.Tensor]:
#         # Convert SimState -> Atoms retaining gradients
#         atoms = ts.io.state_to_atoms(state)[0]
#         n_atoms = len(atoms)
#
#         # descriptors & loss dict
#         desc_out = self.desc(state)
#         loss_dict = self.auglag(desc_out)
#         total_loss = loss_dict["loss"]            # scalar tensor
#         per_atom_energy = total_loss / n_atoms     # stabilise gradients
#
#         # forces = -∇E
#         forces = -torch.autograd.grad(
#             per_atom_energy,
#             #atoms.positions,
#             state.positions,
#             create_graph=True,
#             retain_graph=True,
#             allow_unused=True
#         )[0].unsqueeze(0)                         # (1, N, 3)
#
#         # attach aux for logging
#         state._aux = loss_dict
#         return {"energy": per_atom_energy[None], "forces": forces}
#
#
# def run_optimization(
#     cfg: DictConfig,
#     atoms: Atoms,
#     spectrum_calc: SpectrumCalculator,
#     rdf_data: TargetRDFData,
#     hyper: AugLagHyper,
#     *,
#     device: Union[str, torch.device] = "cpu",
#     steps: int = 500,
#     mace_relax_every: int = 5,
#     mace_relax_steps: int = 10,
#     use_wandb: bool = False,
#     log_every: int = 25,
#     dtype: torch.dtype = torch.float32,
# ) -> Atoms:
#     """Optimise *atoms* using AugLagLoss + interleaved MACE FIRE blocks."""
#
#     device = torch.device(device)
#
#     # --------------------------------------------------------
#     #  Build descriptor & loss
#     # --------------------------------------------------------
#     desc_model = DescriptorModel(
#         spectrum_calc,
#         rdf_data,
#         central=cfg.data.central,
#         neighbour=cfg.data.neighbour,
#         cutoff=cfg.data.cutoff,
#         device=device,
#     ).to(device)
#
#     auglag_loss = AugLagLoss(rdf_data, hyper, device=device).to(device)
#     energy_model = AugLagEnergyModel(desc_model, auglag_loss,  device=device, dtype=dtype)
#
#     # --------------------------------------------------------
#     #  torch_sim states & FIRE initialisation
#     # --------------------------------------------------------
#     state = ts.io.atoms_to_state([atoms], device=device, dtype=dtype)
#     # Enable gradients on positions & cell so autograd can compute forces
#     state.positions.requires_grad_(True)
#     state.cell.requires_grad_(True)
#     init_fn, update_fn = fire(model=energy_model)
#     state = init_fn(state)
#
#     # -- MACE model for fast relaxation ---------------------------------------
#     raw_mace = mace_mp(model=MaceUrls.mace_mpa_medium, return_raw_model=True, default_dtype=dtype, device=device)
#     mace_model = MaceModel(raw_mace, device=device, compute_forces=True, dtype=dtype)
#     mace_init, mace_update = fire(model=mace_model)
#
#     logger.info("— Optimising with external AugLagLoss + FIRE —")
#
#     for step in range(steps):
#         # 1) AugLag FIRE update (forces already inside model)
#         state = update_fn(state)
#         aux = state._aux  # type: ignore[attr-defined]
#
#         # 2) Optional MACE relaxation block
#         if mace_relax_every > 0 and step % mace_relax_every == 0:
#             logger.info(f"[MACE] Relax block at outer step {step}")
#             # create a fresh SimState for MACE from current geometry
#             cur_atoms = ts.io.state_to_atoms(state)[0]
#             mace_state = ts.io.atoms_to_state([cur_atoms], device=device, dtype=dtype)
#             mace_state = mace_init(mace_state)
#             for sub in range(mace_relax_steps):
#                 mace_state = mace_update(mace_state)
#                 if sub % 3 == 0:
#                     e_sub = mace_state.energy[0].item()
#                     logger.info(f"    MACE sub‑step {sub:02d} | E = {e_sub:.6f} eV")
#             # copy relaxed positions back (keep requires_grad)
#             state.positions.data.copy_(mace_state.positions.data)
#             state.cell.data.copy_(mace_state.cell.data)
#
#         # 3) Logging
#         if step % log_every == 0:
#             msg = (
#                 f"step {step:4d} | E = {aux['loss'].item():.6f} "
#                 f"| χ²_corr={aux['chi2_corr'].item():.3e} "
#                 f"| χ²_scatt={aux['chi2_scatt'].item():.3e} "
#                 f"| q_loss={aux['q_loss'].item():.3e}"
#             )
#             logger.info(msg)
#             if use_wandb and wandb is not None:
#                 wandb.log({
#                     "step": step,
#                     "loss": aux["loss"].item(),
#                     "chi2_corr": aux["chi2_corr"].item(),
#                     "chi2_scatt": aux["chi2_scatt"].item(),
#                     "q_loss": aux["q_loss"].item(),
#                 })
#
#     # --------------------------------------------------------
#     #  Return final relaxed Atoms
#     # --------------------------------------------------------
#     final_atoms = ts.io.state_to_atoms(state)[0]
#     atoms.set_positions(final_atoms.positions)
#     atoms.set_cell(final_atoms.cell)
#     return atoms

# -----------------------------------------------------------------------------
#  FIRE‑compatible wrapper around AugLagLoss
# -----------------------------------------------------------------------------
class AugLagEnergyModel(nn.Module):
    """Turn Descriptor → AugLagLoss into per‑atom energy & forces for FIRE."""

    def __init__(self, desc: DescriptorModel, auglag: AugLagLoss, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.desc = desc
        self.auglag = auglag
        self.device = device
        self.dtype = dtype

    def forward(self, state: ts.state.SimState) -> Dict[str, torch.Tensor]:
        n_atoms = state.positions.shape[0]

        desc_out = self.desc(state)
        loss_dict = self.auglag(desc_out)
        e_tot = loss_dict["loss"].reshape(1)          # scalar tensor
        e_pa = e_tot / n_atoms                         # per‑atom energy

        # forces = -∂E/∂R
        (forces,) = torch.autograd.grad(
            e_pa,
            state.positions,
            create_graph=True,
            retain_graph=True,
        )
        forces = -forces.unsqueeze(0)  # (1,N,3)

        state._aux = loss_dict  # attach aux for logging
        return {"energy": e_pa.unsqueeze(0), "forces": forces}


# -----------------------------------------------------------------------------
#  Main optimisation routine
# -----------------------------------------------------------------------------

def run_optimization(
    cfg: DictConfig,
    state: SimState,
    #atoms: Atoms,
    spectrum_calc: SpectrumCalculator,
    rdf_data: TargetRDFData,
    hyper: AugLagHyper,
    *,
    device: Union[str, torch.device] = "cpu",
    steps: int = 500,
    mace_relax_every: int = 5,
    mace_relax_steps: int = 10,
    use_wandb: bool = False,
    log_every: int = 25,
    dtype: torch.dtype = torch.float32,
) -> Atoms:

    device = torch.device(device)

    # -- descriptor + loss ----------------------------------------------------
    desc_model = DescriptorModel(
        spectrum_calc,
        rdf_data,
        central=cfg.data.central,
        neighbour=cfg.data.neighbour,
        cutoff=cfg.data.cutoff,
        device=device,
    ).to(device)

    auglag = AugLagLoss(rdf_data, hyper, device=device).to(device)
    fire_model = AugLagEnergyModel(desc_model, auglag, device=device, dtype=dtype)

    # -- initial SimState + FIRE ---------------------------------------------
    #state = ts.io.atoms_to_state([atoms], device=device, dtype=dtype)
    state.positions.requires_grad_(True)
    state.cell.requires_grad_(True)

    fire_init, fire_step = fire(model=fire_model)
    state = fire_init(state)

    # -- MACE model for fast relaxations -------------------------------------
    raw_mace = mace_mp(MaceUrls.mace_mpa_medium, return_raw_model=True, default_dtype=dtype, device=device)
    mace_model = MaceModel(raw_mace, device=device, compute_forces=True, dtype=dtype)
    mace_init, mace_step = fire(model=mace_model)

    logger.info("— Optimising with AugLagLoss + FIRE —")

    for step in range(steps):
        state = fire_step(state)
        aux = state._aux  # type: ignore[attr-defined]

        # ---- interleaved MACE relaxation -----------------------------------
        if mace_relax_every > 0 and step % mace_relax_every == 0:
            logger.info(f"[MACE] Relax block at outer step {step}")
            s_mace = mace_init(state.clone())
            for sub in range(mace_relax_steps):
                s_mace = mace_step(s_mace)
                if sub % 3 == 0:
                    logger.info(f"    MACE sub‑step {sub:02d} | E = {s_mace.energy[0].item():.6f} eV")
            # safe copy‑back without breaking autograd
            with torch.no_grad():
                state.positions.copy_(s_mace.positions.detach())
                state.cell.copy_(s_mace.cell.detach())

            # with torch.no_grad():
            #     state = ts.io.atoms_to_state([ts.io.state_to_atoms(s_mace)[0]], device=device, dtype=dtype)
            #     state.positions.requires_grad_(True)
            #     state.cell.requires_grad_(True)
            #     state = fire_init(state)

        # ---- logging --------------------------------------------------------
        if step % log_every == 0:
            logger.info(
                f"step {step:4d} | E = {aux['loss'].item():.6f} | "
                f"χ²_corr={aux['chi2_corr'].item():.3e} | "
                f"χ²_scatt={aux['chi2_scatt'].item():.3e} | q_loss={aux['q_loss'].item():.3e}"
            )
            if use_wandb and wandb is not None:
                wandb.log({
                    "step": step,
                    "loss": aux["loss"].item(),
                    "chi2_corr": aux["chi2_corr"].item(),
                    "chi2_scatt": aux["chi2_scatt"].item(),
                    "q_loss": aux["q_loss"].item(),
                })

    return state
