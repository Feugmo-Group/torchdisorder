from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Union

import torch
from torch import nn
import torch_sim as ts
from torch_sim.optimizers import fire
from ase import Atoms

from torchdisorder.model.rdf import SpectrumCalculator
from torchdisorder.model.loss import mean_tetrahedral_q, q_tetrahedral, q_octahedral  # your smoothed implementation
from torchdisorder.model.loss import AugLagHyper, TargetRDFData, chi_squared

__all__ = [
    "DescriptorModel",
]
class DescriptorModel(nn.Module):
    """Compute G(r), T(r), S(Q) and q_tet **directly from SimState tensors**."""

    def __init__(
        self,
        spectrum_calc: SpectrumCalculator,
        rdf_data: TargetRDFData,
        *,
        central: str,
        neighbour: str,
        cutoff: float = 3.5,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.spec = spectrum_calc
        self.rdf_data = rdf_data
        self.central = central
        self.neighbour = neighbour
        self.cutoff = cutoff
        self.device = torch.device(device)

    def forward(self, state: ts.state.SimState) -> Dict[str, torch.Tensor]:
        pos = state.positions               # (N,3) requires_grad=True
        cell = state.cell                   # (3,3) or (B,3,3)

        atoms = ts.io.state_to_atoms(state)[0]
        #symbols = [ts.utils.Z_to_symbol[z.item()] for z in state.atomic_numbers]
        symbols = atoms.get_chemical_symbols()
        # Neutron spectra -------------------------------------------------
        G_r = self.spec.compute_neutron_rdf(symbols, pos, cell, self.rdf_data.r_bins)
        T_r = self.spec.compute_neutron_correlation(G_r, self.rdf_data.r_bins, symbols, cell)
        S_Q = self.spec.compute_neutron_sf(G_r, self.rdf_data.r_bins, self.rdf_data.q_bins, symbols, cell)

        # Tetrahedral OP --------------------------------------------------
        q_tet = mean_tetrahedral_q(
            state=state,
            central=self.central,
            neighbour=self.neighbour,
            cutoff=self.cutoff,
        )

        return {"G_r": G_r, "T_r": T_r, "S_Q": S_Q, "q_tet": q_tet}

# class DescriptorModel(nn.Module):
#     """Compute **G(r)**, **T(r)**, **S(Q)**, and **q_tet** in a single differentiable call.
#
#     Parameters
#     ----------
#     spectrum_calc : SpectrumCalculator
#         Helper object with neutron scattering lengths etc.
#     rdf_data : TargetRDFData
#         Provides r‑bins and q‑bins so forward() doesn’t need extra args.
#     central, neighbour : str
#         Species used for the tetrahedral order parameter.
#     cutoff : float
#         Cut‑off distance for neighbour search.
#     device : torch.device | str
#         Compute device.
#     """
#
#     def __init__(
#         self,
#         spectrum_calc: SpectrumCalculator,
#         rdf_data: TargetRDFData,
#         *,
#         central: str ,
#         neighbour: str ,
#         cutoff: float = 3.5,
#         device: str | torch.device = "cpu",
#     ) -> None:
#         super().__init__()
#         self.spec = spectrum_calc
#         self.rdf_data = rdf_data
#         self.central = central
#         self.neighbour = neighbour
#         self.cutoff = cutoff
#         self.device = torch.device(device)
#
#     def forward(self, atoms: Atoms) -> Dict[str, torch.Tensor]:
#         # Convert ASE → tensors with grad
#         pos = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=self.device, requires_grad=True)
#         cell = torch.tensor(atoms.get_cell().array, dtype=torch.float32, device=self.device, requires_grad=True)
#         symbols = atoms.get_chemical_symbols()
#
#         # Spectra
#         G_r = self.spec.compute_neutron_rdf(symbols, pos, cell, self.rdf_data.r_bins)
#         T_r = self.spec.compute_neutron_correlation(G_r, self.rdf_data.r_bins, symbols, cell)
#         S_Q = self.spec.compute_neutron_sf(G_r, self.rdf_data.r_bins, self.rdf_data.q_bins, symbols, cell)
#
#         # Order parameter (scalar)
#         q_tet = mean_tetrahedral_q(
#             atoms, central=self.central, neighbour=self.neighbour, cutoff=self.cutoff, pbc=True
#         )
#         # Make q_tet tensor on device
#         q_tet = torch.tensor(float(q_tet), device=self.device, requires_grad=True)
#
#         return {
#             "positions": pos,
#             "cell": cell,
#             "symbols": symbols,
#             "G_r": G_r,
#             "T_r": T_r,
#             "S_Q": S_Q,
#             "q_tet": q_tet,
#         }


# # model/gbsd_calculator.py
# """ASE‐compatible calculator that wraps a PyTorch energy function and
# leverages **torch‑sim** utilities for fast force / stress back‑prop. The
# energy function must be written in PyTorch and fully differentiable with
# respect to atomic Cartesian coordinates and (optionally) the cell tensor.
# """
#
# from __future__ import annotations
#
# import time
# from typing import Callable, List, Optional
#
# import numpy as np
# import torch
# from ase import Atoms
# from ase.calculators.calculator import Calculator, all_properties
#
# try:
#     # torch‑sim ≥0.5 style API
#     from torch_sim.gradients import forces as ts_forces
#     from torch_sim.gradients import stress as ts_stress
# except ImportError:  # graceful fallback to torch.autograd
#     ts_forces = None
#     ts_stress = None
#
# from torchdisorder.model.rdf import get_cell_volume  # reuse util already in our project
#
#
# class GBSDCalculator(Calculator):
#     """Generic PyTorch (+ torch‑sim) calculator for ASE optimisation loops.
#
#     Parameters
#     ----------
#     energy_fn : Callable
#         Signature `(positions, species, cell) -> torch.Tensor` (scalar energy).
#     device : str | torch.device, optional
#         CUDA device (e.g. "cuda:0") or "cpu".  If *None*, uses cpu.
#     keep_graph : bool, default False
#         Whether to keep autograd graph after each force/stress evaluation.
#         Useful if you need higher‑order derivatives but increases memory.
#     """
#
#     implemented_properties = ["energy", "free_energy", "forces", "stress"]
#
#     def __init__(
#         self,
#         energy_fn: Callable[[torch.Tensor, List[str], torch.Tensor], torch.Tensor],
#         *,
#         device: str | torch.device | None = None,
#         keep_graph: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__(**kwargs)
#         self.energy_fn = energy_fn
#         self.device = torch.device(device) if device else torch.device("cpu")
#         self.keep_graph = keep_graph
#         self._conversion_time = 0.0  # profile NumPy↔Torch copies
#
#     # ---------------------------------------------------------------------
#     # Helper conversion utilities with timing
#     # ---------------------------------------------------------------------
#     def _np2tensor(self, arr: np.ndarray) -> torch.Tensor:
#         t0 = time.perf_counter()
#         ten = torch.as_tensor(arr, device=self.device, dtype=torch.get_default_dtype())
#         self._conversion_time += time.perf_counter() - t0
#         return ten
#
#     def _tensor2np(self, ten: torch.Tensor) -> np.ndarray:
#         t0 = time.perf_counter()
#         arr = ten.detach().cpu().numpy()
#         self._conversion_time += time.perf_counter() - t0
#         return arr
#
#     # ------------------------------------------------------------------
#     # Core calculation hook
#     # ------------------------------------------------------------------
#     def calculate(
#         self,
#         atoms: Optional[Atoms] = None,
#         properties: Optional[List[str]] = None,
#         system_changes: Optional[List[str]] = None,
#     ) -> None:  # noqa: D401
#         """Populate `self.results` with requested properties."""
#         super().calculate(atoms, properties, system_changes)
#
#         if atoms is None:
#             raise ValueError("GBSDCalculator requires an ASE Atoms object.")
#
#         prop_list = properties or all_properties
#
#         # Convert input to tensors on chosen device
#         scaled_pos = self._np2tensor(atoms.get_scaled_positions())  # (N,3) fractional
#         cell = self._np2tensor(atoms.get_cell().array)             # (3,3)
#         symbols = atoms.get_chemical_symbols()
#
#         # Optionally enable cell strain for differentiable stress
#         strain = torch.zeros((3, 3), device=self.device, dtype=cell.dtype, requires_grad=True)
#         lvecs = cell @ (torch.eye(3, device=self.device) + strain)
#         cart_pos = scaled_pos @ lvecs
#
#         # Forward energy
#         energy = self.energy_fn(cart_pos, symbols, lvecs)
#         self.results["energy"] = float(energy)
#         self.results["free_energy"] = float(energy)  # identical by definition
#
#         # -----------------------------------------------------------------
#         # Forces via torch‑sim (fallback to autograd)
#         # -----------------------------------------------------------------
#         if "forces" in prop_list:
#             if ts_forces is not None:
#                 f = ts_forces(energy, cart_pos, create_graph=self.keep_graph)
#             else:  # autograd fallback (slower)
#                 (grad_pos,) = torch.autograd.grad(
#                     outputs=energy,
#                     inputs=cart_pos,
#                     retain_graph=self.keep_graph,
#                     create_graph=self.keep_graph,
#                 )
#                 f = -grad_pos
#             self.results["forces"] = self._tensor2np(f)
#
#         # -----------------------------------------------------------------
#         # Stress (voigt) via torch‑sim (fallback to autograd)
#         # -----------------------------------------------------------------
#         if "stress" in prop_list:
#             if ts_stress is not None:
#                 s = ts_stress(energy, lvecs, voigt=True, create_graph=self.keep_graph)
#             else:
#                 (grad_strain,) = torch.autograd.grad(
#                     outputs=energy,
#                     inputs=strain,
#                     retain_graph=self.keep_graph,
#                     create_graph=self.keep_graph,
#                 )
#                 volume = get_cell_volume(lvecs)
#                 sigma = grad_strain / volume  # 3x3 tensor
#                 s = torch.stack([sigma[0, 0], sigma[1, 1], sigma[2, 2], sigma[1, 2], sigma[0, 2], sigma[0, 1]])
#             self.results["stress"] = self._tensor2np(s)
#
#     # Convenience for profiling
#     @property
#     def conversion_time(self) -> float:
#         """Total time (s) spent in NumPy↔Torch conversions."""
#         return self._conversion_time
