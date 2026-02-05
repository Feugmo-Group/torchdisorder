"""
XRD Model for computing structure factors and order parameters.

This module provides the XRDModel class for computing G(r), T(r), S(Q)
from SimState tensors, along with order parameter calculations.
"""

import traceback
import warnings
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Callable, Optional, Dict, Tuple, List
import math

import torch
from torch import nn
import torch_sim as ts
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.typing import StateDict
from torch_sim.state import DeformGradMixin, SimState
from torch.nn.utils.rnn import pad_sequence
from ase.data import chemical_symbols
from dataclasses import dataclass, field
import yaml

from torchdisorder.common.target_rdf import TargetRDFData
from torchdisorder.model.rdf import SpectrumCalculator
from torchdisorder.model.loss import AugLagLoss


@dataclass(kw_only=True)
class CooperState(SimState):
    """Extended SimState with loss and descriptor fields for constrained optimization."""
    
    loss: torch.Tensor
    G_r: torch.Tensor
    T_r: torch.Tensor
    S_Q: torch.Tensor
    q_tet: torch.Tensor

    has_G_r: bool = False
    has_T_r: bool = False
    has_S_Q: bool = False
    has_q_tet: bool = False
    diagnostics: dict | None = None

    _global_attributes: ClassVar[set[str]] = SimState._global_attributes | {
        "loss",
        "G_r", "T_r", "S_Q", "q_tet",
        "has_G_r", "has_T_r", "has_S_Q", "has_q_tet",
        "diagnostics",
    }


class XRDModel(nn.Module):
    """Compute G(r), T(r), S(Q) from SimState tensors.
    
    This model computes neutron/X-ray diffraction patterns and local order
    parameters for atomic structures. Supports batched processing of multiple
    systems.
    
    Args:
        spectrum_calc: SpectrumCalculator instance for computing spectra
        rdf_data: TargetRDFData with experimental target data
        neighbor_list_fn: Neighbor list function (default: vesin_nl_ts)
        dtype: PyTorch data type
        device: Device to use for computations
        system_idx: Optional pre-set system indices
        atomic_numbers: Optional pre-set atomic numbers
        compute_q_tet: Whether to compute tetrahedral order parameter
        central: Central atom type for order parameters (e.g., 'Si', 'Ge')
        neighbour: Neighbor atom type for order parameters (e.g., 'O', 'S')
        cutoff: Cutoff distance for neighbor search
    """

    def __init__(
            self,
            spectrum_calc: SpectrumCalculator,
            rdf_data: TargetRDFData,
            *,
            neighbor_list_fn: Callable = vesin_nl_ts,
            dtype: torch.dtype,
            device: str | torch.device = "cuda",
            system_idx: torch.Tensor | None = None,
            atomic_numbers: torch.Tensor | None = None,
            compute_q_tet: bool = True,
            central: str = "Si",
            neighbour: str = "O",
            cutoff: float = 4.0,
    ) -> None:
        super().__init__()
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.spec = spectrum_calc
        self.rdf_data = rdf_data

        self.dtype = dtype
        self.device = device
        self._memory_scales_with = "n_atoms"
        self.neighbor_list_fn = neighbor_list_fn
        
        # Order parameter settings
        self._compute_q_tet = compute_q_tet
        self.central = central
        self.neighbour = neighbour
        self.cutoff = cutoff

        # Set up batch information if atomic numbers are provided
        self.atomic_numbers_in_init = atomic_numbers is not None
        if atomic_numbers is not None:
            if system_idx is None:
                system_idx = torch.zeros(
                    len(atomic_numbers), dtype=torch.long, device=self.device
                )
            self.setup_from_batch(atomic_numbers, system_idx)

    def setup_from_batch(
            self, atomic_numbers: torch.Tensor, system_idx: torch.Tensor
    ) -> None:
        """Set up internal state from atomic numbers and system indices.

        Args:
            atomic_numbers: Atomic numbers tensor with shape [n_atoms].
            system_idx: System indices tensor with shape [n_atoms]
                indicating which system each atom belongs to.
        """
        self.atomic_numbers = atomic_numbers
        self.system_idx = system_idx

        # Determine number of systems and atoms per system
        self.n_systems = system_idx.max().item() + 1

        # Create ptr tensor for system boundaries
        self.n_atoms_per_system = []
        ptr = [0]
        for i in range(self.n_systems):
            system_mask = system_idx == i
            n_atoms = system_mask.sum().item()
            self.n_atoms_per_system.append(n_atoms)
            ptr.append(ptr[-1] + n_atoms)

        self.ptr = torch.tensor(ptr, dtype=torch.long, device=self.device)
        self.total_atoms = atomic_numbers.shape[0]

    def forward(self, state: ts.state.SimState) -> Dict[str, torch.Tensor]:
        """Forward pass computing all spectra.
        
        Args:
            state: SimState with positions, cell, and atomic_numbers
            
        Returns:
            Dictionary with 'G_r', 'T_r', 'S_Q' tensors
        """
        # Handle dict input
        if isinstance(state, dict):
            state = ts.SimState(**state, masses=torch.ones_like(state["positions"]))

        # Validate atomic numbers
        if state.atomic_numbers is None and not self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers must be provided in either the constructor or forward."
            )
        if state.atomic_numbers is not None and self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers cannot be provided in both the constructor and forward."
            )

        # Set system_idx if needed
        if state.system_idx is None:
            if not hasattr(self, "system_idx"):
                raise ValueError("System indices must be provided if not set during initialization")
            state.system_idx = self.system_idx

        # Update batch info if needed
        if (
                state.atomic_numbers is not None
                and not self.atomic_numbers_in_init
                and not torch.equal(
            state.atomic_numbers,
            getattr(self, "atomic_numbers", torch.zeros(0, device=self.device)),
        )
        ):
            self.setup_from_batch(state.atomic_numbers, state.system_idx)

        pos = state.positions
        cell = state.cell
        atomic_numbers = state.atomic_numbers.detach()
        system_idx = state.system_idx.detach()

        results = {}
        G_r_list, T_r_list, S_Q_list = [], [], []

        # Batched descriptor computation
        for b in range(state.n_systems):
            batch_mask = state.system_idx == b
            system_numbers = atomic_numbers[batch_mask]
            pos_b = pos[batch_mask]
            cell_b = cell[b]
            
            # Convert atomic numbers to chemical symbols
            symbols_b = [chemical_symbols[z] for z in system_numbers]

            # Compute spectra: G(r) → T(r) → S(Q)
            G_r = self.spec.compute_rdf(
                symbols_b, pos_b, cell_b, self.rdf_data.r_bins
            )
            T_r = self.spec.compute_correlation(
                G_r, self.rdf_data.r_bins, symbols_b, cell_b
            )
            S_Q = self.spec.compute_sf(
                G_r, self.rdf_data.r_bins, self.rdf_data.q_bins, symbols_b, cell_b
            )

            G_r_list.append(G_r)
            T_r_list.append(T_r)
            S_Q_list.append(S_Q)

        # Stack results
        results["G_r"] = torch.stack(G_r_list)
        results["T_r"] = torch.stack(T_r_list)
        results["S_Q"] = torch.stack(S_Q_list)
        
        return results

    def tetrahedral_q(
            self, pos: torch.Tensor, cell: torch.Tensor, symbols: list[str]
    ) -> torch.Tensor:
        """Compute tetrahedral order parameter using exact formula.
        
        q = 1 - (3/8) * sum_{j<k} (cos(psi_jk) + 1/3)^2
        
        For each central atom, considers the 4 nearest neighbors and computes
        the angle-based order parameter. Perfect tetrahedron gives q=1.
        
        Args:
            pos: Atomic positions [n_atoms, 3]
            cell: Unit cell [3, 3]
            symbols: List of chemical symbols
            
        Returns:
            Tensor of q values for each central atom
        """
        device = pos.device
        cent_idx = torch.tensor(
            [i for i, s in enumerate(symbols) if s == self.central],
            device=device,
            dtype=torch.long,
        )
        neigh_mask = torch.tensor(
            [s == self.neighbour for s in symbols],
            device=device,
            dtype=torch.bool,
        )

        cutoff_tensor = torch.tensor(self.cutoff, device=device, dtype=pos.dtype)
        edge_idx, shifts_idx = self.neighbor_list_fn(
            positions=pos,
            cell=cell,
            pbc=True,
            cutoff=cutoff_tensor,
        )

        shifts = torch.mm(shifts_idx.to(dtype=pos.dtype), cell)
        i, j = edge_idx
        rij = pos[j] + shifts - pos[i]

        q_vals = []

        for ic in cent_idx:
            mask = (i == ic) & neigh_mask[j]
            vecs = rij[mask]
            n = vecs.size(0)

            # Return 0 if fewer than 4 neighbors (can't form tetrahedron)
            if n < 4:
                continue

            # Select exactly 4 nearest neighbors
            distances = torch.norm(vecs, dim=1)
            nearest_4_indices = torch.argsort(distances)[:4]
            vecs_4 = vecs[nearest_4_indices]

            # Compute q using exact formula from paper
            # Sum over j=1..3, k=j+1..4 (6 angle pairs total)
            acc = 0.0
            for j_idx in range(3):
                for k_idx in range(j_idx + 1, 4):
                    v_j = vecs_4[j_idx]
                    v_k = vecs_4[k_idx]

                    # Compute cosine of angle between vectors
                    cos_psi = torch.nn.functional.cosine_similarity(
                        v_j.view(1, -1), v_k.view(1, -1)
                    )
                    cos_psi = torch.clamp(cos_psi, -1.0, 1.0)

                    # Formula: (cos(psi_jk) + 1/3)^2
                    term = (cos_psi + 1.0 / 3.0) ** 2
                    acc += term

            # q = 1 - (3/8) * sum of 6 terms
            q_val = 1.0 - (3.0 / 8.0) * acc
            q_vals.append(q_val)

        if not q_vals:
            return torch.zeros((), device=device, dtype=pos.dtype, requires_grad=True)

        return torch.stack(q_vals)

    def octahedral_q(
            self, pos: torch.Tensor, cell: torch.Tensor, symbols: list[str]
    ) -> torch.Tensor:
        """Compute octahedral order parameter.
        
        q_oct = 1/Norm * {
            sum_{j!=k} [
                3 * H(theta_jk - theta_thr) * exp(-(theta_jk - 180)^2 / (2*d_theta1^2))
            ] +
            sum_{j!=k} sum_{m!=j,k} [
                H(theta_thr - theta_jk) * H(theta_thr - theta_jm) *
                cos^2(2*phi_m) * exp(-(theta_jm - 90)^2 / (2*d_theta2^2))
            ]
        }
        
        Args:
            pos: Atomic positions [n_atoms, 3]
            cell: Unit cell [3, 3]
            symbols: List of chemical symbols
            
        Returns:
            Tensor of q values for each central atom
        """
        device = pos.device
        dtype = pos.dtype

        # Constants (angles in degrees)
        theta_thr_deg = 160.0
        d_theta1_deg = 12.0
        d_theta2_deg = 10.0

        # Convert to radians
        theta_thr = math.radians(theta_thr_deg)

        cent_idx = torch.tensor(
            [i for i, s in enumerate(symbols) if s == self.central],
            device=device,
            dtype=torch.long,
        )
        neigh_mask = torch.tensor(
            [s == self.neighbour for s in symbols],
            device=device,
            dtype=torch.bool,
        )

        cutoff_tensor = torch.tensor(self.cutoff, device=device, dtype=dtype)
        edge_idx, shifts_idx = self.neighbor_list_fn(
            positions=pos,
            cell=cell,
            pbc=True,
            cutoff=cutoff_tensor,
        )

        shifts = torch.mm(shifts_idx.to(dtype=dtype), cell)
        i, j = edge_idx
        rij = pos[j] + shifts - pos[i]

        q_vals = []

        for ic in cent_idx:
            mask = (i == ic) & neigh_mask[j]
            vecs = rij[mask]
            n_ngh = vecs.size(0)

            if n_ngh < 6:  # Need at least 6 neighbors for octahedron
                continue

            total_sum = 0.0

            # Loop over all ordered pairs of neighbors (j, k)
            for j_idx in range(n_ngh):
                r_ij = vecs[j_idx]
                z_axis = r_ij / torch.norm(r_ij)

                for k_idx in range(n_ngh):
                    if k_idx == j_idx:
                        continue

                    r_ik = vecs[k_idx]

                    # Calculate theta_jk (polar angle of k in j's frame)
                    cos_theta_jk = torch.dot(r_ij, r_ik) / (torch.norm(r_ij) * torch.norm(r_ik))
                    cos_theta_jk = torch.clamp(cos_theta_jk, -1.0, 1.0)
                    theta_jk = torch.acos(cos_theta_jk)
                    theta_jk_deg = math.degrees(theta_jk.item())

                    # Term A
                    if theta_jk_deg > theta_thr_deg:
                        exponent = -((theta_jk_deg - 180.0) ** 2) / (2 * d_theta1_deg ** 2)
                        total_sum += 3.0 * torch.exp(torch.tensor(exponent, device=device, dtype=dtype))

                    # Term B
                    if theta_jk_deg < theta_thr_deg:
                        # Define local coordinate system for phi calculation
                        proj_ik_on_ij = torch.dot(r_ik, z_axis) * z_axis
                        x_axis = r_ik - proj_ik_on_ij
                        if torch.norm(x_axis) < 1e-6:
                            continue  # colinear vectors
                        x_axis = x_axis / torch.norm(x_axis)
                        y_axis = torch.cross(z_axis, x_axis)

                        # Loop over other neighbors m
                        for m_idx in range(n_ngh):
                            if m_idx == j_idx or m_idx == k_idx:
                                continue

                            r_im = vecs[m_idx]

                            cos_theta_jm = torch.dot(r_ij, r_im) / (torch.norm(r_ij) * torch.norm(r_im))
                            cos_theta_jm = torch.clamp(cos_theta_jm, -1.0, 1.0)
                            theta_jm = torch.acos(cos_theta_jm)
                            theta_jm_deg = math.degrees(theta_jm.item())

                            if theta_jm_deg < theta_thr_deg:
                                # Calculate phi_m
                                proj_im_on_z = torch.dot(r_im, z_axis) * z_axis
                                r_im_proj_xy = r_im - proj_im_on_z

                                phi_m = torch.atan2(
                                    torch.dot(r_im_proj_xy, y_axis),
                                    torch.dot(r_im_proj_xy, x_axis)
                                )

                                cos2_2phi = torch.cos(2 * phi_m) ** 2
                                exponent = -((theta_jm_deg - 90.0) ** 2) / (2 * d_theta2_deg ** 2)

                                term_B = cos2_2phi * torch.exp(torch.tensor(exponent, device=device, dtype=dtype))
                                total_sum += term_B

            # Normalization
            if n_ngh > 3:
                norm_factor = n_ngh * (3 + (n_ngh - 2) * (n_ngh - 3))
                q_val = total_sum / norm_factor
                q_vals.append(q_val)

        if not q_vals:
            return torch.zeros((), device=device, dtype=dtype, requires_grad=True)

        return torch.stack(q_vals)
