import traceback
import warnings
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any
from torch import nn
import torch
import torch_sim as ts
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts #This is the function I need to use for the neighbour lists
from torch_sim.typing import StateDict

from dataclasses import dataclass, field
from typing import Dict, List
from torchdisorder.common.target_rdf import TargetRDFData
from torchdisorder.model.rdf import SpectrumCalculator
from ase.data import chemical_symbols
import yaml
from torch_sim.state import DeformGradMixin, SimState

from torchdisorder.model.loss import AugLagLoss

from torchdisorder.common.target_rdf import TargetRDFData

from typing import Callable, Optional, Dict, Tuple

import math
#
@dataclass
class CooperState(SimState):
    loss: torch.Tensor
    G_r: Optional[torch.Tensor] = None
    T_r: Optional[torch.Tensor] = None
    S_Q: Optional[torch.Tensor] = None
    q_tet: Optional[torch.Tensor] = None
    diagnostics: Optional[dict] = None
  #                                           THIS NEEDS TO BE FIXED``````



class XRDModel(nn.Module):
    """Compute G(r), T(r), S(Q),  from SimState tensors."""

    def __init__(
            self,
            spectrum_calc: SpectrumCalculator,
            rdf_data: TargetRDFData,
            *,
            neighbor_list_fn: Callable = vesin_nl_ts,
            dtype:torch.dtype,
            device: str | torch.device = "cuda",
            system_idx: torch.Tensor | None = None,
            atomic_numbers: torch.Tensor | None = None,
            compute_q_tet: bool = True,  # Constraint Parameters
            central: str = "Fe",  #NEED TO CHANGE THESE
            neighbour: str = "O",  #Need to change
            # neighbour: str=["O", "N"],
            # cutoff: float = 5.7, #FOR GeO2
            cutoff: float = 4.1, #For FeO3
            # cutoff: float = 3.9, #for SiO2
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
        #adding the constraint stuff
        self.neighbor_list_fn = neighbor_list_fn
        self._compute_q_tet = compute_q_tet  # Set flag
        self.central = central  # Central atom type
        self.neighbour = neighbour  # Neighbor atom type
        self.cutoff = cutoff  # Cutoff for neighbors

        # Set up batch information if atomic numbers are provided
        # Store flag to track if atomic numbers were provided at init
        self.atomic_numbers_in_init = atomic_numbers is not None
        if atomic_numbers is not None:
            if system_idx is None:
                # If batch is not provided, assume all atoms belong to same system
                system_idx = torch.zeros(
                    len(atomic_numbers), dtype=torch.long, device=self.device
                )

            self.setup_from_batch(atomic_numbers, system_idx)

    def setup_from_batch(
            self, atomic_numbers: torch.Tensor, system_idx: torch.Tensor
    ) -> None:
        """Set up internal state from atomic numbers and system indices.

        Processes the atomic numbers and system indices to prepare the model for
        forward pass calculations. Creates the necessary data structures for
        batched processing of multiple systems.

        Args:
            atomic_numbers (torch.Tensor): Atomic numbers tensor with shape [n_atoms].
            system_idx (torch.Tensor): System indices tensor with shape [n_atoms]
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
        atomic_numbers = state.atomic_numbers.detach()   #got rid of cpu and numpy to make compatible with cuda
        system_idx = state.system_idx.detach()



        results = {}
        G_r_list, T_r_list, S_Q_list, q_tet_list= [], [], [], []

        # --- Batched descriptor computation ---
        for b in range(state.n_systems):
            batch_mask = state.system_idx == b
            system_numbers = atomic_numbers[batch_mask]
            pos_b = pos[batch_mask]
            cell_b = cell[b]
            # Convert atomic numbers to chemical symbols
            symbols_b = [chemical_symbols[z] for z in system_numbers]

            G_r = self.spec.compute_neutron_rdf(
                symbols_b, pos_b, cell_b, self.rdf_data.r_bins
            )
            T_r = self.spec.compute_neutron_correlation(
                G_r, self.rdf_data.r_bins, symbols_b, cell_b
            )
            S_Q = self.spec.compute_neutron_sf(
                G_r, self.rdf_data.r_bins, self.rdf_data.q_bins, symbols_b, cell_b
            )

            G_r_list.append(G_r)
            T_r_list.append(T_r)
            S_Q_list.append(S_Q)
            #Uncommenting the constraint line
            # if self._compute_q_tet:
            #     q_tet = self.tetrahedral_q(
            #         pos=pos_b, cell=cell_b, symbols=symbols_b
            #     )
            #     q_tet_list.append(q_tet)
            if self._compute_q_tet:
                q_tet = self.octahedral_q(
                    pos=pos_b, cell=cell_b, symbols=symbols_b
                )
                q_tet_list.append(q_tet)

        # Stack and detach all results
        results["G_r"] = torch.stack([g for g in G_r_list])
        results["T_r"] = torch.stack([t for t in T_r_list])
        results["S_Q"] = torch.stack([s for s in S_Q_list])
        # if self._compute_q_tet:
        #     # Get per-atom q_tet values
        #     q_tet = self.tetrahedral_q(
        #         pos=pos_b, cell=cell_b, symbols=symbols_b
        #     )
        #     results['q_tet'] = q_tet # Shape: (num_Si_atoms,)
        if self._compute_q_tet:
            # Get per-atom q_tet values
            q_tet = self.octahedral_q(
                pos=pos_b, cell=cell_b, symbols=symbols_b
            )
            results['q_tet'] = q_tet # Shape: (num_Si_atoms,)
        return results




    #Zimmerman paper implementation of the mean tetrahdral constraint (with torch no grad)
    # def tetrahedral_q(
    #         self, pos: torch.Tensor, cell: torch.Tensor, symbols: list[str],
    #         delta_theta: float = 12.0
    # ) -> torch.Tensor:
    #     """
    #     Vectorized tetrahedral order parameter using Zimmermann et al. (2015) formula:
    #
    #     q_tet = 1 / [N_ngh(N_ngh-1)(N_ngh-2)] * sum_{j≠k} {
    #         exp[-(θ_k - 109.47°)² / (2Δθ²)] *
    #         sum_{m≠j,k} cos²(1.5φ) * exp[-(θ_m - 109.47°)² / (2Δθ²)]
    #     }
    #
    #     GRADIENT-SAFE: No in-place operations.
    #     """
    #     device = pos.device
    #     dtype = pos.dtype
    #
    #     cent_mask = torch.tensor(
    #         [s == self.central for s in symbols],
    #         device=device, dtype=torch.bool
    #     )
    #     neigh_mask = torch.tensor(
    #         [s == self.neighbour for s in symbols],
    #         device=device, dtype=torch.bool
    #     )
    #
    #     cutoff_tensor = torch.tensor(self.cutoff, device=device, dtype=dtype)
    #     edge_idx, shifts_idx = self.neighbor_list_fn(
    #         positions=pos, cell=cell, pbc=True, cutoff=cutoff_tensor
    #     )
    #
    #     shifts = torch.mm(shifts_idx.to(dtype=dtype), cell)
    #     i, j = edge_idx
    #     rij = pos[j] + shifts - pos[i]
    #     vecs_norm = torch.nn.functional.normalize(rij, dim=1)
    #
    #     # Constants
    #     delta_theta_rad = torch.tensor(delta_theta * torch.pi / 180.0, device=device, dtype=dtype)
    #     theta0_rad = torch.tensor(109.47 * torch.pi / 180.0, device=device, dtype=dtype)
    #     two_delta_theta_sq = 2.0 * delta_theta_rad ** 2
    #
    #     q_vals = []
    #
    #     for ic in torch.where(cent_mask)[0]:
    #         mask = (i == ic) & neigh_mask[j]
    #         vecs = vecs_norm[mask]  # [N_ngh, 3]
    #         n_ngh = vecs.size(0)
    #
    #         if n_ngh < 4:
    #             continue
    #
    #         # Normalization factor
    #         norm = 1.0 / (n_ngh * (n_ngh - 1) * (n_ngh - 2))
    #
    #         # === Vectorized (j, k) pair computation ===
    #         # Create index arrays
    #         idx_range = torch.arange(n_ngh, device=device, dtype=torch.long)
    #         idx_j = idx_range.unsqueeze(1).expand(-1, n_ngh)  # [N, N]
    #         idx_k = idx_range.unsqueeze(0).expand(n_ngh, -1)  # [N, N]
    #         jk_mask = (idx_j != idx_k).float()  # [N, N]
    #
    #         # Pairwise vectors
    #         vecs_j = vecs.unsqueeze(1).expand(-1, n_ngh, -1)  # [N, N, 3]
    #         vecs_k = vecs.unsqueeze(0).expand(n_ngh, -1, -1)  # [N, N, 3]
    #
    #         # Compute polar angles θ_jk between pairs
    #         cos_theta_jk = torch.sum(vecs_j * vecs_k, dim=2).clamp(-1.0, 1.0)
    #         theta_jk = torch.acos(cos_theta_jk)
    #
    #         # Gaussian weight for θ_k
    #         weight_k = torch.exp(-((theta_jk - theta0_rad) ** 2) / two_delta_theta_sq) * jk_mask
    #
    #         # === Vectorized (j, k, m) triple computation ===
    #         vecs_j_3d = vecs.unsqueeze(1).unsqueeze(2).expand(-1, n_ngh, n_ngh, -1)
    #         vecs_k_3d = vecs.unsqueeze(0).unsqueeze(2).expand(n_ngh, -1, n_ngh, -1)
    #         vecs_m_3d = vecs.unsqueeze(0).unsqueeze(1).expand(n_ngh, n_ngh, -1, -1)
    #
    #         # Compute polar angles θ_jm
    #         cos_theta_jm = torch.sum(vecs_j_3d * vecs_m_3d, dim=3).clamp(-1.0, 1.0)
    #         theta_jm = torch.acos(cos_theta_jm)
    #         weight_m = torch.exp(-((theta_jm - theta0_rad) ** 2) / two_delta_theta_sq)
    #
    #         # === Azimuth angle φ computation ===
    #         # Plane normal: n = v_j × v_k
    #         cross_jk = torch.cross(vecs_j_3d, vecs_k_3d, dim=3)
    #         cross_jk_norm = torch.norm(cross_jk, dim=3, keepdim=True).clamp(min=1e-8)
    #         plane_normal = cross_jk / cross_jk_norm
    #
    #         # Orthonormal basis: v_k_perp = normalize(v_k - (v_k·v_j)v_j)
    #         dot_kj = torch.sum(vecs_k_3d * vecs_j_3d, dim=3, keepdim=True)
    #         vecs_k_perp = vecs_k_3d - dot_kj * vecs_j_3d
    #         vecs_k_perp = vecs_k_perp / torch.norm(vecs_k_perp, dim=3, keepdim=True).clamp(min=1e-8)
    #
    #         # Orthonormal basis: v_m_perp = normalize(v_m - (v_m·v_j)v_j)
    #         dot_mj = torch.sum(vecs_m_3d * vecs_j_3d, dim=3, keepdim=True)
    #         vecs_m_perp = vecs_m_3d - dot_mj * vecs_j_3d
    #         vecs_m_perp = vecs_m_perp / torch.norm(vecs_m_perp, dim=3, keepdim=True).clamp(min=1e-8)
    #
    #         # Azimuth angle φ between v_m and plane defined by v_j, v_k
    #         cos_phi = torch.sum(vecs_m_perp * vecs_k_perp, dim=3).clamp(-1.0, 1.0)
    #         cross_km = torch.cross(vecs_k_perp, vecs_m_perp, dim=3)
    #         sin_phi = torch.sum(cross_km * vecs_j_3d, dim=3).clamp(-1.0, 1.0)
    #         phi = torch.atan2(sin_phi, cos_phi)
    #
    #         # === Apply formula: cos²(1.5φ) ===
    #         azimuth_term = torch.cos(1.5 * phi) ** 2
    #
    #         # === m ≠ j AND m ≠ k mask ===
    #         idx_j_3d = idx_j.unsqueeze(2).expand(-1, -1, n_ngh)
    #         idx_k_3d = idx_k.unsqueeze(2).expand(-1, -1, n_ngh)
    #         idx_m_3d = idx_range.view(1, 1, -1).expand(n_ngh, n_ngh, -1)
    #         m_valid = ((idx_m_3d != idx_j_3d) & (idx_m_3d != idx_k_3d)).float()
    #
    #         # Inner sum: sum_{m≠j,k} cos²(1.5φ) * exp[-(θ_m - 109.47°)² / (2Δθ²)]
    #         inner_sum = torch.sum(azimuth_term * weight_m * m_valid, dim=2)
    #
    #         # Outer sum: sum_{j≠k} weight_k * inner_sum
    #         total_sum = torch.sum(weight_k * inner_sum)
    #
    #         # Final q_tet value
    #         q_val = norm * total_sum
    #         q_vals.append(q_val)
    #
    #     if not q_vals:
    #         return torch.zeros((), device=device, dtype=dtype, requires_grad=True)
    #
    #     return torch.stack(q_vals)

    # #Trying the sequential tetrahedral formula for the mean_tetrahedral_q function
    def tetrahedral_q(
            self, pos: torch.Tensor, cell: torch.Tensor, symbols: list[str]
    ) -> torch.Tensor:
        """
        Compute mean tetrahedral order parameter using exact formula:
        q = 1 - (3/8) * sum_{j<k} (cos(psi_jk) + 1/3)^2
        """
        device = pos.device
        cent_idx = torch.tensor(
            [i for i, s in enumerate(symbols) if s == self.central],
            device=device,
            dtype=torch.long,
        )
        # neigh_mask = torch.tensor(
        #     [s == self.neighbour for s in symbols],
        #     device=device,
        #     dtype=torch.bool,
        # )

        # NEW (checks BOTH O and N):
        neigh_mask = torch.tensor(
            [s in ["O", "N"] for s in symbols],  # ← Accept both O and N
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
        """
        Compute mean octahedral order parameter based on the provided formula image.
        q_oct = 1/Norm * {
            sum_{j!=k} [
                3 * H(theta_jk - theta_thr) * exp(-(theta_jk - 180)^2 / (2*d_theta1^2))
            ] +
            sum_{j!=k} sum_{m!=j,k} [
                H(theta_thr - theta_jk) * H(theta_thr - theta_jm) *
                cos^2(2*phi_m) * exp(-(theta_jm - 90)^2 / (2*d_theta2^2))
            ]
        }
        """
        device = pos.device
        dtype = pos.dtype

        # Constants from the formula (angles in degrees)
        theta_thr_deg = 160.0
        d_theta1_deg = 12.0
        d_theta2_deg = 10.0

        # Convert degrees to radians for torch functions
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

                    # --- Calculate theta_jk (polar angle of k in j's frame) ---
                    cos_theta_jk = torch.dot(r_ij, r_ik) / (torch.norm(r_ij) * torch.norm(r_ik))
                    cos_theta_jk = torch.clamp(cos_theta_jk, -1.0, 1.0)
                    theta_jk = torch.acos(cos_theta_jk)
                    theta_jk_deg = math.degrees(theta_jk.item())

                    # --- Term A ---
                    if theta_jk_deg > theta_thr_deg:
                        exponent = -((theta_jk_deg - 180.0) ** 2) / (2 * d_theta1_deg ** 2)
                        total_sum += 3.0 * torch.exp(torch.tensor(exponent, device=device, dtype=dtype))

                    # --- Term B ---
                    if theta_jk_deg < theta_thr_deg:
                        # Define local coordinate system for phi calculation
                        proj_ik_on_ij = torch.dot(r_ik, z_axis) * z_axis
                        x_axis = r_ik - proj_ik_on_ij
                        if torch.norm(x_axis) < 1e-6: continue  # colinear vectors, phi is ill-defined
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

                                phi_m = torch.atan2(torch.dot(r_im_proj_xy, y_axis), torch.dot(r_im_proj_xy, x_axis))

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

    # #Original version
    # def mean_tetrahedral_q(
    #         self, pos: torch.Tensor, cell: torch.Tensor, symbols: list[str]
    # ) -> torch.Tensor:
    #     device, dtype = pos.device, pos.dtype
    #
    #     cent_idx = torch.tensor(
    #         [i for i, s in enumerate(symbols) if s == self.central],
    #         device=device,
    #         dtype=torch.long,
    #     )
    #     neigh_mask = torch.tensor(
    #         [s == self.neighbour for s in symbols],
    #         device=device,
    #         dtype=torch.bool,
    #     )
    #
    #     q_vals = []
    #     #Neighbour list argument needs to be tensor not float
    #     cutoff_tensor = torch.tensor(self.cutoff, device=pos.device, dtype=pos.dtype)
    #     edge_idx, shifts_idx = self.neighbor_list_fn(
    #         positions=pos,
    #         cell=cell,
    #         pbc=True,  # Assume fully periodic
    #         cutoff=cutoff_tensor,
    #     )
    #
    #     shifts = torch.mm(shifts_idx.to(dtype=dtype), cell)
    #     i, j = edge_idx
    #     rij = pos[j] + shifts - pos[i]
    #
    #     for ic in cent_idx:
    #         mask = (i == ic) & neigh_mask[j]
    #         vecs = rij[mask]
    #         n = vecs.size(0)
    #         if n < 3:
    #             continue
    #
    #         v = vecs / vecs.norm(dim=1, keepdim=True)
    #         cos = v @ v.T
    #         cos_pairs = cos[torch.triu_indices(n, n, offset=1)]
    #         q_i = torch.sum((cos_pairs + 1 / 3) ** 2)
    #
    #         norm = 6.0 / (n * (n - 1))
    #         q_vals.append(1.0 - norm * q_i)
    #
    #     if not q_vals:
    #         return torch.zeros((), device=device, dtype=dtype, requires_grad=True)
    #
    #     return torch.stack(q_vals).mean()



    # def q_tetrahedral(self,
    #         pos: torch.Tensor,
    #         cell: torch.Tensor,
    #         symbols: list[str],
    #         central: str,
    #         neighbour: str,
    #         neighbor_list_fn,
    #         cutoff=3.5,
    #         delta_theta=10.0,
    #         theta0=109.47,
    # ) -> torch.Tensor:
    #     device = pos.device
    #
    #     cent_idx = torch.tensor(
    #         [i for i, s in enumerate(symbols) if s == central],
    #         device=device,
    #         dtype=torch.long,
    #     )
    #     neigh_mask = torch.tensor(
    #         [s == neighbour for s in symbols],
    #         device=device,
    #         dtype=torch.bool,
    #     )
    #
    #     edge_idx, shifts = neighbor_list_fn(
    #         positions=pos,
    #         cell=cell,
    #         pbc=True,
    #         cutoff=cutoff,
    #     )
    #     i, j = edge_idx
    #     rij = pos[j] + shifts @ cell - pos[i]
    #
    #     q_vals = []
    #     theta0_rad = torch.tensor(theta0 * torch.pi / 180.0, device=device)
    #     delta_theta_rad = torch.tensor(delta_theta * torch.pi / 180.0, device=device)
    #
    #     for idx in cent_idx:
    #         nbr_mask = (i == idx) & neigh_mask[j]
    #         r_ij = rij[nbr_mask]
    #         n = r_ij.shape[0]
    #
    #         if n < 3:
    #             q_vals.append(torch.tensor(0.0, dtype=pos.dtype, device=device))
    #             continue
    #
    #         acc = 0.0
    #         for j1 in range(n):
    #             for j2 in range(j1 + 1, n):
    #                 v1 = r_ij[j1]
    #                 v2 = r_ij[j2]
    #                 cos_theta = torch.nn.functional.cosine_similarity(v1.view(1, -1), v2.view(1, -1)).clamp(-1.0, 1.0)
    #                 theta = torch.acos(cos_theta)
    #                 weight = torch.exp(-((theta - theta0_rad) ** 2) / (2 * delta_theta_rad ** 2))
    #                 acc += weight
    #
    #         norm = 1.0 / (n * (n - 1) / 2)
    #         q_vals.append(norm * acc)
    #
    #     return torch.stack(q_vals).mean()
    #
    # def _heaviside(self, x: torch.Tensor) -> torch.Tensor:
    #     return (x > 0).float()
    #
    # def _smooth_heaviside(self, x: torch.Tensor, slope: float = 50.0) -> torch.Tensor:
    #     """
    #     Differentiable approximation to the Heaviside step function.
    #     Returns ~0 when x < 0, ~1 when x > 0.
    #
    #     Parameters
    #     ----------
    #     x : torch.Tensor
    #         Input tensor.
    #     slope : float
    #         Controls steepness of transition (higher = sharper).
    #
    #     Returns
    #     -------
    #     torch.Tensor
    #         Smoothed Heaviside output.
    #     """
    #     return torch.sigmoid(slope * x)
    #
    # def q_octahedral(self,
    #         pos: torch.Tensor,
    #         cell: torch.Tensor,
    #         symbols: list[str],
    #         central: str,
    #         neighbour: str,
    #         neighbor_list_fn,
    #         cutoff=3.5,
    #         theta_thr=160.0,
    #         delta1=12.0,
    #         delta2=10.0,
    # ) -> torch.Tensor:
    #     device = pos.device
    #
    #     cent_idx = torch.tensor(
    #         [i for i, s in enumerate(symbols) if s == central],
    #         device=device,
    #         dtype=torch.long,
    #     )
    #     neigh_mask = torch.tensor(
    #         [s == neighbour for s in symbols],
    #         device=device,
    #         dtype=torch.bool,
    #     )
    #
    #     edge_idx, shifts = neighbor_list_fn(
    #         positions=pos,
    #         cell=cell,
    #         pbc=True,
    #         cutoff=cutoff,
    #     )
    #     i, j = edge_idx
    #     rij = pos[j] + shifts @ cell - pos[i]
    #
    #     q_vals = []
    #     theta_thr_rad = torch.tensor(theta_thr * torch.pi / 180.0, device=device)
    #     delta1_rad = torch.tensor(delta1 * torch.pi / 180.0, device=device)
    #     delta2_rad = torch.tensor(delta2 * torch.pi / 180.0, device=device)
    #
    #     for idx in cent_idx:
    #         nbr_mask = (i == idx) & neigh_mask[j]
    #         r_ij = rij[nbr_mask]
    #         n = r_ij.shape[0]
    #
    #         if n < 3:
    #             q_vals.append(torch.tensor(0.0, dtype=pos.dtype, device=device))
    #             continue
    #
    #         acc = 0.0
    #         for j1 in range(n):
    #             for j2 in range(n):
    #                 if j2 == j1:
    #                     continue
    #                 theta_jk = torch.acos(
    #                     torch.nn.functional.cosine_similarity(r_ij[j1].view(1, -1), r_ij[j2].view(1, -1)).clamp(-1.0,
    #                                                                                                             1.0)
    #                 )
    #                 H1 = self._heaviside(theta_jk - theta_thr_rad)
    #                 H2 = self._heaviside(theta_thr_rad - theta_jk)
    #                 term1 = 3 * H1 * torch.exp(-((theta_jk - torch.pi) ** 2) / (2 * delta1_rad ** 2))
    #                 term2 = 0.0
    #                 for j3 in range(n):
    #                     if j3 == j1 or j3 == j2:
    #                         continue
    #                     phi = 1.5
    #                     cos2phi = torch.cos(phi) ** 2
    #                     H3 = self._heaviside(theta_thr_rad - theta_jk)
    #                     theta_j13 = torch.acos(torch.nn.functional.cosine_similarity(
    #                         r_ij[j1].view(1, -1), r_ij[j3].view(1, -1)
    #                     ).clamp(-1.0, 1.0))
    #                     H4 = self._heaviside(theta_thr_rad - theta_j13)
    #                     term2 += H3 * H4 * cos2phi * torch.exp(
    #                         -((theta_jk - torch.pi / 2) ** 2) / (2 * delta2_rad ** 2))
    #                 acc += term1 + term2
    #
    #         denom = n * (3 + (n - 2) * (n - 3))
    #         q_vals.append(acc / denom)
    #
    #     return torch.stack(q_vals).mean()
    #
