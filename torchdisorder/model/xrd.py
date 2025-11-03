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
            device: str | torch.device = "cpu",
            system_idx: torch.Tensor | None = None,
            atomic_numbers: torch.Tensor | None = None,
            compute_q_tet: bool = True,  # Constraint Parameters
            central: str = "Si",  #NEED TO CHANGE THESE
            neighbour: str = "O",
            # cutoff: float = 5.7, #FOR GeO2
            cutoff: float = 3.9,
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
            if self._compute_q_tet:
                q_tet = self.tetrahedral_q(
                    pos=pos_b, cell=cell_b, symbols=symbols_b
                )
                q_tet_list.append(q_tet)


        # Stack and detach all results
        results["G_r"] = torch.stack([g for g in G_r_list])
        results["T_r"] = torch.stack([t for t in T_r_list])
        results["S_Q"] = torch.stack([s for s in S_Q_list])
        if self._compute_q_tet:
            # Get per-atom q_tet values
            q_tet = self.tetrahedral_q(
                pos=pos_b, cell=cell_b, symbols=symbols_b
            )
            results['q_tet'] = q_tet # Shape: (num_Si_atoms,)
        return results

    #Q_tet with the sequential optimization
    # def single_atom_tetrahedral_q(
    #         self, pos: torch.Tensor, cell: torch.Tensor, symbols: list[str],
    #         central_idx: int
    # ) -> torch.Tensor:
    #     """
    #     Compute tetrahedral order parameter for a single central atom using:
    #     q = 1 - (3/8) * sum_{j=1}^3 sum_{k=j+1}^4 (cos(psi_jk) + 1/3)^2
    #
    #     where psi_jk is the angle between lines joining the central atom to
    #     its 4 nearest neighbors j and k.
    #
    #     Returns q=1 for perfect tetrahedral, q=0 for random arrangement.
    #     """
    #     device = pos.device
    #     neigh_mask = torch.tensor(
    #         [s == self.neighbour for s in symbols],
    #         device=device,
    #         dtype=torch.bool,
    #     )
    #
    #     cutoff_tensor = torch.tensor(self.cutoff, device=device, dtype=pos.dtype)
    #     edge_idx, shifts_idx = self.neighbor_list_fn(
    #         positions=pos,
    #         cell=cell,
    #         pbc=True,
    #         cutoff=cutoff_tensor,
    #     )
    #
    #
    #     shifts = torch.mm(shifts_idx.to(dtype=pos.dtype), cell)
    #     i, j = edge_idx
    #     rij = pos[j] + shifts - pos[i]
    #
    #     # Get neighbors for this central atom
    #     mask = (i == central_idx) & neigh_mask[j]
    #     vecs = rij[mask]
    #     n = vecs.size(0)
    #
    #     if n < 4:
    #         # Not enough neighbors - return 0  This could be a problem
    #         return torch.tensor(0.0, device=device, dtype=pos.dtype, requires_grad=True)
    #
    #     # Select exactly 4 nearest neighbors
    #     distances = torch.norm(vecs, dim=1)
    #     nearest_4_indices = torch.argsort(distances)[:4]
    #     vecs_4 = vecs[nearest_4_indices]
    #
    #     # Compute q using exact formula from paper
    #     # Sum over j=1..3, k=j+1..4 (6 angle pairs total)
    #     acc = 0.0
    #     for j_idx in range(3):
    #         for k_idx in range(j_idx + 1, 4):
    #             v_j = vecs_4[j_idx]
    #             v_k = vecs_4[k_idx]
    #
    #             # Compute cosine of angle between vectors
    #             cos_psi = torch.nn.functional.cosine_similarity(
    #                 v_j.view(1, -1), v_k.view(1, -1)
    #             )
    #             cos_psi = torch.clamp(cos_psi, -1.0, 1.0)
    #
    #             # Formula: (cos(psi_jk) + 1/3)^2
    #             term = (cos_psi + 1.0 / 3.0) ** 2
    #             acc += term
    #
    #     # q = 1 - (3/8) * sum of 6 terms
    #     q = 1.0 - (3.0 / 8.0) * acc
    #     return q

    #Zimmerman paper tetrahedral constraint implementation for the single atom constraint application
    # def single_atom_tetrahedral_q(
    #         self, pos: torch.Tensor, cell: torch.Tensor, symbols: list[str],
    #         central_idx: int, delta_theta: float = 12.0
    # ) -> torch.Tensor:
    #     """
    #     Numerically stable Zimmermann formula with NaN protection.
    #     """
    #     device = pos.device
    #     eps = 1e-6  # Increased epsilon for numerical stability
    #
    #     neigh_mask = torch.tensor(
    #         [s == self.neighbour for s in symbols],
    #         device=device,
    #         dtype=torch.bool,
    #     )
    #
    #     cutoff_tensor = torch.tensor(self.cutoff, device=device, dtype=pos.dtype)
    #     edge_idx, shifts_idx = self.neighbor_list_fn(
    #         positions=pos, cell=cell, pbc=True, cutoff=cutoff_tensor,
    #     )
    #
    #     shifts = torch.mm(shifts_idx.to(dtype=pos.dtype), cell)
    #     i, j = edge_idx
    #     rij = pos[j] + shifts - pos[i]
    #
    #     # Get neighbors for this central atom
    #     mask = (i == central_idx) & neigh_mask[j]
    #     vecs = rij[mask]
    #     n_ngh = vecs.size(0)
    #
    #     if n_ngh < 3:
    #         return torch.tensor(0.0, device=device, dtype=pos.dtype, requires_grad=True)
    #
    #     # Normalize vectors with safety check
    #     vec_norms = torch.norm(vecs, dim=1, keepdim=True)
    #     if (vec_norms < eps).any():
    #         # Overlapping atoms - return 0
    #         return torch.tensor(0.0, device=device, dtype=pos.dtype, requires_grad=True)
    #
    #     vecs_norm = vecs / vec_norms
    #
    #     # Convert parameters
    #     delta_theta_rad = delta_theta * torch.pi / 180.0
    #     theta0_rad = 109.47 * torch.pi / 180.0
    #     two_delta_theta_sq = 2 * delta_theta_rad ** 2
    #
    #     total_sum = 0.0
    #
    #     for j_idx in range(n_ngh):
    #         v_j = vecs_norm[j_idx]
    #
    #         for k_idx in range(n_ngh):
    #             if k_idx == j_idx:
    #                 continue
    #
    #             v_k = vecs_norm[k_idx]
    #
    #             # Compute polar angle with safety
    #             cos_theta_k = torch.dot(v_j, v_k).clamp(-1.0 + eps, 1.0 - eps)
    #             theta_k = torch.acos(cos_theta_k)
    #
    #             # Check for NaN
    #             if torch.isnan(theta_k):
    #                 continue
    #
    #             weight_k = torch.exp(-((theta_k - theta0_rad) ** 2) / two_delta_theta_sq)
    #
    #             inner_sum = 0.0
    #
    #             for m_idx in range(n_ngh):
    #                 if m_idx == j_idx or m_idx == k_idx:
    #                     continue
    #
    #                 v_m = vecs_norm[m_idx]
    #
    #                 # Compute polar angle
    #                 cos_theta_m = torch.dot(v_j, v_m).clamp(-1.0 + eps, 1.0 - eps)
    #                 theta_m = torch.acos(cos_theta_m)
    #
    #                 if torch.isnan(theta_m):
    #                     continue
    #
    #                 # Create orthonormal basis with stability checks
    #                 v_k_component = torch.dot(v_k, v_j) * v_j
    #                 v_k_perp = v_k - v_k_component
    #                 v_k_perp_norm = torch.norm(v_k_perp)
    #
    #                 # Skip if vectors are too parallel
    #                 if v_k_perp_norm < eps:
    #                     continue
    #
    #                 v_k_perp = v_k_perp / v_k_perp_norm
    #
    #                 # Project v_m
    #                 v_m_component = torch.dot(v_m, v_j) * v_j
    #                 v_m_perp = v_m - v_m_component
    #                 v_m_perp_norm = torch.norm(v_m_perp)
    #
    #                 if v_m_perp_norm < eps:
    #                     continue
    #
    #                 v_m_perp = v_m_perp / v_m_perp_norm
    #
    #                 # Azimuth angle with safety
    #                 cos_phi = torch.dot(v_m_perp, v_k_perp).clamp(-1.0 + eps, 1.0 - eps)
    #
    #                 # Cross product with check
    #                 cross_prod = torch.cross(v_k_perp, v_m_perp)
    #                 cross_norm = torch.norm(cross_prod)
    #
    #                 if cross_norm < eps:
    #                     # Vectors parallel - azimuth undefined, use 0
    #                     phi = torch.tensor(0.0, device=device)
    #                 else:
    #                     sin_phi = torch.dot(cross_prod, v_j).clamp(-1.0, 1.0)
    #                     phi = torch.atan2(sin_phi, cos_phi)
    #
    #                 if torch.isnan(phi):
    #                     continue
    #
    #                 weight_m = torch.exp(-((theta_m - theta0_rad) ** 2) / two_delta_theta_sq)
    #                 azimuth_term = torch.cos(1.5 * phi) ** 2
    #
    #                 inner_sum += azimuth_term * weight_m
    #
    #             total_sum += weight_k * inner_sum
    #
    #     norm = 1.0 / (n_ngh * (n_ngh - 1) * (n_ngh - 2))
    #     q = norm * total_sum
    #
    #     # Final NaN check
    #     if torch.isnan(q):
    #         return torch.tensor(0.0, device=device, dtype=pos.dtype, requires_grad=True)
    #
    #     return q

    # def sequential_tetrahedral_optimization(
    #         self, pos: torch.Tensor, cell: torch.Tensor, symbols: list[str],
    #         max_steps_per_atom: int = 100,
    #         q_threshold: float = 0.90,
    #         lr: float = 0.01,
    #         start_idx: int = 0,
    #         freeze_strategy: str = "none",  # "none", "central", or "soft"
    #         verbose: bool = True
    # ) -> torch.Tensor:
    #     """
    #     Sequentially optimize each central atom's tetrahedral order.
    #
    #     Args:
    #         pos: atomic positions tensor [n_atoms, 3]
    #         cell: unit cell tensor [3, 3]
    #         symbols: list of chemical symbols
    #         max_steps_per_atom: max optimization steps per central atom
    #         q_threshold: target q value (1.0 = perfect tetrahedral)
    #         lr: learning rate for Adam optimizer
    #         start_idx: starting atom index (will find nearest central atom)
    #         freeze_strategy:
    #             - "none": allow all atoms to move
    #             - "central": freeze the central atom itself
    #             - "soft": apply exponential decay to gradients of optimized atoms
    #         verbose: print progress
    #
    #     Returns:
    #         Updated positions tensor with improved tetrahedral order
    #     """
    #     device = pos.device
    #
    #     # Get all central atom indices
    #     cent_idx = [i for i, s in enumerate(symbols) if s == self.central]
    #
    #     if not cent_idx:
    #         if verbose:
    #             print("No central atoms found!")
    #         return pos
    #
    #     # Find starting central atom (closest to start_idx)
    #     if start_idx not in cent_idx:
    #         distances_to_start = torch.norm(
    #             pos[cent_idx] - pos[start_idx], dim=1
    #         )
    #         start_idx = cent_idx[distances_to_start.argmin().item()]
    #
    #     # Order central atoms by distance from start_idx
    #     distances = torch.norm(pos[cent_idx] - pos[start_idx], dim=1)
    #     sorted_indices = torch.argsort(distances).cpu().tolist()
    #     ordered_cent_idx = [cent_idx[i] for i in sorted_indices]
    #
    #     if verbose:
    #         print(f"\n=== Sequential Tetrahedral Optimization ===")
    #         print(f"Optimizing {len(ordered_cent_idx)} {self.central} atoms")
    #         print(f"Target q >= {q_threshold:.2f}, max {max_steps_per_atom} steps per atom")
    #         print(f"Freeze strategy: {freeze_strategy}\n")
    #
    #     # Track which atoms have been optimized (for soft freezing)
    #     optimized_atoms = set()
    #
    #     # Work with detached copy of positions
    #     pos_opt = pos.detach().clone()
    #
    #     q_initial_list = []
    #     q_final_list = []
    #
    #     for atom_num, ic in enumerate(ordered_cent_idx):
    #         # Compute initial q for this atom
    #         with torch.no_grad():
    #             q_initial = self.single_atom_tetrahedral_q(
    #                 pos_opt, cell, symbols, ic
    #             ).item()
    #         q_initial_list.append(q_initial)
    #
    #         # Create optimizer for this optimization round
    #         pos_local = pos_opt.clone().requires_grad_(True)
    #         optimizer = torch.optim.Adam([pos_local], lr=lr)
    #
    #         best_q = -float('inf')
    #         best_pos = pos_local.detach().clone()
    #         no_improvement = 0
    #
    #         for step in range(max_steps_per_atom):
    #             optimizer.zero_grad()
    #
    #             # Compute q for this specific atom
    #             q = self.single_atom_tetrahedral_q(pos_local, cell, symbols, ic)
    #
    #             # Loss: maximize q (minimize -q)
    #             loss = -q
    #             loss.backward()
    #
    #             # Apply freeze strategy
    #             if freeze_strategy == "central":
    #                 # Freeze the central atom itself
    #                 if pos_local.grad is not None:
    #                     pos_local.grad[ic] = 0.0
    #
    #             elif freeze_strategy == "soft" and optimized_atoms:
    #                 # Apply exponential decay to previously optimized atoms
    #                 with torch.no_grad():
    #                     for opt_idx in optimized_atoms:
    #                         decay_factor = 0.1  # Reduce gradient by 90%
    #                         if pos_local.grad is not None:
    #                             pos_local.grad[opt_idx] *= decay_factor
    #
    #             # Update positions
    #             optimizer.step()
    #
    #             # Track best configuration
    #             current_q = q.item()
    #             if current_q > best_q:
    #                 best_q = current_q
    #                 best_pos = pos_local.detach().clone()
    #                 no_improvement = 0
    #             else:
    #                 no_improvement += 1
    #
    #             # Early stopping if converged or no improvement
    #             if current_q >= q_threshold:
    #                 if verbose and atom_num % 10 == 0:
    #                     print(f"  Atom {atom_num + 1}: converged at step {step}, q={current_q:.4f}")
    #                 break
    #
    #             if no_improvement >= 20:
    #                 break
    #
    #         # Update positions with best found
    #         pos_opt = best_pos.detach().clone()
    #         optimized_atoms.add(ic)
    #         q_final_list.append(best_q)
    #
    #         # Print progress every 10 atoms
    #         if verbose and (atom_num + 1) % 10 == 0:
    #             avg_q_initial = sum(q_initial_list[-10:]) / 10
    #             avg_q_final = sum(q_final_list[-10:]) / 10
    #             print(f"  Processed {atom_num + 1}/{len(ordered_cent_idx)}: "
    #                   f"avg q {avg_q_initial:.3f} → {avg_q_final:.3f}")
    #
    #     if verbose:
    #         avg_q_initial_all = sum(q_initial_list) / len(q_initial_list)
    #         avg_q_final_all = sum(q_final_list) / len(q_final_list)
    #         print(f"\n=== Optimization Complete ===")
    #         print(f"Overall: q {avg_q_initial_all:.4f} → {avg_q_final_all:.4f}")
    #         print(f"Min q: {min(q_final_list):.4f}, Max q: {max(q_final_list):.4f}\n")
    #
    #     return pos_opt

    # #my version of the q_tet with the weighted average
    # def mean_tetrahedral_q(
    #         self, pos: torch.Tensor, cell: torch.Tensor, symbols: list[str],
    #         delta_theta=10.0, theta0=109.47
    # ) -> torch.Tensor:
    #     device = pos.device
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
    #     cutoff_tensor = torch.tensor(self.cutoff, device=device, dtype=pos.dtype)
    #     edge_idx, shifts_idx = self.neighbor_list_fn(
    #         positions=pos,
    #         cell=cell,
    #         pbc=True,
    #         cutoff=cutoff_tensor,
    #     )
    #
    #     shifts = torch.mm(shifts_idx.to(dtype=pos.dtype), cell)
    #     i, j = edge_idx
    #     rij = pos[j] + shifts - pos[i]
    #
    #     theta0_rad = theta0 * torch.pi / 180.0
    #     delta_theta_rad = delta_theta * torch.pi / 180.0
    #
    #     q_vals = []
    #
    #     for ic in cent_idx:
    #         mask = (i == ic) & neigh_mask[j]
    #         vecs = rij[mask]
    #         n = vecs.size(0)
    #         if n < 3:
    #             continue
    #
    #         acc = 0.0
    #         for idx1 in range(n):
    #             for idx2 in range(idx1 + 1, n):
    #                 v1 = vecs[idx1]
    #                 v2 = vecs[idx2]
    #                 cos_theta = torch.nn.functional.cosine_similarity(v1.view(1, -1), v2.view(1, -1))
    #                 cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    #                 theta = torch.acos(cos_theta)
    #                 weight = torch.exp(-((theta - theta0_rad) ** 2) / (2 * delta_theta_rad ** 2))
    #                 acc += weight
    #
    #         norm = 2.0 / (n * (n - 1))
    #         q_val = acc * norm
    #         q_vals.append(q_val)
    #
    #     if not q_vals:
    #         return torch.zeros((), device=device, dtype=pos.dtype, requires_grad=True)
    #
    #     return torch.stack(q_vals).mean()

    #Zimmerman paper implementation of the mean tetrahdral constraint (with torch no grad)
    # def tetrahedral_q(
    #         self, pos: torch.Tensor, cell: torch.Tensor, symbols: list[str],
    #         delta_theta: float = 12.0  # Parameter from paper (in degrees)
    # ) -> torch.Tensor:
    #     """
    #     Compute mean tetrahedral order parameter using Zimmermann et al. (2015) formula:
    #
    #
    #     q_tet = (1 / [N_ngh * (N_ngh-1) * (N_ngh-2)]) *
    #             sum_{j≠k} { exp[-(θ_k - 109.47)² / (2Δθ²)] *
    #                         sum_{m≠j,k} cos²(1.5φ) * exp[-(θ_m - 109.47)² / (2Δθ²)] }
    #
    #
    #     Where:
    #     - θ_k is the polar angle between bonds to neighbors j and k
    #     - φ is the azimuth angle between bond i-m and the plane spanned by i,j,k
    #     - Δθ = 12° controls the Gaussian width for rewarding ideal angles
    #
    #
    #     Returns q≈1 for perfect tetrahedral, q≈0 for non-tetrahedral.
    #     """
    #     device = pos.device
    #
    #     # Find all central atoms (Si)
    #     cent_idx = torch.tensor(
    #         [i for i, s in enumerate(symbols) if s == self.central],
    #         device=device,
    #         dtype=torch.long,
    #     )
    #     print("cent_idx", cent_idx.shape)
    #     print(cent_idx)
    #
    #     # Get the mask of all atoms
    #     neigh_mask = torch.tensor(
    #         [s == self.neighbour for s in symbols],  # Returns [True, True, False, False, True, ...]
    #         device=device,
    #         dtype=torch.bool,
    #     )
    #     print("neigh_mask", neigh_mask.shape)
    #     print(neigh_mask)
    #
    #     # Get neighbor list
    #     cutoff_tensor = torch.tensor(self.cutoff, device=device, dtype=pos.dtype)
    #     edge_idx, shifts_idx = self.neighbor_list_fn(
    #         positions=pos,
    #         cell=cell,
    #         pbc=True,
    #         cutoff=cutoff_tensor,
    #     )
    #
    #     shifts = torch.mm(shifts_idx.to(dtype=pos.dtype), cell)
    #     i, j = edge_idx
    #     rij = pos[j] + shifts - pos[i]
    #
    #     # Convert delta_theta to radians and precompute constants
    #     delta_theta_rad = torch.tensor(delta_theta * torch.pi / 180.0, device=device)
    #     theta0_rad = torch.tensor(109.47 * torch.pi / 180.0, device=device)
    #     two_delta_theta_sq = 2 * delta_theta_rad ** 2
    #
    #     q_vals = []
    #     atoms_with_few_neighbors = 0
    #     neighbor_counts = []
    #     for ic in cent_idx:
    #         # Get vectors to neighbors
    #         mask = (i == ic) & neigh_mask[j]
    #         vecs = rij[mask]
    #         n_ngh = vecs.size(0)
    #         neighbor_counts.append(n_ngh)
    #
    #         # Need at least 3 neighbors
    #         if n_ngh < 4:
    #             atoms_with_few_neighbors += 1
    #             continue
    #
    #         # Normalize vectors
    #         vecs_norm = vecs / torch.norm(vecs, dim=1, keepdim=True)
    #
    #         # Normalization factor from paper
    #         norm = 1.0 / (n_ngh * (n_ngh - 1) * (n_ngh - 2))
    #
    #         total_sum = 0.0
    #
    #         # Outer sum over pairs j, k (j ≠ k)
    #         for j_idx in range(n_ngh):
    #             v_j = vecs_norm[j_idx]
    #
    #             for k_idx in range(n_ngh):
    #                 if k_idx == j_idx:
    #                     continue
    #
    #                 v_k = vecs_norm[k_idx]
    #
    #                 # Compute polar angle θ_k between v_j and v_k
    #                 cos_theta_k = torch.dot(v_j, v_k).clamp(-1.0, 1.0)
    #                 theta_k = torch.acos(cos_theta_k)
    #
    #                 # Gaussian weight for polar angle
    #                 weight_k = torch.exp(-((theta_k - theta0_rad) ** 2) / two_delta_theta_sq)
    #
    #                 # Inner sum over m (m ≠ j, m ≠ k)
    #                 inner_sum = 0.0
    #
    #                 for m_idx in range(n_ngh):
    #                     if m_idx == j_idx or m_idx == k_idx:
    #                         continue
    #
    #                     v_m = vecs_norm[m_idx]
    #
    #                     # Compute polar angle θ_m between v_j and v_m
    #                     cos_theta_m = torch.dot(v_j, v_m).clamp(-1.0, 1.0)
    #                     theta_m = torch.acos(cos_theta_m)
    #
    #                     # Compute azimuth angle φ
    #                     # φ is angle between v_m and the plane spanned by v_j and v_k
    #                     # Plane normal: n = v_j × v_k
    #                     plane_normal = torch.linalg.cross(v_j, v_k)
    #                     plane_normal = plane_normal / (torch.norm(plane_normal) + 1e-8)
    #
    #                     # Project v_m onto plane defined by v_j and v_k
    #                     # φ can be computed from the cross product and dot product
    #                     # For azimuth angle in spherical coords with v_j as north pole:
    #                     # We need angle of v_m in the plane perpendicular to v_j
    #
    #                     # Create orthonormal basis: v_j (north), v_k_perp (east)
    #                     # v_k_perp = normalize(v_k - (v_k·v_j)v_j)
    #                     v_k_component_along_j = torch.dot(v_k, v_j) * v_j
    #                     v_k_perp = v_k - v_k_component_along_j
    #                     v_k_perp = v_k_perp / (torch.norm(v_k_perp) + 1e-8)
    #
    #                     # Project v_m onto plane perpendicular to v_j
    #                     v_m_component_along_j = torch.dot(v_m, v_j) * v_j
    #                     v_m_perp = v_m - v_m_component_along_j
    #                     v_m_perp_norm = torch.norm(v_m_perp) + 1e-8
    #                     v_m_perp = v_m_perp / v_m_perp_norm
    #
    #                     # Azimuth angle φ in plane perpendicular to v_j
    #                     cos_phi = torch.dot(v_m_perp, v_k_perp).clamp(-1.0, 1.0)
    #                     sin_phi = torch.dot(torch.linalg.cross(v_k_perp, v_m_perp), v_j).clamp(-1.0, 1.0)
    #                     phi = torch.atan2(sin_phi, cos_phi)
    #
    #                     # Gaussian weight for polar angle θ_m
    #                     weight_m = torch.exp(-((theta_m - theta0_rad) ** 2) / two_delta_theta_sq)
    #
    #                     # Azimuth contribution: cos²(1.5φ)
    #                     azimuth_term = torch.cos(1.5 * phi) ** 2
    #
    #                     inner_sum += azimuth_term * weight_m
    #
    #                 total_sum += weight_k * inner_sum
    #
    #         q_val = norm * total_sum
    #         q_vals.append(q_val)
    #     print(f"Si atoms skipped due to < 4 neighbors: {atoms_with_few_neighbors}")
    #     print(
    #         f"Neighbor count distribution: min={min(neighbor_counts)}, max={max(neighbor_counts)}, mean={sum(neighbor_counts) / len(neighbor_counts):.2f}")
    #     if not q_vals:
    #         return torch.zeros((), device=device, dtype=pos.dtype, requires_grad=True)
    #
    #     return torch.stack(q_vals)

        #return torch.stack(q_vals)

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
