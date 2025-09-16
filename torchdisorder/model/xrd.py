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
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.typing import StateDict

from dataclasses import dataclass
from typing import Dict, List
from torchdisorder.common.target_rdf import TargetRDFData
from torchdisorder.model.rdf import SpectrumCalculator
from ase.data import chemical_symbols
import yaml
from torch_sim.state import DeformGradMixin, SimState

from torchdisorder.model.loss import AugLagLoss

from torchdisorder.common.target_rdf import TargetRDFData

from typing import Callable, Optional, Dict, Tuple

# @dataclass
# class AugLagState(SimState):
#     loss: torch.Tensor
#     G_r: Optional[torch.Tensor] = None
#     T_r: Optional[torch.Tensor] = None
#     S_Q: Optional[torch.Tensor] = None
#     q_tet: Optional[torch.Tensor] = None
#     diagnostics: Optional[dict] = None

@dataclass
class AugLagState(SimState):
    loss: torch.Tensor
    T_r: torch.Tensor = torch.tensor([])
    G_r: torch.Tensor = torch.tensor([])
    S_q: torch.Tensor = torch.tensor([])
    q_text: torch.Tensor = torch.tensor([])
    diagnostics: Optional[Dict] = None

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
        atomic_numbers = state.atomic_numbers.detach().cpu().numpy()
        system_idx = state.system_idx.detach().cpu().numpy()



        results = {}
        G_r_list, T_r_list, S_Q_list= [], [], []

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
            # if self._compute_q_tet:
            #     q_tet = self.mean_tetrahedral_q(
            #         pos=pos_b, cell=cell_b, symbols=symbols_b
            #     )
            #     q_tet_list.append(q_tet)

        # Stack and detach all results
        results["G_r"] = torch.stack([g for g in G_r_list])
        results["T_r"] = torch.stack([t for t in T_r_list])
        results["S_Q"] = torch.stack([s for s in S_Q_list])
        # if self._compute_q_tet:
        #     results["q_tet"] = torch.stack([q for q in q_tet_list])

        return results

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
    #
    #     edge_idx, shifts_idx = self.neighbor_list_fn(
    #         positions=pos,
    #         cell=cell,
    #         pbc=True,  # Assume fully periodic
    #         cutoff=self.cutoff,
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
    #
    #
    #
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
