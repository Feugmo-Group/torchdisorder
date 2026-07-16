"""
TorchSim Structural Order Parameters
====================================

Supports:
    cn, tet, oct, bcc
    q2, q4, q6 (Steinhardt)

Steinhardt modes
----------------

approx : original fast m=0 approximation
full   : rotationally invariant Steinhardt definition
"""

import torch
import torch.nn as nn
import math
from typing import List, Dict, Optional, Tuple

import torch_sim as ts
from torch_sim.neighbors import torch_nl_linked_cell, torch_nl_n2, torchsim_nl


class TorchSimOrderParameters(nn.Module):

    SUPPORTED_TYPES = [
        "cn",
        "tet",
        "oct",
        "bcc",
        "q2",
        "q4",
        "q6",
    ]

    def __init__(
        self,
        cutoff: float = 3.5,
        device: str = "cpu",
        max_neighbors: int = 64,
        steinhardt_mode: str = "approx",
    ):
        super().__init__()

        self.cutoff = cutoff
        self.device = torch.device(device)
        self.max_neighbors = max_neighbors
        self.steinhardt_mode = steinhardt_mode

        if steinhardt_mode not in ["approx", "full"]:
            raise ValueError("steinhardt_mode must be 'approx' or 'full'")

    # ==========================================================
    # Main forward
    # ==========================================================

    def forward(
        self,
        state: ts.SimState,
        atom_indices: torch.Tensor,
        order_params: List[str],
        element_filter: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:

        for op in order_params:
            if op not in self.SUPPORTED_TYPES:
                raise ValueError(f"Unsupported OP: {op}")

        M = len(atom_indices)

        neighbor_indices, neighbor_pos, valid_mask = self._build_neighbors(
            state, atom_indices
        )

        vectors, distances, thetas, phis = self._compute_geometry(
            state.positions,
            atom_indices,
            neighbor_indices,
            neighbor_pos,
            valid_mask,
        )

        results = {}

        for op in order_params:

            if op == "cn":
                results[op] = valid_mask.sum(dim=1).float()

            elif op == "q2":
                results[op] = self._compute_q2(thetas, phis, valid_mask)

            elif op == "q4":
                results[op] = self._compute_q4(thetas, phis, valid_mask)

            elif op == "q6":
                results[op] = self._compute_q6(thetas, phis, valid_mask)

            else:
                results[op] = torch.zeros(M, device=self.device)

        return results

    # ==========================================================
    # Neighbor construction
    # ==========================================================

    def _build_neighbors(self, state, atom_indices):

        nl = torchsim_nl(
            state.positions,
            cutoff=self.cutoff,
            cell=state.cell,
            pbc=state.pbc,
        )

        neighbor_indices = nl.neighbors[atom_indices]
        neighbor_pos = state.positions[neighbor_indices]

        valid_mask = neighbor_indices >= 0

        neighbor_indices = neighbor_indices[:, : self.max_neighbors]
        neighbor_pos = neighbor_pos[:, : self.max_neighbors]
        valid_mask = valid_mask[:, : self.max_neighbors]

        return neighbor_indices, neighbor_pos, valid_mask

    # ==========================================================
    # Geometry
    # ==========================================================

    def _compute_geometry(
        self,
        positions,
        center_indices,
        neighbor_indices,
        neighbor_positions,
        valid_mask,
    ):

        center_pos = positions[center_indices].unsqueeze(1)

        vectors = neighbor_positions - center_pos

        distances = torch.norm(vectors, dim=2) + 1e-12

        unit_vec = vectors / distances.unsqueeze(-1)

        cos_theta = unit_vec[:, :, 2]
        thetas = torch.acos(torch.clamp(cos_theta, -1, 1))

        phis = torch.atan2(unit_vec[:, :, 1], unit_vec[:, :, 0])

        return vectors, distances, thetas, phis

    # ==========================================================
    # Q2 (full)
    # ==========================================================

    def _compute_q2(self, thetas, phis, valid_mask):

        n_neighbors = valid_mask.sum(dim=1).float()

        cos_t = torch.cos(thetas)

        Y20 = 0.5 * torch.sqrt(5 / math.pi) * (3 * cos_t ** 2 - 1)

        real = (Y20 * valid_mask.float()).sum(dim=1)

        acc = real ** 2

        return torch.sqrt(4 * math.pi * acc / (5 * n_neighbors ** 2 + 1e-10))

    # ==========================================================
    # Spherical harmonics
    # ==========================================================

    def _associated_legendre(self, l, m, x):

        m = abs(m)

        Pmm = torch.ones_like(x)

        if m > 0:
            fact = torch.prod(
                torch.arange(1, 2 * m, 2, device=x.device).float()
            )
            Pmm = ((-1) ** m) * fact * (1 - x ** 2) ** (m / 2)

        if l == m:
            return Pmm

        Pm1m = x * (2 * m + 1) * Pmm

        if l == m + 1:
            return Pm1m

        Pll = None

        for n in range(m + 2, l + 1):
            Pll = ((2 * n - 1) * x * Pm1m - (n + m - 1) * Pmm) / (n - m)
            Pmm = Pm1m
            Pm1m = Pll

        return Pll

    def _spherical_harmonic(self, l, m, theta, phi):

        x = torch.cos(theta)

        P = self._associated_legendre(l, m, x)

        m_abs = abs(m)

        norm = math.sqrt(
            (2 * l + 1)
            / (4 * math.pi)
            * math.factorial(l - m_abs)
            / math.factorial(l + m_abs)
        )

        Y = norm * P

        if m > 0:

            real = Y * torch.cos(m * phi)
            imag = Y * torch.sin(m * phi)

        elif m < 0:

            real = Y * torch.cos(m_abs * phi)
            imag = -Y * torch.sin(m_abs * phi)

        else:

            real = Y
            imag = torch.zeros_like(Y)

        return real, imag

    # ==========================================================
    # Full Steinhardt Q_l
    # ==========================================================

    def _compute_ql_full(self, l, thetas, phis, valid_mask):

        n_neighbors = valid_mask.sum(dim=1).float().clamp(min=1)

        acc = torch.zeros(thetas.shape[0], device=self.device)

        for m in range(-l, l + 1):

            real, imag = self._spherical_harmonic(l, m, thetas, phis)

            real_sum = (real * valid_mask.float()).sum(dim=1)
            imag_sum = (imag * valid_mask.float()).sum(dim=1)

            acc += real_sum ** 2 + imag_sum ** 2

        return torch.sqrt(
            4 * math.pi * acc / ((2 * l + 1) * n_neighbors ** 2 + 1e-10)
        )

    # ==========================================================
    # Q4
    # ==========================================================

    def _compute_q4(self, thetas, phis, valid_mask):

        if self.steinhardt_mode == "full":
            return self._compute_ql_full(4, thetas, phis, valid_mask)

        n_neighbors = valid_mask.sum(dim=1, keepdim=True).float()

        cos_t = torch.cos(thetas)

        pre = (3 / 16.0) * math.sqrt(1 / math.pi) * (
            35 * cos_t ** 4 - 30 * cos_t ** 2 + 3
        )

        real = (pre * valid_mask.float()).sum(dim=1)

        acc = real ** 2

        return torch.sqrt(
            4 * math.pi * acc / (9 * n_neighbors.squeeze() ** 2 + 1e-10)
        )

    # ==========================================================
    # Q6
    # ==========================================================

    def _compute_q6(self, thetas, phis, valid_mask):

        if self.steinhardt_mode == "full":
            return self._compute_ql_full(6, thetas, phis, valid_mask)

        n_neighbors = valid_mask.sum(dim=1, keepdim=True).float()

        cos_t = torch.cos(thetas)

        pre = (1 / 32.0) * math.sqrt(13 / math.pi) * (
            231 * cos_t ** 6
            - 315 * cos_t ** 4
            + 105 * cos_t ** 2
            - 5
        )

        real = (pre * valid_mask.float()).sum(dim=1)

        acc = real ** 2

        return torch.sqrt(
            4 * math.pi * acc / (13 * n_neighbors.squeeze() ** 2 + 1e-10)
        )