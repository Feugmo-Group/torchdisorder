
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List
from torchdisorder.common.target_rdf import TargetRDFData
import torch
import yaml
from pathlib import Path

import torch
from ase import Atoms
from typing import List
from torchdisorder.common.neighbors import standard_nl
from torchdisorder.model.rdf import get_distance_matrix
import torch_sim as ts
from torch_sim.state import DeformGradMixin, SimState

import torch
import torch_sim as ts
from torch import Tensor
import time
from pathlib import Path
from typing import Dict, List, Union

import torch
from torch import nn
import torch_sim as ts
from torch_sim.optimizers import fire
from ase import Atoms
from dataclasses import dataclass
from omegaconf import OmegaConf
from pathlib import Path
from typing import Union
from torchdisorder.model.rdf import SpectrumCalculator


__all__ = [
    "AugLagLoss",
 "AugLagHyper"
]


from torchdisorder.model.rdf import (
    SpectrumCalculator,
    get_distance_matrix,
)

# -----------------------------------------------------------------------------
# χ² utility
# -----------------------------------------------------------------------------

def chi_squared(estimate: torch.Tensor, target: torch.Tensor, uncertainty: torch.Tensor | float) -> torch.Tensor:
    if isinstance(uncertainty, (float, int)):
        uncertainty = torch.tensor(uncertainty, device=estimate.device, dtype=estimate.dtype)
    return torch.sum((estimate - target) ** 2 / (uncertainty ** 2))
#normalize

# -----------------------------------------------------------------------------
# Hyper‑parameter dataclass
# -----------------------------------------------------------------------------

@dataclass
class AugLagHyper:
    rho: float = 1e-3
    rho_factor: float = 5.0
    tol: float = 1e-4
    update_every: int = 10
    scale_scatt_init: float = 0.02
    scale_q_init: float = 1.0
    q_target: float = 0.7
    q_uncert: float = 0.05
    
    # @classmethod
    # def from_yaml(cls, path: str | Path) -> "AugLagHyper":
    #     with open(path, "r") as f:
    #         data = yaml.safe_load(f)
    #     return cls(**data)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AugLagHyper":
        cfg = OmegaConf.load(path)
        # Convert OmegaConf DictConfig to standard python dict and instantiate
        return cls(**OmegaConf.to_container(cfg, resolve=True))

# -----------------------------------------------------------------------------
# Tetrahedral order parameter (mean q)
# -----------------------------------------------------------------------------

def _as_3x3(cell: torch.Tensor) -> torch.Tensor:
    """Return a (3,3) lattice matrix no matter what was supplied."""
    if cell.ndim == 3:          # (B,3,3) -> drop batch
        cell = cell[0]
    if cell.ndim == 2 and 1 in cell.shape:   # (1,3) or (3,1)
        cell = torch.diag(cell.flatten())
    if cell.ndim == 1:          # (3,)
        cell = torch.diag(cell)
    return cell

def _ensure_3x3_cell(cell: torch.Tensor) -> torch.Tensor:
    """
    Promote any 1‑D / 3×1 / 1×3 / 1×3×3 cell to a (3,3) matrix suitable for
    neighbour‑list code.
    """
    if cell.ndim == 3:                 # (B,3,3) → take first (batch = 1)
        cell = cell[0]
    if cell.ndim == 2 and 1 in cell.shape:   # (3,1) or (1,3) → diag
        cell = torch.diag(cell.flatten())
    if cell.ndim == 1:                 # (3,) → diag
        cell = torch.diag(cell)
    return cell  # now guaranteed (3,3)




def mean_tetrahedral_q(
    *,
    state: SimState,
    central: str,
    neighbour: str,
    cutoff: float = 3.5,
    pbc: bool = True,
) -> torch.Tensor:
    """
    Differentiable ⟨q_tet⟩ computed directly from tensors.
    Returns a **scalar tensor** so gradients flow into `positions`.
    """

    positions = state.positions
    cell = state.cell

    cell_mat = cell[0]

    atoms = ts.io.state_to_atoms(state)[0]
    # symbols = [ts.utils.Z_to_symbol[z.item()] for z in state.atomic_numbers]
    symbols = atoms.get_chemical_symbols()
    device, dtype = positions.device, positions.dtype
    # cell_mat = _as_3x3(cell.to(device=device, dtype=dtype))

    # ---- neighbour list ------------------------------------------------
    mapping, shifts = standard_nl(
        positions,
        cell_mat,
        pbc,
        torch.tensor(cutoff, device=device, dtype=dtype),
        sort_id=False,
    )
    i, j = mapping
    rij = positions[j] + shifts @ cell_mat - positions[i]

    # ---- masks ---------------------------------------------------------
    cent_idx = torch.tensor(
        [k for k, s in enumerate(symbols) if s == central],
        device=device, dtype=torch.long
    )
    neigh_mask = torch.tensor(
        [s == neighbour for s in symbols],
        device=device, dtype=torch.bool
    )

    q_vals = []
    for ic in cent_idx:
        mask = (i == ic) & neigh_mask[j]
        vecs = rij[mask]
        n = vecs.size(0)
        if n < 3:                # need ≥3 neighbours
            continue

        v = vecs / vecs.norm(dim=1, keepdim=True)
        cos = v @ v.T
        cos_pairs = cos[torch.triu_indices(n, n, offset=1)]
        q_i = torch.sum((cos_pairs + 1 / 3) ** 2)

        norm = 6.0 / (n * (n - 1))
        q_vals.append(1.0 - norm * q_i)

    if not q_vals:  # no valid centres → return zero with gradient
        return torch.zeros((), device=device, dtype=dtype, requires_grad=True)

    return torch.stack(q_vals).mean()

def tetrahedral_q(
    atoms: Atoms,
    central: str ,
    neighbour: str ,
    cutoff: float = 3.5,
    pbc: bool = True,
) -> float:
    """
    Compute the average tetrahedral order parameter q from an ASE Atoms object,
    for a given central atom and neighbor species.

    Parameters
    ----------
    atoms : ASE Atoms
        Atomic configuration.
    central : str
        Chemical symbol of the central atom (e.g., "Si").
    neighbour : str
        Chemical symbol of the neighbor atoms (e.g., "O").
    cutoff : float
        Neighbor cutoff distance (in angstrom).
    pbc : bool
        Whether to apply periodic boundary conditions.

    Returns
    -------
    float
        Mean tetrahedrality value for all central atoms.
    """
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    cell = torch.tensor(atoms.get_cell().array, dtype=torch.float32)
    symbols = atoms.get_chemical_symbols()
    mapping, shifts = standard_nl(pos, cell, pbc, torch.tensor(cutoff))

    i, j = mapping
    rij = pos[j] + shifts @ cell - pos[i]

    # pre‑compute masks
    central_idx = [k for k, s in enumerate(symbols) if s == central]
    neigh_mask_all = torch.tensor([s == neighbour for s in symbols],
                                  dtype=torch.bool, device=pos.device)

    q_vals = torch.zeros(len(central_idx), dtype=torch.float32)

    for idx, ic in enumerate(central_idx):
        mask = (i == ic) & neigh_mask_all[j]  # neighbours of *this* central atom
        vecs = rij[mask]

        n = vecs.shape[0]
        if n < 3:  # need at least 3 neighbours for q
            continue

        # normalised vectors
        v = vecs / vecs.norm(dim=1, keepdim=True)
        cos = v @ v.T  # pairwise cosines
        cos_pairs = cos[torch.triu_indices(n, n, offset=1)]
        q_i = torch.sum((cos_pairs + 1 / 3) ** 2)

        norm = 6.0 / (n * (n - 1))
        q_vals[idx] = 1.0 - norm * q_i

    return q_vals




def heaviside(x):
    return (x > 0).float()

def smooth_heaviside(x: torch.Tensor, slope: float = 50.0) -> torch.Tensor:
    """
    Differentiable approximation to the Heaviside step function.
    Returns ~0 when x < 0, ~1 when x > 0.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    slope : float
        Controls steepness of transition (higher = sharper).

    Returns
    -------
    torch.Tensor
        Smoothed Heaviside output.
    """
    return torch.sigmoid(slope * x)


def q_tetrahedral(atoms: Atoms, central: str, neighbor: str, cutoff=3.5, pbc=True, delta_theta=10.0,
                        theta0=109.47):
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    cell = torch.tensor(atoms.get_cell().array, dtype=torch.float32)
    symbols = atoms.get_chemical_symbols()

    mapping, shifts = standard_nl(pos, cell, pbc, torch.tensor(cutoff))
    i, j = mapping
    rij = pos[j] + shifts @ cell - pos[i]

    central_indices = [idx for idx, sym in enumerate(symbols) if sym == central]
    symbol_tensor = torch.tensor([1 if s == neighbor else 0 for s in symbols], dtype=torch.bool)

    q_vals = []

    for idx in central_indices:
        nbr_mask = (i == idx) & symbol_tensor[j]
        r_ij = rij[nbr_mask]
        n = r_ij.shape[0]

        if n < 3:
            q_vals.append(torch.tensor(0.0, dtype=torch.float32))
            continue

        theta0_rad = torch.tensor(theta0 * torch.pi / 180.0)
        delta_theta_rad = torch.tensor(delta_theta * torch.pi / 180.0)

        acc = 0.0
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                v1 = r_ij[j1]
                v2 = r_ij[j2]
                cos_theta = torch.nn.functional.cosine_similarity(v1.view(1, -1), v2.view(1, -1)).clamp(-1.0, 1.0)
                theta = torch.acos(cos_theta)
                weight = torch.exp(-((theta - theta0_rad) ** 2) / (2 * delta_theta_rad ** 2))
                acc += weight

        norm = 1.0 / (n * (n - 1) / 2)
        q_vals.append(norm * acc)

    return torch.stack(q_vals).mean()


def q_octahedral(
        atoms: Atoms, central: str, neighbor: str, cutoff=3.5,
        pbc=True, theta_thr=160.0, delta1=12.0, delta2=10.0
):
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    cell = torch.tensor(atoms.get_cell().array, dtype=torch.float32)
    symbols = atoms.get_chemical_symbols()

    mapping, shifts = standard_nl(pos, cell, pbc, torch.tensor(cutoff))
    i, j = mapping
    rij = pos[j] + shifts @ cell - pos[i]

    central_indices = [idx for idx, sym in enumerate(symbols) if sym == central]
    symbol_tensor = torch.tensor([1 if s == neighbor else 0 for s in symbols], dtype=torch.bool)

    q_vals = []

    for idx in central_indices:
        nbr_mask = (i == idx) & symbol_tensor[j]
        r_ij = rij[nbr_mask]
        n = r_ij.shape[0]

        if n < 3:
            q_vals.append(torch.tensor(0.0, dtype=torch.float32))
            continue

        theta_thr_rad = torch.tensor(theta_thr * torch.pi / 180.0)
        delta1_rad = torch.tensor(delta1 * torch.pi / 180.0)
        delta2_rad = torch.tensor(delta2 * torch.pi / 180.0)
        acc = 0.0

        for j1 in range(n):
            for j2 in range(n):
                if j2 == j1:
                    continue
                theta_jk = torch.acos(
                    torch.nn.functional.cosine_similarity(r_ij[j1].view(1, -1), r_ij[j2].view(1, -1)).clamp(-1.0, 1.0)
                )
                H1 = heaviside(theta_jk - theta_thr_rad)
                H2 = heaviside(theta_thr_rad - theta_jk)
                term1 = 3 * H1 * torch.exp(-((theta_jk - torch.pi) ** 2) / (2 * delta1_rad ** 2))
                term2 = 0.0
                for j3 in range(n):
                    if j3 == j1 or j3 == j2:
                        continue
                    phi = 1.5  # Use constant φ = 1.5 as placeholder
                    cos2phi = torch.cos(phi) ** 2
                    H3 = heaviside(theta_thr_rad - theta_jk)
                    H4 = heaviside(theta_thr_rad - torch.acos(torch.nn.functional.cosine_similarity(
                        r_ij[j1].view(1, -1), r_ij[j3].view(1, -1)).clamp(-1.0, 1.0)))
                    term2 += H3 * H4 * cos2phi * torch.exp(-((theta_jk - torch.pi / 2) ** 2) / (2 * delta2_rad ** 2))
                acc += term1 + term2

        denom = n * (3 + (n - 2) * (n - 3))
        q_vals.append(acc / denom)

    return torch.stack(q_vals).mean()



def chi_squared(estimate: torch.Tensor, target: torch.Tensor, uncertainty: torch.Tensor | float) -> torch.Tensor:
    """
        Compute the unnormalized chi-squared statistic between estimated and target values.

        The chi-squared value is calculated as:

            χ² = Σ [(estimate - target)² / uncertainty²]

        where each squared deviation is normalized by the square of the uncertainty.
        This metric is commonly used to evaluate the goodness-of-fit of a model to experimental data,
        accounting for the variance in measurements.

        Parameters
        ----------
        estimate : torch.Tensor
            Predicted values (e.g., from a model), shape (...,).
        target : torch.Tensor
            Ground truth or observed values to compare against, same shape as `estimate`.
        uncertainty : torch.Tensor or float
            Measurement uncertainty. Can be:
            - A scalar, applied uniformly
            - A tensor of the same shape as `estimate`, for pointwise uncertainty

        Returns
        -------
        torch.Tensor
            A scalar tensor containing the total chi-squared loss.
        """
    if isinstance(uncertainty, (float, int)):
        uncertainty = torch.tensor(uncertainty, device=estimate.device, dtype=estimate.dtype)
    return torch.sum((estimate - target) ** 2 / (uncertainty ** 2))


@dataclass
class AugLagHyper:
    rho: float = 1e-3
    rho_factor: float = 5.0
    tol: float = 1e-4
    update_every: int = 10
    scale_scatt_init: float = 0.02
    scale_q_init: float = 1.0
    q_target: float = 0.7
    q_uncert: float = 0.05

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AugLagHyper":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

# --- Augmented Lagrangian Loss ---
class AugLagLoss(nn.Module):
    def __init__(
        self,
        rdf_data: TargetRDFData,
        hyper: AugLagHyper = AugLagHyper(),
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.rdf = rdf_data
        self.hyper = hyper
        self.device = torch.device(device)

        self.register_buffer("lambda_corr", torch.zeros(1))
        self.register_buffer("rho", torch.tensor(float(hyper.rho)))
        self.register_buffer("scale_scatt", torch.tensor(float(hyper.scale_scatt_init)))
        self.register_buffer("scale_q", torch.tensor(float(hyper.scale_q_init)))

        self.iter_counter = 0

    def forward(self, desc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        chi2_corr = chi_squared(desc["T_r"], self.rdf.T_r_target, 0.05) / desc["T_r"].numel()
        chi2_scatt = chi_squared(desc["S_Q"], self.rdf.F_q_target, self.rdf.F_q_uncert) / desc["S_Q"].numel()
        q_loss = chi_squared(
            desc["q_tet"],
            torch.tensor(self.hyper.q_target, device=self.device),
            self.hyper.q_uncert,
        )

        total = (
            self.scale_scatt * chi2_scatt
            + self.lambda_corr * chi2_corr
            + 0.5 * self.rho * chi2_corr ** 2
            + self.scale_q * q_loss
        )

        return {
            "loss": total,
            "chi2_corr": chi2_corr,
            "chi2_scatt": chi2_scatt,
            "q_loss": q_loss,
            "scale_q": self.scale_q,
            "scale_scatt": self.scale_scatt,
            "rho": self.rho,
            "lambda_corr": self.lambda_corr,
        }

    def update_penalties(self, loss_dict: dict) -> None:
        g_val = loss_dict["chi2_corr"].detach()
        with torch.no_grad():
            self.lambda_corr += self.rho * g_val
            self.iter_counter += 1
            # print("Type of self.iter_counter:", type(self.iter_counter))
            # print("Type of self.hyper.update_every:", type(self.hyper.update_every))
            # print("Type of g_val:", type(g_val))
            # print("Type of g_val.abs():", type(g_val.abs()))
            # print("Type of self.hyper.tol:", type(self.hyper.tol))
            if self.iter_counter % self.hyper.update_every == 0 and g_val.abs() > self.hyper.tol:
                self.rho *= self.hyper.rho_factor
                self.scale_scatt *= self.hyper.rho_factor ** 0.5
                self.scale_q *= self.hyper.rho_factor ** 0.5







class ChiSquaredObjective(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model  # should be XRDModel with .rdf_data

    def forward(self, state: ts.state.SimState) -> torch.Tensor:
        out = self.model(state)
        T_r = out["T_r"]
        S_Q = out["S_Q"]
        target = self.model.rdf_data

        chi2_corr = chi_squared(T_r, target.T_r_target, 0.05) / T_r.numel()
        chi2_scatt = chi_squared(S_Q, target.F_q_target, target.F_q_uncert) / S_Q.numel()

        return chi2_corr + chi2_scatt


class ConstraintChiSquared(nn.Module):
    def __init__(self, model: nn.Module, chi2_threshold: float = 0.1):
        super().__init__()
        self.model = model  # should be XRDModel with .rdf_data
        self.chi2_threshold = chi2_threshold

    def forward(self, state: ts.state.SimState) -> list[torch.Tensor]:
        out = self.model(state)
        T_r = out["T_r"]
        S_Q = out["S_Q"]
        target = self.model.rdf_data

        chi2_corr = chi_squared(T_r, target.T_r_target, 0.05) / T_r.numel()
        chi2_scatt = chi_squared(S_Q, target.F_q_target, target.F_q_uncert) / S_Q.numel()

        return [chi2_corr - self.chi2_threshold, chi2_scatt - self.chi2_threshold]




class Qtetconstraint(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model  # should be XRDModel with .rdf_data
        self.q_tet_target = torch.tensor(0.5, device=self.model.device)

    def forward(self, state: ts.state.SimState) -> torch.Tensor:
        out = self.model(state)
        return out["q_tet"]-self.q_tet_target





# Objective and constraints

def make_objective_and_constraints(rdf_data,desc):
    def objective(rdf_data,desc):

        chi2_corr = chi_squared(desc["T_r"], rdf_data.T_r_target, 0.05) / desc["T_r"].numel()
        chi2_scatt = chi_squared(desc["S_Q"], rdf_data.rdf.F_q_target, rdf_data.F_q_uncert) / desc["S_Q"].numel()
        return chi2_corr + chi2_scatt

    def constraint_corr(rdf_data,desc):
        return chi_squared(desc["T_r"], rdf_data.T_r_target, 0.05) / desc["T_r"].numel() - 0.1

    def constraint_scatt(rdf_data,desc):
        return chi_squared(desc["S_Q"], rdf_data.rdf.F_q_target, rdf_data.F_q_uncert) / desc["S_Q"].numel() - 0.1

    return objective, [constraint_corr, constraint_scatt]

#Loss function for the cooper constraint
#returns the chi^2 loss
#modeled off the AugLagLoss
class CooperLoss(nn.Module):
    def __init__(self, target_data, q_target=0.7, scale_q=0.2, device="gpu"):
        super().__init__()
        self.target = target_data
        self.q_target = torch.tensor(q_target, device=device)
        self.scale_q = scale_q
        self.device = torch.device(device)

    def forward(self, desc: dict) -> dict:
        chi2_corr = chi_squared(desc["T_r"], self.target.T_r_target, 0.05) / desc["T_r"].numel()
        chi2_scatt = chi_squared(desc["S_Q"], self.target.F_q_target, self.target.F_q_uncert) / desc["S_Q"].numel()
        # q_loss = chi_squared(desc["q_tet"], self.q_target, 0.05)  # You could adjust uncertainty if needed
        # print(f"q_tet value: {desc['q_tet'].item():.4f}, q_loss contribution: {q_loss.item():.6e}")


        total_loss = chi2_scatt * 0.02 + chi2_corr * 1.0 #+ q_loss * 0.1
        # total_loss = chi2_corr

   # for T_r
        # total_loss = chi2_scatt   #for s_Q

        return {
            "loss": total_loss,
            "chi2_corr": chi2_corr,
            "chi2_scatt": chi2_scatt,
            # "q_loss": q_loss,
        }

