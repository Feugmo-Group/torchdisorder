# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Note that importing this module has two side effects:
1. It sets the environment variable `PROJECT_ROOT` to the root of the explorers project.
2. It registers a new resolver for OmegaConf, `eval`, which allows us to use `eval` in our config files.
"""
import os
from functools import lru_cache
from pathlib import Path
from pathlib import Path
from ase import Atoms
from torch_sim.state import DeformGradMixin, SimState
import plotly.graph_objects as go
from ase.io import write, Trajectory
import torch
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from typing import Callable, Dict
from ase.data import chemical_symbols
import torch
import torch.nn as nn
from typing import Callable, Dict, Optional
from torch_sim.neighbors import vesin_nl_ts
from ase.data import chemical_symbols
import yaml
from torch_sim.state import DeformGradMixin, SimState
@lru_cache
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache
def get_pyg_device() -> torch.device:
    """
    Some operations of pyg don't work on MPS, so fall back to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


MODELS_PROJECT_ROOT = Path(__file__).resolve().parents[2]
print(f"MODELS_PROJECT_ROOT: {MODELS_PROJECT_ROOT}")

# Set environment variable PROJECT_ROOT so that hydra / OmegaConf can access it.
os.environ["PROJECT_ROOT"] = str(MODELS_PROJECT_ROOT)  # for hydra

#DEFAULT_SAMPLING_CONFIG_PATH = Path(__file__).resolve().parents[3] / "sampling_conf"


SELECTED_ATOMIC_NUMBERS = [
    1,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    37,
    38,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    55,
    56,
    57,
    58,
    59,
    60,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
]
MAX_ATOMIC_NUM = 100


# Set `eval` resolver
def try_eval(s):
    """This is a custom resolver for OmegaConf that allows us to use `eval` in our config files
    with the syntax `${eval:'${foo} + ${bar}'}

    See:
    https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html#id1
    """
    try:
        return eval(s)
    except Exception as e:
        print(f"Calling eval on string {s} raised exception {e}")
        raise


OmegaConf.register_new_resolver("eval", try_eval)


def write_trajectories(atoms: Atoms, traj_path: str, xdatcar_path: str):
    # Write ASE trajectory file (.traj) for internal use
    write(traj_path, atoms, format="traj", append=True)

    # Write to VASP XDATCAR format
    write(xdatcar_path, atoms, format="vasp-xdatcar", append=True)





class OrderParameter(nn.Module):
    def __init__(
        self,
        central: str,
        neighbour: str,
        cutoff: float,
        dtype: torch.dtype,
        device: str | torch.device = "cpu",
        neighbor_list_fn: Callable = vesin_nl_ts,
        compute_q_tet: bool = False,
        compute_q_tetrahedral: bool = False,
        compute_q_octahedral: bool = False,
    ):
        super().__init__()
        self.central = central
        self.neighbour = neighbour
        self.cutoff = cutoff
        self.dtype = dtype
        self.device = device
        self.neighbor_list_fn = neighbor_list_fn

        self.compute_q_tet = compute_q_tet
        self.compute_q_tetrahedral = compute_q_tetrahedral
        self.compute_q_octahedral = compute_q_octahedral

    def setup_from_state(self, state):
        self.positions = state.positions
        self.cell = state.cell
        self.atomic_numbers = state.atomic_numbers
        self.system_idx = state.system_idx
        self.n_systems = state.n_systems

    def forward(self, state) -> Dict[str, torch.Tensor]:
        self.setup_from_state(state)

        pos = self.positions
        cell = self.cell
        atomic_numbers = self.atomic_numbers.detach().cpu().numpy()
        results = {}

        for b in range(self.n_systems):
            mask_b = self.system_idx == b
            pos_b = pos[mask_b]
            cell_b = cell[b]
            symbols_b = [chemical_symbols[z] for z in atomic_numbers[mask_b]]

            if self.compute_q_tet:
                results.setdefault("q", []).append(self.q_tet(pos_b, cell_b, symbols_b))
            if self.compute_q_tetrahedral:
                results.setdefault("q", []).append(self.q_tetrahedral(pos_b, cell_b, symbols_b))
            if self.compute_q_octahedral:
                results.setdefault("q", []).append(self.q_octahedral(pos_b, cell_b, symbols_b))

        return {k: torch.cat(v) for k, v in results.items()}

    def q_tet(self, pos, cell, symbols):
        cent_idx = torch.tensor([i for i, s in enumerate(symbols) if s == self.central], device=self.device)
        neigh_mask = torch.tensor([s == self.neighbour for s in symbols], device=self.device)

        edge_idx, shifts = self.neighbor_list_fn(pos, cell, True, self.cutoff)
        shifts = shifts @ cell
        rij = pos[edge_idx[1]] + shifts - pos[edge_idx[0]]

        q_vals = []
        for ic in cent_idx:
            mask = (edge_idx[0] == ic) & neigh_mask[edge_idx[1]]
            vecs = rij[mask]
            n = vecs.size(0)
            if n < 3:
                continue
            v = vecs / vecs.norm(dim=1, keepdim=True)
            cos_theta = (v @ v.T)[torch.triu_indices(n, n, offset=1)]
            q = 1.0 - (6.0 / (n * (n - 1))) * torch.sum((cos_theta + 1/3) ** 2)
            q_vals.append(q)

        return torch.stack(q_vals) if q_vals else torch.zeros(0, device=self.device, dtype=self.dtype)

    def q_tetrahedral(self, pos, cell, symbols, theta0=109.47, delta_theta=10.0):
        cent_idx = torch.tensor([i for i, s in enumerate(symbols) if s == self.central], device=self.device)
        neigh_mask = torch.tensor([s == self.neighbour for s in symbols], device=self.device)

        edge_idx, shifts = self.neighbor_list_fn(pos, cell, True, self.cutoff)
        rij = pos[edge_idx[1]] + shifts @ cell - pos[edge_idx[0]]

        theta0_rad = theta0 * torch.pi / 180
        delta_rad = delta_theta * torch.pi / 180

        q_vals = []
        for ic in cent_idx:
            mask = (edge_idx[0] == ic) & neigh_mask[edge_idx[1]]
            r_ij = rij[mask]
            n = r_ij.size(0)
            if n < 3:
                q_vals.append(torch.tensor(0.0, device=self.device, dtype=self.dtype))
                continue
            acc = 0.0
            for j1 in range(n):
                for j2 in range(j1 + 1, n):
                    theta = torch.acos(torch.nn.functional.cosine_similarity(
                        r_ij[j1].unsqueeze(0), r_ij[j2].unsqueeze(0)).clamp(-1, 1))
                    acc += torch.exp(-((theta - theta0_rad)**2) / (2 * delta_rad**2))
            norm = 1.0 / (n * (n - 1) / 2)
            q_vals.append(norm * acc)

        return torch.stack(q_vals) if q_vals else torch.zeros(0, device=self.device, dtype=self.dtype)

    def q_octahedral(self, pos, cell, symbols, theta_thr=160.0, delta1=12.0, delta2=10.0):
        cent_idx = torch.tensor([i for i, s in enumerate(symbols) if s == self.central], device=self.device)
        neigh_mask = torch.tensor([s == self.neighbour for s in symbols], device=self.device)

        edge_idx, shifts = self.neighbor_list_fn(pos, cell, True, self.cutoff)
        rij = pos[edge_idx[1]] + shifts @ cell - pos[edge_idx[0]]

        theta_thr = theta_thr * torch.pi / 180
        delta1 = delta1 * torch.pi / 180
        delta2 = delta2 * torch.pi / 180

        q_vals = []
        for ic in cent_idx:
            mask = (edge_idx[0] == ic) & neigh_mask[edge_idx[1]]
            r_ij = rij[mask]
            n = r_ij.size(0)
            if n < 3:
                q_vals.append(torch.tensor(0.0, device=self.device, dtype=self.dtype))
                continue

            acc = 0.0
            for j1 in range(n):
                for j2 in range(n):
                    if j1 == j2: continue
                    theta = torch.acos(torch.nn.functional.cosine_similarity(
                        r_ij[j1].unsqueeze(0), r_ij[j2].unsqueeze(0)).clamp(-1, 1))
                    term1 = 3 * (theta > theta_thr).float() * torch.exp(-((theta - torch.pi) ** 2) / (2 * delta1 ** 2))
                    term2 = 0.0
                    for j3 in range(n):
                        if j3 in [j1, j2]: continue
                        theta13 = torch.acos(torch.nn.functional.cosine_similarity(
                            r_ij[j1].unsqueeze(0), r_ij[j3].unsqueeze(0)).clamp(-1, 1))
                        term2 += (theta < theta_thr).float() * (theta13 < theta_thr).float() * (1.5**2) * torch.exp(
                            -((theta - torch.pi / 2) ** 2) / (2 * delta2 ** 2))
                    acc += term1 + term2
            denom = n * (3 + (n - 2) * (n - 3))
            q_vals.append(acc / denom)

        return torch.stack(q_vals) if q_vals else torch.zeros(0, device=self.device, dtype=self.dtype)



# Helper to build inequality constraints from OrderParameter output
def constraint_q(order_model: nn.Module, key: str, threshold: float, is_ge: bool = False):
    """
    Create a vector-valued constraint function for batched systems.

    Returns:
        Tuple (constraint_fn, is_ge)
    """
    def constraint_fn(state) -> torch.Tensor:
        # Expect output to be (n_atoms,) or (n_atoms, 1)
        # Reduce per system using scatter or global mean per system
        q_val = order_model(state)[key]  # shape: (n_atoms,) or (n_atoms, 1)
        system_idx = state.system_idx    # shape: (n_atoms,)
        n_systems = state.n_systems

        # Compute average q per system
        q_mean = torch.zeros(n_systems, device=q_val.device)
        for b in range(n_systems):
            q_mean[b] = q_val[system_idx == b]

        return q_mean - threshold

    return (constraint_fn, is_ge)



def constraint_order_param(order_param_model: nn.Module, state) -> torch.Tensor:
    """
    Constraint function: enforces order parameter < 0.5 per atom.

    Args:
        order_param_model (nn.Module): Instance of OrderParameter.
        state: TorchSim-compatible state with .positions, .cell, etc.

    Returns:
        torch.Tensor: Constraint values g(x), where g_i(x) = order_param_i - 0.5 <= 0

    use:
    constraint_fn = lambda state: constraint_order_param(order_param_model, state)

        aug_lag = AugmentedLagrangian(
            objective_fn=...,
            constraint_fns=[constraint_fn],
            ...
        )
    """
    result = order_param_model(state)
    # Assuming only one order parameter is computed (e.g., q_tet)
    key = list(result.keys())[0]
    per_atom_values = result[key]
    return per_atom_values - 0.5
