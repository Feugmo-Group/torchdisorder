from __future__ import annotations

from typing import Callable, Optional, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt

from torch_sim.state import DeformGradMixin, SimState

from torchdisorder.model.loss import AugLagLoss

from torchdisorder.common.target_rdf import TargetRDFData

from typing import Callable, Optional, Dict, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt

import torch_sim as ts
from torch_sim.io import atoms_to_state, state_to_atoms
import cooper






# --- State ---
@dataclass
class AugLagState(SimState):
    loss: Optional[torch.Tensor] = None
    G_r: Optional[torch.Tensor] = None
    T_r: Optional[torch.Tensor] = None
    S_Q: Optional[torch.Tensor] = None
    q: Optional[torch.Tensor] = None
    diagnostics: Optional[dict] = None
    system_idx: Optional[torch.Tensor] = None
    n_systems:Optional[torch.Tensor] = None


# --- Augmented Lagrangian Optimizer Class ---
def aug_lag(
    model: nn.Module,
    #loss_fn: Callable[[dict], dict],
    loss_fn: Callable[[AugLagLoss],AugLagLoss] = None,
    lr: float = 1e-2,
    max_steps: int = 300,
    tol: float = 1e-6,
    verbose: bool = True,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    stopping_criterion: Optional[Callable[[AugLagState, int], bool]] = None,
    optimize_cell: bool = False,
    #lag_loss: Optional[AugLagLoss] = None,
    lag_loss: Callable[[AugLagLoss],AugLagLoss] = None,
) -> Tuple[Callable[[SimState], AugLagState], Callable[[AugLagState], AugLagState]]:

    def init_fn(state: SimState) -> AugLagState:
        state.positions.requires_grad_(True)
        desc = model(state)
        loss_dict = loss_fn(desc)
        return AugLagState(
            positions=state.positions,
            masses=state.masses.clone(),
            cell=state.cell.clone(),
            atomic_numbers=state.atomic_numbers.clone(),
            system_idx=state.system_idx.clone(),
            n_systems=state.n_systems.clone(),
            pbc=state.pbc,
            loss=loss_dict["loss"],
            G_r=desc.get("G_r", None),
            T_r=desc.get("T_r", None),
            S_Q=desc.get("S_Q", None),
            q=desc.get("q", None),
            diagnostics=loss_dict,
        )

    def update_fn(state: AugLagState,verbose=False) -> AugLagState:
        state.positions.requires_grad_(True)
        if optimize_cell:
            state.cell.requires_grad_(True)

        desc = model(state)
        loss_dict = loss_fn(desc)
        loss = loss_dict["loss"]
        loss.backward()

        with torch.no_grad():
            state.positions -= lr * state.positions.grad
            if optimize_cell:
                state.cell -= lr * state.cell.grad

        if scheduler:
            scheduler.step()

        if lag_loss is not None:
            lag_loss.update_penalties(loss_dict)

        new_state = AugLagState(
            positions=state.positions, #state.positions.detach(),
            masses=state.masses.clone(),
            cell=state.cell.detach() if optimize_cell else state.cell.clone(),
            atomic_numbers=state.atomic_numbers.clone(),
            system_idx=state.system_idx.clone(),
            n_systems=state.n_systems.clone(),
            pbc=state.pbc,
            loss=loss_dict["loss"],
            G_r=desc.get("G_r", None),
            T_r=desc.get("T_r", None),
            S_Q=desc.get("S_Q", None),
            q=desc.get("q", None),
            diagnostics=loss_dict,
        )


        #wandb.log(loss_dict)

        if verbose:
            msg = f"Loss: {new_state.loss.item():.6f}"
            if "chi2_corr" in new_state.diagnostics:
                msg += f", χ2_corr: {new_state.diagnostics['chi2_corr']:.4e}"
            #print(msg)

        return new_state

    return init_fn, update_fn



# --- Augmented Lagrangian Wrapper ---
class AugmentedLagrangian:
    def __init__(
        self,
        objective: Callable[[torch.Tensor], torch.Tensor],
        lam: list[float],
        sigma: list[float],
        constraints_eq: list[Callable[[torch.Tensor], torch.Tensor]] = [],
        constraints_ineq: list[Callable[[torch.Tensor], torch.Tensor]] = [],
    ):
        self.objective = objective
        self.lam_eq = torch.tensor(lam[:len(constraints_eq)], dtype=torch.float32)
        self.lam_ineq = torch.tensor(lam[len(constraints_eq):], dtype=torch.float32)
        self.sigma_eq = torch.tensor(sigma[:len(constraints_eq)], dtype=torch.float32)
        self.sigma_ineq = torch.tensor(sigma[len(constraints_eq):], dtype=torch.float32)
        self.constraints_eq = constraints_eq
        self.constraints_ineq = constraints_ineq

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        loss = self.objective(x)

        for i, c in enumerate(self.constraints_eq):
            ci = c(x)
            loss = loss + self.lam_eq[i] * ci + 0.5 * self.sigma_eq[i] * ci**2

        for i, c in enumerate(self.constraints_ineq):
            ci = torch.clamp(c(x), min=0.0)  # Only penalize if violated
            loss = loss + self.lam_ineq[i] * ci + 0.5 * self.sigma_ineq[i] * ci**2

        return loss

    def update_penalties(self, x: torch.Tensor):
        with torch.no_grad():
            for i, c in enumerate(self.constraints_eq):
                ci = c(x)
                self.lam_eq[i] += self.sigma_eq[i] * ci.item()

            for i, c in enumerate(self.constraints_ineq):
                ci = c(x).item()
                self.lam_ineq[i] = max(0.0, self.lam_ineq[i] + self.sigma_ineq[i] * ci)

        print("Updated Lagrange Multipliers (λ):", self.lam_eq.tolist() + self.lam_ineq.tolist())
        print("Penalty Coefficients (σ):", self.sigma_eq.tolist() + self.sigma_ineq.tolist())


# --- Augmented Lagrangian Optimizer Class ---
def aug_lagg(
    model: nn.Module,
    loss_fn: Callable[[AugLagLoss], AugLagLoss] = None,
    lr: float = 1e-2,
    max_steps: int = 300,
    tol: float = 1e-6,
    verbose: bool = True,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    stopping_criterion: Optional[Callable[[AugLagState, int], bool]] = None,
    optimize_cell: bool = False,
    lag_loss: Optional[AugmentedLagrangian] = None,
) -> Tuple[Callable[[SimState], AugLagState], Callable[[AugLagState], AugLagState]]:

    def init_fn(state: SimState) -> AugLagState:
        state.positions.requires_grad_(True)
        desc = model(state)
        loss_dict = loss_fn(desc)
        return AugLagState(
            positions=state.positions,
            masses=state.masses.clone(),
            cell=state.cell.clone(),
            atomic_numbers=state.atomic_numbers.clone(),
            system_idx=state.system_idx.clone(),
            n_systems=state.n_systems.clone(),
            pbc=state.pbc,
            loss=loss_dict["loss"],
            G_r=desc.get("G_r", None),
            T_r=desc.get("T_r", None),
            S_Q=desc.get("S_Q", None),
            q=desc.get("q", None),
            diagnostics=loss_dict,
        )

    def update_fn(state: AugLagState, verbose=False) -> AugLagState:
        state.positions.requires_grad_(True)
        if optimize_cell:
            state.cell.requires_grad_(True)

        desc = model(state)
        loss_dict = loss_fn(desc)
        loss = loss_dict["loss"]
        loss.backward()

        with torch.no_grad():
            state.positions -= lr * state.positions.grad
            if optimize_cell:
                state.cell -= lr * state.cell.grad

        if scheduler:
            scheduler.step()

        if lag_loss is not None:
            lag_loss.update_penalties(state.positions)

        new_state = AugLagState(
            positions=state.positions,
            masses=state.masses.clone(),
            cell=state.cell.detach() if optimize_cell else state.cell.clone(),
            atomic_numbers=state.atomic_numbers.clone(),
            system_idx=state.system_idx.clone(),
            n_systems=state.n_systems.clone(),
            pbc=state.pbc,
            loss=loss_dict["loss"],
            G_r=desc.get("G_r", None),
            T_r=desc.get("T_r", None),
            S_Q=desc.get("S_Q", None),
            q=desc.get("q", None),
            diagnostics=loss_dict,
        )

        return new_state

    return init_fn, update_fn





#
# # --- State ---
# @dataclass
# class AugLagState(SimState):
#     loss: torch.Tensor
#     G_r: Optional[torch.Tensor] = None
#     T_r: Optional[torch.Tensor] = None
#     S_Q: Optional[torch.Tensor] = None
#     q_tet: Optional[torch.Tensor] = None
#     diagnostics: Optional[dict] = None
#
#
# def aug_lag_init(state: SimState, model: nn.Module, loss_fn: Callable[[dict], dict]) -> AugLagState:
#     state.positions.requires_grad_(True)
#     desc = model(state)
#     loss_dict = loss_fn(desc)
#     return AugLagState(
#         positions=state.positions.detach(),
#         masses=state.masses.clone(),
#         cell=state.cell.clone(),
#         atomic_numbers=state.atomic_numbers.clone(),
#         system_idx=state.system_idx.clone(),
#         pbc=state.pbc,
#         loss=loss_dict["loss"].detach(),
#         G_r=desc.get("G_r", None),
#         T_r=desc.get("T_r", None),
#         S_Q=desc.get("S_Q", None),
#         q_tet=desc.get("q_tet", None),
#         diagnostics=loss_dict,
#     )
#
#
# # --- Augmented Lagrangian Optimizer ---
# def aug_lag(
#     state: SimState,
#     model: torch.nn.Module,
#     loss_fn: Callable[[dict], dict],
#     lr: float = 1e-2,
#     max_steps: int = 300,
#     tol: float = 1e-6,
#     verbose: bool = True,
#     scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
#     stopping_criterion: Optional[Callable[[AugLagState, int], bool]] = None,
#     optimize_cell: bool = False,
#     lag_loss: Optional[AugLagLoss] = None,
# ) -> AugLagState:
#
#     state = aug_lag_init(state, model, loss_fn)
#     prev_loss = state.loss.item()
#     loss_history = []
#     sq_history = []
#
#     # wandb.log({"init_loss": prev_loss})
#     # plt.ion()
#     fig, ax = plt.subplots()
#
#     for step in range(max_steps):
#         state.positions.requires_grad_(True)
#         if optimize_cell:
#             state.cell.requires_grad_(True)
#
#         # Forward pass
#         desc = model(state)
#         loss_dict = loss_fn(desc)
#         loss = loss_dict["loss"]
#         loss.backward()
#
#         with torch.no_grad():
#             state.positions -= lr * state.positions.grad
#             if optimize_cell:
#                 state.cell -= lr * state.cell.grad
#
#         # Learning rate scheduler step
#         if scheduler:
#             scheduler.step()
#
#         # Update lagrangian multipliers and penalty terms
#         if lag_loss is not None:
#             lag_loss.update_penalties(loss_dict)
#
#         state = AugLagState(
#             positions=state.positions.detach(),
#             masses=state.masses.clone(),
#             cell=state.cell.detach() if optimize_cell else state.cell.clone(),
#             atomic_numbers=state.atomic_numbers.clone(),
#             system_idx=state.system_idx.clone(),
#             pbc=state.pbc,
#             loss=loss.detach(),
#             G_r=desc.get("G_r", None),
#             T_r=desc.get("T_r", None),
#             S_Q=desc.get("S_Q", None),
#             q_tet=desc.get("q_tet", None),
#             diagnostics=loss_dict,
#         )
#
#         state.positions.grad = None
#         if optimize_cell:
#             state.cell.grad = None
#
#         current_loss = state.loss.item()
#         # loss_history.append(current_loss)
#         # sq_history.append(state.S_Q.cpu().numpy())
#
#         # wandb.log(loss_dict | {"step": step})
#
#         # # Live plot of S(Q)
#         # ax.clear()
#         # if state.S_Q is not None:
#         #     ax.plot(state.S_Q.cpu().numpy(), label="S(Q)")
#         #     ax.legend()
#         #     ax.set_title(f"S(Q) at step {step}")
#         #     plt.pause(0.01)
#
#         if verbose:
#             log_msg = f"[{step:03d}] Loss: {current_loss:.6f}"
#             if "chi2_corr" in state.diagnostics:
#                 log_msg += f", χ2_corr: {state.diagnostics['chi2_corr']:.4e}"
#             print(log_msg)
#
#         # Convergence check
#         if abs(current_loss - prev_loss) < tol:
#             if verbose:
#                 print("Converged (loss change below tolerance).")
#             break
#
#         if stopping_criterion and stopping_criterion(state, step):
#             if verbose:
#                 print("Stopping criterion triggered.")
#             break
#
#         prev_loss = current_loss
#
#     # plt.ioff()
#     return state

# Cooper state initial draft
# Needs to be a state




#Defining the cooper problem

class StructureFactorCMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self, model, base_state, target_vec: torch.Tensor, target_kind: str, q_bins: torch.Tensor, loss_fn):
        super().__init__()
        self.model = model
        self.base_state = base_state  # This is for the simstate
        self.target = target_vec
        self.kind = target_kind  # "S_Q" or "F_Q"
        self.q_bins = q_bins
        self.loss_fn = loss_fn  # callable returning scalar loss

    def compute_cmp_state(self, positions: torch.Tensor, cell: torch.Tensor) -> tuple:
        self.base_state.positions = positions
        self.base_state.cell = cell
        out = self.model(self.base_state)
        loss_dict = self.loss_fn(out)
        loss = loss_dict["loss"]

        misc = dict(
            Q=self.q_bins.detach().cpu(),
            Y=out[self.kind].detach().cpu(),
            loss=loss.detach().cpu(),
            kind=self.kind,
        )
        cmp_state = cooper.CMPState(loss=loss, observed_constraints={}, misc=misc)

        return cmp_state


#Optimizer for the cooper constraint
#contains init and update functions
#helps for iterative gradient updates

def cooper_optimizer(
    cmp_problem: cooper.ConstrainedMinimizationProblem,
    lr: float = 1e-6,
    max_steps: int = 1000,
    tol: float = 1e-6,
    optimize_cell: bool = False,
    verbose: bool = True,
):
    def init_fn(state: SimState) -> cooper.CMPState:
        state.positions.requires_grad_(True)
        return cmp_problem.compute_cmp_state(state.positions, state.cell)

    def update_fn(cmp_state: cooper.CMPState, sim_state: SimState) -> tuple:
        base_state = cmp_problem.base_state

        base_state.positions.requires_grad_(True)
        if optimize_cell:
            base_state.cell.requires_grad_(True)

        # Compute loss and descriptors
        cmp_state = cmp_problem.compute_cmp_state(base_state.positions, base_state.cell)
        updated_state = cmp_problem.base_state  # Access the updated simulation state

        loss = cmp_state.loss
        print(f"Loss before backward: {loss.item()}")

        max_norm = 100  # tune this value as needed

        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_state.positions, max_norm)

        pos_grad_norm = base_state.positions.grad.norm().item() if base_state.positions.grad is not None else float(
            'nan')
        print(f"Gradient norm positions: {pos_grad_norm}")

        if optimize_cell:
            cell_grad_norm = base_state.cell.grad.norm().item() if base_state.cell.grad is not None else float('nan')
            print(f"Gradient norm cell: {cell_grad_norm}")

        with torch.no_grad():
            base_state.positions -= lr * base_state.positions.grad
            if optimize_cell:
                base_state.cell -= lr * base_state.cell.grad

            base_state.positions.grad = None
            if optimize_cell:
                base_state.cell.grad = None

        print(f"Position norm after update: {base_state.positions.norm().item()}")

        if verbose:
            print(f"Loss: {loss.item():.6f}")

        return cmp_state, updated_state

    return init_fn, update_fn


#Defining the FIRE MACE relaxation every few steps
# Updated relaxation function
def perform_fire_relaxation(sim_state, mace_model, device, dtype, max_steps=1000):
    """
    Performs FIRE relaxation on the current simulation state using the MACE model.

    Args:
        sim_state: The current simulation state (with positions, cell, atomic numbers).
        mace_model: Pre-loaded MaceModel used for relaxation.
        device: Torch device.
        dtype: Torch dtype.
        max_steps: Maximum steps for the FIRE optimizer.

    Returns:
        Updated simulation state with relaxed atomic positions and cell.
    """
    atoms_list_curr = state_to_atoms(sim_state)  # Convert current sim state to ASE atoms list
    print(f"Starting FIRE relaxation for {len(atoms_list_curr)} structures...")

    relaxed_atoms_list = []
    for i, atoms in enumerate(atoms_list_curr):
        print(f"Relaxing structure {i + 1}...")
        relaxed_state = ts.optimize(
            system=atoms,
            model=mace_model,
            optimizer=ts.frechet_cell_fire,
            max_steps=max_steps,
            autobatcher=False,
        )
        print(f"Structure {i + 1} relaxation complete.")
        # Append single Atoms or extend if list
        new_atoms = relaxed_state.to_atoms()
        if isinstance(new_atoms, list):
            relaxed_atoms_list.extend(new_atoms)  # Flatten if list of atoms
        else:
            relaxed_atoms_list.append(new_atoms)

    # Convert relaxed ASE atoms back to simulation state
    new_state = atoms_to_state(relaxed_atoms_list, device=device, dtype=dtype)
    new_state.positions.requires_grad_(True)
    new_state.cell.requires_grad_(True)

    print("Performed FIRE relaxation with MACE.")
    print(f"Updated positions norm: {new_state.positions.norm().item()}")
    print(f"Updated cell tensor: {new_state.cell}")

    return new_state





