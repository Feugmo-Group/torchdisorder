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