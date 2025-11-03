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


import torch_sim as ts
from torch_sim.io import atoms_to_state, state_to_atoms
import cooper
from ase.data import chemical_symbols
from cooper.penalty_coefficients import PenaltyCoefficient






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


class ConstantPenalty(PenaltyCoefficient):
    """Constant penalty coefficient for AugmentedLagrangian."""

    def __init__(self, value: float, device='cuda'):
        super().__init__(init=torch.tensor(value, device=device))
        self.expects_constraint_features = False

    def __call__(self, constraint_features=None):
        """Return the constant penalty value."""
        return self.init


class StructureFactorCMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self, model, base_state, target_vec, target_kind, q_bins, loss_fn,
                 q_threshold=0.7, device='cuda', penalty_rho=10.0):
        super().__init__()
        self.model = model
        self.base_state = base_state

        # Store both target values and uncertainty from rdf_data object
        self.target = target_vec.F_q_target
        self.target_uncert = target_vec.F_q_uncert

        self.kind = target_kind
        self.q_bins = q_bins
        self.loss_fn = loss_fn
        self.q_threshold = q_threshold
        self.device = device

        # Detect placeholder region (where dF was originally zero)
        self.placeholder_mask = (self.target_uncert < 1e-6).to(device)
        n_placeholder = self.placeholder_mask.sum().item()
        print(f"Found {n_placeholder} placeholder points where dF = 1e-7")

        # Count Si atoms for per-atom constraints
        symbols = [chemical_symbols[int(z)] for z in base_state.atomic_numbers.cpu()]
        num_central = sum(1 for s in symbols if s == 'Si')
        print(f"Number of central atoms: {num_central}")

        # CREATE MULTIPLIER (λ tensor)
        multiplier = cooper.multipliers.DenseMultiplier(
            num_constraints=num_central,
            device=device
        )
        penalty_coeff = ConstantPenalty(penalty_rho, device=device)

        # CREATE CONSTRAINT
        self.q_tet_constraint = cooper.Constraint(
            multiplier=multiplier,
            constraint_type=cooper.ConstraintType.INEQUALITY,
            formulation_type=cooper.formulations.AugmentedLagrangian,
            penalty_coefficient=penalty_coeff,
        )

    def compute_cmp_state(self, positions, cell, step=None):
        self.base_state.positions = positions
        self.base_state.cell = cell
        out = self.model(self.base_state)

        # Get predicted S(Q)
        pred_sq = out[self.kind].squeeze()

        # Clamp prediction to experimental value in placeholder region
        pred_sq_clamped = pred_sq.clone()
        pred_sq_clamped[self.placeholder_mask] = self.target[self.placeholder_mask]

        # Replace in output dict for loss computation
        out[self.kind] = pred_sq_clamped.unsqueeze(0) if pred_sq_clamped.dim() == 1 else pred_sq_clamped

        # Call loss_fn with clamped output (placeholder region contributes 0 to chi²)
        loss_dict = self.loss_fn(out)
        chi2_loss = loss_dict["chi2_scatt"]
        # chi2_loss = loss_dict["chi2_corr"]  # Uncomment for T(r) optimization

        # Total loss is just chi²
        total_loss = chi2_loss

        # COMPUTE CONSTRAINT VIOLATION
        q_tet_per_atom = out['q_tet']
        violation = self.q_threshold - q_tet_per_atom

        q_tet_state = cooper.ConstraintState(violation=violation)
        observed_constraints = {self.q_tet_constraint: q_tet_state}

        misc = dict(
            Q=self.q_bins,
            Y=pred_sq_clamped,
            loss=total_loss,
            chi2_loss=chi2_loss,
            kind=self.kind,
            qtet=q_tet_per_atom
        )

        return cooper.CMPState(loss=total_loss, observed_constraints=observed_constraints, misc=misc)



#Optimizer for the cooper constraint
#contains init and update functions
#helps for iterative gradient updates

#Cooper optimizer for Adam
def cooper_optimizer(
        cmp_problem: cooper.ConstrainedMinimizationProblem,
        base_state: SimState,
        monitor=None,  # ← ADD monitor parameter
        primal_lr: float = 1e-6,
        dual_lr: float = 1e-2,
        max_steps: int = 1000,
        optimize_cell: bool = False,
        verbose: bool = True,
):
    """Proper Cooper optimizer using SimultaneousOptimizer."""

    # Setup parameters
    base_state.positions.requires_grad_(True)
    primal_params = [base_state.positions]

    if optimize_cell:
        base_state.cell.requires_grad_(True)
        primal_params.append(base_state.cell)

    # CREATE PRIMAL OPTIMIZER (for positions)
    primal_optimizer = torch.optim.Adam(primal_params, lr=primal_lr)

    # CREATE DUAL OPTIMIZER (for λ)
    dual_optimizer = torch.optim.SGD(
        cmp_problem.dual_parameters(),
        lr=dual_lr,
        maximize=True
    )

    # CREATE COOPER OPTIMIZER
    cooper_opt = cooper.optim.SimultaneousOptimizer(
        cmp=cmp_problem,
        primal_optimizers=primal_optimizer,
        dual_optimizers=dual_optimizer
    )

    # TRAINING LOOP
    for step in range(max_steps):
        # roll() handles everything: zero_grad, compute_cmp_state, backward, updates
        roll_out = cooper_opt.roll(
            compute_cmp_state_kwargs={
                "positions": base_state.positions,
                "cell": base_state.cell
            }
        )

        # Extract results
        cmp_state = roll_out.cmp_state
        loss = cmp_state.loss
        violations = list(cmp_state.observed_constraints.values())[0].violation

        # Compute metrics
        avg_violation = violations.mean().item()
        max_violation = violations.max().item()
        num_violated = (violations > 0).sum().item()

        # LOGGING
        if verbose and step % 1 == 0:
            print(f"Step {step}: Loss={loss.item():.6f}, "
                  f"Avg q_tet violation={avg_violation:.6f}, "
                  f"Max q_tet violation={max_violation:.6f}")

        # ========== UPDATE MONITOR ==========
        if monitor is not None and step % 1 == 0:
            pred_sq = cmp_state.misc.get("Y")
            if pred_sq is not None:
                pred_sq_np = pred_sq.detach().cpu().numpy().flatten()
                monitor.update_data(
                    step=step,
                    loss=loss.item(),
                    pred_sq=pred_sq_np,
                    num_violated=num_violated
                )

    return base_state

# #Cooper optimizer with LBFG
# def cooper_optimizer(
#         cmp_problem: cooper.ConstrainedMinimizationProblem,
#         base_state: SimState,
#         monitor=None,  # ← ADD THIS
#         primal_lr: float = 1.0,
#         dual_lr: float = 1e-2,
#         max_steps: int = 1000,
#         optimize_cell: bool = False,
#         verbose: bool = True,
# ):
#     """L-BFGS optimizer with AugmentedLagrangian constraints."""
#
#     # Setup parameters
#     base_state.positions.requires_grad_(True)
#     primal_params = [base_state.positions]
#
#     if optimize_cell:
#         base_state.cell.requires_grad_(True)
#         primal_params.append(base_state.cell)
#
#     # CREATE OPTIMIZERS
#     primal_optimizer = torch.optim.LBFGS(
#         primal_params,
#         lr=primal_lr,
#         max_iter=5,
#         history_size=10,
#         line_search_fn="strong_wolfe"
#     )
#
#     dual_optimizer = torch.optim.SGD(
#         cmp_problem.dual_parameters(),
#         lr=dual_lr,
#         maximize=True
#     )
#
#     # TRAINING LOOP
#     for step in range(max_steps):
#
#         # ========== PRIMAL STEP (L-BFGS) ==========
#         def closure():
#             primal_optimizer.zero_grad()
#
#             # Compute CMP state
#             cmp_state = cmp_problem.compute_cmp_state(
#                 positions=base_state.positions,
#                 cell=base_state.cell
#             )
#
#             # Compute Augmented Lagrangian manually
#             loss = cmp_state.loss
#             lagrangian = loss
#
#             # Add AugmentedLagrangian penalty for constraints
#             for constraint, constraint_state in cmp_state.observed_constraints.items():
#                 violation = constraint_state.violation  # Shape: [num_Ge]
#                 lambda_val = constraint.multiplier.weight  # Shape: [num_Ge]
#                 rho = constraint.penalty_coefficient()  # Scalar (your penalty_rho)
#
#                 # Augmented Lagrangian formula: λ*c + (ρ/2)*c²
#                 penalty = lambda_val * violation + 0.5 * rho * (violation ** 2)
#                 lagrangian = lagrangian + penalty.sum()
#
#             # Backward pass
#             lagrangian.backward()
#             return lagrangian
#
#         # L-BFGS step
#         primal_optimizer.step(closure)
#
#         # ========== DUAL STEP (Update multipliers) ==========
#         dual_optimizer.zero_grad()
#
#         # Recompute state for dual update
#         with torch.no_grad():
#             cmp_state = cmp_problem.compute_cmp_state(
#                 positions=base_state.positions,
#                 cell=base_state.cell
#             )
#
#         # Compute Lagrangian again for dual gradients
#         loss = cmp_state.loss
#         lagrangian_dual = loss
#
#         for constraint, constraint_state in cmp_state.observed_constraints.items():
#             violation = constraint_state.violation
#             lambda_val = constraint.multiplier.weight
#             rho = constraint.penalty_coefficient()
#
#             penalty = lambda_val * violation + 0.5 * rho * (violation ** 2)
#             lagrangian_dual = lagrangian_dual + penalty.sum()
#
#         lagrangian_dual.backward()
#         dual_optimizer.step()
#
#         # ========== PROJECT MULTIPLIERS (λ ≥ 0 for inequality) ==========
#         with torch.no_grad():
#             for constraint in cmp_problem.constraints():
#                 if constraint.constraint_type == cooper.ConstraintType.INEQUALITY:
#                     constraint.multiplier.weight.clamp_(min=0.0)
#
#         # ========== COMPUTE STATE FOR LOGGING/MONITORING (MOVE THIS UP) ==========
#         with torch.no_grad():
#             cmp_state = cmp_problem.compute_cmp_state(
#                 positions=base_state.positions,
#                 cell=base_state.cell
#             )
#             loss = cmp_state.loss
#             violations = list(cmp_state.observed_constraints.values())[0].violation
#
#             avg_violation = violations.mean().item()
#             max_violation = violations.max().item()
#             num_violated = (violations > 0).sum().item()
#
#         # ========== LOGGING ==========
#         if verbose and step % 1 == 0:
#             print(f"Step {step}: Loss={loss.item():.6f}, "
#                   f"Avg q_tet violation={avg_violation:.6f}, "
#                   f"Max q_tet violation={max_violation:.6f}")
#
#         # ========== UPDATE MONITOR ==========
#         if monitor is not None and step % 1 == 0:
#             pred_sq = cmp_state.misc.get("Y")
#             if pred_sq is not None:
#                 pred_sq_np = pred_sq.detach().cpu().numpy().flatten()
#                 monitor.update_data(
#                     step=step,
#                     loss=loss.item(),
#                     pred_sq=pred_sq_np,
#                     num_violated=num_violated
#                 )
#
#     return base_state

# #Defining the FIRE MACE relaxation every few steps
# # Updated relaxation function
# def perform_fire_relaxation(sim_state, mace_model, device, dtype, max_steps=1000):
#     """
#     Performs FIRE relaxation on the current simulation state using the MACE model.
#
#     Args:
#         sim_state: The current simulation state (with positions, cell, atomic numbers).
#         mace_model: Pre-loaded MaceModel used for relaxation.
#         device: Torch device.
#         dtype: Torch dtype.
#         max_steps: Maximum steps for the FIRE optimizer.
#
#     Returns:
#         Updated simulation state with relaxed atomic positions and cell.
#     """
#     atoms_list_curr = state_to_atoms(sim_state)  # Convert current sim state to ASE atoms list
#     print(f"Starting FIRE relaxation for {len(atoms_list_curr)} structures...")
#
#     relaxed_atoms_list = []
#     for i, atoms in enumerate(atoms_list_curr):
#         print(f"Relaxing structure {i + 1}...")
#         relaxed_state = ts.optimize(
#             system=atoms,
#             model=mace_model,
#             optimizer=ts.frechet_cell_fire,
#             max_steps=max_steps,
#             autobatcher=False,
#         )
#         print(f"Structure {i + 1} relaxation complete.")
#         # Append single Atoms or extend if list
#         new_atoms = relaxed_state.to_atoms()
#         if isinstance(new_atoms, list):
#             relaxed_atoms_list.extend(new_atoms)  # Flatten if list of atoms
#         else:
#             relaxed_atoms_list.append(new_atoms)
#
#     # Convert relaxed ASE atoms back to simulation state
#     new_state = atoms_to_state(relaxed_atoms_list, device=device, dtype=dtype)
#     new_state.positions.requires_grad_(True)
#     new_state.cell.requires_grad_(True)
#
#     print("Performed FIRE relaxation with MACE.")
#     print(f"Updated positions norm: {new_state.positions.norm().item()}")
#     print(f"Updated cell tensor: {new_state.cell}")
#
#     return new_state
#
# #Function to optimize atom per atom as per the sequential constraint
# def apply_sequential_tetrahedral_constraint(
#         sim_state,
#         xrd_model,
#         device,
#         dtype,
#         max_steps_per_atom: int = 100,
#         q_threshold: float = 0.85,
#         lr: float = 0.01,
#         freeze_strategy: str = "soft",
#         verbose: bool = True
# ):
#     """
#     Apply sequential tetrahedral optimization to all systems in sim_state.
#     """
#     from ase.data import chemical_symbols
#
#     print("\n" + "=" * 60)
#     print("APPLYING SEQUENTIAL TETRAHEDRAL CONSTRAINT")
#     print("=" * 60)
#
#     # Handle n_systems attribute - check if it exists, otherwise assume single system
#     n_systems = getattr(sim_state, 'n_systems', 1)
#
#     # Detach positions to avoid gradient issues
#     positions_updated = sim_state.positions.detach().clone()
#
#     # Apply sequential optimization to each system
#     for b in range(n_systems):
#         if verbose:
#             print(f"\n--- System {b + 1}/{n_systems} ---")
#
#         # Handle single vs multi-system cases
#         if n_systems == 1:
#             batch_mask = torch.ones(len(sim_state.positions), dtype=torch.bool, device=device)
#             pos_b = positions_updated
#             cell_b = sim_state.cell[0] if sim_state.cell.dim() > 2 else sim_state.cell
#             atomic_numbers = sim_state.atomic_numbers.detach().cpu().numpy()
#         else:
#             batch_mask = sim_state.system_idx == b
#             pos_b = positions_updated[batch_mask]
#             cell_b = sim_state.cell[b] if sim_state.cell.dim() > 2 else sim_state.cell
#             atomic_numbers = sim_state.atomic_numbers[batch_mask].detach().cpu().numpy()
#
#         # Get symbols for this system
#         symbols_b = [chemical_symbols[z] for z in atomic_numbers]
#
#         # Apply sequential optimization
#         pos_optimized = xrd_model.sequential_tetrahedral_optimization(
#             pos=pos_b,
#             cell=cell_b,
#             symbols=symbols_b,
#             max_steps_per_atom=max_steps_per_atom,
#             q_threshold=q_threshold,
#             lr=lr,
#             freeze_strategy=freeze_strategy,
#             verbose=verbose,
#         )
#
#         # Update in the cloned tensor
#         if n_systems == 1:
#             positions_updated = pos_optimized
#         else:
#             positions_updated[batch_mask] = pos_optimized
#
#     # Replace the entire positions tensor (not in-place)
#     sim_state.positions = positions_updated.requires_grad_(True)
#
#     print("=" * 60)
#     print("SEQUENTIAL CONSTRAINT COMPLETE")
#     print("=" * 60 + "\n")
#
#     return sim_state
#
#
# def compute_mace_energy(sim_state, mace_calculator, device, dtype):
#     """
#     Compute MACE potential energy for structures in sim_state.
#     Returns a differentiable torch tensor.
#
#     Args:
#         sim_state: SimState object with positions, cell, atomic_numbers
#         mace_calculator: MACE calculator instance
#         device: torch device
#         dtype: torch dtype
#
#     Returns:
#         energy_tensor: Total energy in eV as torch tensor with gradients
#     """
#     from ase import Atoms
#     import torch
#
#     # Convert sim_state to ASE Atoms
#     positions = sim_state.positions.detach().cpu().numpy()
#     cell = sim_state.cell.detach().cpu().numpy()
#     atomic_numbers = sim_state.atomic_numbers.detach().cpu().numpy()
#
#     # Handle batched vs single structure
#     if cell.ndim == 3:
#         cell = cell[0]  # Take first system if batched
#
#     # Create ASE Atoms object
#     atoms = Atoms(
#         numbers=atomic_numbers,
#         positions=positions,
#         cell=cell,
#         pbc=True
#     )
#
#     # Attach MACE calculator
#     atoms.calc = mace_calculator
#
#     # Get energy and forces (forces give gradients)
#     energy = atoms.get_potential_energy()  # In eV
#     forces = atoms.get_forces()  # In eV/Å
#
#     # Convert to torch tensors
#     energy_tensor = torch.tensor(
#         energy,
#         device=device,
#         dtype=dtype,
#         requires_grad=True
#     )
#
#     # Store forces for gradient computation if needed
#     # (Cooper optimizer will handle this through autograd)
#
#     return energy_tensor



