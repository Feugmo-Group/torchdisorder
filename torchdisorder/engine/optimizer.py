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

    def __init__(self, value: float):
        # Convert float to torch tensor
        super().__init__(init=torch.tensor(value))
        self.expects_constraint_features = False

    def __call__(self, constraint_features=None):
        """Return the constant penalty value."""
        return self.init  # Returns the tensor


class StructureFactorCMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self, model, base_state, target_vec, target_kind, q_bins, loss_fn,
                 q_threshold=0.8, device='cuda', penalty_rho=1.0):
        super().__init__()
        self.model = model
        self.base_state = base_state
        self.target = target_vec
        self.kind = target_kind
        self.q_bins = q_bins
        self.loss_fn = loss_fn
        self.q_threshold = q_threshold

        # Count Ge atoms for per-atom constraints
        symbols = [chemical_symbols[int(z)] for z in base_state.atomic_numbers.cpu()]
        num_si = sum(1 for s in symbols if s == 'Ge')
        print(num_si)

        # CREATE MULTIPLIER (λ tensor)
        multiplier = cooper.multipliers.DenseMultiplier(
            num_constraints=num_si,
            device=device
        )
        penalty_coeff = ConstantPenalty(penalty_rho)

        # CREATE CONSTRAINT
        self.q_tet_constraint = cooper.Constraint(
            multiplier=multiplier,
            constraint_type=cooper.ConstraintType.INEQUALITY,
            formulation_type=cooper.formulations.AugmentedLagrangian,
            penalty_coefficient=penalty_coeff,
        )

    def compute_cmp_state(self, positions, cell):
        self.base_state.positions = positions
        self.base_state.cell = cell
        out = self.model(self.base_state)

        # NO MASKING - Use predicted S(Q) directly
        pred_sq = out[self.kind].squeeze()

        # Call loss_fn with unmodified output
        loss_dict = self.loss_fn(out)
        loss = loss_dict["chi2_scatt"]

        # COMPUTE CONSTRAINT VIOLATION
        q_tet_per_atom = out['q_tet']
        violation = self.q_threshold - q_tet_per_atom

        q_tet_state = cooper.ConstraintState(violation=violation)
        observed_constraints = {self.q_tet_constraint: q_tet_state}

        misc = dict(
            Q=self.q_bins.detach().cpu(),
            Y=pred_sq.detach().cpu(),
            loss=loss.detach().cpu(),
            kind=self.kind,
            qtet=q_tet_per_atom.detach().cpu()
        )

        return cooper.CMPState(loss=loss, observed_constraints=observed_constraints, misc=misc)



#Optimizer for the cooper constraint
#contains init and update functions
#helps for iterative gradient updates

#Cooper optimizer for Adam
def cooper_optimizer(
        cmp_problem: cooper.ConstrainedMinimizationProblem,
        base_state: SimState,
        primal_lr: float = 1e-4,
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
        cmp_problem.dual_parameters(),  # Gets λ automatically!
        lr=dual_lr,
        maximize=True  # CRITICAL: dual is maximization
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
        loss = roll_out.cmp_state.loss
        violations = list(roll_out.cmp_state.observed_constraints.values())[0].violation

        if verbose and step % 100 == 0:
            avg_violation = violations.mean().item()
            max_violation = violations.max().item()
            print(f"Step {step}: Loss={loss.item():.6f}, "
                  f"Avg q_tet violation={avg_violation:.6f}, "
                  f"Max q_tet violation={max_violation:.6f}")

    return base_state







# #Cooper optimizer with LBFG
# def cooper_optimizer(
#         cmp_problem: cooper.ConstrainedMinimizationProblem,
#         base_state: SimState,
#         primal_lr: float = 1.0,  #LFBG
#         dual_lr: float = 1e-2,
#         max_steps: int = 1000,
#         optimize_cell: bool = False,
#         verbose: bool = True,
# ):
#     """Proper Cooper optimizer using SimultaneousOptimizer and LBFG."""
#
#     # Setup parameters
#     base_state.positions.requires_grad_(True)
#     primal_params = [base_state.positions]
#
#     if optimize_cell:
#         base_state.cell.requires_grad_(True)
#         primal_params.append(base_state.cell)
#
#     # CREATE PRIMAL OPTIMIZER for LBFG
#     primal_optimizer = torch.optim.LBFGS(
#         primal_params,
#         lr=primal_lr,
#         max_iter=20,  # Max line search iterations per step [web:247]
#         history_size=10,  # How many previous gradients to store [web:247]
#         line_search_fn="strong_wolfe"  # Better line search [web:247]
#     )
#
#     # CREATE DUAL OPTIMIZER (for λ)
#     dual_optimizer = torch.optim.SGD(
#         cmp_problem.dual_parameters(),  # Gets λ automatically!
#         lr=dual_lr,
#         maximize=True  # CRITICAL: dual is maximization
#     )
#
#     # CREATE COOPER OPTIMIZER
#     cooper_opt = cooper.optim.SimultaneousOptimizer(
#         cmp=cmp_problem,
#         primal_optimizers=primal_optimizer,
#         dual_optimizers=dual_optimizer
#     )
#
#     # TRAINING LOOP
#     # ↓↓↓ MODIFIED TRAINING LOOP for L-BFGS
#     for step in range(max_steps):
#         # L-BFGS needs a closure that re-evaluates the model
#         def closure():
#             # Zero gradients
#             primal_optimizer.zero_grad()
#             dual_optimizer.zero_grad()
#
#             # Compute CMP state
#             cmp_state = cmp_problem.compute_cmp_state(
#                 positions=base_state.positions,
#                 cell=base_state.cell
#             )
#
#             # Get Lagrangian (loss + penalty)
#             lagrangian = cooper_opt.compute_lagrangian(cmp_state)
#
#             # Backward pass
#             lagrangian.backward()
#
#             return lagrangian
#
#         # Step with closure
#         primal_optimizer.step(closure)
#
#         # Dual step (standard, no closure needed)
#         dual_optimizer.step()
#
#         # Get current state for logging
#         with torch.no_grad():
#             cmp_state = cmp_problem.compute_cmp_state(
#                 positions=base_state.positions,
#                 cell=base_state.cell
#             )
#             loss = cmp_state.loss
#             violations = list(cmp_state.observed_constraints.values())[0].violation
#
#         if verbose and step % 100 == 0:
#             avg_violation = violations.mean().item()
#             max_violation = violations.max().item()
#             print(f"Step {step}: Loss={loss.item():.6f}, "
#                   f"Avg q_tet violation={avg_violation:.6f}, "
#                   f"Max q_tet violation={max_violation:.6f}")
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



