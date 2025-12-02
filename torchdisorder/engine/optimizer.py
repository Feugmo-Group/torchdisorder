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

import torch
import numpy as np
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units
from ase.io import write
from mace.calculators.foundations_models import mace_mp






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
                 q_threshold=0.7, device='cuda', penalty_rho=40.0):
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
        num_central = sum(1 for s in symbols if s == 'Fe') #Change depending on whatever the central atom is
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




#Defining the FIRE MACE relaxation every few steps
# Updated relaxation function
def perform_fire_relaxation(sim_state, mace_model, device, dtype, max_steps=50):
    """
    Performs FIRE relaxation ONLY on atomic positions (cell frozen).
    """
    atoms_list_curr = state_to_atoms(sim_state)
    print(f"Starting FIRE relaxation for {len(atoms_list_curr)} structures...")

    relaxed_atoms_list = []
    for i, atoms in enumerate(atoms_list_curr):
        print(f"Relaxing structure {i + 1}...")
        relaxed_state = ts.optimize(
            system=atoms,
            model=mace_model,
            optimizer=ts.optimizers.fire,  # ← Only atoms, not cell
            max_steps=max_steps,
            autobatcher=False,
            pbar=True,
        )
        print(f"Structure {i + 1} relaxation complete.")
        new_atoms = relaxed_state.to_atoms()
        if isinstance(new_atoms, list):
            relaxed_atoms_list.extend(new_atoms)
        else:
            relaxed_atoms_list.append(new_atoms)

    new_state = atoms_to_state(relaxed_atoms_list, device=device, dtype=dtype)
    new_state.positions.requires_grad_(True)
    new_state.cell.requires_grad_(True)

    print("Performed FIRE relaxation with MACE.")
    print(f"Updated positions norm: {new_state.positions.norm().item()}")

    return new_state


"""
Melt-quench simulation using MACE for escaping local minima.
"""

def perform_melt_quench(
        sim_state,
        mace_model,
        device,
        dtype,
        melt_temp=2000,
        quench_temp=300,
        melt_steps=1000,
        quench_steps=2000,
        timestep=1.0,
        save_prefix="melt_quench"
):
    """
    Performs MACE-driven melt-quench simulation to escape local minima.

    Args:
        sim_state: Current SimState with positions and cell
        mace_model: MaceModel instance from torch_sim
        device: torch device
        dtype: torch dtype
        melt_temp: Melting temperature in Kelvin (default: 2000)
        quench_temp: Final quenching temperature in Kelvin (default: 300)
        melt_steps: Number of MD steps at melt temperature (default: 1000)
        quench_steps: Number of steps for linear quench (default: 2000)
        timestep: MD timestep in femtoseconds (default: 1.0)
        save_prefix: Prefix for saved structure files

    Returns:
        new_state: SimState with melt-quenched structure
    """
    print(f"\n{'=' * 70}")
    print(f"Starting melt-quench simulation...")
    print(f"{'=' * 70}")

    # Convert to ASE Atoms
    atoms_list = state_to_atoms(sim_state)

    quenched_atoms_list = []

    for i, atoms in enumerate(atoms_list):
        print(f"\nProcessing structure {i + 1}/{len(atoms_list)}...")

        # Save pre-melt structure
        write(f'{save_prefix}_pre_melt_{i}.xyz', atoms)

        # Create ASE-compatible MACE calculator
        ase_calc = mace_mp(
            model="small",
            device=str(device),
            default_dtype='float32'
        )

        atoms.calc = ase_calc

        # ===== MELTING PHASE =====
        print(f"  Phase 1: Melting at {melt_temp} K for {melt_steps} steps...")

        # Initialize velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=melt_temp)

        # Langevin dynamics at high temperature
        dyn = Langevin(
            atoms,
            timestep=timestep * units.fs,
            temperature_K=melt_temp,
            friction=0.01
        )

        # Run melting
        for step in range(melt_steps):
            dyn.run(1)
            if (step + 1) % 200 == 0:
                temp = atoms.get_temperature()
                energy = atoms.get_potential_energy()
                print(f"    Melt step {step + 1}/{melt_steps}, T={temp:.1f} K, E={energy:.2f} eV")

        # ===== QUENCHING PHASE =====
        print(f"  Phase 2: Quenching to {quench_temp} K over {quench_steps} steps...")

        # Linear temperature ramp
        temps = np.linspace(melt_temp, quench_temp, quench_steps)

        for step, temp in enumerate(temps):
            dyn = Langevin(
                atoms,
                timestep=timestep * units.fs,
                temperature_K=temp,
                friction=0.01
            )
            dyn.run(1)

            if (step + 1) % 400 == 0:
                energy = atoms.get_potential_energy()
                print(f"    Quench step {step + 1}/{quench_steps}, T={temp:.1f} K, E={energy:.2f} eV")

        print(f"  Phase 3: Equilibration at {quench_temp} K for 500 steps...")
        dyn = Langevin(
            atoms,
            timestep=timestep * units.fs,
            temperature_K=quench_temp,
            friction=0.01
        )
        dyn.run(500)

        final_temp = atoms.get_temperature()
        final_energy = atoms.get_potential_energy()
        print(f"    Final: T={final_temp:.1f} K, E={final_energy:.2f} eV")

        # Save quenched structure
        write(f'{save_prefix}_post_quench_{i}.xyz', atoms)
        write(f'{save_prefix}_post_quench_{i}.XDATCAR', atoms)

        # Remove calculator before returning
        atoms.calc = None

        quenched_atoms_list.append(atoms)
        print(f"  ✓ Structure {i + 1} melt-quench complete!")

    # Convert back to SimState
    new_state = atoms_to_state(quenched_atoms_list, device=device, dtype=dtype)
    new_state.positions.requires_grad_(True)
    new_state.cell.requires_grad_(True)

    print(f"\n✓ MELT-QUENCH COMPLETE")
    print(f"  Structures saved with prefix: {save_prefix}")
    print(f"  Updated positions norm: {new_state.positions.norm().item():.4f}")
    print(f"{'=' * 70}\n")

    return new_state
