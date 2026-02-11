from __future__ import annotations
from typing import Callable, Optional, Dict, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import json

import torch
import torch.nn as nn
import numpy as np
import wandb
import matplotlib.pyplot as plt
import cooper
from cooper.penalty_coefficients import PenaltyCoefficient

import torch_sim as ts
from torch_sim.state import DeformGradMixin, SimState
from torch_sim.io import atoms_to_state, state_to_atoms
from ase.data import chemical_symbols
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units
from ase.io import write
from mace.calculators.foundations_models import mace_mp

from torchdisorder.model.loss import AugLagLoss
from torchdisorder.common.target_rdf import TargetRDFData
from torchdisorder.engine.order_params import TorchSimOrderParameters



class ConstantPenalty(PenaltyCoefficient):
   """Constant penalty coefficient for AugmentedLagrangian."""


   def __init__(self, value: float, device='cuda'):
       super().__init__(init=torch.tensor(value, device=device))
       self.expects_constraint_features = False


   def __call__(self, constraint_features=None):
       """Return the constant penalty value."""
       return self.init


class ScalarAdaptivePenalty(PenaltyCoefficient):
    """
    Scalar adaptive penalty coefficient for Augmented Lagrangian.
    
    A single penalty value applied to all constraints that adapts based
    on overall constraint satisfaction progress.
    
    For per-constraint adaptive penalties, use AdaptivePenalty from
    constrained_optimizer module.
    
    Parameters
    ----------
    init : float
        Initial penalty value
    growth_rate : float
        Multiply penalty by this when constraints not improving
    decay_rate : float
        Multiply penalty by this when constraints improving
    max_penalty : float
        Upper bound on penalty
    min_penalty : float
        Lower bound on penalty
    patience : int
        Steps without improvement before increasing penalty
    device : str
        Device for tensors
    """
    
    def __init__(
        self,
        init: float = 10.0,
        growth_rate: float = 1.5,
        decay_rate: float = 0.95,
        max_penalty: float = 1000.0,
        min_penalty: float = 1.0,
        patience: int = 10,
        device: str = 'cuda'
    ):
        # Don't call super().__init__ to avoid the value setter issue
        # Just set the attributes we need
        self.expects_constraint_features = False
        
        self.growth_rate = growth_rate
        self.decay_rate = decay_rate
        self.max_penalty = max_penalty
        self.min_penalty = min_penalty
        self.patience = patience
        self.device = device
        self._init_value = init
        
        # Tracking state
        self._current_value = init
        self._best_violation = float('inf')
        self._steps_without_improvement = 0
    
    def __call__(self, constraint_features=None):
        """Return the current penalty value."""
        return torch.tensor(self._current_value, device=self.device)
    
    def update(self, total_violation: float):
        """
        Update penalty based on constraint violation.
        
        Call this after each optimization step with the sum of
        squared constraint violations.
        
        Parameters
        ----------
        total_violation : float
            Sum of squared constraint violations
        """
        if total_violation < self._best_violation * 0.99:  # 1% improvement threshold
            # Constraints improving - decrease penalty
            self._best_violation = total_violation
            self._steps_without_improvement = 0
            self._current_value = max(
                self.min_penalty,
                self._current_value * self.decay_rate
            )
        else:
            # Not improving
            self._steps_without_improvement += 1
            if self._steps_without_improvement >= self.patience:
                # Increase penalty
                self._current_value = min(
                    self.max_penalty,
                    self._current_value * self.growth_rate
                )
                self._steps_without_improvement = 0
    
    def reset(self, init: float = None):
        """Reset penalty to initial value."""
        if init is not None:
            self._current_value = init
        else:
            self._current_value = self._init_value
        self._best_violation = float('inf')
        self._steps_without_improvement = 0
    
    @property
    def current_penalty(self) -> float:
        """Current penalty value."""
        return self._current_value




class StructureFactorCMPWithConstraints(cooper.ConstrainedMinimizationProblem):
   """
   Constrained Minimization Problem for RDF/Structure Factor matching
   with per-atom order parameter constraints.


   Uses Cooper library for augmented Lagrangian optimization with:
   - RDF/Structure factor matching objective
   - Per-atom order parameter constraints from JSON file
   - Environment-specific constraint targets


   Updated for torch_sim   with batched neighbor lists.


   Usage:
       >>> cmp = StructureFactorCMPWithConstraints(
       ...     model=xrd_model,
       ...     base_state=state,
       ...     target_vec=rdf_data,
       ...     constraints_file='my_structure_constraints.json',
       ...     loss_fn=loss_fn,
       ...     device='cuda'
       ... )
       >>>
       >>> # In optimization loop
       >>> cmp_state = cmp.compute_cmp_state(state.positions, state.cell)
       >>> # Cooper handles the rest


   Note:
       The order parameter calculator automatically handles the new torch_sim
       neighbor list API internally. No changes needed in your optimization loop.
   """


   def __init__(
           self,
           model,
           base_state: ts.SimState,
           target_vec: TargetRDFData,
           constraints_file: Union[str, Path],
           loss_fn,
           target_kind: str = 'S_Q',
           q_bins: Optional[torch.Tensor] = None,
           device: str = 'cuda',
           penalty_config: Union[float, Dict] = 10.0,
           violation_type: str = 'soft',  # 'soft' or 'hard'
           constraint_warmup_steps: int = 0,  # Ramp constraints over N steps
   ):
       """
       Initialize constrained minimization problem.
       
       Parameters
       ----------
       penalty_config : float or dict
           If float: constant penalty value
           If dict: adaptive penalty with keys:
               - init: Initial penalty value
               - growth_rate: Multiply when constraints not improving
               - decay_rate: Multiply when constraints improving
               - max_penalty: Upper bound
               - min_penalty: Lower bound
               - patience: Steps before increasing penalty
       """
       super().__init__()


       self.model = model
       self.base_state = base_state
       self.kind = target_kind
       self.q_bins = q_bins
       self.loss_fn = loss_fn
       self.device = device
       self.violation_type = violation_type
       self.constraint_warmup_steps = constraint_warmup_steps
       
       # Select correct target based on target_kind
       # Map target_kind to the corresponding attribute in TargetRDFData
       target_map = {
           'S_Q': ('S_Q_target', 'S_Q_uncert'),
           'F_Q': ('F_q_target', 'F_q_uncert'),
           'T_r': ('T_r_target', 'T_r_uncert'),
           'g_r': ('g_r_target', 'g_r_uncert'),
           'G_r': ('G_r_target', 'G_r_uncert'),  # G(r) = 4πρr[g(r) - 1]
       }
       
       if target_kind in target_map:
           target_attr, uncert_attr = target_map[target_kind]
           self.target = getattr(target_vec, target_attr)
           self.target_uncert = getattr(target_vec, uncert_attr)
           print(f"  Using target: {target_attr} with shape {self.target.shape}")
       else:
           # Fallback to F_q for backward compatibility
           print(f"  Warning: Unknown target_kind '{target_kind}', defaulting to F_q")
           self.target = target_vec.F_q_target
           self.target_uncert = target_vec.F_q_uncert


       # Load constraints from JSON
       self.constraints_data = self._load_constraints(constraints_file)


       # Initialize order parameter calculator
       cutoff = self.constraints_data['cutoff']
       self.op_calc = TorchSimOrderParameters(cutoff=cutoff, device=str(device))


       # Detect placeholder region (where uncertainty is negligible)
       self.placeholder_mask = (self.target_uncert < 1e-6).to(device)
       n_placeholder = self.placeholder_mask.sum().item()
       print(f"  Found {n_placeholder} placeholder points (uncertainty < 1e-6)")


       # Set up constraints based on JSON file
       self._setup_constraints(penalty_config)


       print(f"\nInitialized StructureFactorCMPWithConstraints:")
       print(f"  Constrained atoms: {len(self.constraints_data['atom_constraints'])}")
       print(f"  Order parameters: {', '.join(self.constraints_data['metadata']['order_parameter_types'])}")
       print(f"  Total Cooper constraints: {len(self.constraint_dict)}")
       if self.constraint_warmup_steps > 0:
           print(f"  Constraint warmup: {self.constraint_warmup_steps} steps")


   def _load_constraints(self, constraints_file: Union[str, Path]) -> Dict:
       """Load constraint specifications from JSON file."""
       with open(constraints_file, 'r') as f:
           constraints = json.load(f)
       return constraints


   def _setup_constraints(self, penalty_config: Union[float, Dict]):
       """
       Set up Cooper constraint objects for all atom-constraint pairs.

       Parameters
       ----------
       penalty_config : float or dict
           If float: constant penalty value
           If dict: adaptive penalty configuration

       Strategy: Group constraints by order parameter type to reduce
       the number of Cooper constraint objects.
       """
       # Create penalty coefficient based on config type
       if isinstance(penalty_config, dict):
           self.penalty_coeff = ScalarAdaptivePenalty(
               init=penalty_config.get('init', 10.0),
               growth_rate=penalty_config.get('growth_rate', 1.5),
               decay_rate=penalty_config.get('decay_rate', 0.95),
               max_penalty=penalty_config.get('max_penalty', 1000.0),
               min_penalty=penalty_config.get('min_penalty', 1.0),
               patience=penalty_config.get('patience', 10),
               device=self.device
           )
           print(f"  Using ScalarAdaptivePenalty: init={penalty_config.get('init', 10.0)}, "
                 f"growth={penalty_config.get('growth_rate', 1.5)}, "
                 f"decay={penalty_config.get('decay_rate', 0.95)}")
       else:
           self.penalty_coeff = ConstantPenalty(float(penalty_config), device=self.device)
           print(f"  Using ConstantPenalty: {penalty_config}")
       
       # Group atoms by order parameter type
       op_to_atoms = defaultdict(list)


       for atom_idx_str, atom_constraint in self.constraints_data['atom_constraints'].items():
           atom_idx = int(atom_idx_str)
           for op_name in atom_constraint['order_parameters'].keys():
               op_to_atoms[op_name].append(atom_idx)


       # Create one Cooper constraint per order parameter type
       self.constraint_dict = {}


       for op_name, atom_list in op_to_atoms.items():
           n_atoms = len(atom_list)


           # Create multiplier for all atoms with this order parameter
           multiplier = cooper.multipliers.DenseMultiplier(
               num_constraints=n_atoms,
               device=self.device
           )


           # Create constraint using shared penalty coefficient
           constraint = cooper.Constraint(
               multiplier=multiplier,
               constraint_type=cooper.ConstraintType.INEQUALITY,
               formulation_type=cooper.formulations.AugmentedLagrangian,
               penalty_coefficient=self.penalty_coeff,
           )


           self.constraint_dict[op_name] = {
               'constraint': constraint,
               'atom_indices': atom_list
           }


           print(f"  Created constraint for {op_name}: {n_atoms} atoms")

   def _compute_violations(self, op_results: Dict[str, torch.Tensor], constrained_atom_indices: torch.Tensor, step: Optional[int] = None):
       violations = {}

       # Compute warmup factor: ramps from 0→1 over warmup_steps
       if self.constraint_warmup_steps > 0 and step is not None:
           warmup_factor = min(1.0, step / self.constraint_warmup_steps)
       else:
           warmup_factor = 1.0

       for op_name, constraint_info in self.constraint_dict.items():
           atom_list = constraint_info['atom_indices']
           op_v = []

           for atom_idx in atom_list:
               atom_constraint = self.constraints_data['atom_constraints'][str(atom_idx)]
               op_params = atom_constraint['order_parameters'][op_name]

               pos = (constrained_atom_indices == atom_idx).nonzero(as_tuple=True)[0]
               if pos.numel() == 0:
                   # keep this as a tensor on device (still no grad, but rare)
                   op_v.append(torch.zeros((), device=self.device))
                   continue

               pos = pos[0]  # tensor index (no .item())
               value = op_results[op_name][pos]  # tensor value (no .item())

               # Guard: if the order parameter is NaN (e.g., atom had 0 neighbors),
               # skip this constraint — use zero violation (no gradient contribution).
               if not torch.isfinite(value):
                   op_v.append(torch.zeros((), device=self.device))
                   continue

               weight = torch.tensor(op_params.get('weight', 1.0), device=self.device, dtype=value.dtype)

               if 'min' in op_params and 'max' in op_params:
                   minv = torch.tensor(op_params['min'], device=self.device, dtype=value.dtype)
                   maxv = torch.tensor(op_params['max'], device=self.device, dtype=value.dtype)

                   # "soft" box violation: relu(min - x) + relu(x - max)
                   v = torch.relu(minv - value) + torch.relu(value - maxv)

                   if self.violation_type == 'hard':
                       v = torch.relu(minv - value) + torch.relu(value - maxv)

               else:
                   target = torch.tensor(op_params['target'], device=self.device, dtype=value.dtype)
                   tol = torch.tensor(op_params.get('tolerance', 0.1), device=self.device, dtype=value.dtype)
                   v = torch.relu(torch.abs(value - target) - tol)

               op_v.append(weight * v)

           violation_tensor = torch.stack(op_v).to(dtype=torch.float32)
           # Apply warmup scaling
           if warmup_factor < 1.0:
               violation_tensor = violation_tensor * warmup_factor
           violations[op_name] = violation_tensor

       return violations

   def compute_cmp_state(
           self,
           positions: torch.Tensor,
           cell: torch.Tensor,
           step: Optional[int] = None
   ) -> cooper.CMPState:
       """
       Compute the constrained minimization problem state.


       Args:
           positions: Atomic positions
           cell: Unit cell
           step: Current optimization step (for logging)


       Returns:
           Cooper CMPState with loss and constraint violations
       """
       # Update base state
       self.base_state.positions = positions
       self.base_state.cell = cell


       # Forward pass through model
       out = self.model(self.base_state)


       # Get predicted S(Q) or T(r)
       pred_sq = out[self.kind].squeeze()


       # Clamp prediction to experimental value in placeholder region
       pred_sq_clamped = pred_sq.clone()
       pred_sq_clamped[self.placeholder_mask] = self.target[self.placeholder_mask]


       # Replace in output dict
       out[self.kind] = pred_sq_clamped.unsqueeze(0) if pred_sq_clamped.dim() == 1 else pred_sq_clamped


       # Compute chi-squared loss
       loss_dict = self.loss_fn(out)
       # Support both old and new loss key formats
       chi2_loss = loss_dict.get("total_loss", 
                   loss_dict.get("chi2_loss",
                   loss_dict.get("chi2_scatt", 
                   loss_dict.get("chi2_corr", 
                   loss_dict.get("loss")))))


       # ====================================================================
       # Compute order parameter constraints
       # ====================================================================


       # Get element filter and atom indices
       element_filter = self.constraints_data['element_filter']
       atom_mask = torch.zeros(len(self.base_state.atomic_numbers), dtype=torch.bool, device=self.device)
       for Z in element_filter:
           atom_mask |= (self.base_state.atomic_numbers == Z)


       constrained_atom_indices = torch.where(atom_mask)[0]


       # Determine which order parameters to compute
       all_op_types = set()
       for atom_constraint in self.constraints_data['atom_constraints'].values():
           all_op_types.update(atom_constraint['order_parameters'].keys())

       # Compute all needed order parameters at once
       op_results = self.op_calc(
           self.base_state,
           constrained_atom_indices,
           list(all_op_types),
           element_filter=element_filter
       )


       # Compute violations for each constraint group
       violations = self._compute_violations(op_results, constrained_atom_indices, step=step)


       # ====================================================================
       # Create Cooper constraint states
       # ====================================================================


       observed_constraints = {}


       for op_name, violation_tensor in violations.items():
           constraint = self.constraint_dict[op_name]['constraint']
           constraint_state = cooper.ConstraintState(violation=violation_tensor)
           observed_constraints[constraint] = constraint_state


       # ====================================================================
       # Create misc dict with diagnostics
       # ====================================================================


       # Compute violation statistics
       all_violations = torch.cat([v for v in violations.values()])
       n_violations = (all_violations > 1e-4).sum().item()
       avg_violation = all_violations.mean().item()
       max_violation = all_violations.max().item()


       misc = dict(
           Q=self.q_bins,
           Y=pred_sq_clamped,
           loss=chi2_loss,
           chi2_loss=chi2_loss,
           kind=self.kind,
           n_violations=n_violations,
           avg_violation=avg_violation,
           max_violation=max_violation,
           op_results={k: v.detach() for k, v in op_results.items()},
           step=step
       )


       # Add per-environment statistics if step is provided
       if step is not None and step % 50 == 0:
           env_stats = self._get_environment_statistics(op_results, constrained_atom_indices)
           misc['environment_stats'] = env_stats


       return cooper.CMPState(
           loss=chi2_loss,
           observed_constraints=observed_constraints,
           misc=misc
       )


   def _get_environment_statistics(
           self,
           op_results: Dict[str, torch.Tensor],
           constrained_atom_indices: torch.Tensor
   ) -> Dict:
       """
       Get detailed statistics about constraint satisfaction by environment type.
       """
       stats_by_env = defaultdict(lambda: {
           'total': 0,
           'satisfied': 0,
           'violations': []
       })


       for atom_idx in constrained_atom_indices.tolist():
           atom_idx_str = str(atom_idx)
           if atom_idx_str not in self.constraints_data['atom_constraints']:
               continue


           atom_constraint = self.constraints_data['atom_constraints'][atom_idx_str]
           env_type = atom_constraint.get('environment_type', 'unknown')


           stats_by_env[env_type]['total'] += 1


           # Check if all constraints satisfied for this atom
           pos = (constrained_atom_indices == atom_idx).nonzero(as_tuple=True)[0][0].item()


           has_violation = False
           for op_name, op_params in atom_constraint['order_parameters'].items():
               if op_name not in op_results:
                   continue


               value = op_results[op_name][pos].item()


               # Check violation
               if 'min' in op_params and 'max' in op_params:
                   if value < op_params['min'] or value > op_params['max']:
                       has_violation = True
                       violation = min(op_params['min'] - value, value - op_params['max'])
                       stats_by_env[env_type]['violations'].append(abs(violation))
               else:
                   target = op_params['target']
                   tolerance = op_params.get('tolerance', 0.1)
                   if abs(value - target) > tolerance:
                       has_violation = True
                       stats_by_env[env_type]['violations'].append(abs(value - target))


           if not has_violation:
               stats_by_env[env_type]['satisfied'] += 1


       # Compute satisfaction rates
       results = {}
       for env_type, stats in stats_by_env.items():
           satisfied_pct = 100.0 * stats['satisfied'] / stats['total'] if stats['total'] > 0 else 0
           avg_viol = sum(stats['violations']) / len(stats['violations']) if stats['violations'] else 0


           results[env_type] = {
               'total_atoms': stats['total'],
               'satisfied_atoms': stats['satisfied'],
               'satisfaction_rate': satisfied_pct,
               'avg_violation': avg_viol,
               'max_violation': max(stats['violations']) if stats['violations'] else 0
           }


       return results


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

