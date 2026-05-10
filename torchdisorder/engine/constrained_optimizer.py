"""
engine/constrained_optimizer.py – Environment-Based Constraint Optimization
============================================================================

This module implements an improved constrained optimization framework that:

1. **Groups constraints by local environment** (PS₄³⁻, P₂S₇⁴⁻, PS₃⁻, etc.)
   rather than by order parameter type. This provides better physics because
   atoms in the same environment should satisfy similar constraints together.

2. **Uses adaptive penalty coefficients** that increase exponentially for
   persistently violated constraints while keeping penalties low for satisfied ones.

3. **Supports environment-specific constraint weighting** to prioritize certain
   structural motifs (e.g., maintaining tetrahedral PS₄ units).

Theory:
    The augmented Lagrangian method solves:
    
    min_x L(x, λ, ρ) = f(x) + Σ_k [λ_k^T g_k(x) + (ρ_k/2)||g_k(x)||²]
    
    where:
    - f(x) = χ²(S_pred, S_exp) is the structure factor matching loss
    - g_k(x) are constraint violations grouped by environment
    - λ_k are Lagrange multipliers (learned)
    - ρ_k are penalty coefficients (adaptive)

Usage:
    >>> cmp = EnvironmentConstrainedOptimizer(
    ...     model=xrd_model,
    ...     state=sim_state,
    ...     target=rdf_data,
    ...     constraints_file='constraints.json',
    ...     device='cuda',
    ... )
    >>> cmp_state = cmp.compute_cmp_state(positions, cell, step=100)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import json
import math

import torch
import torch.nn as nn
import cooper
from cooper.penalty_coefficients import PenaltyCoefficient

import torch_sim as ts
from torch_sim.io import state_to_atoms

from torchdisorder.common.target_rdf import TargetRDFData
from torchdisorder.engine.order_params import TorchSimOrderParameters


# =============================================================================
# Adaptive Penalty Coefficient
# =============================================================================

class AdaptivePenalty(PenaltyCoefficient):
    """
    Adaptive penalty coefficient that grows for persistently violated constraints.
    
    Each constraint (environment group) has its own penalty that:
    - Starts at `init` value
    - Increases by factor `growth_rate` when violation persists
    - Caps at `max_penalty` to prevent numerical instability
    - Decreases by `decay_rate` when constraint is satisfied
    
    Args:
        init: Initial penalty value (default: 10.0)
        growth_rate: Factor to increase penalty when violated (default: 1.5)
        decay_rate: Factor to decrease penalty when satisfied (default: 0.95)
        max_penalty: Maximum penalty cap (default: 1000.0)
        min_penalty: Minimum penalty floor (default: 1.0)
        patience: Steps before increasing penalty (default: 10)
    """
    
    def __init__(
        self,
        num_constraints: int,
        device: str = 'cuda',
        init: float = 10.0,
        growth_rate: float = 1.5,
        decay_rate: float = 0.95,
        max_penalty: float = 1000.0,
        min_penalty: float = 1.0,
        patience: int = 10,
    ):
        # Initialize with per-constraint penalties
        super().__init__(
            init=torch.full((num_constraints,), init, device=device)
        )
        self.expects_constraint_features = True
        
        self.growth_rate = growth_rate
        self.decay_rate = decay_rate
        self.max_penalty = max_penalty
        self.min_penalty = min_penalty
        self.patience = patience
        
        # Track violation history for adaptive updates
        self.device = device
        self.violation_count = torch.zeros(num_constraints, device=device, dtype=torch.long)
        self.satisfied_count = torch.zeros(num_constraints, device=device, dtype=torch.long)
        
    def __call__(self, constraint_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return current penalty values."""
        return self.init
    
    def update(self, violations: torch.Tensor, threshold: float = 1e-4):
        """
        Update penalties based on current violations.
        
        Args:
            violations: (n_constraints,) tensor of violation magnitudes
            threshold: Violation below this is considered satisfied
        """
        with torch.no_grad():
            violated = violations > threshold
            satisfied = ~violated
            
            # Update counts
            self.violation_count[violated] += 1
            self.violation_count[satisfied] = 0
            self.satisfied_count[satisfied] += 1
            self.satisfied_count[violated] = 0
            
            # Increase penalties for persistent violations
            persistent_violators = self.violation_count >= self.patience
            self.init[persistent_violators] = torch.clamp(
                self.init[persistent_violators] * self.growth_rate,
                max=self.max_penalty
            )
            self.violation_count[persistent_violators] = 0
            
            # Decrease penalties for satisfied constraints
            long_satisfied = self.satisfied_count >= self.patience
            self.init[long_satisfied] = torch.clamp(
                self.init[long_satisfied] * self.decay_rate,
                min=self.min_penalty
            )
            self.satisfied_count[long_satisfied] = 0


class ConstantPenalty(PenaltyCoefficient):
    """Simple constant penalty (backward compatibility)."""
    
    def __init__(self, value: float, device: str = 'cuda'):
        super().__init__(init=torch.tensor(value, device=device))
        self.expects_constraint_features = False
    
    def __call__(self, constraint_features=None) -> torch.Tensor:
        return self.init


# =============================================================================
# Environment Constraint Data Structure
# =============================================================================

@dataclass
class EnvironmentConstraint:
    """
    Constraint specification for a local atomic environment.
    
    Attributes:
        env_type: Environment name (e.g., 'PS4_tet', 'P2S7_bridge')
        atom_indices: Atom indices belonging to this environment
        order_params: Dict of order parameter constraints
            Each entry: {op_name: {'target': float, 'tolerance': float, 'weight': float}}
            or: {op_name: {'min': float, 'max': float, 'weight': float}}
        priority: Constraint priority (higher = more important)
    """
    env_type: str
    atom_indices: List[int]
    order_params: Dict[str, Dict[str, float]]
    priority: float = 1.0
    
    @property
    def num_atoms(self) -> int:
        return len(self.atom_indices)
    
    @property
    def op_names(self) -> List[str]:
        return list(self.order_params.keys())


@dataclass
class ConstraintState:
    """Current state of constraints during optimization."""
    violations: Dict[str, torch.Tensor]  # env_type -> violation vector
    penalties: Dict[str, torch.Tensor]   # env_type -> penalty vector
    satisfaction_rates: Dict[str, float] # env_type -> % satisfied
    total_violation: float
    num_violated: int
    num_total: int


# =============================================================================
# Environment-Based Constrained Optimizer
# =============================================================================

class EnvironmentConstrainedOptimizer(cooper.ConstrainedMinimizationProblem):
    """
    Constrained optimization with environment-based constraint grouping.

    Supports two initialization modes detected automatically at runtime:

    **from_cif mode** (atom indices in JSON match P atoms in structure):
        Constraints are applied to specific P atoms as recorded by lps_generator.py.

    **random_icp mode** (atom indices in JSON don't match P atoms):
        The JSON's per-atom index table is ignored. Instead, the target environment
        fractions from ``global_constraints.p_environment_distribution`` are used to
        distribute environment types proportionally across all P atoms in the structure.
        Order parameter targets per environment type are taken from the first occurrence
        of each env type recorded in ``atom_constraints``.

    Example environments for Li-P-S glass:
        - PS4_isolated: Tetrahedral PS₄³⁻ (P surrounded by 4 S)
        - P2S7_bridge: Bridging P in P₂S₇⁴⁻
        - PS3_terminal: Terminal PS₃⁻ (3-coordinated P)

    Args:
        model: XRD model for computing spectra
        base_state: Initial SimState
        target: Target RDF/spectrum data
        constraints_file: Path to JSON constraints file
        loss_fn: Loss function for spectrum matching
        device: Computation device
        penalty_config: Adaptive penalty configuration dict
        warmup_steps: Steps before applying full constraint strength
    """
    
    # Maps the lps_generator label strings (used in global_constraints fractions)
    # to the short env_type keys used in atom_constraints entries.
    _LABEL_TO_ENV: Dict[str, str] = {
        'PS4^3-':              'P4',
        'P2S7^4- (dimer)':    'Pa',
        'P2S6^4- (dumbbell)': 'P2',
        'PS3^-':               'P3',
    }
    # Atomic number of phosphorus — used to locate P atoms in SimState.
    _P_ATOMIC_NUMBER: int = 15

    def __init__(
        self,
        model: nn.Module,
        base_state: ts.SimState,
        target: TargetRDFData,
        constraints_file: Union[str, Path],
        loss_fn: nn.Module,
        target_kind: str = 'S_Q',
        q_bins: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        penalty_config: Optional[Dict] = None,
        warmup_steps: int = 100,
        use_adaptive_penalty: bool = True,
    ):
        super().__init__()

        self.model = model
        self.base_state = base_state
        self.target = target.F_q_target if hasattr(target, 'F_q_target') else target.S_Q_target
        self.target_uncert = target.F_q_uncert if hasattr(target, 'F_q_uncert') else target.S_Q_uncert
        self.kind = target_kind
        self.q_bins = q_bins
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.warmup_steps = warmup_steps
        self.use_adaptive_penalty = use_adaptive_penalty

        # Default penalty config
        self.penalty_config = penalty_config or {
            'init': 10.0,
            'growth_rate': 1.5,
            'decay_rate': 0.95,
            'max_penalty': 1000.0,
            'min_penalty': 1.0,
            'patience': 10,
        }

        # Load and parse constraints — auto-detects from_cif vs random_icp mode
        self.constraints_data = self._load_constraints(constraints_file)
        self.env_constraints = self._parse_environment_constraints(base_state.atomic_numbers)

        # Initialize order parameter calculator
        cutoff = self.constraints_data.get('cutoff', 3.5)
        self.op_calc = TorchSimOrderParameters(cutoff=cutoff, device=str(device))

        # Set up Cooper constraints grouped by environment
        self._setup_environment_constraints()

        self._print_summary()
    
    def _load_constraints(self, path: Union[str, Path]) -> Dict:
        """Load constraints from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _parse_environment_constraints(
        self,
        atomic_numbers: torch.Tensor,
    ) -> Dict[str, EnvironmentConstraint]:
        """
        Parse constraints into environment-grouped structure.

        Auto-detects the initialization mode:
        - **from_cif**: JSON atom indices all map to P atoms in the structure →
          use indices as-is (per-atom assignment from lps_generator).
        - **random_icp**: Any JSON atom index maps to a non-P atom →
          redistribute environment types across all P atoms by target fraction.

        Supports v6 (``"environment"`` key) and v5 (``"environment_type"`` key).
        """
        atom_constraints = self.constraints_data.get('atom_constraints', {})
        priorities = self.constraints_data.get('environment_priorities', {})

        # --- detect mode ---------------------------------------------------
        p_index_set = set(
            int(i) for i, z in enumerate(atomic_numbers.tolist())
            if int(z) == self._P_ATOMIC_NUMBER
        )
        json_indices = [int(k) for k in atom_constraints.keys()]
        # random_icp: any JSON index is out-of-range or not a P atom
        is_random = bool(json_indices) and any(
            idx not in p_index_set for idx in json_indices
        )

        if is_random:
            print("[ConstrainedOptimizer] random_icp mode detected — "
                  "redistributing environments by target fraction.")
            return self._parse_from_global_fractions(p_index_set, atom_constraints, priorities)

        # --- from_cif: use JSON indices directly ---------------------------
        print("[ConstrainedOptimizer] from_cif mode — using per-atom JSON assignments.")
        env_to_atoms: Dict[str, list] = defaultdict(list)
        env_to_ops: Dict[str, dict] = {}

        for atom_idx_str, constraint_data in atom_constraints.items():
            atom_idx = int(atom_idx_str)
            env_type = (constraint_data.get('environment')
                        or constraint_data.get('environment_type', 'default'))
            env_to_atoms[env_type].append(atom_idx)
            if env_type not in env_to_ops:
                env_to_ops[env_type] = constraint_data.get('order_parameters', {})

        return {
            env_type: EnvironmentConstraint(
                env_type=env_type,
                atom_indices=atom_list,
                order_params=env_to_ops.get(env_type, {}),
                priority=priorities.get(env_type, 1.0),
            )
            for env_type, atom_list in env_to_atoms.items()
        }

    def _parse_from_global_fractions(
        self,
        p_index_set: set,
        atom_constraints: dict,
        priorities: dict,
    ) -> Dict[str, EnvironmentConstraint]:
        """
        Distribute P atoms across environment types by target fraction.

        Reads ``global_constraints.p_environment_distribution.target_fractions``
        from the JSON, maps label strings to env_type keys, then assigns
        P atoms sequentially (sorted by index) in proportion to each fraction.
        Order parameter targets are taken from the first occurrence of each
        env_type in ``atom_constraints``.
        """
        # --- collect order_params template per env_type from JSON ----------
        env_to_ops: Dict[str, dict] = {}
        for constraint_data in atom_constraints.values():
            env_type = (constraint_data.get('environment')
                        or constraint_data.get('environment_type', 'default'))
            if env_type not in env_to_ops:
                env_to_ops[env_type] = constraint_data.get('order_parameters', {})

        # --- read target fractions from global_constraints -----------------
        global_fracs: Dict[str, float] = (
            self.constraints_data
            .get('global_constraints', {})
            .get('p_environment_distribution', {})
            .get('target_fractions', {})
        )
        # Map label → env_type, skip zero-fraction or unmapped entries
        env_fracs: Dict[str, float] = {}
        for label, frac in global_fracs.items():
            env_type = self._LABEL_TO_ENV.get(label)
            if env_type and frac > 0.0:
                env_fracs[env_type] = frac / 100.0  # stored as percentages

        if not env_fracs:
            # No usable fractions — fall back to treating all P atoms as P4
            print("[ConstrainedOptimizer] Warning: no target fractions found; "
                  "assigning all P atoms to P4 environment.")
            env_fracs = {'P4': 1.0}

        # --- distribute P atoms by fraction --------------------------------
        p_indices = sorted(p_index_set)
        n_p = len(p_indices)
        total_frac = sum(env_fracs.values())
        env_to_atoms: Dict[str, list] = {}
        start = 0
        items = list(env_fracs.items())
        for i, (env_type, frac) in enumerate(items):
            if i == len(items) - 1:
                # Last environment gets all remaining atoms (avoids rounding gaps)
                count = n_p - start
            else:
                count = round(n_p * frac / total_frac)
            env_to_atoms[env_type] = p_indices[start: start + count]
            start += count

        print(f"[ConstrainedOptimizer] P atom assignment ({n_p} total):")
        for env_type, indices in env_to_atoms.items():
            print(f"  {env_type}: {len(indices)} atoms "
                  f"({100*len(indices)/n_p:.1f}%)")

        return {
            env_type: EnvironmentConstraint(
                env_type=env_type,
                atom_indices=atom_list,
                order_params=env_to_ops.get(env_type, {}),
                priority=priorities.get(env_type, 1.0),
            )
            for env_type, atom_list in env_to_atoms.items()
            if atom_list  # skip empty groups
        }
    
    def _setup_environment_constraints(self):
        """
        Create Cooper constraints grouped by environment.
        
        Each environment gets one Cooper Constraint object with:
        - Multiplier for each atom in that environment
        - Adaptive or constant penalty coefficient
        - Environment-specific priority weighting
        """
        self.constraint_dict = {}
        
        for env_type, env_constraint in self.env_constraints.items():
            n_atoms = env_constraint.num_atoms
            n_ops = len(env_constraint.order_params)
            
            # Total number of scalar constraints = atoms × order_parameters
            n_constraints = n_atoms * n_ops
            
            if n_constraints == 0:
                continue
            
            # Create multiplier
            multiplier = cooper.multipliers.DenseMultiplier(
                num_constraints=n_constraints,
                device=self.device
            )
            
            # Create penalty coefficient
            if self.use_adaptive_penalty:
                penalty = AdaptivePenalty(
                    num_constraints=n_constraints,
                    device=str(self.device),
                    **self.penalty_config
                )
            else:
                penalty = ConstantPenalty(
                    value=self.penalty_config.get('init', 10.0),
                    device=str(self.device)
                )
            
            # Create constraint
            constraint = cooper.Constraint(
                multiplier=multiplier,
                constraint_type=cooper.ConstraintType.INEQUALITY,
                formulation_type=cooper.formulations.AugmentedLagrangian,
                penalty_coefficient=penalty,
            )
            
            self.constraint_dict[env_type] = {
                'constraint': constraint,
                'env_constraint': env_constraint,
                'penalty': penalty,
            }
    
    def _print_summary(self):
        """Print constraint summary."""
        print(f"\n{'=' * 60}")
        print("Environment-Based Constrained Optimizer")
        print(f"{'=' * 60}")
        
        total_atoms = 0
        total_constraints = 0
        
        for env_type, info in self.constraint_dict.items():
            env_c = info['env_constraint']
            n_atoms = env_c.num_atoms
            n_ops = len(env_c.order_params)
            n_cons = n_atoms * n_ops
            
            print(f"  {env_type}:")
            print(f"    Atoms: {n_atoms}")
            print(f"    Order params: {env_c.op_names}")
            print(f"    Constraints: {n_cons}")
            print(f"    Priority: {env_c.priority:.1f}")
            
            total_atoms += n_atoms
            total_constraints += n_cons
        
        print(f"\nTotal: {total_atoms} atoms, {total_constraints} constraints")
        print(f"Adaptive penalties: {self.use_adaptive_penalty}")
        print(f"Warmup steps: {self.warmup_steps}")
        print(f"{'=' * 60}\n")
    
    def compute_cmp_state(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        step: Optional[int] = None,
    ) -> cooper.CMPState:
        """
        Compute the constrained minimization problem state.
        
        Args:
            positions: Current atomic positions
            cell: Current cell vectors
            step: Current optimization step (for warmup)
        
        Returns:
            CMPState with loss and observed constraints
        """
        # Update state
        self.base_state.positions = positions
        self.base_state.cell = cell
        
        # Compute spectrum and loss
        results = self.model(self.base_state)
        loss_dict = self.loss_fn(results)
        chi2_loss = loss_dict['total_loss']
        
        # Compute order parameters for constrained atoms
        all_constrained_indices = []
        for info in self.constraint_dict.values():
            all_constrained_indices.extend(info['env_constraint'].atom_indices)
        
        constrained_indices = torch.tensor(
            sorted(set(all_constrained_indices)),
            device=self.device, dtype=torch.long
        )
        
        # Get all required order parameters
        all_ops = set()
        for info in self.constraint_dict.values():
            all_ops.update(info['env_constraint'].op_names)
        
        if len(constrained_indices) > 0 and len(all_ops) > 0:
            op_results = self.op_calc(
                self.base_state,
                constrained_indices,
                list(all_ops),
                element_filter=None,
            )
        else:
            op_results = {}
        
        # Compute violations per environment
        violations_dict = self._compute_environment_violations(
            op_results, constrained_indices, step
        )
        
        # Create Cooper constraint states
        observed_constraints = {}
        for env_type, violation_tensor in violations_dict.items():
            constraint = self.constraint_dict[env_type]['constraint']
            constraint_state = cooper.ConstraintState(violation=violation_tensor)
            observed_constraints[constraint] = constraint_state
            
            # Update adaptive penalties
            if self.use_adaptive_penalty and step is not None and step > 0:
                penalty = self.constraint_dict[env_type]['penalty']
                if hasattr(penalty, 'update'):
                    penalty.update(violation_tensor)
        
        # Compute diagnostics
        all_violations = torch.cat([v for v in violations_dict.values()]) if violations_dict else torch.zeros(1, device=self.device)
        
        misc = {
            'chi2_loss': chi2_loss,
            'S_Q': results.get('S_Q'),
            'T_r': results.get('T_r'),
            'G_r': results.get('G_r'),
            'avg_violation': all_violations.mean().item(),
            'max_violation': all_violations.max().item(),
            'num_violated': (all_violations > 1e-4).sum().item(),
        }
        
        return cooper.CMPState(
            loss=chi2_loss,
            observed_constraints=observed_constraints,
            misc=misc,
        )
    
    def _compute_environment_violations(
        self,
        op_results: Dict[str, torch.Tensor],
        constrained_indices: torch.Tensor,
        step: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute constraint violations grouped by environment.
        
        Returns dict mapping environment type to violation tensor.
        """
        violations = {}
        
        # Warmup factor
        if self.warmup_steps > 0 and step is not None:
            warmup_factor = min(1.0, step / self.warmup_steps)
        else:
            warmup_factor = 1.0
        
        for env_type, info in self.constraint_dict.items():
            env_constraint = info['env_constraint']
            env_violations = []
            
            for atom_idx in env_constraint.atom_indices:
                # Find position in constrained_indices
                pos = (constrained_indices == atom_idx).nonzero(as_tuple=True)[0]
                if pos.numel() == 0:
                    # Atom not in computed set - zero violations
                    for _ in env_constraint.order_params:
                        env_violations.append(torch.zeros((), device=self.device))
                    continue
                
                pos = pos[0]
                
                for op_name, op_params in env_constraint.order_params.items():
                    if op_name not in op_results:
                        env_violations.append(torch.zeros((), device=self.device))
                        continue
                    
                    value = op_results[op_name][pos]
                    
                    # Skip NaN values
                    if not torch.isfinite(value):
                        env_violations.append(torch.zeros((), device=self.device))
                        continue
                    
                    weight = torch.tensor(
                        op_params.get('weight', 1.0) * env_constraint.priority,
                        device=self.device, dtype=value.dtype
                    )
                    
                    # Compute violation
                    if 'min' in op_params and 'max' in op_params:
                        # Box constraint
                        minv = torch.tensor(op_params['min'], device=self.device, dtype=value.dtype)
                        maxv = torch.tensor(op_params['max'], device=self.device, dtype=value.dtype)
                        v = torch.relu(minv - value) + torch.relu(value - maxv)
                    else:
                        # Tolerance constraint
                        target = torch.tensor(op_params['target'], device=self.device, dtype=value.dtype)
                        tol = torch.tensor(op_params.get('tolerance', 0.1), device=self.device, dtype=value.dtype)
                        v = torch.relu(torch.abs(value - target) - tol)
                    
                    env_violations.append(weight * v * warmup_factor)
            
            if env_violations:
                violations[env_type] = torch.stack(env_violations)
        
        return violations
    
    def get_constraint_state(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
    ) -> ConstraintState:
        """
        Get detailed constraint state for diagnostics.
        """
        cmp_state = self.compute_cmp_state(positions, cell)
        
        violations = {}
        penalties = {}
        satisfaction_rates = {}
        
        total_violated = 0
        total_constraints = 0
        total_violation = 0.0
        
        for env_type, info in self.constraint_dict.items():
            constraint = info['constraint']
            
            # Get violations from CMP state
            for c, cs in cmp_state.observed_constraints.items():
                if c is constraint:
                    v = cs.violation
                    violations[env_type] = v
                    
                    n_violated = (v > 1e-4).sum().item()
                    n_total = v.numel()
                    satisfaction_rates[env_type] = 100 * (1 - n_violated / n_total) if n_total > 0 else 100.0
                    
                    total_violated += n_violated
                    total_constraints += n_total
                    total_violation += v.sum().item()
                    break
            
            # Get penalty values
            penalty = info['penalty']
            if hasattr(penalty, 'init'):
                penalties[env_type] = penalty.init
        
        return ConstraintState(
            violations=violations,
            penalties=penalties,
            satisfaction_rates=satisfaction_rates,
            total_violation=total_violation,
            num_violated=total_violated,
            num_total=total_constraints,
        )
    
    def log_constraint_summary(self, step: int, positions: torch.Tensor, cell: torch.Tensor):
        """Log constraint summary to console."""
        state = self.get_constraint_state(positions, cell)
        
        print(f"\nStep {step} - Constraint Summary:")
        print(f"  Total violation: {state.total_violation:.4f}")
        print(f"  Violated: {state.num_violated}/{state.num_total}")
        
        for env_type in state.satisfaction_rates:
            rate = state.satisfaction_rates[env_type]
            penalty = state.penalties.get(env_type)
            penalty_str = f", penalty={penalty.mean().item():.1f}" if penalty is not None else ""
            print(f"  {env_type}: {rate:.1f}% satisfied{penalty_str}")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'AdaptivePenalty',
    'ConstantPenalty',
    'EnvironmentConstraint',
    'ConstraintState',
    'EnvironmentConstrainedOptimizer',
]
