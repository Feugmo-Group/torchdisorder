"""
Cooper-based Constrained Minimization Problem with Per-Atom Order Parameter Constraints
Uses TorchSimOrderParameters and JSON constraint files

"""

from __future__ import annotations
import json
import torch
import cooper
from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import defaultdict
from ase.data import chemical_symbols

import torch_sim as ts
from cooper.penalty_coefficients import PenaltyCoefficient

# Import your existing modules
from torchdisorder.common.target_rdf import TargetRDFData

# Import order parameter calculator for constraints
from order_params import TorchSimOrderParameters


class ConstantPenalty(PenaltyCoefficient):
    """Constant penalty coefficient for AugmentedLagrangian."""

    def __init__(self, value: float, device='cuda'):
        super().__init__(init=torch.tensor(value, device=device))
        self.expects_constraint_features = False

    def __call__(self, constraint_features=None):
        """Return the constant penalty value."""
        return self.init


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
        penalty_rho: float = 40.0,
        violation_type: str = 'soft'  # 'soft' or 'hard'
    ):
        super().__init__()
        
        self.model = model
        self.base_state = base_state
        self.target = target_vec.F_q_target
        self.target_uncert = target_vec.F_q_uncert
        self.kind = target_kind
        self.q_bins = q_bins
        self.loss_fn = loss_fn
        self.device = device
        self.violation_type = violation_type
        
        # Load constraints from JSON
        self.constraints_data = self._load_constraints(constraints_file)
        
        # Initialize order parameter calculator
        cutoff = self.constraints_data['cutoff']
        self.op_calc = TorchSimOrderParameters(cutoff=cutoff, device=str(device))
        
        # Detect placeholder region (where dF was originally zero)
        self.placeholder_mask = (self.target_uncert < 1e-6).to(device)
        n_placeholder = self.placeholder_mask.sum().item()
        print(f"Found {n_placeholder} placeholder points where dF = 1e-7")
        
        # Set up constraints based on JSON file
        self._setup_constraints(penalty_rho)
        
        print(f"\nInitialized StructureFactorCMPWithConstraints:")
        print(f"  Constrained atoms: {len(self.constraints_data['atom_constraints'])}")
        print(f"  Order parameters: {', '.join(self.constraints_data['metadata']['order_parameter_types'])}")
        print(f"  Total Cooper constraints: {len(self.constraint_dict)}")
    
    def _load_constraints(self, constraints_file: Union[str, Path]) -> Dict:
        """Load constraint specifications from JSON file."""
        with open(constraints_file, 'r') as f:
            constraints = json.load(f)
        return constraints
    
    def _setup_constraints(self, penalty_rho: float):
        """
        Set up Cooper constraint objects for all atom-constraint pairs.
        
        Strategy: Group constraints by order parameter type to reduce
        the number of Cooper constraint objects.
        """
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
            
            # Create penalty coefficient
            penalty_coeff = ConstantPenalty(penalty_rho, device=self.device)
            
            # Create constraint
            constraint = cooper.Constraint(
                multiplier=multiplier,
                constraint_type=cooper.ConstraintType.INEQUALITY,
                formulation_type=cooper.formulations.AugmentedLagrangian,
                penalty_coefficient=penalty_coeff,
            )
            
            self.constraint_dict[op_name] = {
                'constraint': constraint,
                'atom_indices': atom_list
            }
            
            print(f"  Created constraint for {op_name}: {n_atoms} atoms")
    
    def _compute_violations(
        self,
        op_results: Dict[str, torch.Tensor],
        constrained_atom_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute constraint violations for all atoms.
        
        Returns:
            violations: {op_name: tensor of violations per atom}
        """
        violations = {}
        
        for op_name, constraint_info in self.constraint_dict.items():
            atom_list = constraint_info['atom_indices']
            op_violations = []
            
            for atom_idx in atom_list:
                atom_idx_str = str(atom_idx)
                atom_constraint = self.constraints_data['atom_constraints'][atom_idx_str]
                op_params = atom_constraint['order_parameters'][op_name]
                
                # Find position in constrained_atom_indices tensor
                pos = (constrained_atom_indices == atom_idx).nonzero(as_tuple=True)[0]
                if len(pos) == 0:
                    # Atom not in computed results (shouldn't happen)
                    op_violations.append(0.0)
                    continue
                
                pos = pos[0].item()
                value = op_results[op_name][pos].item()
                
                # Compute violation based on constraint type
                if 'min' in op_params and 'max' in op_params:
                    # Box constraint: violation if outside [min, max]
                    if self.violation_type == 'soft':
                        # Soft constraint: smooth penalty outside bounds
                        if value < op_params['min']:
                            violation = op_params['min'] - value
                        elif value > op_params['max']:
                            violation = value - op_params['max']
                        else:
                            violation = 0.0
                    else:
                        # Hard constraint: violation = distance to nearest bound
                        mid = (op_params['min'] + op_params['max']) / 2
                        if value < mid:
                            violation = max(0, op_params['min'] - value)
                        else:
                            violation = max(0, value - op_params['max'])
                else:
                    # Equality constraint with tolerance
                    target = op_params['target']
                    tolerance = op_params.get('tolerance', 0.1)
                    violation = max(0.0, abs(value - target) - tolerance)
                
                # Apply weight
                weight = op_params.get('weight', 1.0)
                op_violations.append(weight * violation)
            
            # Convert to tensor
            violations[op_name] = torch.tensor(op_violations, device=self.device, dtype=torch.float32)
        
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
        chi2_loss = loss_dict.get("chi2_scatt", loss_dict.get("chi2_corr", loss_dict.get("loss")))
        
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
        violations = self._compute_violations(op_results, constrained_atom_indices)
        
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


# ============================================================================
# Backward-compatible wrapper 
# ============================================================================

def create_cmp_with_constraints(
    model,
    base_state: ts.SimState,
    target_vec: TargetRDFData,
    constraints_file: str,
    loss_fn,
    target_kind: str = 'S_Q',
    device: str = 'cuda',
    penalty_rho: float = 40.0,
    **kwargs
) -> StructureFactorCMPWithConstraints:
    """
    Factory function to create CMP with constraints.
    
    Drop-in replacement for your old StructureFactorCMP initialization.
    
    Example:
        >>> # OLD CODE:
        >>> # cmp = StructureFactorCMP(
        >>> #     model, base_state, target_vec, 'S_Q', q_bins, loss_fn,
        >>> #     q_threshold=0.7, device='cuda', penalty_rho=40.0
        >>> # )
        >>> 
        >>> # NEW CODE:
        >>> cmp = create_cmp_with_constraints(
        ...     model, base_state, target_vec,
        ...     constraints_file='my_structure_constraints.json',
        ...     loss_fn=loss_fn,
        ...     target_kind='S_Q',
        ...     device='cuda',
        ...     penalty_rho=40.0
        ... )
    """
    return StructureFactorCMPWithConstraints(
        model=model,
        base_state=base_state,
        target_vec=target_vec,
        constraints_file=constraints_file,
        loss_fn=loss_fn,
        target_kind=target_kind,
        device=device,
        penalty_rho=penalty_rho,
        **kwargs
    )


# ============================================================================
# Example usage
# ============================================================================

def example_usage():
    """Example showing how to use with Cooper."""
    
    print("="*70)
    print("EXAMPLE: Cooper CMP with Per-Atom Constraints")
    print("="*70)
    
    # Your existing setup (unchanged)
    import torch_sim as ts
    from pymatgen.core import Structure
    
    structure = Structure.from_file('your_structure.cif')
    state = ts.initialize_state(structure, device='cuda', dtype=torch.float64)
    
    # Your model and RDF data (unchanged)
    # model = YourXRDModel(...)
    # rdf_data = TargetRDFData(...)
    # loss_fn = your_loss_function
    
    # ===================================================================
    # CHANGE: Use new CMP with constraints from JSON
    # ===================================================================
    
    # OLD CODE (commented out):
    # cmp = StructureFactorCMP(
    #     model, state, rdf_data, 'S_Q', q_bins, loss_fn,
    #     q_threshold=0.7, device='cuda', penalty_rho=40.0
    # )
    
    # NEW CODE:
    cmp = StructureFactorCMPWithConstraints(
        model=model,
        base_state=state,
        target_vec=rdf_data,
        constraints_file='my_structure_constraints.json',  # Generated by glass_generator.py
        loss_fn=loss_fn,
        target_kind='S_Q',
        device='cuda',
        penalty_rho=40.0
    )
    
    # Rest of Cooper setup (unchanged)
    formulation = cooper.formulations.AugmentedLagrangian()
    constrained_optimizer = cooper.optim.ConstrainedOptimizer(
        optimizer_class=torch.optim.Adam,
        params=[state.positions],
        lr=0.01,
        constrained_optimizer_kwargs={'primal_optimizer_kwargs': {'lr': 0.01}}
    )
    
    # Optimization loop (unchanged except no manual q_tet)
    for iteration in range(1000):
        # Compute CMP state (order parameters computed automatically!)
        cmp_state = cmp.compute_cmp_state(
            state.positions,
            state.cell,
            step=iteration
        )
        
        # Cooper optimization step
        constrained_optimizer.zero_grad()
        
        lagrangian = formulation.composite_objective(
            cmp_state.loss,
            cmp_state.observed_constraints
        )
        
        formulation.custom_backward(lagrangian)
        constrained_optimizer.step()
        
        # Logging
        if iteration % 10 == 0:
            misc = cmp_state.misc
            print(f"\nIter {iteration}:")
            print(f"  Loss: {cmp_state.loss.item():.6f}")
            print(f"  Violations: {misc['n_violations']}")
            print(f"  Avg: {misc['avg_violation']:.4f}")
            print(f"  Max: {misc['max_violation']:.4f}")
        
        # Detailed environment statistics
        if iteration % 50 == 0 and 'environment_stats' in cmp_state.misc:
            print("\n  Environment satisfaction:")
            for env, stats in cmp_state.misc['environment_stats'].items():
                print(f"    {env}: {stats['satisfaction_rate']:.1f}%")
    
    print("\nOptimization complete!")


if __name__ == "__main__":
    example_usage()
