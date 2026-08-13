from collections import defaultdict
from typing import TypeVar, Optional, Dict, Any, List
import os

import numpy as np
import torch
from tqdm.auto import tqdm

"""
Callbacks for optimization monitoring and control.
"""


class EarlyStoppingCallback:
    """
    Early stopping callback for optimization.
    
    Stops training when the monitored metric stops improving.
    """
    
    def __init__(
        self,
        monitor: str = 'loss',
        patience: int = 1000,
        min_delta: float = 1e-6,
        mode: str = 'min',
        verbose: bool = True,
    ):
        """
        Args:
            monitor: Metric to monitor
            patience: Number of steps with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether lower or higher is better
            verbose: Whether to print messages
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False
        
    def __call__(self, step: int, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop.
        
        Args:
            step: Current step
            metrics: Dictionary of metrics
            
        Returns:
            True if training should stop
        """
        if self.monitor not in metrics:
            return False
            
        current = metrics[self.monitor]
        
        if self.mode == 'min':
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta
            
        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.verbose:
                print(f"\n{'=' * 70}")
                print(f"⏹ EARLY STOPPING at step {step}")
                print(f"  No improvement in {self.monitor} for {self.patience} steps")
                print(f"  Best value: {self.best_value:.6f}")
                print(f"{'=' * 70}\n")
            self.should_stop = True
            return True
            
        return False
        
    def reset(self):
        """Reset the callback state."""
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False


class CheckpointCallback:
    """
    Checkpoint callback for saving model state during optimization.
    """
    
    def __init__(
        self,
        save_dir: str,
        save_interval: int = 200,
        save_best: bool = True,
        monitor: str = 'loss',
        mode: str = 'min',
        verbose: bool = True,
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            save_interval: Save every N steps
            save_best: Whether to save the best model
            monitor: Metric to monitor for best model
            mode: 'min' or 'max'
            verbose: Whether to print messages
        """
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
        os.makedirs(save_dir, exist_ok=True)
        
    def __call__(
        self,
        step: int,
        state: Dict[str, Any],
        metrics: Dict[str, float],
    ) -> Optional[str]:
        """
        Save checkpoint if needed.
        
        Args:
            step: Current step
            state: State dictionary to save
            metrics: Dictionary of metrics
            
        Returns:
            Path to saved checkpoint, or None
        """
        saved_path = None
        
        # Regular interval save
        if step > 0 and step % self.save_interval == 0:
            path = os.path.join(self.save_dir, f'checkpoint_step_{step}.pt')
            torch.save(state, path)
            if self.verbose:
                print(f"  💾 Checkpoint saved: {path}")
            saved_path = path
            
        # Best model save
        if self.save_best and self.monitor in metrics:
            current = metrics[self.monitor]
            
            if self.mode == 'min':
                is_best = current < self.best_value
            else:
                is_best = current > self.best_value
                
            if is_best:
                self.best_value = current
                path = os.path.join(self.save_dir, 'best_model.pt')
                torch.save(state, path)
                if self.verbose:
                    print(f"  ⭐ New best model saved: {self.monitor}={current:.6f}")
                saved_path = path
                
        return saved_path
        
    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint."""
        path = os.path.join(self.save_dir, 'best_model.pt')
        if os.path.exists(path):
            return torch.load(path)
        return None


class PlateauDetector:
    """Detects training plateaus and triggers melt-quench."""
    
    def __init__(self, window=200, melt_quench_fn=None, max_melt_quench=3):
        self.window = window  # Number of steps to track for plateau
        self.melt_quench_fn = melt_quench_fn
        self.max_melt_quench = max_melt_quench
        self.reduction_history = []
        self.n_melt_quench = 0

    def check_and_trigger(self, step, current_reduction, current_state):
        # Round to 1 decimal place before storing
        rounded_reduction = round(current_reduction, 1)
        self.reduction_history.append(rounded_reduction)

        if len(self.reduction_history) >= self.window:
            recent_window = self.reduction_history[-self.window:]

            # Check if all values in recent_window are exactly the same
            if len(set(recent_window)) == 1 and self.n_melt_quench < self.max_melt_quench:
                print(f"\n{'=' * 70}")
                print(f"⚠️ PLATEAU DETECTED at step {step}")
                print(f"   Percentage reduction stable at {rounded_reduction}% for last {self.window} steps")
                print(f"   Triggering melt-quench #{self.n_melt_quench + 1}/{self.max_melt_quench}")
                print(f"{'=' * 70}\n")

                updated_state = self.melt_quench_fn(current_state, self.n_melt_quench)
                self.n_melt_quench += 1
                self.reduction_history = []  # Reset history after melt-quench
                return updated_state, True

        return current_state, False


class FISFeedbackCallback:
    """Dynamically adjusts environment constraint priorities based on per-environment F_IS error.

    For each structural environment (e.g. Fe4, Fe6, Ta6, PO4, PO3N), computes the
    mean F_IS of the atoms in that environment and scales its ``priority`` proportionally
    to how far it is from the F_IS target.  Environments with large F_IS error receive
    higher penalty weight so the optimizer focuses more effort there.

    Priority update rule (applied every ``update_interval`` steps, after ``warmup_steps``):
        error_e  = |mean_F_IS_e − target|
        scale_e  = clip(1 + feedback_strength * error_e, min_scale, max_scale)
        priority_e = base_priority_e * scale_e

    Args:
        optimizer:          EnvironmentConstrainedOptimizer instance.
        fis_target:         Global F_IS target value from config.
        central_z:          Atomic number of the central species (e.g. 26 for Fe).
        neighbor_z:         Atomic number of the neighbor species (None = all).
        update_interval:    Steps between priority updates.
        feedback_strength:  Scales how aggressively priorities shift (≥0).
        min_scale:          Minimum priority multiplier (keeps base floor).
        max_scale:          Maximum priority multiplier (prevents explosion).
        warmup_steps:       Do not update until after this many steps.
    """

    def __init__(
        self,
        optimizer,
        fis_target: float,
        central_z: int,
        neighbor_z: Optional[int] = None,
        update_interval: int = 200,
        feedback_strength: float = 2.0,
        min_scale: float = 0.5,
        max_scale: float = 5.0,
        warmup_steps: int = 500,
    ):
        self.optimizer = optimizer
        self.fis_target = fis_target
        self.central_z = central_z
        self.neighbor_z = neighbor_z
        self.update_interval = update_interval
        self.feedback_strength = feedback_strength
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.warmup_steps = warmup_steps

        # Snapshot the initial priorities so we always scale from the original baseline
        self._base_priorities: Dict[str, float] = {
            env_type: info["env_constraint"].priority
            for env_type, info in optimizer.constraint_dict.items()
        }

        # Running log: list of dicts {step, fis_by_env, scales}
        self.history: List[Dict] = []

    def __call__(self, step: int, state) -> Dict[str, float]:
        """Compute per-environment F_IS and update priorities.

        Returns a dict mapping env_type → mean F_IS (empty dict if not an update step).
        """
        if step < self.warmup_steps or step % self.update_interval != 0:
            return {}

        fis_by_env = self._compute_fis_by_env(state)
        if not fis_by_env:
            return {}

        scales: Dict[str, float] = {}
        for env_type, fis_mean in fis_by_env.items():
            error = abs(fis_mean - self.fis_target)
            scale = float(np.clip(1.0 + self.feedback_strength * error,
                                  self.min_scale, self.max_scale))
            new_priority = self._base_priorities.get(env_type, 1.0) * scale
            self.optimizer.constraint_dict[env_type]["env_constraint"].priority = new_priority
            scales[env_type] = scale

        self.history.append({"step": step, "fis_by_env": dict(fis_by_env), "scales": scales})
        return fis_by_env

    def _compute_fis_by_env(self, state) -> Dict[str, float]:
        """Return mean F_IS per environment type using the optimizer's own op_calc."""
        all_indices = []
        for info in self.optimizer.constraint_dict.values():
            all_indices.extend(info["env_constraint"].atom_indices)
        if not all_indices:
            return {}

        device = state.positions.device
        constrained_indices = torch.tensor(
            sorted(set(all_indices)), dtype=torch.long, device=device
        )

        element_filter = [self.neighbor_z] if self.neighbor_z is not None else None
        try:
            with torch.no_grad():
                op_results = self.optimizer.op_calc(
                    state,
                    constrained_indices,
                    order_params=["fis"],
                    element_filter=element_filter,
                )
        except Exception:
            return {}

        fis_values = op_results.get("fis")
        if fis_values is None or fis_values.numel() == 0:
            return {}

        idx_map = {int(a): k for k, a in enumerate(constrained_indices.cpu().tolist())}

        fis_by_env: Dict[str, float] = {}
        for env_type, info in self.optimizer.constraint_dict.items():
            vals = [fis_values[idx_map[ai]].item()
                    for ai in info["env_constraint"].atom_indices
                    if ai in idx_map]
            if vals:
                fis_by_env[env_type] = float(np.mean(vals))

        return fis_by_env


class StructureHealthCallback:
    """Warns as soon as a refinement starts producing physically impossible geometry.

    Fitting a 1-D scattering function is underdetermined, so chi-squared can fall
    happily while atoms pass through one another.  An audit of 35 archived runs
    found 25 of them ended with overlapping atoms that no loss curve revealed.
    This callback surfaces that while the run is still cheap to abandon.

    It only ever *reports* — the decision to stop belongs to the caller, which
    reads :attr:`failed`.  Set ``raise_on_fail`` to turn the first violation into
    an exception instead.

    Args:
        check_interval:  Steps between checks (each one builds a neighbour list,
                         so keep this well above 1).
        central/neighbour: Species for the coordination-plateau check; omit to
                         check overlap only.
        expected_cn:     Optional target coordination for the plateau check.
        overlap_tol:     Fraction of summed covalent radii below which a contact
                         counts as an overlap.
        raise_on_fail:   Raise ``RuntimeError`` on the first failed check.
        verbose:         Print the full report when a check fails.
    """

    def __init__(
        self,
        check_interval: int = 200,
        central=None,
        neighbour=None,
        expected_cn: Optional[float] = None,
        overlap_tol: float = 0.6,
        raise_on_fail: bool = False,
        verbose: bool = True,
    ):
        self.check_interval = max(int(check_interval), 1)
        self.central = central
        self.neighbour = neighbour
        self.expected_cn = expected_cn
        self.overlap_tol = overlap_tol
        self.raise_on_fail = raise_on_fail
        self.verbose = verbose

        self.failed = False
        self.first_failure_step: Optional[int] = None
        self.history: List[Dict[str, Any]] = []

    def __call__(self, step: int, state) -> Dict[str, float]:
        """Validate the current state.  Returns metrics suitable for wandb.log."""
        if step % self.check_interval:
            return {}

        from torchdisorder.common.validation import validate_structure

        report = validate_structure(
            state,
            overlap_tol=self.overlap_tol,
            check_plateau=self.central is not None,
            central=self.central,
            neighbour=self.neighbour,
            expected_cn=self.expected_cn,
        )

        metrics = {
            "health/n_overlaps": float(report.n_overlaps),
            "health/min_distance": float(report.min_distance),
            "health/worst_ratio": float(report.worst_ratio),
        }
        self.history.append({"step": step, **metrics})

        if not report:
            if not self.failed:
                self.first_failure_step = step
            self.failed = True
            message = f"[structure health] step {step}: {report.summary()}"
            if self.raise_on_fail:
                raise RuntimeError(message)
            if self.verbose:
                print(f"\n{message}\n")

        return metrics


# import wandb
# import torch
# from torchdisorder.model.rdf import compute_rdf
# from torchdisorder.data.target_rdf import load_target_rdf
# from torchdisorder.model import generator
#
# class Trainer:
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.coords = generator.init_coords(cfg).to(cfg.device).requires_grad_()
#         self.optimizer = instantiate(cfg.optimizer, [self.coords])
#         self.target_r, self.target_gr = load_target_rdf(cfg.experiment.target_rdf_path)
#         wandb.init(**cfg.wandb)
#
#     def step(self, step_idx):
#         self.optimizer.zero_grad()
#         r, g_r = compute_rdf(self.coords, self.cfg.system.box_length, self.cfg.experiment.rdf_bins)
#         loss = ((g_r - self.target_gr) ** 2).mean()
#         loss.backward()
#         self.optimizer.step()
#
#         wandb.log({"loss": loss.item()}, step=step_idx)
#         if step_idx % self.cfg.wandb.log_rdf_every == 0:
#             wandb.log({"g_r": wandb.plot.line_series(xs=r.cpu(),
#                                                      ys=[self.target_gr.cpu(), g_r.detach().cpu()],
#                                                      keys=["target", "model"],
#                                                      title=f"RDF @ step {step_idx}")})

