from collections import defaultdict
from typing import TypeVar, Optional, Dict, Any
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
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

