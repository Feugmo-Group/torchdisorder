from collections import defaultdict
from typing import TypeVar

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from tqdm.auto import tqdm

"""
Callbacks for optimization monitoring and control.
"""

class PlateauDetector:
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

