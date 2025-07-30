from collections import defaultdict
from typing import TypeVar

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from tqdm.auto import tqdm


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
