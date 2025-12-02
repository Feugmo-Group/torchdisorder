from __future__ import annotations
import torch
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path
import yaml
def load_target_rdf(path):
    df = pd.read_csv(path)
    r = torch.tensor(df['r'].values)
    g_r = torch.tensor(df['g_r'].values)
    return r, g_r



@dataclass
class TargetRDFData:
    r_bins: torch.Tensor
    T_r_target: torch.Tensor
    q_bins: torch.Tensor
    F_q_target: torch.Tensor
    F_q_uncert: torch.Tensor
    device: torch.device

    @classmethod
    def from_yaml(cls, path: str | Path, *, device: str | torch.device = "cuda", stride_r: int = 1, stride_q: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with open(path, "r") as f:
            cfg: Dict[str, Any] = yaml.safe_load(f)
        return cls.from_dict(cfg, device=device, stride_r=stride_r, stride_q=stride_q).as_tuple()

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], *, device: str | torch.device = "cuda", stride_r: int = 1, stride_q: int = 1) -> "TargetRDFData":
        device = torch.device(device)
        df_T = pd.read_csv(cfg["t_of_r_path"])
        df_F = pd.read_csv(cfg["f_of_q_path"])

        r_bins = torch.tensor(df_T["r"].to_numpy(dtype="float32")[::stride_r], device=device)
        T_r = torch.tensor(df_T["T"].to_numpy(dtype="float32")[::stride_r], device=device)

        q_bins = torch.tensor(df_F["Q"].to_numpy(dtype="float32")[::stride_q], device=device)
        F_q = torch.tensor(df_F["F"].to_numpy(dtype="float32")[::stride_q], device=device)
        dF_q = torch.tensor(df_F["dF"].to_numpy(dtype="float32")[::stride_q], device=device)


        return cls(r_bins, T_r, q_bins, F_q, dF_q, device)

    def to(self, new_device: str | torch.device) -> "TargetRDFData":
        new_device = torch.device(new_device)
        return TargetRDFData(
            r_bins=self.r_bins.to(new_device),
            T_r_target=self.T_r_target.to(new_device),
            q_bins=self.q_bins.to(new_device),
            F_q_target=self.F_q_target.to(new_device),
            F_q_uncert=self.F_q_uncert.to(new_device),
            device=new_device,
        )

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.T_r_target, self.F_q_target, self.q_bins, self.r_bins, self.F_q_uncert
