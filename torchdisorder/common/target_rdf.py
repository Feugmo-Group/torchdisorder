"""
Target RDF Data Module
======================

Handles loading and storing target diffraction data for structure optimization.

Supported target types:
    - S(Q): Structure factor in reciprocal space (Q-space)
    - T(r): Total correlation function in real space (r-space)
    - g(r): Pair distribution function / radial distribution function (r-space)

Relationships between functions:
    - T(r) = 4πρr[g(r) - 1]  where ρ is number density
    - S(Q) is the Fourier transform of g(r)
"""

from __future__ import annotations
import torch
import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class TargetRDFData:
    """
    Container for target diffraction data.
    
    Attributes
    ----------
    q_bins : torch.Tensor
        Q values (Å⁻¹) for reciprocal space data
    S_Q_target : torch.Tensor
        Target S(Q) structure factor values
    S_Q_uncert : torch.Tensor
        Uncertainty in S(Q)
        
    r_bins : torch.Tensor
        r values (Å) for real space data
    T_r_target : torch.Tensor
        Target T(r) total correlation function values
    T_r_uncert : torch.Tensor
        Uncertainty in T(r)
        
    g_r_target : torch.Tensor
        Target g(r) pair distribution function values
    g_r_uncert : torch.Tensor
        Uncertainty in g(r)
        
    device : torch.device
        Device where tensors are stored
    """
    # Q-space data
    q_bins: torch.Tensor
    S_Q_target: torch.Tensor
    S_Q_uncert: torch.Tensor
    
    # r-space data
    r_bins: torch.Tensor
    T_r_target: torch.Tensor
    T_r_uncert: torch.Tensor
    g_r_target: torch.Tensor
    g_r_uncert: torch.Tensor
    
    device: torch.device

    @classmethod
    def from_yaml(cls, path: str | Path, *, device: str | torch.device = "cuda", 
                  stride_r: int = 1, stride_q: int = 1) -> "TargetRDFData":
        """Load target data from YAML config file."""
        with open(path, "r") as f:
            cfg: Dict[str, Any] = yaml.safe_load(f)
        return cls.from_dict(cfg, device=device, stride_r=stride_r, stride_q=stride_q)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], *, device: str | torch.device = "cuda", 
                  stride_r: int = 1, stride_q: int = 1) -> "TargetRDFData":
        """
        Load target RDF data from CSV files specified in config dictionary.
        
        Parameters
        ----------
        cfg : dict
            Configuration dictionary with keys:
                - s_of_q_path or f_of_q_path: Path to S(Q) CSV file
                - t_of_r_path: Path to T(r) CSV file (optional)
                - g_of_r_path: Path to g(r) CSV file (optional)
                - r_max: Maximum r for auto-generated r_bins (default: 10.0 Å)
                - n_r_bins: Number of r bins if auto-generated (default: 500)
        device : str or torch.device
            Device to place tensors on
        stride_r : int
            Stride for subsampling r-space data
        stride_q : int
            Stride for subsampling Q-space data
            
        Returns
        -------
        TargetRDFData
            Loaded target data container
            
        Notes
        -----
        Supported CSV column names:
            Q-space: Q/q for bins, F/SQ/S/S(Q)/F(Q) for values, dF/dSQ/dS/uncertainty/error for uncertainty
            T(r): r/R for bins, T/Tr/T(r) for values, dT/uncertainty/error for uncertainty
            g(r): r/R for bins, g/gr/g(r)/G/G(r) for values, dg/uncertainty/error for uncertainty
        """
        device = torch.device(device)
        
        def find_column(df, possible_names, required=True):
            """Find column by trying multiple possible names (case-insensitive)."""
            df_cols_lower = {c.lower(): c for c in df.columns}
            for name in possible_names:
                if name.lower() in df_cols_lower:
                    return df_cols_lower[name.lower()]
            if required:
                raise KeyError(f"Could not find column. Tried: {possible_names}. Available: {list(df.columns)}")
            return None
        
        def load_csv_safe(path):
            """Load CSV handling common issues like BOM."""
            return pd.read_csv(path, encoding='utf-8-sig')
        
        # Initialize empty tensors
        r_bins = torch.tensor([], dtype=torch.float32, device=device)
        T_r_target = torch.tensor([], dtype=torch.float32, device=device)
        T_r_uncert = torch.tensor([], dtype=torch.float32, device=device)
        g_r_target = torch.tensor([], dtype=torch.float32, device=device)
        g_r_uncert = torch.tensor([], dtype=torch.float32, device=device)
        q_bins = torch.tensor([], dtype=torch.float32, device=device)
        S_Q_target = torch.tensor([], dtype=torch.float32, device=device)
        S_Q_uncert = torch.tensor([], dtype=torch.float32, device=device)
        
        r_bins_loaded = False
        
        # =====================================================================
        # Load T(r) data
        # =====================================================================
        t_path = cfg.get("t_of_r_path")
        if t_path and str(t_path).lower() not in ['null', 'none', ''] and Path(t_path).exists():
            df = load_csv_safe(t_path)
            r_col = find_column(df, ['r', 'R'])
            T_col = find_column(df, ['T', 'Tr', 'T(r)'])
            
            r_bins = torch.tensor(df[r_col].to_numpy(dtype="float32")[::stride_r], device=device)
            T_r_target = torch.tensor(df[T_col].to_numpy(dtype="float32")[::stride_r], device=device)
            r_bins_loaded = True
            
            # Optional uncertainty
            dT_col = find_column(df, ['dT', 'uncertainty', 'error'], required=False)
            if dT_col:
                T_r_uncert = torch.tensor(df[dT_col].to_numpy(dtype="float32")[::stride_r], device=device)
            else:
                T_r_uncert = torch.full_like(T_r_target, 0.05)
            
            print(f"  ✓ Loaded T(r): {len(r_bins)} bins from {Path(t_path).name}")
        
        # =====================================================================
        # Load g(r) data
        # =====================================================================
        g_path = cfg.get("g_of_r_path")
        if g_path and str(g_path).lower() not in ['null', 'none', ''] and Path(g_path).exists():
            df = load_csv_safe(g_path)
            r_col = find_column(df, ['r', 'R'])
            g_col = find_column(df, ['g', 'gr', 'g(r)', 'G', 'G(r)', 'rdf', 'RDF'])
            
            r_bins_g = torch.tensor(df[r_col].to_numpy(dtype="float32")[::stride_r], device=device)
            g_r_target = torch.tensor(df[g_col].to_numpy(dtype="float32")[::stride_r], device=device)
            
            # Use g(r) r_bins if T(r) not loaded
            if not r_bins_loaded:
                r_bins = r_bins_g
                r_bins_loaded = True
            
            # Optional uncertainty
            dg_col = find_column(df, ['dg', 'dgr', 'uncertainty', 'error'], required=False)
            if dg_col:
                g_r_uncert = torch.tensor(df[dg_col].to_numpy(dtype="float32")[::stride_r], device=device)
            else:
                g_r_uncert = torch.full_like(g_r_target, 0.05)
            
            print(f"  ✓ Loaded g(r): {len(g_r_target)} bins from {Path(g_path).name}")
        
        # =====================================================================
        # Auto-generate r_bins if no r-space data provided
        # =====================================================================
        if not r_bins_loaded:
            r_max = cfg.get("r_max", 10.0)
            n_r_bins = cfg.get("n_r_bins", cfg.get("r_bins", 500))
            r_bins = torch.linspace(0.01, r_max, n_r_bins, device=device, dtype=torch.float32)
            print(f"  ⚙ Auto-generated r_bins: {n_r_bins} bins, r_max={r_max} Å")
        
        # =====================================================================
        # Load S(Q) data
        # =====================================================================
        # Support both 's_of_q_path' and legacy 'f_of_q_path'
        s_path = cfg.get("s_of_q_path") or cfg.get("f_of_q_path")
        if s_path and str(s_path).lower() not in ['null', 'none', ''] and Path(s_path).exists():
            df = load_csv_safe(s_path)
            
            Q_col = find_column(df, ['Q', 'q'])
            S_col = find_column(df, ['F', 'SQ', 'S', 'S(Q)', 'F(Q)'])
            
            q_bins = torch.tensor(df[Q_col].to_numpy(dtype="float32")[::stride_q], device=device)
            S_Q_target = torch.tensor(df[S_col].to_numpy(dtype="float32")[::stride_q], device=device)
            
            # Optional uncertainty
            dS_col = find_column(df, ['dF', 'dSQ', 'dS', 'uncertainty', 'error'], required=False)
            if dS_col:
                S_Q_uncert = torch.tensor(df[dS_col].to_numpy(dtype="float32")[::stride_q], device=device)
            else:
                print(f"  Note: {Path(s_path).name} has no uncertainty column. Using default 0.05")
                S_Q_uncert = torch.full_like(S_Q_target, 0.05)
            
            print(f"  ✓ Loaded S(Q): {len(q_bins)} bins from {Path(s_path).name}")

        return cls(
            q_bins=q_bins,
            S_Q_target=S_Q_target,
            S_Q_uncert=S_Q_uncert,
            r_bins=r_bins,
            T_r_target=T_r_target,
            T_r_uncert=T_r_uncert,
            g_r_target=g_r_target,
            g_r_uncert=g_r_uncert,
            device=device,
        )

    def to(self, new_device: str | torch.device) -> "TargetRDFData":
        """Move all tensors to a new device."""
        new_device = torch.device(new_device)
        return TargetRDFData(
            q_bins=self.q_bins.to(new_device),
            S_Q_target=self.S_Q_target.to(new_device),
            S_Q_uncert=self.S_Q_uncert.to(new_device),
            r_bins=self.r_bins.to(new_device),
            T_r_target=self.T_r_target.to(new_device),
            T_r_uncert=self.T_r_uncert.to(new_device),
            g_r_target=self.g_r_target.to(new_device),
            g_r_uncert=self.g_r_uncert.to(new_device),
            device=new_device,
        )

    # =========================================================================
    # Helper methods
    # =========================================================================
    
    def has_S_Q(self) -> bool:
        """Check if S(Q) data is available."""
        return self.q_bins.numel() > 0 and self.S_Q_target.numel() > 0
    
    def has_T_r(self) -> bool:
        """Check if T(r) target data is available."""
        return self.r_bins.numel() > 0 and self.T_r_target.numel() > 0
    
    def has_g_r(self) -> bool:
        """Check if g(r) target data is available."""
        return self.r_bins.numel() > 0 and self.g_r_target.numel() > 0
    
    def has_r_bins(self) -> bool:
        """Check if r_bins is available (for computing G(r))."""
        return self.r_bins.numel() > 0
    
    def summary(self) -> str:
        """Return a summary string of loaded data."""
        lines = []
        if self.has_S_Q():
            lines.append(f"S(Q): {len(self.q_bins)} bins, Q=[{self.q_bins.min():.2f}, {self.q_bins.max():.2f}] Å⁻¹")
        if self.has_T_r():
            lines.append(f"T(r): {len(self.T_r_target)} bins, r=[{self.r_bins.min():.2f}, {self.r_bins.max():.2f}] Å")
        if self.has_g_r():
            lines.append(f"g(r): {len(self.g_r_target)} bins, r=[{self.r_bins.min():.2f}, {self.r_bins.max():.2f}] Å")
        if self.has_r_bins() and not self.has_T_r() and not self.has_g_r():
            lines.append(f"r_bins: {len(self.r_bins)} bins (auto-generated)")
        return "\n".join(lines) if lines else "No data loaded"
    
    # =========================================================================
    # Backward compatibility aliases
    # =========================================================================
    
    @property
    def F_q_target(self) -> torch.Tensor:
        """Alias for S_Q_target (backward compatibility)."""
        return self.S_Q_target
    
    @property
    def F_q_uncert(self) -> torch.Tensor:
        """Alias for S_Q_uncert (backward compatibility)."""
        return self.S_Q_uncert
    
    def has_q_data(self) -> bool:
        """Alias for has_S_Q (backward compatibility)."""
        return self.has_S_Q()
    
    def has_r_data(self) -> bool:
        """Alias for has_T_r (backward compatibility)."""
        return self.has_T_r()
