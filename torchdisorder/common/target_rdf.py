"""
Target RDF Data Module
======================

Handles loading and storing target diffraction data for structure optimization.

Supported target types:
    - S(Q): Structure factor in reciprocal space (Q-space), oscillates around 1
    - F(Q): Reduced structure factor F(Q) = Q[S(Q) - 1], oscillates around 0
    - T(r): Total correlation function in real space (r-space)
    - g(r): Pair distribution function / radial distribution function (r-space)

Relationships between functions:
    - F(Q) = Q[S(Q) - 1]  (reduced structure factor, → 0 as Q → ∞)
    - S(Q) = 1 + F(Q)/Q   (structure factor, → 1 as Q → ∞)
    - T(r) = 4πρr[g(r) - 1]  where ρ is number density
    - S(Q) is the Fourier transform of g(r)
"""

from __future__ import annotations
import torch
import pandas as pd
import numpy as np

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class TargetRDFData:
    """
    Container for target diffraction data.
    
    Stores ORIGINAL data as loaded, with properties to compute derived quantities.
    This avoids numerical issues from back-and-forth conversions.
    
    Attributes
    ----------
    q_bins : torch.Tensor
        Q values (Å⁻¹) for reciprocal space data
    _q_space_target : torch.Tensor
        Original Q-space target values (either F(Q) or S(Q))
    _q_space_uncert : torch.Tensor
        Original uncertainty values
    _input_is_F_Q : bool
        True if original data was F(Q), False if S(Q)
        
    r_bins : torch.Tensor
        r values (Å) for real space data
    T_r_target : torch.Tensor
        Target T(r) total correlation function values
    T_r_uncert : torch.Tensor
        Uncertainty in T(r)
        
    g_r_target : torch.Tensor
        Target g(r) pair distribution function values (g(r) → 1 as r → ∞)
    g_r_uncert : torch.Tensor
        Uncertainty in g(r)
    
    G_r_target : torch.Tensor
        Target G(r) reduced PDF values: G(r) = 4πρr[g(r) - 1] (G(r) → 0 as r → ∞)
    G_r_uncert : torch.Tensor
        Uncertainty in G(r)
        
    device : torch.device
        Device where tensors are stored
    """
    # Q-space data (original format)
    q_bins: torch.Tensor
    _q_space_target: torch.Tensor  # Original data (F(Q) or S(Q))
    _q_space_uncert: torch.Tensor  # Original uncertainty
    
    # r-space data
    r_bins: torch.Tensor
    T_r_target: torch.Tensor
    T_r_uncert: torch.Tensor
    g_r_target: torch.Tensor
    g_r_uncert: torch.Tensor
    G_r_target: torch.Tensor
    G_r_uncert: torch.Tensor
    
    device: torch.device
    
    # Default field must come last
    _input_is_F_Q: bool = False    # Flag for original data type

    # =========================================================================
    # Computed properties for S(Q) and F(Q)
    # =========================================================================
    
    @property
    def S_Q_target(self) -> torch.Tensor:
        """
        Get S(Q) target values.
        
        If original data was F(Q), converts: S(Q) = 1 + F(Q)/Q
        If original data was S(Q), returns as-is.
        """
        if self._q_space_target.numel() == 0:
            return torch.tensor([], device=self.device)
        
        if self._input_is_F_Q:
            # Convert F(Q) -> S(Q): S(Q) = 1 + F(Q)/Q
            # Avoid division by zero for small Q
            q_safe = torch.clamp(self.q_bins, min=0.1)
            return 1.0 + self._q_space_target / q_safe
        else:
            return self._q_space_target
    
    @property
    def S_Q_uncert(self) -> torch.Tensor:
        """
        Get S(Q) uncertainty.
        
        If original data was F(Q), converts: dS = dF/Q
        If original data was S(Q), returns as-is.
        """
        if self._q_space_uncert.numel() == 0:
            return torch.tensor([], device=self.device)
        
        if self._input_is_F_Q:
            # Propagate uncertainty: dS = dF/Q
            q_safe = torch.clamp(self.q_bins, min=0.1)
            return self._q_space_uncert / q_safe
        else:
            return self._q_space_uncert
    
    @property
    def F_q_target(self) -> torch.Tensor:
        """
        Get F(Q) = Q[S(Q) - 1] target values.
        
        If original data was F(Q), returns as-is.
        If original data was S(Q), converts: F(Q) = Q * (S(Q) - 1)
        
        Note: F(Q) → 0 as Q → ∞ (reduced structure factor)
        """
        if self._q_space_target.numel() == 0:
            return torch.tensor([], device=self.device)
        
        if self._input_is_F_Q:
            return self._q_space_target
        else:
            # Convert S(Q) -> F(Q): F(Q) = Q * (S(Q) - 1)
            return self.q_bins * (self._q_space_target - 1.0)
    
    @property
    def F_q_uncert(self) -> torch.Tensor:
        """
        Get F(Q) uncertainty.
        
        If original data was F(Q), returns as-is.
        If original data was S(Q), converts: dF = Q * dS
        """
        if self._q_space_uncert.numel() == 0:
            return torch.tensor([], device=self.device)
        
        if self._input_is_F_Q:
            return self._q_space_uncert
        else:
            # Propagate uncertainty: dF = Q * dS
            return self.q_bins * self._q_space_uncert

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
        Load target data from a configuration dictionary.
        
        The dictionary should contain paths to CSV files with experimental data.
        Data is kept in ORIGINAL format (F(Q) or S(Q)) and converted on-the-fly.
        
        Parameters
        ----------
        cfg : dict
            Configuration dictionary with keys like 's_of_q_path', 't_of_r_path', etc.
        device : str or torch.device
            Device to store tensors on
        stride_r : int
            Subsampling factor for r-space data
        stride_q : int
            Subsampling factor for Q-space data
        """
        device = torch.device(device)
        
        # Helper functions
        def load_csv_safe(path):
            """Load CSV with flexible delimiter detection."""
            try:
                df = pd.read_csv(path)
                if len(df.columns) == 1:
                    df = pd.read_csv(path, sep=r'\s+')
                return df
            except Exception as e:
                raise IOError(f"Failed to load {path}: {e}")
        
        def find_column(df, possible_names, required=True):
            """Find column by checking possible names (case-insensitive)."""
            df_cols_lower = {c.lower(): c for c in df.columns}
            for name in possible_names:
                if name.lower() in df_cols_lower:
                    return df_cols_lower[name.lower()]
            if required:
                raise KeyError(f"Could not find column matching {possible_names} in {list(df.columns)}")
            return None
        
        # Initialize empty tensors
        q_bins = torch.tensor([], dtype=torch.float32, device=device)
        q_space_target = torch.tensor([], dtype=torch.float32, device=device)
        q_space_uncert = torch.tensor([], dtype=torch.float32, device=device)
        detected_F_Q = False
        
        r_bins = torch.tensor([], dtype=torch.float32, device=device)
        T_r_target = torch.tensor([], dtype=torch.float32, device=device)
        T_r_uncert = torch.tensor([], dtype=torch.float32, device=device)
        g_r_target = torch.tensor([], dtype=torch.float32, device=device)
        g_r_uncert = torch.tensor([], dtype=torch.float32, device=device)
        G_r_target = torch.tensor([], dtype=torch.float32, device=device)
        G_r_uncert = torch.tensor([], dtype=torch.float32, device=device)
        
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
            
            # Optional uncertainty
            dT_col = find_column(df, ['dT', 'uncertainty', 'error'], required=False)
            if dT_col:
                T_r_uncert = torch.tensor(df[dT_col].to_numpy(dtype="float32")[::stride_r], device=device)
            else:
                T_r_uncert = torch.full_like(T_r_target, 0.1)
            
            print(f"  ✓ Loaded T(r): {len(r_bins)} bins from {Path(t_path).name}")
        
        # =====================================================================
        # Load g(r) or G(r) data
        # =====================================================================
        g_path = cfg.get("g_of_r_path")
        if g_path and str(g_path).lower() not in ['null', 'none', ''] and Path(g_path).exists():
            df = load_csv_safe(g_path)
            
            r_col = find_column(df, ['r', 'R'])
            
            # Try to find G(r) column first (reduced PDF), then g(r)
            G_col = find_column(df, ['Gr', 'G_r', 'G(r)', 'G'], required=False)
            g_col = find_column(df, ['g', 'gr', 'g(r)', 'rdf'], required=False)
            
            r_data = torch.tensor(df[r_col].to_numpy(dtype="float32")[::stride_r], device=device)
            
            # Determine if data is G(r) or g(r)
            if G_col is not None:
                # Data is G(r) = 4πρr[g(r) - 1]
                G_r_target = torch.tensor(df[G_col].to_numpy(dtype="float32")[::stride_r], device=device)
                # For g(r), we'd need number density ρ to convert: g(r) = G(r)/(4πρr) + 1
                # Store as G(r) and leave g(r) empty for now
                g_r_target = torch.tensor([], dtype=torch.float32, device=device)
                g_r_uncert = torch.tensor([], dtype=torch.float32, device=device)
                
                # Optional uncertainty
                dG_col = find_column(df, ['dG', 'dGr', 'uncertainty', 'error'], required=False)
                if dG_col:
                    loaded_uncert = torch.tensor(df[dG_col].to_numpy(dtype="float32")[::stride_r], device=device)
                    # Check if loaded uncertainty is valid (not all zeros/negligible)
                    if loaded_uncert.abs().max() > 1e-6:
                        G_r_uncert = loaded_uncert
                        print(f"    G(r) uncertainty: from column '{dG_col}'")
                    else:
                        G_r_uncert = torch.full_like(G_r_target, 0.05)
                        print(f"    Note: '{dG_col}' column has negligible values, using default uncertainty 0.05")
                else:
                    G_r_uncert = torch.full_like(G_r_target, 0.05)
                
                print(f"  ✓ Loaded G(r): {len(r_data)} bins from {Path(g_path).name}")
                print(f"    G(r) range: [{G_r_target.min():.3f}, {G_r_target.max():.3f}]")
                
            elif g_col is not None:
                # Data is g(r), standard pair distribution function
                g_r_target = torch.tensor(df[g_col].to_numpy(dtype="float32")[::stride_r], device=device)
                G_r_target = torch.tensor([], dtype=torch.float32, device=device)
                G_r_uncert = torch.tensor([], dtype=torch.float32, device=device)
                
                # Optional uncertainty
                dg_col = find_column(df, ['dg', 'uncertainty', 'error'], required=False)
                if dg_col:
                    loaded_uncert = torch.tensor(df[dg_col].to_numpy(dtype="float32")[::stride_r], device=device)
                    # Check if loaded uncertainty is valid (not all zeros/negligible)
                    if loaded_uncert.abs().max() > 1e-6:
                        g_r_uncert = loaded_uncert
                        print(f"    g(r) uncertainty: from column '{dg_col}'")
                    else:
                        g_r_uncert = torch.full_like(g_r_target, 0.05)
                        print(f"    Note: '{dg_col}' column has negligible values, using default uncertainty 0.05")
                else:
                    g_r_uncert = torch.full_like(g_r_target, 0.05)
                
                print(f"  ✓ Loaded g(r): {len(r_data)} bins from {Path(g_path).name}")
                print(f"    g(r) range: [{g_r_target.min():.3f}, {g_r_target.max():.3f}]")
            else:
                raise ValueError(f"Could not find g(r) or G(r) column in {g_path}. "
                               f"Available columns: {list(df.columns)}")
            
            # Use this r_bins if not already loaded
            if r_bins.numel() == 0:
                r_bins = r_data
        
        # =====================================================================
        # Auto-generate r_bins if not loaded but needed
        # =====================================================================
        r_min = cfg.get("r_min", 0.01)
        r_max = cfg.get("r_max", 10.0)
        
        if r_bins.numel() == 0:
            n_r_bins = cfg.get("n_r_bins", 500)
            r_bins = torch.linspace(r_min, r_max, n_r_bins, device=device, dtype=torch.float32)
            print(f"  ⚙ Auto-generated r_bins: {n_r_bins} bins, r=[{r_min}, {r_max}] Å")
        else:
            # Filter existing r_bins to [r_min, r_max] range
            r_mask = (r_bins >= r_min) & (r_bins <= r_max)
            if r_mask.sum() < r_bins.numel():
                old_len = r_bins.numel()
                r_bins = r_bins[r_mask]
                if T_r_target.numel() > 0:
                    T_r_target = T_r_target[r_mask]
                    if T_r_uncert.numel() > 0:
                        T_r_uncert = T_r_uncert[r_mask]
                if g_r_target.numel() > 0:
                    g_r_target = g_r_target[r_mask]
                    if g_r_uncert.numel() > 0:
                        g_r_uncert = g_r_uncert[r_mask]
                if G_r_target.numel() > 0:
                    G_r_target = G_r_target[r_mask]
                    if G_r_uncert.numel() > 0:
                        G_r_uncert = G_r_uncert[r_mask]
                print(f"  ⚙ Filtered r_bins to [{r_min}, {r_max}] Å: {old_len} → {r_bins.numel()} bins")
        
        # =====================================================================
        # Load S(Q) or F(Q) data - KEEP ORIGINAL FORMAT
        # =====================================================================
        s_path = cfg.get("s_of_q_path") or cfg.get("f_of_q_path")
        input_is_F_Q = cfg.get("input_is_F_Q", None)  # None = auto-detect
        
        if s_path and str(s_path).lower() not in ['null', 'none', ''] and Path(s_path).exists():
            df = load_csv_safe(s_path)
            
            Q_col = find_column(df, ['Q', 'q'])
            S_col = find_column(df, ['F', 'SQ', 'S', 'S(Q)', 'F(Q)'])
            
            q_bins = torch.tensor(df[Q_col].to_numpy(dtype="float32")[::stride_q], device=device)
            q_space_target = torch.tensor(df[S_col].to_numpy(dtype="float32")[::stride_q], device=device)
            
            # Optional uncertainty
            dS_col = find_column(df, ['dF', 'dSQ', 'dS', 'uncertainty', 'error'], required=False)
            if dS_col:
                q_space_uncert = torch.tensor(df[dS_col].to_numpy(dtype="float32")[::stride_q], device=device)
            else:
                print(f"  Note: {Path(s_path).name} has no uncertainty column. Using default 0.05")
                q_space_uncert = torch.full_like(q_space_target, 0.05)
            
            # Auto-detect F(Q) vs S(Q) if not specified
            # F(Q) oscillates around 0 and → 0 as Q → ∞
            # S(Q) oscillates around 1 and → 1 as Q → ∞
            if input_is_F_Q is None:
                # Check mean of high-Q region (last 20%)
                high_q_idx = int(0.8 * len(q_space_target))
                high_q_mean = q_space_target[high_q_idx:].mean().item()
                
                # If mean is closer to 0 than to 1, it's likely F(Q)
                detected_F_Q = abs(high_q_mean) < abs(high_q_mean - 1)
                
                if detected_F_Q:
                    print(f"  ⚙ Auto-detected as F(Q) (high-Q mean ≈ {high_q_mean:.3f})")
                else:
                    print(f"  ⚙ Auto-detected as S(Q) (high-Q mean ≈ {high_q_mean:.3f})")
            else:
                detected_F_Q = input_is_F_Q
                data_type = "F(Q)" if detected_F_Q else "S(Q)"
                print(f"  ✓ Loaded as {data_type} (user-specified): {len(q_bins)} bins from {Path(s_path).name}")
            
            # Filter q_bins to [q_min, q_max] range if specified
            q_min = cfg.get("q_min", None)
            q_max = cfg.get("q_max", None)
            
            if q_min is not None or q_max is not None:
                q_min_val = q_min if q_min is not None else q_bins.min().item()
                q_max_val = q_max if q_max is not None else q_bins.max().item()
                
                q_mask = (q_bins >= q_min_val) & (q_bins <= q_max_val)
                if q_mask.sum() < q_bins.numel():
                    old_len = q_bins.numel()
                    q_bins = q_bins[q_mask]
                    q_space_target = q_space_target[q_mask]
                    if q_space_uncert.numel() > 0:
                        q_space_uncert = q_space_uncert[q_mask]
                    print(f"  ⚙ Filtered q_bins to [{q_min_val:.2f}, {q_max_val:.2f}] Å⁻¹: {old_len} → {q_bins.numel()} bins")

        return cls(
            q_bins=q_bins,
            _q_space_target=q_space_target,
            _q_space_uncert=q_space_uncert,
            r_bins=r_bins,
            T_r_target=T_r_target,
            T_r_uncert=T_r_uncert,
            g_r_target=g_r_target,
            g_r_uncert=g_r_uncert,
            G_r_target=G_r_target,
            G_r_uncert=G_r_uncert,
            device=device,
            _input_is_F_Q=detected_F_Q,
        )

    def to(self, new_device: str | torch.device) -> "TargetRDFData":
        """Move all tensors to a new device."""
        new_device = torch.device(new_device)
        return TargetRDFData(
            q_bins=self.q_bins.to(new_device),
            _q_space_target=self._q_space_target.to(new_device),
            _q_space_uncert=self._q_space_uncert.to(new_device),
            r_bins=self.r_bins.to(new_device),
            T_r_target=self.T_r_target.to(new_device),
            T_r_uncert=self.T_r_uncert.to(new_device),
            g_r_target=self.g_r_target.to(new_device),
            g_r_uncert=self.g_r_uncert.to(new_device),
            G_r_target=self.G_r_target.to(new_device),
            G_r_uncert=self.G_r_uncert.to(new_device),
            device=new_device,
            _input_is_F_Q=self._input_is_F_Q,
        )

    # =========================================================================
    # Helper methods
    # =========================================================================
    
    def has_S_Q(self) -> bool:
        """Check if S(Q) data is available."""
        return self.q_bins.numel() > 0 and self._q_space_target.numel() > 0
    
    def has_F_Q(self) -> bool:
        """Check if F(Q) data is available."""
        return self.has_S_Q()  # F(Q) can be derived from S(Q) and vice versa
    
    def has_T_r(self) -> bool:
        """Check if T(r) target data is available."""
        return self.r_bins.numel() > 0 and self.T_r_target.numel() > 0
    
    def has_g_r(self) -> bool:
        """Check if g(r) target data is available."""
        return self.r_bins.numel() > 0 and self.g_r_target.numel() > 0
    
    def has_G_r(self) -> bool:
        """Check if G(r) target data is available."""
        return self.r_bins.numel() > 0 and self.G_r_target.numel() > 0
    
    def has_r_bins(self) -> bool:
        """Check if r_bins is available (for computing G(r))."""
        return self.r_bins.numel() > 0
    
    def input_was_F_Q(self) -> bool:
        """Check if original input data was F(Q) format."""
        return self._input_is_F_Q
    
    def summary(self) -> str:
        """Return a summary string of loaded data."""
        lines = []
        if self.has_S_Q():
            data_type = "F(Q)" if self._input_is_F_Q else "S(Q)"
            lines.append(f"Q-space: {len(self.q_bins)} bins, Q=[{self.q_bins.min():.2f}, {self.q_bins.max():.2f}] Å⁻¹ (original: {data_type})")
        if self.has_T_r():
            lines.append(f"T(r): {len(self.T_r_target)} bins, r=[{self.r_bins.min():.2f}, {self.r_bins.max():.2f}] Å")
        if self.has_g_r():
            lines.append(f"g(r): {len(self.g_r_target)} bins, r=[{self.r_bins.min():.2f}, {self.r_bins.max():.2f}] Å")
        if self.has_G_r():
            lines.append(f"G(r): {len(self.G_r_target)} bins, r=[{self.r_bins.min():.2f}, {self.r_bins.max():.2f}] Å")
        if self.has_r_bins() and not self.has_T_r() and not self.has_g_r() and not self.has_G_r():
            lines.append(f"r_bins: {len(self.r_bins)} bins (auto-generated)")
        return "\n".join(lines) if lines else "No data loaded"
    
    # =========================================================================
    # Backward compatibility aliases
    # =========================================================================
    
    def has_q_data(self) -> bool:
        """Alias for has_S_Q (backward compatibility)."""
        return self.has_S_Q()
    
    def has_r_data(self) -> bool:
        """Alias for has_T_r (backward compatibility)."""
        return self.has_T_r()
