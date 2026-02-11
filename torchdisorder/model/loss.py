"""
model/loss.py – Loss Functions for Structure Optimization
=========================================================

Provides chi-squared and related loss functions for matching computed
spectra to experimental targets.

Supports multiple target types:
    - S_Q: Structure factor
    - T_r: Total correlation function
    - g_r: Pair distribution function
    - F_Q: Reduced structure factor
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from torchdisorder.common.target_rdf import TargetRDFData


# =============================================================================
# Chi-Squared Utility
# =============================================================================

def chi_squared(
    estimate: torch.Tensor,
    target: torch.Tensor,
    uncertainty: Union[torch.Tensor, float],
    normalize: bool = False,
) -> torch.Tensor:
    """
    Compute chi-squared statistic.
    
    χ² = Σ (estimate - target)² / σ²
    
    Args:
        estimate: Predicted values
        target: Target values
        uncertainty: Per-point or constant uncertainty
        normalize: If True, return χ²/N (reduced chi-squared)
    
    Returns:
        Scalar chi-squared value
    """
    # Handle empty tensors
    if estimate.numel() == 0 or target.numel() == 0:
        return torch.tensor(float('inf'), device=estimate.device, dtype=estimate.dtype)
    
    # Flatten
    estimate = estimate.reshape(-1)
    target = target.reshape(-1)
    
    # Handle size mismatch
    if estimate.shape[0] != target.shape[0]:
        min_len = min(estimate.shape[0], target.shape[0])
        estimate = estimate[:min_len]
        target = target[:min_len]
    
    # Handle uncertainty
    if isinstance(uncertainty, (float, int)):
        sigma = torch.full_like(estimate, uncertainty)
    else:
        sigma = uncertainty.reshape(-1)
        if sigma.shape[0] != estimate.shape[0]:
            sigma = sigma[:estimate.shape[0]] if sigma.numel() > 1 else sigma.expand_as(estimate)
    
    # Clamp to avoid division by zero
    sigma = torch.clamp(sigma, min=1e-6)
    
    chi2 = torch.sum((estimate - target) ** 2 / sigma ** 2)
    
    if torch.isnan(chi2):
        return torch.tensor(float('inf'), device=estimate.device, dtype=estimate.dtype)
    
    if normalize:
        return chi2 / estimate.numel()
    
    return chi2


def r_squared(
    estimate: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute coefficient of determination R².
    
    R² = 1 - SS_res / SS_tot
    """
    estimate = estimate.reshape(-1)
    target = target.reshape(-1)
    
    ss_res = torch.sum((target - estimate) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    
    return 1 - ss_res / (ss_tot + 1e-10)


def rmse(
    estimate: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute root mean squared error."""
    estimate = estimate.reshape(-1)
    target = target.reshape(-1)
    return torch.sqrt(torch.mean((estimate - target) ** 2))


# =============================================================================
# Loss Function for Cooper
# =============================================================================

class CooperLoss(nn.Module):
    """
    Loss function for constrained optimization with Cooper.
    
    Computes chi-squared loss between predicted and target spectra.
    
    Supported targets:
        - S_Q: Structure factor
        - T_r: Total correlation function
        - g_r: Pair distribution function
        - F_Q: Reduced structure factor
    
    Args:
        target_data: TargetRDFData with target spectra
        target_type: Primary target ('S_Q', 'T_r', 'g_r', 'G_r', 'F_Q')
        device: Computation device
        uncertainty_floor: Minimum uncertainty value
    """
    
    VALID_TARGETS = ['S_Q', 'T_r', 'g_r', 'G_r', 'F_Q']
    
    def __init__(
        self,
        target_data: TargetRDFData,
        target_type: str = 'S_Q',
        device: str = 'cuda',
        uncertainty_floor: float = 0.01,
    ):
        super().__init__()
        
        if target_type not in self.VALID_TARGETS:
            raise ValueError(f"target_type must be one of {self.VALID_TARGETS}")
        
        self.target_data = target_data
        self.target_type = target_type
        self.device = torch.device(device)
        self.uncertainty_floor = uncertainty_floor
        
        self._logged = False
    
    def forward(self, results: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss from model results.
        
        Args:
            results: Dict from XRDModel with spectra
        
        Returns:
            Dict with 'total_loss', 'chi2_loss', and individual losses
        """
        losses = {}
        
        # S(Q) loss
        if 'S_Q' in results and self.target_data.has_S_Q():
            pred = results['S_Q']
            target = self.target_data.S_Q_target
            uncert = self.target_data.S_Q_uncert
            if uncert is None or uncert.numel() == 0:
                uncert = self.uncertainty_floor
            losses['S_Q_loss'] = chi_squared(pred, target, uncert)
        
        # T(r) loss
        if 'T_r' in results and self.target_data.has_T_r():
            pred = results['T_r']
            target = self.target_data.T_r_target
            uncert = self.target_data.T_r_uncert
            if uncert is None or uncert.numel() == 0:
                uncert = self.uncertainty_floor
            losses['T_r_loss'] = chi_squared(pred, target, uncert)
        
        # g(r) loss
        if 'g_r' in results and self.target_data.has_g_r():
            pred = results['g_r']
            target = self.target_data.g_r_target
            uncert = self.target_data.g_r_uncert
            if uncert is None or uncert.numel() == 0:
                uncert = self.uncertainty_floor
            losses['g_r_loss'] = chi_squared(pred, target, uncert)
        
        # G(r) loss (reduced PDF)
        if 'G_r' in results and self.target_data.has_G_r():
            pred = results['G_r']
            target = self.target_data.G_r_target
            uncert = self.target_data.G_r_uncert
            if uncert is None or uncert.numel() == 0:
                uncert = self.uncertainty_floor
            losses['G_r_loss'] = chi_squared(pred, target, uncert)
        
        # F(Q) loss
        if 'F_Q' in results and self.target_data.has_F_Q():
            pred = results['F_Q']
            target = self.target_data.F_q_target
            uncert = self.target_data.F_q_uncert
            if uncert is None or uncert.numel() == 0:
                uncert = self.uncertainty_floor
            losses['F_Q_loss'] = chi_squared(pred, target, uncert)
        
        # Select primary loss
        primary_key = f'{self.target_type}_loss'
        if primary_key in losses:
            total_loss = losses[primary_key]
        else:
            # Fallback to first available
            available = [k for k in losses.keys()]
            if available:
                total_loss = losses[available[0]]
                if not self._logged:
                    print(f"Warning: {self.target_type} not available, using {available[0]}")
                    self._logged = True
            else:
                total_loss = torch.tensor(1e6, device=self.device, requires_grad=True)
        
        # Log once
        if not self._logged:
            print(f"\nCooperLoss:")
            print(f"  Target type: {self.target_type}")
            print(f"  Available losses: {list(losses.keys())}")
            print(f"  Primary loss key: {primary_key}")
            self._logged = True
        
        return {
            'total_loss': total_loss,
            'chi2_loss': total_loss,
            **losses,
        }


# =============================================================================
# Augmented Lagrangian Loss (Legacy)
# =============================================================================

@dataclass
class AugLagHyper:
    """Hyperparameters for augmented Lagrangian optimization."""
    rho: float = 1e-3
    rho_factor: float = 5.0
    tol: float = 1e-4
    update_every: int = 10
    scale_scatt_init: float = 0.02
    scale_q_init: float = 1.0
    q_target: float = 0.7
    q_uncert: float = 0.05
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'AugLagHyper':
        cfg = OmegaConf.load(path)
        return cls(**OmegaConf.to_container(cfg, resolve=True))


class AugLagLoss(nn.Module):
    """
    Augmented Lagrangian loss for structure optimization.
    
    Legacy implementation for backward compatibility.
    Consider using CooperLoss with EnvironmentConstrainedOptimizer instead.
    """
    
    def __init__(
        self,
        target_data: TargetRDFData,
        hyper: AugLagHyper,
        device: str = 'cuda',
    ):
        super().__init__()
        
        self.target_data = target_data
        self.hyper = hyper
        self.device = torch.device(device)
        
        # Augmented Lagrangian variables
        self.rho = hyper.rho
        self.lambda_corr = torch.tensor(0.0, device=device)
        
        # Scaling factors
        self.scale_scatt = torch.tensor(hyper.scale_scatt_init, device=device)
        self.scale_q = torch.tensor(hyper.scale_q_init, device=device)
        
        self.iter_counter = 0
    
    def forward(self, results: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute augmented Lagrangian loss."""
        S_Q = results.get('S_Q')
        T_r = results.get('T_r')
        
        losses = {}
        
        # Structure factor loss
        if S_Q is not None and self.target_data.has_S_Q():
            chi2_scatt = chi_squared(
                S_Q, 
                self.target_data.S_Q_target,
                self.target_data.S_Q_uncert or 0.05
            ) / S_Q.numel()
            losses['chi2_scatt'] = chi2_scatt
        
        # Correlation function loss
        if T_r is not None and self.target_data.has_T_r():
            chi2_corr = chi_squared(
                T_r,
                self.target_data.T_r_target,
                0.05
            ) / T_r.numel()
            losses['chi2_corr'] = chi2_corr
        
        # Combine losses
        total = torch.tensor(0.0, device=self.device)
        if 'chi2_scatt' in losses:
            total = total + self.scale_scatt * losses['chi2_scatt']
        if 'chi2_corr' in losses:
            # Augmented Lagrangian for correlation constraint
            g = losses['chi2_corr'] - 0.1  # g(x) ≤ 0 means chi2_corr ≤ 0.1
            total = total + self.lambda_corr * g + (self.rho / 2) * g ** 2
        
        losses['total_loss'] = total
        return losses
    
    def update_penalties(self, loss_dict: Dict[str, torch.Tensor]):
        """Update Lagrange multipliers and penalties."""
        if 'chi2_corr' not in loss_dict:
            return
        
        g_val = loss_dict['chi2_corr'].detach()
        
        with torch.no_grad():
            # Update multiplier
            self.lambda_corr = self.lambda_corr + self.rho * g_val
            
            # Increase penalty if constraint still violated
            self.iter_counter += 1
            if self.iter_counter % self.hyper.update_every == 0:
                if g_val.abs() > self.hyper.tol:
                    self.rho *= self.hyper.rho_factor


# =============================================================================
# Backward Compatibility Classes (v5)
# =============================================================================

class ChiSquaredObjective(nn.Module):
    """
    Chi-squared objective combining T(r) and S(Q) losses.
    
    For backward compatibility with v5.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, state) -> torch.Tensor:
        out = self.model(state)
        T_r = out["T_r"]
        S_Q = out["S_Q"]
        target = self.model.rdf_data

        chi2_corr = chi_squared(T_r, target.T_r_target, 0.05) / T_r.numel()
        chi2_scatt = chi_squared(S_Q, target.F_q_target, target.F_q_uncert) / S_Q.numel()

        return chi2_corr + chi2_scatt


class ConstraintChiSquared(nn.Module):
    """
    Constraint that chi-squared is below threshold.
    
    For backward compatibility with v5.
    """
    def __init__(self, model: nn.Module, chi2_threshold: float = 0.1):
        super().__init__()
        self.model = model
        self.chi2_threshold = chi2_threshold

    def forward(self, state) -> List[torch.Tensor]:
        out = self.model(state)
        T_r = out["T_r"]
        S_Q = out["S_Q"]
        target = self.model.rdf_data

        chi2_corr = chi_squared(T_r, target.T_r_target, 0.05) / T_r.numel()
        chi2_scatt = chi_squared(S_Q, target.F_q_target, target.F_q_uncert) / S_Q.numel()

        return [chi2_corr - self.chi2_threshold, chi2_scatt - self.chi2_threshold]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'chi_squared',
    'r_squared',
    'rmse',
    'CooperLoss',
    'AugLagHyper',
    'AugLagLoss',
    # Backward compatibility
    'ChiSquaredObjective',
    'ConstraintChiSquared',
]
