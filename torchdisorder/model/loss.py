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
from torchdisorder.model.aggregators import Aggregator, build_aggregator  # noqa: F401


# =============================================================================
# Chi-Squared Utility
# =============================================================================

def chi_squared(
    estimate: torch.Tensor,
    target: torch.Tensor,
    uncertainty: Union[torch.Tensor, float],
    normalize: bool = False,
    sigma_mode: str = "data",
    sigma_floor_frac: float = 0.0,
) -> torch.Tensor:
    """
    Compute chi-squared statistic.

    χ² = Σ (estimate - target)² / σ²

    Args:
        estimate: Predicted values
        target: Target values
        uncertainty: Per-point or constant uncertainty
        normalize: If True, return χ²/N (reduced chi-squared)
        sigma_mode: How to treat the uncertainties.
            ``"data"``       — use them as given (default, historical behaviour).
            ``"fractional"`` — use ``max(sigma, sigma_floor_frac * max|target|)``.
        sigma_floor_frac: Fraction of the target's peak amplitude used as an error
            floor in ``"fractional"`` mode.

    Why ``sigma_mode`` exists
    ------------------------
    The SiO2 target carries a median σ of 6e-4, and ``source.txt`` records that
    zero-valued uncertainties were hand-patched to 1e-7.  Forwarding the published
    GAP glass — a structure independently validated as correct — gives an RMS
    residual of ~0.04, so χ²/point is ~4800 even for a structure we believe.  χ² can
    then never approach 1, and minimising it drives the optimizer to chase
    differences far below the real accuracy of either the data or the model.  A
    fractional floor makes χ² interpretable again; it is a modelling choice and
    should be stated as one in any write-up.

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

    # Optional error floor tied to the signal's own amplitude.  Without it, points
    # whose published sigma is near zero dominate chi^2 entirely: two such points
    # in the SiO2 target can contribute ~1e12 on their own.
    if sigma_mode == "fractional" and sigma_floor_frac > 0.0:
        floor = sigma_floor_frac * target.abs().max()
        sigma = torch.clamp(sigma, min=float(floor))

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

    Computes chi-squared loss between predicted and target spectra, with an
    optional F_IS (local inversion symmetry) regularization term that steers
    the optimizer toward a target mean F_IS value derived from a reference
    structure (e.g. melt-quench MD or a prior TorchDisorder run).

    Supported scattering targets:
        S_Q, T_r, g_r, G_r, F_Q

    F_IS regularization:
        When ``fis_target`` is set, the loss gains an additional term::

            fis_weight * (mean_fis - fis_target)²

        where ``mean_fis`` is computed over ``fis_central_z`` atoms at each
        forward pass.  Because F_IS is fully differentiable, gradients flow
        directly into atomic positions alongside the scattering chi-squared.

        Reference: Milkus & Zaccone, Phys. Rev. B 93, 094204 (2016).
        https://doi.org/10.1103/PhysRevB.93.094204

    Args:
        target_data: TargetRDFData with experimental spectra.
        target_type: Primary scattering target ('S_Q', 'T_r', 'g_r', 'G_r', 'F_Q').
        device: Computation device.
        uncertainty_floor: Minimum uncertainty value to avoid division by zero.
        fis_target: Target mean F_IS value.  ``None`` disables the F_IS term.
            Use the F_IS mean of a reference glass (e.g. ~0.0 for a-SiO₂).
        fis_weight: Weight of the F_IS loss relative to the scattering chi².
        fis_cutoff: Neighbor cutoff in Å for F_IS (should match the first
            coordination shell of the central species).
        fis_central_z: Atomic number of the central atoms over which F_IS is
            averaged (e.g. 14 for Si in SiO₂, 15 for P in Li-P-S).
        fis_neighbor_z: Atomic number used to filter neighbors (e.g. 8 for O).
            ``None`` uses all neighbors within the cutoff.
        fis_mode: Weighting scheme — ``'variable_R'`` (recommended, JCTC 2026)
            or ``'milkus2016'`` (original uniform weights, PRB 2016).
        fis_max_neighbors: Max neighbors per atom for the F_IS calculator.
        aggregator: Optional ``Aggregator`` instance from
            ``torchdisorder.model.aggregators``.  When set, the scattering
            chi-squared and F_IS loss are combined via the aggregator rather
            than a fixed weighted sum.  Build one with::

                from torchdisorder.model.aggregators import build_aggregator
                agg = build_aggregator('relobralo', params=[], num_losses=2)

            Recommended strategies for TorchDisorder:
              - ``'relobralo'``  — tracks relative progress from t=0 (robust default)
              - ``'brdr'``       — equalises relative decay rates of all terms
              - ``'ema'``        — lightweight EMA magnitude normalisation
              - ``'soft_adapt'`` — up-weights whichever term is improving slowest

            Gradient-based strategies (``grad_norm``, ``lr_annealing``, ``ntk``)
            work but require an extra backward pass per term per step — expensive
            for large structures.
    """

    VALID_TARGETS = ['S_Q', 'T_r', 'g_r', 'G_r', 'F_Q']

    def __init__(
        self,
        target_data: TargetRDFData,
        target_type: str = 'S_Q',
        device: str = 'cuda',
        uncertainty_floor: float = 0.01,
        sigma_mode: str = 'data',
        sigma_floor_frac: float = 0.02,
        normalize_for_weighting: bool = False,
        # F_IS regularization
        fis_target: Optional[float] = None,
        fis_weight: float = 1.0,
        fis_cutoff: float = 2.2,
        fis_central_z: int = 14,
        fis_neighbor_z: Optional[int] = None,
        fis_mode: str = 'variable_R',
        fis_max_neighbors: int = 16,
        # Adaptive loss weighting
        aggregator: Optional[Aggregator] = None,
    ):
        super().__init__()

        if target_type not in self.VALID_TARGETS:
            raise ValueError(f"target_type must be one of {self.VALID_TARGETS}")

        self.target_data = target_data
        self.target_type = target_type
        self.device = torch.device(device)
        self.uncertainty_floor = uncertainty_floor
        self.sigma_mode = sigma_mode
        self.sigma_floor_frac = sigma_floor_frac
        # Divide chi^2 by a running scale before combining it with the other
        # terms, so the aggregator and the constraint penalty see an O(1)
        # objective.  Rescaling the penalty to chase an chi^2 of ~1e8 was tried
        # and measured worse than leaving the penalty alone; normalising the
        # objective instead fixes the mismatch at its source and leaves the
        # aggregators operating on the magnitudes they were designed for.
        self.normalize_for_weighting = normalize_for_weighting
        self._chi2_scale = None

        # F_IS regularization
        self.fis_target = fis_target
        self.fis_weight = fis_weight
        self.fis_central_z = fis_central_z
        self.fis_neighbor_z = fis_neighbor_z
        self._fis_calc = None
        if fis_target is not None:
            from torchdisorder.engine.order_params import TorchSimOrderParameters  # lazy — avoids circular import
            self._fis_calc = TorchSimOrderParameters(
                cutoff=fis_cutoff,
                device=device,
                max_neighbors=fis_max_neighbors,
                fis_mode=fis_mode,
            )

        # Adaptive aggregator (optional)
        self.aggregator = aggregator
        self._step = 0          # internal step counter for the aggregator
        self._logged = False

    def forward(
        self,
        results: Dict[str, torch.Tensor],
        state=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss from model results.

        Args:
            results: Dict from XRDModel containing scattering spectra.
            state: Optional torch_sim SimState.  Required when F_IS
                regularization is enabled (``fis_target`` is not None).

        Returns:
            Dict with keys 'total_loss', 'chi2_loss', optional 'fis_loss',
            and individual per-target losses.
        """
        # Extract state stashed by optimizer (if present); explicit arg wins
        if state is None:
            state = results.pop('_sim_state', None)
        else:
            results.pop('_sim_state', None)  # discard stashed copy if explicit provided

        losses = {}

        # S(Q) loss
        if 'S_Q' in results and self.target_data.has_S_Q():
            pred = results['S_Q']
            target = self.target_data.S_Q_target
            uncert = self.target_data.S_Q_uncert
            if uncert is None or uncert.numel() == 0:
                uncert = self.uncertainty_floor
            losses['S_Q_loss'] = chi_squared(
                pred, target, uncert,
                sigma_mode=self.sigma_mode,
                sigma_floor_frac=self.sigma_floor_frac,
            )
            losses['S_Q_rms'] = rmse(pred, target).detach()
        
        # T(r) loss
        if 'T_r' in results and self.target_data.has_T_r():
            pred = results['T_r']
            target = self.target_data.T_r_target
            uncert = self.target_data.T_r_uncert
            if uncert is None or uncert.numel() == 0:
                uncert = self.uncertainty_floor
            losses['T_r_loss'] = chi_squared(
                pred, target, uncert,
                sigma_mode=self.sigma_mode,
                sigma_floor_frac=self.sigma_floor_frac,
            )
            losses['T_r_rms'] = rmse(pred, target).detach()
        
        # g(r) loss
        if 'g_r' in results and self.target_data.has_g_r():
            pred = results['g_r']
            target = self.target_data.g_r_target
            uncert = self.target_data.g_r_uncert
            if uncert is None or uncert.numel() == 0:
                uncert = self.uncertainty_floor
            losses['g_r_loss'] = chi_squared(
                pred, target, uncert,
                sigma_mode=self.sigma_mode,
                sigma_floor_frac=self.sigma_floor_frac,
            )
            losses['g_r_rms'] = rmse(pred, target).detach()
        
        # G(r) loss (reduced PDF)
        if 'G_r' in results and self.target_data.has_G_r():
            pred = results['G_r']
            target = self.target_data.G_r_target
            uncert = self.target_data.G_r_uncert
            if uncert is None or uncert.numel() == 0:
                uncert = self.uncertainty_floor
            losses['G_r_loss'] = chi_squared(
                pred, target, uncert,
                sigma_mode=self.sigma_mode,
                sigma_floor_frac=self.sigma_floor_frac,
            )
            losses['G_r_rms'] = rmse(pred, target).detach()
        
        # F(Q) loss
        if 'F_Q' in results and self.target_data.has_F_Q():
            pred = results['F_Q']
            target = self.target_data.F_q_target
            uncert = self.target_data.F_q_uncert
            if uncert is None or uncert.numel() == 0:
                uncert = self.uncertainty_floor
            losses['F_Q_loss'] = chi_squared(
                pred, target, uncert,
                sigma_mode=self.sigma_mode,
                sigma_floor_frac=self.sigma_floor_frac,
            )
            losses['F_Q_rms'] = rmse(pred, target).detach()
        
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
        
        # F_IS regularization term
        if self._fis_calc is not None and state is not None:
            central_idx = torch.where(state.atomic_numbers == self.fis_central_z)[0]
            neighbor_filter = [self.fis_neighbor_z] if self.fis_neighbor_z is not None else None
            fis_vals = self._fis_calc(state, central_idx, ['fis'],
                                      element_filter=neighbor_filter)['fis']
            fis_mean = fis_vals.mean()
            fis_loss = self.fis_weight * (fis_mean - self.fis_target) ** 2
            losses['fis_loss'] = fis_loss
            losses['fis_mean'] = fis_mean.detach()

        # Normalise the objective before it is weighted against anything else.
        #
        # chi^2 for an unnormalised F(Q) fit is O(1e8) while the constraint and F_IS
        # terms are O(1).  Every scheme that tried to scale the OTHER terms up to
        # meet it measured worse than leaving them alone.  Dividing chi^2 by a fixed
        # reference -- its value at the first step -- makes the objective O(1), so
        # relative weights mean what they say and the aggregators receive the
        # comparable magnitudes they assume.  The scale is captured once and held
        # constant, so the gradient direction is unchanged and the loss stays
        # comparable across steps.
        if self.normalize_for_weighting and primary_key in losses:
            if self._chi2_scale is None:
                v = float(losses[primary_key].detach())
                self._chi2_scale = max(abs(v), 1e-30)
            losses[primary_key] = losses[primary_key] / self._chi2_scale
            total_loss = losses[primary_key]

        # Combine scattering + F_IS via aggregator or fixed sum
        agg_inputs = {k: v for k, v in losses.items()
                      if k in (primary_key, 'fis_loss') and isinstance(v, torch.Tensor)
                      and v.requires_grad}

        if self.aggregator is not None and len(agg_inputs) > 1:
            total_loss = self.aggregator(agg_inputs, self._step)
            losses['agg_weights'] = torch.tensor(
                self.aggregator.current_weights or [], dtype=torch.float32
            )
        elif 'fis_loss' in losses:
            total_loss = total_loss + losses['fis_loss']
        # (else: total_loss already set above from primary scattering key)

        if self.training:
            self._step += 1

        # Log once
        if not self._logged:
            print(f"\nCooperLoss:")
            print(f"  Target type: {self.target_type}")
            print(f"  Available losses: {list(losses.keys())}")
            print(f"  Primary loss key: {primary_key}")
            if self.fis_target is not None:
                print(f"  F_IS target: {self.fis_target:.4f}  weight: {self.fis_weight}")
            if self.aggregator is not None:
                print(f"  Aggregator: {type(self.aggregator).__name__}")
            self._logged = True

        return {
            'total_loss': total_loss,
            'chi2_loss': losses.get(primary_key, total_loss),
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
    # Adaptive aggregators (re-exported for convenience)
    'Aggregator',
    'build_aggregator',
    # Backward compatibility
    'ChiSquaredObjective',
    'ConstraintChiSquared',
]
