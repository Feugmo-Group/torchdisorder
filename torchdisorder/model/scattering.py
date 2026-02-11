"""
model/scattering.py – Unified differentiable scattering & RDF utilities
========================================================================

This module provides a single, unified interface for computing all common
scattering functions from atomic structures:

Reciprocal Space (Q-space):
    - S(Q): Structure factor (Faber-Ziman convention, →1 at high Q)
    - F(Q): Reduced structure factor F(Q) = Q[S(Q) - 1]

Real Space (r-space):
    - g(r): Pair distribution function (→1 at large r)
    - G(r): Reduced pair distribution function G(r) = 4πρr[g(r) - 1]
    - T(r): Total correlation function T(r) = 4πρr·g(r)

Relationships:
    S(Q) ↔ Fourier Transform ↔ g(r)
    F(Q) = Q[S(Q) - 1]
    G(r) = 4πρr[g(r) - 1]  
    T(r) = G(r) + 4πρr = 4πρr·g(r)

All functions support:
    - Neutron scattering (using coherent scattering lengths)
    - X-ray scattering (using atomic form factors)
    - Full gradient flow through PyTorch autograd

References:
    - Keen (2001) J. Appl. Cryst. 34, 172-177
    - Fischer et al. (2006) Rep. Prog. Phys. 69, 233-299
"""

from itertools import combinations_with_replacement
from functools import cache
from typing import Generator, Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn


# =============================================================================
# Enumerations
# =============================================================================

class ScatteringType(Enum):
    """Type of scattering experiment."""
    NEUTRON = "neutron"
    XRAY = "xray"


class OutputType(Enum):
    """Type of output function."""
    # Q-space
    S_Q = "S_Q"      # Structure factor
    F_Q = "F_Q"      # Reduced structure factor
    # r-space  
    g_r = "g_r"      # Pair distribution function
    G_r = "G_r"      # Reduced PDF
    T_r = "T_r"      # Total correlation function


# =============================================================================
# CUDA Utilities
# =============================================================================

@cache
def get_device_count() -> int:
    """Return number of available NVIDIA GPUs."""
    try:
        import nvidia_smi
        nvidia_smi.nvmlInit()
        return nvidia_smi.nvmlDeviceGetCount()
    except:
        return torch.cuda.device_count() if torch.cuda.is_available() else 1


def split_chunk(tensor: torch.Tensor) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Split tensor across CUDA devices for memory-efficient computation."""
    device_count = max(1, get_device_count())
    for device_idx, chunk in enumerate(torch.chunk(tensor, device_count)):
        device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
        yield device, chunk.to(device)


# =============================================================================
# Distance Utilities
# =============================================================================

def get_distance_matrix(
    points_1: torch.Tensor, 
    points_2: torch.Tensor, 
    cell: torch.Tensor
) -> torch.Tensor:
    """
    Compute pairwise distance matrix under minimum image convention.

    Args:
        points_1: (M, 3) positions
        points_2: (N, 3) positions  
        cell: (3, 3) lattice vectors

    Returns:
        (M, N) distance matrix
    """
    diff = points_2[None, :, :] - points_1[:, None, :]
    cell_inv = torch.linalg.inv(cell.float())
    frac = torch.einsum('ijk,kl->ijl', diff.float(), cell_inv)
    frac = frac - torch.round(frac)
    mip = torch.einsum('ijk,kl->ijl', frac, cell.float())
    # Use sqrt(sum_sq + eps) to avoid NaN gradients at zero distance
    return torch.sqrt((mip * mip).sum(dim=-1) + 1e-10).to(diff.dtype)


def get_observed_distances(
    points_1: torch.Tensor, 
    points_2: torch.Tensor, 
    cell: torch.Tensor
) -> torch.Tensor:
    """
    Extract observed distances, handling self-pairs symmetry.
    
    If points_1 is points_2, returns upper-triangular excluding diagonal.
    """
    dist = get_distance_matrix(points_1, points_2, cell)
    if points_1 is points_2:
        mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
        return dist[mask]
    return dist.flatten()


def get_cell_volume(cell: torch.Tensor) -> torch.Tensor:
    """Calculate volume of periodic cell."""
    try:
        return cell.det().abs()
    except NotImplementedError:
        return cell.cpu().det().abs().to(cell.device)


# =============================================================================
# Partial RDF via Gaussian KDE
# =============================================================================

def get_partial_rdf(
    points_1: torch.Tensor,
    points_2: torch.Tensor,
    cell: torch.Tensor,
    bins: torch.Tensor,
    *,
    kernel_width: float,
) -> torch.Tensor:
    """
    Compute partial RDF g_αβ(r) between two species via Gaussian KDE.

    Args:
        points_1: (M, 3) positions of species α
        points_2: (N, 3) positions of species β
        cell: (3, 3) lattice vectors
        bins: (n_r,) r values to evaluate on
        kernel_width: Gaussian bandwidth

    Returns:
        (n_r,) partial RDF
    """
    obs = get_observed_distances(points_1, points_2, cell)
    min_bin = torch.min(bins) - 3 * kernel_width
    max_bin = torch.max(bins) + 3 * kernel_width
    in_window = obs[(min_bin < obs) & (obs < max_bin)]

    try:
        gauss = torch.exp(
            -0.5 * ((bins[None, :] - in_window[:, None]) / kernel_width) ** 2
        ) / (kernel_width * (2 * torch.pi) ** 0.5)
        summed = gauss.sum(dim=0)
    except torch.cuda.OutOfMemoryError:
        summed = torch.zeros_like(bins)
        for device, chunk in split_chunk(in_window):
            g = torch.exp(
                -0.5 * ((bins.to(device)[None, :] - chunk[:, None]) / kernel_width) ** 2
            ) / (kernel_width * (2 * torch.pi) ** 0.5)
            summed = summed.to(device) + g.sum(dim=0)
        summed = summed.to(bins.device)

    vol = get_cell_volume(cell)
    n_pairs = obs.numel()
    return (vol / n_pairs) * summed / (4 * torch.pi * bins ** 2) if n_pairs > 0 else torch.zeros_like(bins)


# =============================================================================
# Neutron Scattering Functions
# =============================================================================

def compute_neutron_g_r(
    chemical_symbols: List[str],
    positions: torch.Tensor,
    cell: torch.Tensor,
    r_bins: torch.Tensor,
    *,
    kernel_width: float,
    scattering_lengths: Dict[str, float],
) -> torch.Tensor:
    """
    Compute neutron-weighted total pair distribution function g(r).
    
    g(r) = Σ_αβ c_α c_β b_α b_β g_αβ(r) / <b>²
    
    where c = concentration, b = scattering length, <b> = Σ c_α b_α
    """
    n = len(chemical_symbols)
    elems = set(chemical_symbols)
    frac = {e: chemical_symbols.count(e) / n for e in elems}
    masks = {e: torch.tensor([s == e for s in chemical_symbols], 
                             dtype=torch.bool, device=positions.device) for e in elems}
    
    # Mean scattering length squared (fm² → barn conversion: 0.01)
    b_mean_sq = (sum(frac[e] * scattering_lengths[e] for e in elems) ** 2) * 0.01

    g = torch.zeros_like(r_bins)
    for e1, e2 in combinations_with_replacement(elems, 2):
        pos1 = positions[masks[e1]]
        b1 = scattering_lengths[e1]
        f1 = frac[e1]
        
        if e1 != e2:
            pos2 = positions[masks[e2]]
            b2 = scattering_lengths[e2]
            f2 = frac[e2]
        else:
            pos2 = pos1
            b2 = b1
            f2 = f1
        
        # Weight: c_α c_β b_α b_β / <b>² (with symmetry factor 2 for α≠β)
        coeff = f1 * f2 * b1 * b2 * 0.01 / b_mean_sq * (2 if e1 != e2 else 1)
        g += coeff * get_partial_rdf(pos1, pos2, cell, r_bins, kernel_width=kernel_width)
    
    return g


def compute_neutron_G_r(
    chemical_symbols: List[str],
    positions: torch.Tensor,
    cell: torch.Tensor,
    r_bins: torch.Tensor,
    *,
    kernel_width: float,
    scattering_lengths: Dict[str, float],
) -> torch.Tensor:
    """
    Compute reduced pair distribution function G(r) = 4πρr[g(r) - 1].
    
    This is the distinct part of the correlation function (G(r) → 0 as r → ∞).
    """
    n = len(chemical_symbols)
    elems = set(chemical_symbols)
    frac = {e: chemical_symbols.count(e) / n for e in elems}
    masks = {e: torch.tensor([s == e for s in chemical_symbols], 
                             dtype=torch.bool, device=positions.device) for e in elems}

    # The original implementation computed G(r) directly
    G = torch.zeros_like(r_bins)
    for e1, e2 in combinations_with_replacement(elems, 2):
        pos1 = positions[masks[e1]]
        b1 = scattering_lengths[e1]
        f1 = frac[e1]
        
        if e1 != e2:
            pos2 = positions[masks[e2]]
            b2 = scattering_lengths[e2]
            f2 = frac[e2]
        else:
            pos2 = pos1
            b2 = b1
            f2 = f1
        
        # Coefficient includes the -1 from [g(r) - 1]
        coeff = f1 * f2 * b1 * b2 * 0.01 * (2 if e1 != e2 else 1)
        G += coeff * (get_partial_rdf(pos1, pos2, cell, r_bins, kernel_width=kernel_width) - 1)
    
    return G


def compute_neutron_T_r(
    G_r: torch.Tensor,
    r_bins: torch.Tensor,
    chemical_symbols: List[str],
    cell: torch.Tensor,
    *,
    scattering_lengths: Dict[str, float],
) -> torch.Tensor:
    """
    Compute total correlation function T(r) from G(r).
    
    T(r) = 4πρr·g(r) = G(r) + 4πρr·<b>²
    
    Args:
        G_r: Reduced PDF G(r) = 4πρr[g(r) - 1]
        r_bins: r values
        chemical_symbols: Atom symbols
        cell: Lattice vectors
        scattering_lengths: Dict of b values (fm)
    """
    n = len(chemical_symbols)
    elems = set(chemical_symbols)
    frac = {e: chemical_symbols.count(e) / n for e in elems}
    rho = n / get_cell_volume(cell)
    
    # <b>² term (fm² → barn)
    b_mean_sq = (sum(frac[e] * scattering_lengths[e] for e in elems) ** 2) * 0.01
    
    return 4 * torch.pi * r_bins * rho * (G_r / (4 * torch.pi * r_bins * rho + 1e-10) + b_mean_sq)


def compute_neutron_S_Q(
    G_r: torch.Tensor,
    r_bins: torch.Tensor,
    q_bins: torch.Tensor,
    chemical_symbols: List[str],
    cell: torch.Tensor,
) -> torch.Tensor:
    """
    Compute structure factor S(Q) from G(r) via Fourier transform.
    
    Faber-Ziman convention: S(Q) → 1 as Q → ∞
    
    S(Q) = 1 + (4πρ/Q) ∫ r·G(r)·sin(Qr) dr
    """
    rho = len(chemical_symbols) / get_cell_volume(cell)
    
    # Integrand: r·G(r)·sin(Qr)/Q
    integrand = (r_bins[None, :] * G_r[None, :] * 
                 torch.sin(q_bins[:, None] * r_bins[None, :]) / 
                 (q_bins[:, None] + 1e-10))
    
    return 1.0 + 4 * torch.pi * rho * torch.trapezoid(integrand, r_bins)


def compute_neutron_F_Q(
    S_Q: torch.Tensor,
    q_bins: torch.Tensor,
) -> torch.Tensor:
    """
    Compute reduced structure factor F(Q) = Q[S(Q) - 1].
    
    F(Q) → 0 as Q → ∞
    """
    return q_bins * (S_Q - 1)


def compute_neutron_S_Q_direct(
    chemical_symbols: List[str],
    positions: torch.Tensor,
    cell: torch.Tensor,
    q_bins: torch.Tensor,
    *,
    scattering_lengths: Dict[str, float],
) -> torch.Tensor:
    """
    Compute S(Q) directly using Debye formula (no r-space intermediate).
    
    S(Q) = (1/N) Σ_{i,j} (b_i b_j / <b>²) sin(Qr_ij)/(Qr_ij)
    
    More efficient for Q-only calculations but O(N²) in memory.
    """
    device = positions.device
    dtype = positions.dtype
    n_atoms = len(chemical_symbols)
    
    b = torch.tensor(
        [scattering_lengths.get(s, 0.0) for s in chemical_symbols],
        device=device, dtype=dtype
    )
    b_mean = b.mean()
    
    if b_mean.abs() < 1e-10:
        return torch.ones(len(q_bins), device=device, dtype=dtype)
    
    # Compute distances with MIC
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    cell_inv = torch.linalg.inv(cell.float())
    diff_frac = torch.einsum('ijk,kl->ijl', diff.float(), cell_inv)
    diff_frac = diff_frac - torch.round(diff_frac)
    diff = torch.einsum('ijk,kl->ijl', diff_frac, cell.float()).to(dtype)
    r_ij = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-10)
    
    # Weight matrix
    weights = torch.outer(b, b) / (b_mean ** 2)
    
    # Debye formula
    Q = q_bins.unsqueeze(-1).unsqueeze(-1)
    r = r_ij.unsqueeze(0)
    Qr = Q * r
    
    sinc_Qr = torch.where(
        Qr.abs() < 1e-8,
        torch.ones_like(Qr),
        torch.sin(Qr) / (Qr + 1e-10)
    )
    
    return (weights.unsqueeze(0) * sinc_Qr).sum(dim=(-2, -1)) / n_atoms


# =============================================================================
# X-ray Scattering Functions
# =============================================================================

def _compute_form_factors(
    elems: set,
    q_bins: torch.Tensor,
    form_factor_params: Dict[str, Dict[str, List[float]]],
) -> Dict[str, torch.Tensor]:
    """
    Compute atomic form factors f(Q) using Cromer-Mann coefficients.
    
    f(Q) = Σ_i a_i exp(-b_i (Q/4π)²) + c
    """
    ff = {}
    for el in elems:
        params = form_factor_params[el]
        a = torch.tensor(params["a"], device=q_bins.device)
        b = torch.tensor(params["b"], device=q_bins.device)
        c = params["c"][0] if isinstance(params["c"], list) else params["c"]
        
        s = q_bins / (4 * torch.pi)  # s = Q / 4π
        ff[el] = c + torch.sum(
            a[:, None] * torch.exp(-b[:, None] * s[None, :] ** 2), 
            dim=0
        )
    return ff


def compute_xray_S_Q(
    chemical_symbols: List[str],
    positions: torch.Tensor,
    cell: torch.Tensor,
    q_bins: torch.Tensor,
    r_bins: torch.Tensor,
    *,
    kernel_width: float,
    form_factor_params: Dict[str, Dict[str, List[float]]],
) -> torch.Tensor:
    """
    Compute X-ray structure factor S(Q) using Faber-Ziman convention.
    
    S(Q) = 1 + ρ Σ_αβ w_αβ(Q) ∫ 4πr² [g_αβ(r) - 1] sinc(Qr) dr
    
    where w_αβ(Q) = c_α c_β f_α(Q) f_β(Q) / <f(Q)>²
    and <f(Q)> = Σ_γ c_γ f_γ(Q)
    
    This ensures S(Q) → 1 as Q → ∞ (Faber-Ziman convention).
    """
    n = len(chemical_symbols)
    rho = n / get_cell_volume(cell)
    elems = set(chemical_symbols)
    frac = {e: chemical_symbols.count(e) / n for e in elems}
    masks = {e: torch.tensor([s == e for s in chemical_symbols], 
                             dtype=torch.bool, device=positions.device) for e in elems}
    
    # Compute Q-dependent form factors f_α(Q)
    ff = _compute_form_factors(elems, q_bins, form_factor_params)
    
    # <f(Q)> = Σ c_α f_α(Q)  - average form factor
    f_mean = sum(frac[e] * ff[e] for e in elems)
    
    # <f(Q)>² for normalization (avoid division by zero)
    f_mean_sq = f_mean ** 2 + 1e-10
    
    # Precompute sinc matrix
    qr = r_bins[:, None] * q_bins[None, :]
    sinc_qr = torch.where(
        qr.abs() < 1e-8,
        torch.ones_like(qr),
        torch.sin(qr) / (qr + 1e-10)
    )
    
    # Compute weighted sum over all pairs
    S_distinct = torch.zeros_like(q_bins)
    
    for e1, e2 in combinations_with_replacement(elems, 2):
        pos1 = positions[masks[e1]]
        c1, f1 = frac[e1], ff[e1]
        
        if e1 != e2:
            pos2 = positions[masks[e2]]
            c2, f2 = frac[e2], ff[e2]
            multiplicity = 2.0  # Count α-β and β-α
        else:
            pos2 = pos1
            c2, f2 = c1, f1
            multiplicity = 1.0
        
        # Faber-Ziman weight: w_αβ(Q) = c_α c_β f_α(Q) f_β(Q) / <f(Q)>²
        weight = multiplicity * c1 * c2 * f1 * f2 / f_mean_sq
        
        # Compute partial RDF g_αβ(r)
        g_partial = get_partial_rdf(pos1, pos2, cell, r_bins, kernel_width=kernel_width)
        
        # Integrate: ρ ∫ 4πr² [g_αβ(r) - 1] sinc(Qr) dr
        integrand = 4 * torch.pi * r_bins[:, None] ** 2 * (g_partial[:, None] - 1) * sinc_qr
        integral = rho * torch.trapezoid(integrand, r_bins, dim=0)
        
        S_distinct += weight * integral
    
    # S(Q) = 1 + distinct part  (Faber-Ziman: S(Q) → 1 as Q → ∞)
    return 1.0 + S_distinct


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class ScatteringConfig:
    """Configuration for scattering calculations."""
    neutron_scattering_lengths: Dict[str, float]
    xray_form_factor_params: Dict[str, Dict[str, List[float]]]
    kernel_width: float
    
    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "ScatteringConfig":
        return cls(
            neutron_scattering_lengths=cfg["neutron_scattering_lengths"],
            xray_form_factor_params=cfg["xray_form_factor_params"],
            kernel_width=cfg["kernel_width"],
        )


# =============================================================================
# Unified Spectrum Calculator
# =============================================================================

class UnifiedSpectrumCalculator(nn.Module):
    """
    Unified calculator for all scattering functions.
    
    Provides consistent interface for computing any output (S(Q), F(Q), g(r), G(r), T(r))
    from atomic structure with neutron or X-ray weighting.
    
    Usage:
        >>> config = ScatteringConfig.from_dict(cfg)
        >>> calc = UnifiedSpectrumCalculator(config)
        >>> 
        >>> # Compute structure factor
        >>> S_Q = calc.compute(symbols, positions, cell, q_bins=q, output='S_Q')
        >>> 
        >>> # Compute all outputs at once
        >>> results = calc.compute_all(symbols, positions, cell, r_bins=r, q_bins=q)
    """
    
    def __init__(self, config: ScatteringConfig):
        super().__init__()
        self.scattering_lengths = config.neutron_scattering_lengths
        self.form_factor_params = config.xray_form_factor_params
        self.kernel_width = config.kernel_width
    
    @classmethod
    def from_config_dict(cls, cfg: Dict[str, Any]) -> "UnifiedSpectrumCalculator":
        return cls(ScatteringConfig.from_dict(cfg))
    
    def compute(
        self,
        symbols: List[str],
        positions: torch.Tensor,
        cell: torch.Tensor,
        *,
        r_bins: Optional[torch.Tensor] = None,
        q_bins: Optional[torch.Tensor] = None,
        output: str = 'S_Q',
        scattering_type: str = 'neutron',
    ) -> torch.Tensor:
        """
        Compute a single output spectrum.
        
        Args:
            symbols: Chemical symbols
            positions: (N, 3) atomic positions
            cell: (3, 3) lattice vectors
            r_bins: r values for r-space outputs
            q_bins: Q values for Q-space outputs
            output: 'S_Q', 'F_Q', 'g_r', 'G_r', or 'T_r'
            scattering_type: 'neutron' or 'xray'
        """
        # Validate inputs
        if output in ['S_Q', 'F_Q'] and q_bins is None:
            raise ValueError(f"q_bins required for output='{output}'")
        if output in ['g_r', 'G_r', 'T_r'] and r_bins is None:
            raise ValueError(f"r_bins required for output='{output}'")
        
        # Route to appropriate computation
        if scattering_type == 'neutron':
            return self._compute_neutron(symbols, positions, cell, r_bins, q_bins, output)
        else:
            return self._compute_xray(symbols, positions, cell, r_bins, q_bins, output)
    
    def _compute_neutron(
        self,
        symbols: List[str],
        positions: torch.Tensor,
        cell: torch.Tensor,
        r_bins: Optional[torch.Tensor],
        q_bins: Optional[torch.Tensor],
        output: str,
    ) -> torch.Tensor:
        """Compute neutron scattering output."""
        
        if output == 'g_r':
            return compute_neutron_g_r(
                symbols, positions, cell, r_bins,
                kernel_width=self.kernel_width,
                scattering_lengths=self.scattering_lengths,
            )
        
        if output == 'G_r':
            return compute_neutron_G_r(
                symbols, positions, cell, r_bins,
                kernel_width=self.kernel_width,
                scattering_lengths=self.scattering_lengths,
            )
        
        if output == 'T_r':
            G_r = compute_neutron_G_r(
                symbols, positions, cell, r_bins,
                kernel_width=self.kernel_width,
                scattering_lengths=self.scattering_lengths,
            )
            return compute_neutron_T_r(
                G_r, r_bins, symbols, cell,
                scattering_lengths=self.scattering_lengths,
            )
        
        if output == 'S_Q':
            if r_bins is not None:
                # Via Fourier transform
                G_r = compute_neutron_G_r(
                    symbols, positions, cell, r_bins,
                    kernel_width=self.kernel_width,
                    scattering_lengths=self.scattering_lengths,
                )
                return compute_neutron_S_Q(G_r, r_bins, q_bins, symbols, cell)
            else:
                # Direct Debye formula
                return compute_neutron_S_Q_direct(
                    symbols, positions, cell, q_bins,
                    scattering_lengths=self.scattering_lengths,
                )
        
        if output == 'F_Q':
            S_Q = self._compute_neutron(symbols, positions, cell, r_bins, q_bins, 'S_Q')
            return compute_neutron_F_Q(S_Q, q_bins)
        
        raise ValueError(f"Unknown output: {output}")
    
    def _compute_xray(
        self,
        symbols: List[str],
        positions: torch.Tensor,
        cell: torch.Tensor,
        r_bins: Optional[torch.Tensor],
        q_bins: Optional[torch.Tensor],
        output: str,
    ) -> torch.Tensor:
        """Compute X-ray scattering output."""
        
        if output == 'S_Q':
            if r_bins is None:
                raise ValueError("r_bins required for X-ray S(Q)")
            return compute_xray_S_Q(
                symbols, positions, cell, q_bins, r_bins,
                kernel_width=self.kernel_width,
                form_factor_params=self.form_factor_params,
            )
        
        if output == 'F_Q':
            S_Q = self._compute_xray(symbols, positions, cell, r_bins, q_bins, 'S_Q')
            return compute_neutron_F_Q(S_Q, q_bins)  # Same formula
        
        # For r-space, use neutron (or implement X-ray g(r) if needed)
        raise NotImplementedError(f"X-ray {output} not implemented")
    
    def compute_all(
        self,
        symbols: List[str],
        positions: torch.Tensor,
        cell: torch.Tensor,
        *,
        r_bins: torch.Tensor,
        q_bins: torch.Tensor,
        scattering_type: str = 'neutron',
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all output spectra efficiently.
        
        Returns dict with keys: 'g_r', 'G_r', 'T_r', 'S_Q', 'F_Q'
        """
        results = {}
        
        if scattering_type == 'neutron':
            # Compute G(r) first (base for everything)
            G_r = compute_neutron_G_r(
                symbols, positions, cell, r_bins,
                kernel_width=self.kernel_width,
                scattering_lengths=self.scattering_lengths,
            )
            results['G_r'] = G_r
            
            # g(r) from G(r)
            rho = len(symbols) / get_cell_volume(cell)
            results['g_r'] = G_r / (4 * torch.pi * r_bins * rho + 1e-10) + 1
            
            # T(r) from G(r)
            results['T_r'] = compute_neutron_T_r(
                G_r, r_bins, symbols, cell,
                scattering_lengths=self.scattering_lengths,
            )
            
            # S(Q) from G(r)
            results['S_Q'] = compute_neutron_S_Q(G_r, r_bins, q_bins, symbols, cell)
            
            # F(Q) from S(Q)
            results['F_Q'] = compute_neutron_F_Q(results['S_Q'], q_bins)
        
        else:
            # X-ray: compute S(Q) first, then derive G(r) via Fourier transform
            # This ensures G(r) and S(Q) are consistent Fourier pairs
            
            # Step 1: Compute S(Q) with X-ray form factor weighting (Faber-Ziman)
            S_Q = compute_xray_S_Q(
                symbols, positions, cell, q_bins, r_bins,
                kernel_width=self.kernel_width,
                form_factor_params=self.form_factor_params,
            )
            results['S_Q'] = S_Q
            
            # Step 2: F(Q) = Q[S(Q) - 1] (reduced structure factor)
            F_Q = q_bins * (S_Q - 1)
            results['F_Q'] = F_Q
            
            # Step 3: G(r) via inverse Fourier transform of F(Q)
            # G(r) = (2/π) ∫ F(Q) sin(Qr) dQ
            # This ensures G(r) and S(Q) are consistent Fourier pairs!
            qr = q_bins[None, :] * r_bins[:, None]  # [n_r, n_q]
            integrand = F_Q[None, :] * torch.sin(qr)  # [n_r, n_q]
            G_r = (2 / torch.pi) * torch.trapezoid(integrand, q_bins, dim=1)
            results['G_r'] = G_r
            
            # Step 4: g(r) from G(r)
            n = len(symbols)
            rho = n / get_cell_volume(cell)
            # G(r) = 4πρr[g(r) - 1]  =>  g(r) = G(r)/(4πρr) + 1
            g_r = G_r / (4 * torch.pi * r_bins * rho + 1e-10) + 1
            results['g_r'] = g_r
            
            # Step 5: T(r) = 4πρr·g(r) = G(r) + 4πρr
            results['T_r'] = G_r + 4 * torch.pi * r_bins * rho
            
            # Also compute neutron S(Q) for comparison/debugging
            G_r_neutron = compute_neutron_G_r(
                symbols, positions, cell, r_bins,
                kernel_width=self.kernel_width,
                scattering_lengths=self.scattering_lengths,
            )
            results['S_Q_neutron'] = compute_neutron_S_Q(G_r_neutron, r_bins, q_bins, symbols, cell)
        
        return results
    
    # =========================================================================
    # Convenience aliases (backward compatibility)
    # =========================================================================
    
    def compute_rdf(self, symbols, positions, cell, r_bins):
        """Alias for G(r) computation (backward compat)."""
        return self.compute(symbols, positions, cell, r_bins=r_bins, output='G_r')
    
    def compute_sf(self, G_r, r_bins, q_bins, symbols, cell):
        """Compute S(Q) from precomputed G(r)."""
        return compute_neutron_S_Q(G_r, r_bins, q_bins, symbols, cell)
    
    def compute_correlation(self, G_r, r_bins, symbols, cell):
        """Compute T(r) from precomputed G(r)."""
        return compute_neutron_T_r(
            G_r, r_bins, symbols, cell,
            scattering_lengths=self.scattering_lengths,
        )


# =============================================================================
# Legacy Alias
# =============================================================================

# Backward compatibility with v5
SpectrumCalculator = UnifiedSpectrumCalculator


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ScatteringType',
    'OutputType',
    'ScatteringConfig',
    'UnifiedSpectrumCalculator',
    'SpectrumCalculator',
    'get_distance_matrix',
    'get_cell_volume',
    'get_partial_rdf',
    'compute_neutron_g_r',
    'compute_neutron_G_r',
    'compute_neutron_T_r',
    'compute_neutron_S_Q',
    'compute_neutron_F_Q',
    'compute_neutron_S_Q_direct',
    'compute_xray_S_Q',
]
