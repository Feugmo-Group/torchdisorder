"""
model/rdf.py – Differentiable RDF & scattering utilities using PyTorch.

Contains methods for calculating spectra from structures and converting
between RDF, total correlation T(r), and scattering structure factor S(Q).
Functions support gradient flow through PyTorch.
"""

from itertools import combinations_with_replacement
from functools import cache
from typing import Generator, Dict, List, Optional,  Any
from dataclasses import dataclass
import torch




# -----------------------------------------------------------------------------
# 0. CUDA utility functions for chunking large tensors
# -----------------------------------------------------------------------------

@cache
def get_device_count() -> int:
    """Return number of available NVIDIA GPUs."""
    import nvidia_smi
    nvidia_smi.nvmlInit()
    return nvidia_smi.nvmlDeviceGetCount()

def split_chunk(tensor: torch.Tensor) -> Generator[tuple[str, torch.Tensor], None, None]:
    """
    Split tensor into chunks distributed across CUDA devices for memory-efficient computation.

    Yields (device_name, chunk) pairs.
    """
    device_count = get_device_count()
    for device_idx, chunk in enumerate(torch.chunk(tensor, device_count)):
        device = f"cuda:{device_idx}"
        yield device, chunk.to(device)


# -----------------------------------------------------------------------------
# 1. Helper functions: distances and cell volume
# -----------------------------------------------------------------------------

def get_distance_matrix(points_1: torch.Tensor, points_2: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise distance matrix between two point sets under minimum image convention.

    Args:
        points_1: Tensor of shape (M, D)
        points_2: Tensor of shape (N, D)
        cell: Tensor of shape (D, D) representing supercell vectors

    Returns:
        Tensor of shape (M, N) with distances
    """
    diff = points_2[None, :, :] - points_1[:, None, :]
    shifts = torch.round(diff @ cell.inverse()) @ cell
    mip = diff - shifts
    # Use sqrt(sum_sq + eps) instead of linalg.norm to avoid NaN gradients
    # at zero distance (self-pairs in same-element distance matrices).
    # The gradient of sqrt(x+eps) is finite: 1/(2*sqrt(x+eps)), whereas
    # the gradient of norm(x) at x=0 is 0/0 = NaN.
    return torch.sqrt((mip * mip).sum(dim=-1) + 1e-10)


def get_observed_distances(points_1: torch.Tensor, points_2: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """
    Extract observed distances between points_1 and points_2 accounting for symmetry.

    If points_1 is points_2, returns upper-triangular distances excluding diagonal.

    Args:
        points_1: Tensor (M, D)
        points_2: Tensor (N, D)
        cell: Tensor (D, D)

    Returns:
        1D tensor of distances
    """
    dist = get_distance_matrix(points_1, points_2, cell)
    if points_1 is points_2:
        mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
        return dist[mask]
    return dist.flatten()


def get_cell_volume(cell: torch.Tensor) -> torch.Tensor:
    """
    Calculate volume of a periodic cell.

    Args:
        cell: Tensor (D, D)

    Returns:
        Scalar tensor volume
    """
    try:
        return cell.det().abs()
    except NotImplementedError:  # e.g., on MPS backend
        return cell.cpu().det().abs().to(cell.device)


# -----------------------------------------------------------------------------
# 2. Partial RDF via Gaussian KDE (differentiable)
# -----------------------------------------------------------------------------

def get_partial_rdf(
    points_1: torch.Tensor,
    points_2: torch.Tensor,
    cell: torch.Tensor,
    bins: torch.Tensor,
    *,
    kernel_width: float ,
) -> torch.Tensor:
    """
    Compute partial radial distribution function (RDF) between two point sets via Gaussian KDE.

    Args:
        points_1: Tensor (M, D)
        points_2: Tensor (N, D)
        cell: Tensor (D, D)
        bins: 1D tensor of r values to evaluate RDF on
        kernel_width: Gaussian kernel bandwidth

    Returns:
        1D tensor RDF evaluated at bins
    """
    obs = get_observed_distances(points_1, points_2, cell)
    min_bin = torch.min(bins) - 3 * kernel_width
    max_bin = torch.max(bins) + 3 * kernel_width
    in_window = obs[(min_bin < obs) & (obs < max_bin)]

    try:
        gauss = torch.exp(-0.5 * ((bins[None, :] - in_window[:, None]) / kernel_width) ** 2) / (kernel_width * (2 * torch.pi) ** 0.5)
        summed = gauss.sum(dim=0)
    except torch.cuda.OutOfMemoryError:
        summed = torch.zeros_like(bins)
        for device, chunk in split_chunk(in_window):
            g = torch.exp(-0.5 * ((bins.to(device)[None, :] - chunk[:, None]) / kernel_width) ** 2) / (kernel_width * (2 * torch.pi) ** 0.5)
            summed = summed.to(device) + g.sum(dim=0)
        summed = summed.to(bins.device)

    vol = get_cell_volume(cell)
    return (vol / obs.numel()) * summed / (4 * torch.pi * bins ** 2)


# -----------------------------------------------------------------------------
# 3. Neutron total RDF, total correlation function, and scattering SF
# -----------------------------------------------------------------------------

def get_neutron_total_rdf(
    chemical_symbols: List[str],
    positions: torch.Tensor,
    cell: torch.Tensor,
    bins: torch.Tensor,
    *,
    kernel_width: float ,
    scattering_lengths: Dict[str, float],
) -> torch.Tensor:
    """
    Compute neutron total radial distribution function G(r).

    Args:
        chemical_symbols: List of atomic symbols
        positions: Tensor (N, D)
        cell: Tensor (D, D)
        bins: 1D tensor of r values
        kernel_width: Gaussian KDE kernel width
        scattering_lengths: Dict mapping element to neutron scattering length (fm)

    Returns:
        1D tensor G(r)
    """
    n = len(chemical_symbols)
    elems = set(chemical_symbols)
    frac = {e: chemical_symbols.count(e) / n for e in elems}
    masks = {e: [s == e for s in chemical_symbols] for e in elems}

    G = torch.zeros_like(bins)
    for e1, e2 in combinations_with_replacement(elems, 2):
        pos1 = positions[masks[e1]]; f1 = frac[e1]; b1 = scattering_lengths[e1]
        if e1 != e2:
            pos2 = positions[masks[e2]]; f2 = frac[e2]; b2 = scattering_lengths[e2]
        else:
            pos2 = pos1; f2 = f1; b2 = b1
        coeff = f1 * f2 * b1 * b2 * 0.01 * (2 if e1 != e2 else 1)
        G += coeff * (get_partial_rdf(pos1, pos2, cell, bins, kernel_width=kernel_width) - 1)
    return G

# def get_partial_rdf(
#     points_1: torch.Tensor,
#     points_2: torch.Tensor,
#     cell: torch.Tensor,
#     bins: torch.Tensor,
#     *,
#     kernel_width: float,
# ) -> torch.Tensor:
#     """
#     Compute neutron total radial distribution function G(r).
#
#     Args:
#         chemical_symbols: List of atomic symbols
#         positions: Tensor (N, D)
#         cell: Tensor (D, D)
#         bins: 1D tensor of r values
#         kernel_width: Gaussian KDE kernel width
#         scattering_lengths: Dict mapping element to neutron scattering length (fm)
#
#     Returns:
#         1D tensor G(r)
#     """
#     n = len(chemical_symbols)
#     elems = set(chemical_symbols)
#     frac = {e: chemical_symbols.count(e) / n for e in elems}
#     masks = {e: [s == e for s in chemical_symbols] for e in elems}
#
#     G = torch.zeros_like(bins)
#     for e1, e2 in combinations_with_replacement(elems, 2):
#         pos1 = positions[masks[e1]]; f1 = frac[e1]; b1 = scattering_lengths[e1]
#         if e1 != e2:
#             pos2 = positions[masks[e2]]; f2 = frac[e2]; b2 = scattering_lengths[e2]
#         else:
#             pos2 = pos1; f2 = f1; b2 = b1
#         coeff = f1 * f2 * b1 * b2 * 0.01 * (2 if e1 != e2 else 1)
#         G += coeff * (get_partial_rdf(pos1, pos2, cell, bins, kernel_width=kernel_width) - 1)
#     return G


def get_neutron_total_correlation_function(
    total_rdf: torch.Tensor,
    r_bins: torch.Tensor,
    chemical_symbols: List[str],
    cell: torch.Tensor,
    *,
    scattering_lengths: Dict[str, float],
) -> torch.Tensor:
    """
    Compute neutron total correlation function T(r) from total RDF.

    Args:
        total_rdf: 1D tensor G(r)
        r_bins: 1D tensor r values
        chemical_symbols: List of atomic symbols
        cell: Tensor (D, D)
        scattering_lengths: Dict mapping element to neutron scattering length (fm)

    Returns:
        1D tensor T(r)
    """
    n = len(chemical_symbols)
    elems = set(chemical_symbols)
    frac = {e: chemical_symbols.count(e) / n for e in elems}
    rho = n / get_cell_volume(cell)
    b_mean = 0.01 * sum(frac[e] * scattering_lengths[e] for e in elems) ** 2
    return 4 * torch.pi * r_bins * rho * (total_rdf + b_mean)


def get_neutron_total_scattering_sf(
    total_rdf: torch.Tensor,
    r_bins: torch.Tensor,
    q_bins: torch.Tensor,
    chemical_symbols: List[str],
    cell: torch.Tensor,
) -> torch.Tensor:
    """
    Compute neutron total scattering structure factor S(Q) from total RDF.
    
    Uses the Faber-Ziman convention where S(Q) → 1 as Q → ∞.
    
    The relationship is:
        S(Q) = 1 + 4πρ ∫ r·G(r)·sin(Qr)/Q dr
    
    where G(r) is the distinct part of the correlation function (G(r) → 0 as r → ∞).

    Args:
        total_rdf: 1D tensor G(r) - the distinct correlation function
        r_bins: 1D tensor r values (Å)
        q_bins: 1D tensor Q values (Å⁻¹)
        chemical_symbols: List of atomic symbols
        cell: Tensor (D, D) - unit cell vectors

    Returns:
        1D tensor S(Q) in Faber-Ziman convention (→ 1 at high Q)
    """
    rho = len(chemical_symbols) / get_cell_volume(cell)
    integrand = r_bins[None, :] * total_rdf[None, :] * torch.sin(q_bins[:, None] * r_bins[None, :]) / q_bins[:, None]
    # Faber-Ziman S(Q) = 1 + Fourier transform of G(r)
    return 1.0 + 4 * torch.pi * rho * torch.trapz(integrand, r_bins)


# -----------------------------------------------------------------------------
# 4. X-ray total scattering SF and RDF (optional)
# -----------------------------------------------------------------------------

def _scaled_form_factors(elems: set, q_bins: torch.Tensor, frac: Dict[str, float], form_factor_params: Dict[str, Dict[str, List[float]]]) -> Dict[str, torch.Tensor]:
    """
    Compute scaled atomic form factors for X-ray scattering.

    Args:
        elems: Set of elements
        q_bins: 1D tensor Q values
        frac: Element fractions dict
        form_factor_params: Dict with parameters 'a', 'b', 'c' per element

    Returns:
        Dict mapping element to scaled form factor tensor
    """
    ff = {}
    for el in elems:
        a = torch.tensor(form_factor_params[el]["a"], device=q_bins.device)
        b = torch.tensor(form_factor_params[el]["b"], device=q_bins.device)
        c = form_factor_params[el]["c"][0]
        ff_el = c + torch.sum(a[:, None] * torch.exp(-b[:, None] * (q_bins[None, :] / (4 * torch.pi)) ** 2), dim=0)
        ff[el] = ff_el
    total = sum(frac[el] * ff[el] for el in elems)
    return {el: frac[el] * ff[el] / total for el in elems}


def get_xray_total_scattering_sf(
    chemical_symbols: List[str],
    positions: torch.Tensor,
    cell: torch.Tensor,
    q_bins: torch.Tensor,
    r_bins: torch.Tensor,
    *,
    kernel_width: float ,
    form_factor_params: Dict[str, Dict[str, List[float]]],
) -> torch.Tensor:
    """
    Compute X-ray total scattering structure factor S(Q).

    Args:
        chemical_symbols: List of atomic symbols
        positions: Tensor (N, D)
        cell: Tensor (D, D)
        q_bins: 1D tensor Q values
        r_bins: 1D tensor r values
        kernel_width: KDE Gaussian width
        form_factor_params: Dict with X-ray form factor parameters

    Returns:
        1D tensor S(Q)
    """
    n = len(chemical_symbols)
    rho = n / get_cell_volume(cell)
    elems = set(chemical_symbols)
    frac = {e: chemical_symbols.count(e) / n for e in elems}
    masks = {e: [s == e for s in chemical_symbols] for e in elems}
    ff = _scaled_form_factors(elems, q_bins, frac, form_factor_params)

    qr = r_bins[:, None] * q_bins[None, :]
    S = torch.zeros_like(q_bins)
    for e1, e2 in combinations_with_replacement(elems, 2):
        pos1 = positions[masks[e1]]
        f1 = ff[e1]
        if e1 != e2:
            pos2 = positions[masks[e2]]
            f2 = ff[e2]
        else:
            pos2 = pos1
            f2 = f1
        coeff = rho * f1 * f2 * (2 if e1 != e2 else 1)
        prdf = get_partial_rdf(pos1, pos2, cell, r_bins, kernel_width=kernel_width)
        integ = 4 * torch.pi * r_bins[:, None] ** 2 * (prdf[:, None] - 1) * torch.sin(qr) / qr
        S += coeff * torch.trapz(integ, r_bins, dim=0)
    return S


def get_xray_total_rdf(
    total_sf: torch.Tensor,
    r_bins: torch.Tensor,
    q_bins: torch.Tensor,
    chemical_symbols: List[str],
    cell: torch.Tensor,
) -> torch.Tensor:
    """
    Compute X-ray total RDF G(r) from total scattering SF.

    Args:
        total_sf: 1D tensor S(Q)
        r_bins: 1D tensor r values
        q_bins: 1D tensor Q values
        chemical_symbols: List of atomic symbols
        cell: Tensor (D, D)

    Returns:
        1D tensor G(r)
    """
    rho = len(chemical_symbols) / get_cell_volume(cell)
    qr = q_bins[None, :] * r_bins[:, None]
    integ = 4 * torch.pi * q_bins[None, :] ** 2 * total_sf[None, :] * torch.sin(qr) / qr
    return torch.trapz(integ, q_bins) / ((2 * torch.pi) ** 3 * rho)


def get_approximate_xray_total_rdf(
    chemical_symbols: List[str],
    positions: torch.Tensor,
    cell: torch.Tensor,
    r_bins: torch.Tensor,
    *,
    kernel_width: float ,
    form_factor_params: Optional[Dict[str, Dict[str, List[float]]]] = None,
) -> torch.Tensor:
    """
    Approximate X-ray total RDF using electron counts.

    Args:
        chemical_symbols: List of atomic symbols
        positions: Tensor (N, D)
        cell: Tensor (D, D)
        r_bins: 1D tensor r values
        kernel_width: Gaussian KDE width
        form_factor_params: Optional dict with form factor parameters for electron counts

    Returns:
        1D tensor RDF
    """
    n = len(chemical_symbols)
    elems = set(chemical_symbols)
    frac = {e: chemical_symbols.count(e) / n for e in elems}
    masks = {e: [s == e for s in chemical_symbols] for e in elems}

    if form_factor_params is None:
        raise ValueError("form_factor_params must be provided")

    electrons = {e: float(form_factor_params[e]["c"][0] + torch.tensor(form_factor_params[e]["a"]).sum()) for e in elems}
    total_e = sum(frac[e] * electrons[e] for e in elems)
    scale = {e: frac[e] * electrons[e] / total_e for e in elems}

    Gx = torch.zeros_like(r_bins)
    for e1, e2 in combinations_with_replacement(elems, 2):
        pos1 = positions[masks[e1]]
        s1 = scale[e1]
        if e1 != e2:
            pos2 = positions[masks[e2]]
            s2 = scale[e2]
        else:
            pos2 = pos1
            s2 = s1
        coeff = s1 * s2 * (2 if e1 != e2 else 1)
        Gx += coeff * (get_partial_rdf(pos1, pos2, cell, r_bins, kernel_width=kernel_width) - 1)
    return Gx


# -----------------------------------------------------------------------------
# 5. Class wrapper for convenient usage
# -----------------------------------------------------------------------------

@dataclass
class ScatteringConfig:
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


class SpectrumCalculator:
    def __init__(self, config: ScatteringConfig):
        self.neutron_scattering_lengths = config.neutron_scattering_lengths
        self.xray_form_factor_params = config.xray_form_factor_params
        self.kernel_width = config.kernel_width

    @classmethod
    def from_config_dict(cls, cfg: Dict[str, Any]) -> "SpectrumCalculator":
        return cls(ScatteringConfig.from_dict(cfg))

    def compute_structure_factor_direct(
        self, 
        symbols: List[str], 
        positions: torch.Tensor, 
        cell: torch.Tensor, 
        q_bins: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the neutron structure factor S(Q) directly using the Debye formula.
        
        This is more efficient when only S(Q) is needed (no r-space data).
        
        S(Q) = 1/N * |Σ_j b_j * exp(i Q·r_j)|² averaged over Q directions
        
        For powder diffraction (isotropic average):
        S(Q) = 1 + (1/N) * Σ_{i≠j} b_i*b_j * sin(Q*r_ij)/(Q*r_ij) / <b>²
        
        Parameters
        ----------
        symbols : List[str]
            Chemical symbols for each atom
        positions : torch.Tensor
            Atomic positions (N, 3)
        cell : torch.Tensor
            Unit cell (3, 3)
        q_bins : torch.Tensor
            Q values to evaluate S(Q) at
            
        Returns
        -------
        torch.Tensor
            S(Q) values
        """
        device = positions.device
        dtype = positions.dtype
        n_atoms = len(symbols)
        n_q = len(q_bins)
        
        # Get scattering lengths
        b = torch.tensor(
            [self.neutron_scattering_lengths.get(s, 0.0) for s in symbols],
            device=device, dtype=dtype
        )
        b_mean = b.mean()
        b_sq_mean = (b ** 2).mean()
        
        if b_mean.abs() < 1e-10:
            return torch.ones(n_q, device=device, dtype=dtype)
        
        # Compute all pairwise distances
        # Use efficient vectorized computation
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 3)
        
        # Apply minimum image convention for periodic boundaries
        # Convert to fractional coordinates
        cell_inv = torch.linalg.inv(cell)
        diff_frac = torch.einsum('ijk,kl->ijl', diff, cell_inv)
        diff_frac = diff_frac - torch.round(diff_frac)
        diff = torch.einsum('ijk,kl->ijl', diff_frac, cell)
        
        # Compute distances
        r_ij = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-10)  # (N, N)
        
        # Get weight matrix b_i * b_j / <b>²
        weights = torch.outer(b, b) / (b_mean ** 2)  # (N, N)
        
        # Compute S(Q) using Debye formula
        # S(Q) = (1/N) * Σ_{i,j} w_ij * sin(Q*r_ij)/(Q*r_ij)
        S_Q = torch.zeros(n_q, device=device, dtype=dtype)
        
        # Vectorized computation over Q
        Q = q_bins.unsqueeze(-1).unsqueeze(-1)  # (n_q, 1, 1)
        r = r_ij.unsqueeze(0)  # (1, N, N)
        Qr = Q * r  # (n_q, N, N)
        
        # sin(Qr)/(Qr) with limit handling for Qr->0
        sinc_Qr = torch.where(
            Qr.abs() < 1e-8,
            torch.ones_like(Qr),
            torch.sin(Qr) / (Qr + 1e-10)
        )
        
        # Weight and sum
        w = weights.unsqueeze(0)  # (1, N, N)
        S_Q = (w * sinc_Qr).sum(dim=(-2, -1)) / n_atoms
        
        return S_Q

    def compute_neutron_rdf(self, symbols, positions, cell, r_bins):
        return get_neutron_total_rdf(
            symbols,
            positions,
            cell,
            r_bins,
            kernel_width=self.kernel_width,
            scattering_lengths=self.neutron_scattering_lengths,
        )
    
    # Simpler alias
    compute_rdf = compute_neutron_rdf

    def compute_neutron_sf(self, total_rdf: torch.Tensor, r_bins: torch.Tensor, q_bins: torch.Tensor, symbols: List[str], cell: torch.Tensor) -> torch.Tensor:
        return get_neutron_total_scattering_sf(
            total_rdf,
            r_bins,
            q_bins,
            symbols,
            cell,
        )
    
    # Simpler alias
    compute_sf = compute_neutron_sf

    def compute_xray_sf(self, symbols: List[str], positions: torch.Tensor, cell: torch.Tensor, q_bins: torch.Tensor, r_bins: torch.Tensor, kernel_width: float ) -> torch.Tensor:

        return get_xray_total_scattering_sf(
            symbols,
            positions,
            cell,
            q_bins,
            r_bins,
            kernel_width=self.kernel_width,
            form_factor_params=self.xray_form_factor_params,
        )

    def compute_xray_rdf(self, total_sf: torch.Tensor, r_bins: torch.Tensor, q_bins: torch.Tensor, symbols: List[str], cell: torch.Tensor) -> torch.Tensor:
        return get_xray_total_rdf(total_sf, r_bins, q_bins, symbols, cell)

    def compute_approximate_xray_rdf(self, symbols: List[str], positions: torch.Tensor, cell: torch.Tensor, r_bins: torch.Tensor, kernel_width: float ) -> torch.Tensor:
        return get_approximate_xray_total_rdf(
            symbols,
            positions,
            cell,
            r_bins,
            kernel_width=self.kernel_width,
            form_factor_params=self.xray_form_factor_params,
        )

    def compute_neutron_correlation(
            self,
            total_rdf: torch.Tensor,
            r_bins: torch.Tensor,
            chemical_symbols: list[str],
            cell: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the neutron total‑correlation function **T(r)** from a
        pre‑computed total RDF **G(r)**.

        Parameters
        ----------
        total_rdf : torch.Tensor
            1‑D tensor containing G(r) evaluated on `r_bins`.
        r_bins : torch.Tensor
            1‑D tensor of radial distances (Å) that correspond to `total_rdf`.
        chemical_symbols : list[str]
            List of element symbols matching the order of atoms in `cell`.
        cell : torch.Tensor
            (3, 3) tensor of lattice vectors in Å.

        Returns
        -------
        torch.Tensor
            1‑D tensor T(r) evaluated on `r_bins`.
        """
        return get_neutron_total_correlation_function(
            total_rdf,
            r_bins,
            chemical_symbols,
            cell,
            scattering_lengths=self.neutron_scattering_lengths,
        )
    
    # Simpler alias
    compute_correlation = compute_neutron_correlation




