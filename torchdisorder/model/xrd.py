"""
model/xrd.py – XRD/Neutron Diffraction Model for Structure Optimization
=======================================================================

Differentiable model that computes scattering spectra from atomic structures.
Supports both neutron and X-ray scattering with proper weighting factors.

Outputs:
    - G_r: Reduced pair distribution function
    - T_r: Total correlation function  
    - S_Q: Structure factor
    - F_Q: Reduced structure factor

Usage:
    >>> model = XRDModel(
    ...     symbols=['Li', 'P', 'S'],
    ...     config=config_dict,
    ...     r_bins=r_bins,
    ...     q_bins=q_bins,
    ... )
    >>> results = model(sim_state)
    >>> S_Q = results['S_Q']
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import torch
import torch.nn as nn

import torch_sim as ts
from torch_sim.io import state_to_atoms

from torchdisorder.model.scattering import UnifiedSpectrumCalculator, ScatteringConfig


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class XRDModelConfig:
    """Configuration for XRD model."""
    
    # Scattering parameters
    neutron_scattering_lengths: Dict[str, float]
    xray_form_factor_params: Dict[str, Dict[str, List[float]]]
    kernel_width: float = 0.1
    
    # Output control
    compute_neutron: bool = True
    compute_xray: bool = False
    
    # Scattering type for primary output: 'neutron' or 'xray'
    scattering_type: str = 'neutron'
    
    # Species filter for neighbors (e.g., only S for P-S correlations)
    neighbor_species: Optional[List[str]] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'XRDModelConfig':
        return cls(
            neutron_scattering_lengths=d.get('neutron_scattering_lengths', {}),
            xray_form_factor_params=d.get('xray_form_factor_params', {}),
            kernel_width=d.get('kernel_width', 0.1),
            compute_neutron=d.get('compute_neutron', True),
            compute_xray=d.get('compute_xray', False),
            scattering_type=d.get('scattering_type', 'neutron'),
            neighbor_species=d.get('neighbor_species', None),
        )


# =============================================================================
# XRD Model
# =============================================================================

class XRDModel(nn.Module):
    """
    Differentiable model for computing scattering spectra.
    
    Takes atomic structure (positions, cell) and computes:
    - G_r: Reduced pair distribution function G(r) = 4πρr[g(r) - 1]
    - T_r: Total correlation function T(r) = 4πρr·g(r)
    - S_Q: Structure factor S(Q)
    - F_Q: Reduced structure factor F(Q) = Q[S(Q) - 1]
    
    Args:
        symbols: List of element symbols in structure
        config: Configuration dictionary or XRDModelConfig
        r_bins: Tensor of r values for real-space functions
        q_bins: Tensor of Q values for reciprocal-space functions
        rdf_data: Optional target RDF data (for backward compat)
        device: Computation device
    
    Example:
        >>> model = XRDModel(
        ...     symbols=['Li', 'P', 'S'],
        ...     config={'neutron_scattering_lengths': {'Li': -1.90, 'P': 5.13, 'S': 2.847}},
        ...     r_bins=torch.linspace(0.5, 10.0, 200),
        ...     q_bins=torch.linspace(0.5, 15.0, 300),
        ... )
        >>> results = model(sim_state)
    """
    
    def __init__(
        self,
        symbols: List[str],
        config: Dict[str, Any],
        r_bins: torch.Tensor,
        q_bins: torch.Tensor,
        rdf_data: Optional[Any] = None,  # TargetRDFData for backward compat
        device: str = 'cuda',
    ):
        super().__init__()
        
        self.symbols = symbols
        self.device = torch.device(device)
        
        # Parse config
        if isinstance(config, XRDModelConfig):
            self.config = config
        else:
            self.config = XRDModelConfig.from_dict(config)
        
        # Store bins as buffers (not parameters)
        self.register_buffer('r_bins', r_bins.to(self.device))
        self.register_buffer('q_bins', q_bins.to(self.device))
        
        # Create spectrum calculator
        scatt_config = ScatteringConfig(
            neutron_scattering_lengths=self.config.neutron_scattering_lengths,
            xray_form_factor_params=self.config.xray_form_factor_params,
            kernel_width=self.config.kernel_width,
        )
        self.calculator = UnifiedSpectrumCalculator(scatt_config)
        
        # Store RDF data for backward compatibility
        self.rdf_data = rdf_data
        
        # Logging flag
        self._logged_once = False
    
    def forward(
        self,
        state: ts.SimState,
        compute_all: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute scattering spectra from atomic structure.
        
        Args:
            state: torch_sim SimState with positions and cell
            compute_all: If True, compute all spectra. If False, only S_Q.
        
        Returns:
            Dict with keys: 'G_r', 'T_r', 'S_Q', 'F_Q' (depending on config)
        """
        # Extract structure info
        positions = state.positions
        cell = state.cell
        
        # Handle cell dimensions
        if cell.ndim == 3:
            cell = cell[0]
        
        # Get symbols (use stored or extract from state)
        if hasattr(state, 'atomic_numbers') and state.atomic_numbers is not None:
            symbols = self._atomic_numbers_to_symbols(state.atomic_numbers)
        else:
            # Try to get from atoms
            try:
                atoms = state_to_atoms(state)[0]
                symbols = atoms.get_chemical_symbols()
            except:
                symbols = self.symbols
        
        if not self._logged_once:
            print(f"\nXRDModel forward pass:")
            print(f"  Positions: {positions.shape}")
            print(f"  Cell: {cell.shape}")
            print(f"  Symbols: {len(symbols)} atoms ({set(symbols)})")
            print(f"  r_bins: {self.r_bins.shape}")
            print(f"  q_bins: {self.q_bins.shape}")
            self._logged_once = True
        
        # Compute spectra
        results = {}
        
        # Use configured scattering type
        scattering_type = self.config.scattering_type
        
        if compute_all:
            # Compute all spectra efficiently
            all_spectra = self.calculator.compute_all(
                symbols=symbols,
                positions=positions,
                cell=cell,
                r_bins=self.r_bins,
                q_bins=self.q_bins,
                scattering_type=scattering_type,
            )
            results.update(all_spectra)
        else:
            # Just compute S(Q)
            results['S_Q'] = self.calculator.compute(
                symbols=symbols,
                positions=positions,
                cell=cell,
                r_bins=self.r_bins,
                q_bins=self.q_bins,
                output='S_Q',
                scattering_type=scattering_type,
            )
        
        # Compute both neutron and X-ray if requested
        if self.config.compute_xray and scattering_type != 'xray':
            results['S_Q_xray'] = self.calculator.compute(
                symbols=symbols,
                positions=positions,
                cell=cell,
                r_bins=self.r_bins,
                q_bins=self.q_bins,
                output='S_Q',
                scattering_type='xray',
            )
        
        if self.config.compute_neutron and scattering_type != 'neutron':
            results['S_Q_neutron'] = self.calculator.compute(
                symbols=symbols,
                positions=positions,
                cell=cell,
                r_bins=self.r_bins,
                q_bins=self.q_bins,
                output='S_Q',
                scattering_type='neutron',
            )
        
        return results
    
    def _atomic_numbers_to_symbols(self, atomic_numbers: torch.Tensor) -> List[str]:
        """Convert atomic numbers tensor to symbol list."""
        from ase.data import chemical_symbols as ase_symbols
        return [ase_symbols[z] for z in atomic_numbers.cpu().tolist()]
    
    def compute_structure_factor(
        self,
        state: ts.SimState,
        method: str = 'via_rdf',
    ) -> torch.Tensor:
        """
        Compute S(Q) with choice of method.
        
        Args:
            state: SimState
            method: 'via_rdf' (default) or 'direct' (Debye formula)
        """
        positions = state.positions
        cell = state.cell if state.cell.ndim == 2 else state.cell[0]
        
        if hasattr(state, 'atomic_numbers') and state.atomic_numbers is not None:
            symbols = self._atomic_numbers_to_symbols(state.atomic_numbers)
        else:
            symbols = self.symbols
        
        if method == 'direct':
            return self.calculator.compute(
                symbols=symbols,
                positions=positions,
                cell=cell,
                q_bins=self.q_bins,
                output='S_Q',
                scattering_type='neutron',
            )
        else:
            return self.calculator.compute(
                symbols=symbols,
                positions=positions,
                cell=cell,
                r_bins=self.r_bins,
                q_bins=self.q_bins,
                output='S_Q',
                scattering_type='neutron',
            )


# =============================================================================
# Factory Functions
# =============================================================================

def create_xrd_model(
    atoms,
    config: Dict[str, Any],
    r_range: tuple = (0.5, 10.0),
    q_range: tuple = (0.5, 15.0),
    n_r: int = 200,
    n_q: int = 300,
    device: str = 'cuda',
) -> XRDModel:
    """
    Factory function to create XRD model from ASE Atoms.
    
    Args:
        atoms: ASE Atoms object
        config: Configuration dict with scattering parameters
        r_range: (r_min, r_max) for r-space
        q_range: (q_min, q_max) for Q-space
        n_r: Number of r points
        n_q: Number of Q points
        device: Computation device
    """
    symbols = atoms.get_chemical_symbols()
    
    r_bins = torch.linspace(*r_range, n_r, device=device)
    q_bins = torch.linspace(*q_range, n_q, device=device)
    
    return XRDModel(
        symbols=symbols,
        config=config,
        r_bins=r_bins,
        q_bins=q_bins,
        device=device,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'XRDModelConfig',
    'XRDModel',
    'create_xrd_model',
]
