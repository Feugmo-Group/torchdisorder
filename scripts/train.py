"""
TorchDisorder Training Script v2
================================
Constrained optimization for amorphous structure generation from diffraction data.

Uses Augmented Lagrangian method to minimize chi-squared error between computed
and experimental diffraction spectra while satisfying structural constraints.

Workflow:
    1. Load structure from CIF file
    2. Load experimental diffraction data (S(Q), T(r), or g(r))
    3. Load structural constraints from JSON (optional)
    4. Optimize atomic positions to match experimental data
    5. Track progress with wandb live plots

Target Spectra:
    - S_Q: Structure factor S(Q) in reciprocal space [most common]
    - T_r: Total correlation function T(r) in real space
    - g_r: Pair distribution function g(r) in real space

Relationships:
    S(Q) <-> Fourier transform <-> g(r)
    T(r) = 4πρr[g(r) - 1]

Usage:
    python train.py                                  # Default: S(Q) target
    python train.py target=T_r                       # Train on T(r)
    python train.py target=g_r                       # Train on g(r)
    python train.py constraints.enabled=false        # No constraints
    python train.py constraints.use_types=[tet,cn]   # Only specific constraints
    python train.py max_steps=100000                 # Longer training
    python train.py wandb=disabled                   # Disable wandb logging

WandB Logging:
    - loss: Total optimization loss
    - loss_reduction_pct: Percentage reduction from initial
    - S_Q_loss / T_r_loss / g_r_loss: Individual spectrum losses
    - avg_violation, max_violation, num_violated: Constraint metrics
    - spectrum/S(Q): Live spectrum comparison plot
    - metrics/R2, metrics/RMSE: Fit quality metrics
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
import numpy as np
from pathlib import Path
import time
import signal
import sys
import json
from typing import Dict, List, Optional, Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plotly for HTML animations
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("WARNING: Plotly not available, HTML animations disabled")

from torch_sim.io import atoms_to_state, state_to_atoms
from ase.data import chemical_symbols
from ase.io import write, read
import cooper

from torchdisorder.common.utils import MODELS_PROJECT_ROOT
from torchdisorder.common.target_rdf import TargetRDFData
from torchdisorder.model.generator import generate_atoms_from_config
from torchdisorder.model.xrd import XRDModel
from torchdisorder.model.loss import CooperLoss
from torchdisorder.engine.optimizer import StructureFactorCMPWithConstraints


def to_dict(obj):
    """Safely convert OmegaConf or dict to plain dict."""
    if obj is None:
        return {}
    # Check if it's an OmegaConf object
    if hasattr(OmegaConf, 'is_config') and OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    # Legacy check for older OmegaConf versions
    if hasattr(obj, '_iter_ex'):
        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return dict(obj)  # Make a copy
    # Try to convert to dict if possible
    try:
        return dict(obj)
    except (TypeError, ValueError):
        return {}


# =============================================================================
# TARGET SPECTRUM CONFIGURATION
# =============================================================================
# Maps target type to data attributes and plot labels.
# The model computes all spectra (G_r, T_r, S_Q) but loss is computed only
# for the selected target.

TARGET_CONFIG = {
    'S_Q': {
        'name': 'Structure Factor',
        'symbol': 'S(Q)',
        'xlabel': 'Q (Å⁻¹)',
        'ylabel': 'S(Q)',
        'space': 'Q',
        'target_attr': 'S_Q_target',
        'uncert_attr': 'S_Q_uncert',
        'bins_attr': 'q_bins',
        'output_key': 'S_Q',  # Key in model output dict
    },
    'F_Q': {
        'name': 'Reduced Structure Factor',
        'symbol': 'F(Q)',
        'xlabel': 'Q (Å⁻¹)',
        'ylabel': 'F(Q) = Q[S(Q)-1]',
        'space': 'Q',
        'target_attr': 'F_q_target',  # Use F(Q) target directly
        'uncert_attr': 'F_q_uncert',  # Use F(Q) uncertainty directly
        'bins_attr': 'q_bins',
        'output_key': 'F_Q',  # Model computes F(Q) = Q[S(Q)-1]
    },
    'T_r': {
        'name': 'Total Correlation Function',
        'symbol': 'T(r)',
        'xlabel': 'r (Å)',
        'ylabel': 'T(r)',
        'space': 'r',
        'target_attr': 'T_r_target',
        'uncert_attr': 'T_r_uncert',
        'bins_attr': 'r_bins',
        'output_key': 'T_r',
    },
    'g_r': {
        'name': 'Pair Distribution Function',
        'symbol': 'g(r)',
        'xlabel': 'r (Å)',
        'ylabel': 'g(r)',
        'space': 'r',
        'target_attr': 'g_r_target',
        'uncert_attr': 'g_r_uncert',
        'bins_attr': 'r_bins',
        'output_key': 'g_r',  # Model outputs g_r
    },
    'G_r': {
        'name': 'Reduced Pair Distribution Function',
        'symbol': 'G(r)',
        'xlabel': 'r (Å)',
        'ylabel': 'G(r) = 4πρr[g(r)-1]',
        'space': 'r',
        'target_attr': 'G_r_target',
        'uncert_attr': 'G_r_uncert',
        'bins_attr': 'r_bins',
        'output_key': 'G_r',  # Model outputs G_r
    },
}


# =============================================================================
# PLOTTING CLASS
# =============================================================================

class SpectraPlotter:
    """Plotter for diffraction spectra with focus on selected target."""
    
    ELEMENT_COLORS = {
        'Si': '#F0E68C', 'O': '#FF6347', 'Ge': '#9370DB', 'P': '#FFA500',
        'S': '#FFFF00', 'Li': '#00FF00', 'Na': '#0000FF', 'Fe': '#8B4513',
        'N': '#00CED1', 'Cl': '#32CD32', 'Ta': '#708090',
    }
    ELEMENT_SIZES = {
        'Si': 100, 'O': 60, 'Ge': 120, 'P': 90, 'S': 80,
        'Li': 50, 'Na': 70, 'Fe': 100, 'N': 55, 'Cl': 75, 'Ta': 130
    }
    
    def __init__(self, output_dir: Path, target_type: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_type = target_type
        self.target_config = TARGET_CONFIG.get(target_type, TARGET_CONFIG['S_Q'])
        
        # Loss history
        self.history = {
            'step': [], 'total_loss': [], 'chi2_loss': [],
            'avg_violation': [], 'max_violation': [], 'num_violated': [],
        }
        
        # Spectrum evolution for animation
        self.spectrum_history = []  # List of (step, x, y_pred, y_target)
    
    def update_history(self, step: int, losses: Dict[str, float]):
        """Record loss values."""
        self.history['step'].append(step)
        for key in ['total_loss', 'chi2_loss', 'avg_violation', 'max_violation', 'num_violated']:
            self.history[key].append(losses.get(key, 0.0))
    
    def record_spectrum(self, step: int, x: np.ndarray, y_pred: np.ndarray, y_target: np.ndarray):
        """Record spectrum for animation."""
        self.spectrum_history.append({
            'step': step,
            'x': x.copy(),
            'y_pred': y_pred.copy(),
            'y_target': y_target.copy()
        })
    
    def log_spectrum_to_wandb(
        self,
        rdf_data: TargetRDFData,
        prediction: torch.Tensor,
        step: int
    ):
        """Log spectrum comparison to wandb as a custom chart."""
        cfg = self.target_config
        
        # Get bins
        bins_attr = cfg['bins_attr']
        if hasattr(rdf_data, bins_attr) and getattr(rdf_data, bins_attr) is not None:
            x = getattr(rdf_data, bins_attr).cpu().numpy()
        else:
            x = np.arange(len(prediction.reshape(-1)))
        
        # Get target
        target_attr = cfg['target_attr']
        if hasattr(rdf_data, target_attr) and getattr(rdf_data, target_attr) is not None:
            y_target = getattr(rdf_data, target_attr).cpu().numpy()
        else:
            return
        
        # Get prediction
        y_pred = prediction.detach().cpu().numpy().reshape(-1)
        
        # Ensure same length
        min_len = min(len(x), len(y_target), len(y_pred))
        x, y_target, y_pred = x[:min_len], y_target[:min_len], y_pred[:min_len]
        
        # Record for animation
        self.record_spectrum(step, x, y_pred, y_target)
        
        # Compute R² and RMSE
        mask = ~np.isnan(y_target) & ~np.isnan(y_pred)
        if mask.sum() > 0:
            ss_res = np.sum((y_target[mask] - y_pred[mask])**2)
            ss_tot = np.sum((y_target[mask] - y_target[mask].mean())**2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            rmse = np.sqrt(np.mean((y_target[mask] - y_pred[mask])**2))
        else:
            r2, rmse = 0.0, 0.0
        
        # Log as wandb table for custom chart
        # Subsample to avoid too many points
        subsample = max(1, len(x) // 200)
        x_sub = x[::subsample]
        y_target_sub = y_target[::subsample]
        y_pred_sub = y_pred[::subsample]
        
        # Create wandb table
        data = [[xi, yt, yp] for xi, yt, yp in zip(x_sub, y_target_sub, y_pred_sub)]
        table = wandb.Table(data=data, columns=[cfg['xlabel'], "Target", "Predicted"])
        
        # Log line plot
        wandb.log({
            f"spectrum/{cfg['symbol']}": wandb.plot.line_series(
                xs=x_sub.tolist(),
                ys=[y_target_sub.tolist(), y_pred_sub.tolist()],
                keys=["Target (Exp)", "Predicted (Model)"],
                title=f"{cfg['symbol']} - Step {step}",
                xname=cfg['xlabel']
            ),
            f"metrics/{cfg['symbol']}_R2": r2,
            f"metrics/{cfg['symbol']}_RMSE": rmse,
        }, step=step)

    def plot_experimental_spectrum(self, rdf_data: TargetRDFData, data_files: Dict[str, str]):
        """Plot the experimental spectrum that was loaded."""
        cfg = self.target_config
        
        # Get bins
        bins_attr = cfg['bins_attr']
        if hasattr(rdf_data, bins_attr) and getattr(rdf_data, bins_attr) is not None:
            bins_data = getattr(rdf_data, bins_attr)
            if bins_data.numel() == 0:
                return {}
            x = bins_data.cpu().numpy()
        else:
            return {}
        
        # Get target
        target_attr = cfg['target_attr']
        if hasattr(rdf_data, target_attr) and getattr(rdf_data, target_attr) is not None:
            target_data = getattr(rdf_data, target_attr)
            if target_data.numel() == 0:
                print(f"  Warning: {target_attr} is empty, skipping experimental plot")
                return {}
            y_target = target_data.cpu().numpy()
        else:
            return {}
        
        # Get uncertainty if available
        uncert_attr = cfg['uncert_attr']
        y_uncert = None
        if hasattr(rdf_data, uncert_attr) and getattr(rdf_data, uncert_attr) is not None:
            uncert_data = getattr(rdf_data, uncert_attr)
            if uncert_data.numel() > 0 and len(uncert_data) == len(y_target):
                y_uncert = uncert_data.cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot uncertainty band
        if y_uncert is not None:
            ax.fill_between(x, y_target - y_uncert, y_target + y_uncert,
                           alpha=0.3, color='blue', label='Uncertainty (σ)')
        
        # Plot target
        ax.plot(x, y_target, 'b-', linewidth=2, label='Experimental Data', alpha=0.9)
        
        # Labels
        ax.set_xlabel(cfg['xlabel'], fontsize=14)
        ax.set_ylabel(cfg['ylabel'], fontsize=14)
        
        # Title with file info
        file_info = data_files.get(f'{cfg["target_attr"]}_file', 'Unknown')
        title = f"Experimental {cfg['name']} {cfg['symbol']}\nFile: {Path(file_info).name if file_info else 'N/A'}"
        ax.set_title(title, fontsize=14)
        
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        png_path = self.output_dir / f"experimental_{self.target_type}.png"
        pdf_path = self.output_dir / f"experimental_{self.target_type}.pdf"
        
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        
        # Log to wandb
        wandb.log({"experimental_spectrum": wandb.Image(str(png_path))})
        
        return {'png': png_path, 'pdf': pdf_path}
    
    def create_animation_html(self, prefix: str = "") -> Optional[Path]:
        """Create Plotly HTML animation of spectrum evolution."""
        if not PLOTLY_AVAILABLE or len(self.spectrum_history) < 2:
            return None
        
        cfg = self.target_config
        
        # Create figure with animation frames
        fig = go.Figure()
        
        # Get target (constant across all frames)
        y_target = self.spectrum_history[0]['y_target']
        x = self.spectrum_history[0]['x']
        
        # Add target trace (always visible)
        fig.add_trace(go.Scatter(
            x=x, y=y_target,
            mode='lines',
            name='Target (Experimental)',
            line=dict(color='blue', width=2),
            hovertemplate=f'{cfg["xlabel"]}: %{{x:.3f}}<br>{cfg["ylabel"]}: %{{y:.4f}}<extra>Target</extra>'
        ))
        
        # Add initial predicted trace
        fig.add_trace(go.Scatter(
            x=x, y=self.spectrum_history[0]['y_pred'],
            mode='lines',
            name='Predicted (Model)',
            line=dict(color='red', width=2),
            hovertemplate=f'{cfg["xlabel"]}: %{{x:.3f}}<br>{cfg["ylabel"]}: %{{y:.4f}}<extra>Predicted</extra>'
        ))
        
        # Create frames for animation
        frames = []
        for record in self.spectrum_history:
            step = record['step']
            y_pred = record['y_pred']
            
            # Compute R² for this frame
            mask = ~np.isnan(y_target) & ~np.isnan(y_pred)
            if mask.sum() > 0:
                ss_res = np.sum((y_target[mask] - y_pred[mask])**2)
                ss_tot = np.sum((y_target[mask] - y_target[mask].mean())**2)
                r2 = 1 - ss_res / (ss_tot + 1e-10)
            else:
                r2 = 0.0
            
            frame = go.Frame(
                data=[
                    go.Scatter(x=x, y=y_target, mode='lines', line=dict(color='blue', width=2)),
                    go.Scatter(x=x, y=y_pred, mode='lines', line=dict(color='red', width=2))
                ],
                name=str(step),
                layout=go.Layout(
                    title=f"{cfg['name']} {cfg['symbol']} - Step {step} (R² = {r2:.4f})"
                )
            )
            frames.append(frame)
        
        fig.frames = frames
        
        # Add slider and play/pause buttons
        fig.update_layout(
            title=f"{cfg['name']} {cfg['symbol']} - Training Evolution",
            xaxis_title=cfg['xlabel'],
            yaxis_title=cfg['ylabel'],
            hovermode='x unified',
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=1.15,
                    x=0.5,
                    xanchor='center',
                    buttons=[
                        dict(
                            label='▶ Play',
                            method='animate',
                            args=[None, dict(
                                frame=dict(duration=200, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=100)
                            )]
                        ),
                        dict(
                            label='⏸ Pause',
                            method='animate',
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=False),
                                mode='immediate',
                                transition=dict(duration=0)
                            )]
                        )
                    ]
                )
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor='top',
                    xanchor='left',
                    currentvalue=dict(
                        font=dict(size=16),
                        prefix='Step: ',
                        visible=True,
                        xanchor='center'
                    ),
                    transition=dict(duration=100),
                    pad=dict(b=10, t=50),
                    len=0.9,
                    x=0.05,
                    y=0,
                    steps=[
                        dict(
                            args=[[str(record['step'])], dict(
                                frame=dict(duration=100, redraw=True),
                                mode='immediate',
                                transition=dict(duration=100)
                            )],
                            label=str(record['step']),
                            method='animate'
                        ) for record in self.spectrum_history
                    ]
                )
            ]
        )
        
        # Save HTML
        html_path = self.output_dir / f"{prefix}spectrum_evolution.html"
        fig.write_html(str(html_path), include_plotlyjs='cdn')
        
        print(f"Animation saved: {html_path}")
        return html_path
    
    def plot_target_spectrum(
        self,
        rdf_data: TargetRDFData,
        prediction: torch.Tensor,
        step: Optional[int] = None,
        prefix: str = ""
    ) -> Dict[str, Path]:
        """Plot the target spectrum (target vs predicted)."""
        
        cfg = self.target_config
        
        # Get bins
        bins_attr = cfg['bins_attr']
        if hasattr(rdf_data, bins_attr) and getattr(rdf_data, bins_attr) is not None:
            x = getattr(rdf_data, bins_attr).cpu().numpy()
        else:
            x = np.arange(len(prediction))
        
        # Get target
        target_attr = cfg['target_attr']
        if hasattr(rdf_data, target_attr) and getattr(rdf_data, target_attr) is not None:
            y_target = getattr(rdf_data, target_attr).cpu().numpy()
        else:
            print(f"WARNING: Target {target_attr} not available in rdf_data")
            return {}
        
        # Get prediction
        y_pred = prediction.detach().cpu().numpy().reshape(-1)
        
        # Ensure same length
        min_len = min(len(x), len(y_target), len(y_pred))
        x, y_target, y_pred = x[:min_len], y_target[:min_len], y_pred[:min_len]
        
        # Get uncertainty if available
        uncert_attr = cfg['uncert_attr']
        y_uncert = None
        if hasattr(rdf_data, uncert_attr) and getattr(rdf_data, uncert_attr) is not None:
            y_uncert = getattr(rdf_data, uncert_attr).cpu().numpy()[:min_len]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot uncertainty band
        if y_uncert is not None:
            ax.fill_between(x, y_target - y_uncert, y_target + y_uncert,
                           alpha=0.3, color='blue', label='Target ± σ')
        
        # Plot target and prediction
        ax.plot(x, y_target, 'b-', linewidth=2, label='Target (Experimental)', alpha=0.8)
        ax.plot(x, y_pred, 'r-', linewidth=2, label='Predicted (Model)', alpha=0.9)
        
        # Compute metrics
        mask = ~np.isnan(y_target) & ~np.isnan(y_pred)
        if mask.sum() > 0:
            ss_res = np.sum((y_target[mask] - y_pred[mask])**2)
            ss_tot = np.sum((y_target[mask] - y_target[mask].mean())**2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            rmse = np.sqrt(np.mean((y_target[mask] - y_pred[mask])**2))
            
            textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', horizontalalignment='right', bbox=props)
        
        # Labels
        ax.set_xlabel(cfg['xlabel'], fontsize=12)
        ax.set_ylabel(cfg['ylabel'], fontsize=12)
        
        title = f"{cfg['name']} {cfg['symbol']}"
        if step is not None:
            title += f' - Step {step}'
        ax.set_title(title, fontsize=14)
        
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        base = f"{prefix}{self.target_type}"
        png_path = self.output_dir / f"{base}.png"
        pdf_path = self.output_dir / f"{base}.pdf"
        
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        
        return {'png': png_path, 'pdf': pdf_path}
    
    def plot_loss_history(self, prefix: str = "") -> Optional[Dict[str, Path]]:
        """Plot loss curves over training."""
        
        steps = np.array(self.history['step'])
        if len(steps) < 2:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax = axes[0]
        losses = np.array(self.history['total_loss'])
        ax.semilogy(steps, losses, 'b-', linewidth=2, label=f'{self.target_config["symbol"]} Loss')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Loss (log scale)', fontsize=12)
        ax.set_title(f'Training Loss - Target: {self.target_config["symbol"]}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Violations plot
        ax = axes[1]
        if any(v > 0 for v in self.history['avg_violation']):
            ax.plot(steps, self.history['avg_violation'], 'b-', linewidth=2, label='Avg Violation')
            ax.plot(steps, self.history['max_violation'], 'r--', linewidth=1.5, label='Max Violation')
            ax.legend()
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Violation', fontsize=12)
        ax.set_title('Constraint Violations', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        png_path = self.output_dir / f'{prefix}loss_history.png'
        pdf_path = self.output_dir / f'{prefix}loss_history.pdf'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        
        return {'png': png_path, 'pdf': pdf_path}
    
    def plot_structure_3d(self, atoms, title: str = "Structure", prefix: str = "") -> Dict[str, Path]:
        """Plot 3D structure."""
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        unique_symbols = list(set(symbols))
        
        for symbol in unique_symbols:
            mask = np.array([s == symbol for s in symbols])
            pos = positions[mask]
            color = self.ELEMENT_COLORS.get(symbol, '#808080')
            size = self.ELEMENT_SIZES.get(symbol, 80)
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=color, s=size, label=symbol, alpha=0.8, edgecolors='black', linewidths=0.5)
        
        # Draw unit cell
        cell = atoms.get_cell()
        origin = np.array([0, 0, 0])
        a, b, c = cell[0], cell[1], cell[2]
        edges = [
            (origin, origin + a), (origin, origin + b), (origin, origin + c),
            (origin + a, origin + a + b), (origin + a, origin + a + c),
            (origin + b, origin + b + a), (origin + b, origin + b + c),
            (origin + c, origin + c + a), (origin + c, origin + c + b),
            (origin + a + b, origin + a + b + c),
            (origin + a + c, origin + a + b + c),
            (origin + b + c, origin + a + b + c),
        ]
        for start, end in edges:
            ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                     'k-', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('x (Å)')
        ax.set_ylabel('y (Å)')
        ax.set_zlabel('z (Å)')
        ax.set_title(title)
        ax.legend(loc='upper left')
        plt.tight_layout()
        
        # Save
        png_path = self.output_dir / f'{prefix}structure_3d.png'
        pdf_path = self.output_dir / f'{prefix}structure_3d.pdf'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        
        return {'png': png_path, 'pdf': pdf_path}


# =============================================================================
# CONSTRAINT FILTERING
# =============================================================================

def filter_constraints(constraints_data: Dict, use_types, enabled: bool = True) -> Dict:
    """Filter constraints based on config settings."""
    
    if not enabled:
        return {
            'cutoff': constraints_data.get('cutoff', 3.5),
            'element_filter': constraints_data.get('element_filter', []),
            'atom_constraints': {},
            'metadata': {'order_parameter_types': [], 'filtered': True}
        }
    
    # Get all available types
    all_types = set()
    for ac in constraints_data.get('atom_constraints', {}).values():
        all_types.update(ac.get('order_parameters', {}).keys())
    all_types = list(all_types)
    
    # Determine selected types
    if use_types is None or use_types == 'all' or (isinstance(use_types, list) and 'all' in use_types):
        selected = all_types
    else:
        selected = [t for t in use_types if t in all_types]
    
    if not selected:
        return filter_constraints(constraints_data, None, enabled=False)
    
    # Filter
    filtered_ac = {}
    for idx, ac in constraints_data.get('atom_constraints', {}).items():
        filtered_ops = {k: v for k, v in ac.get('order_parameters', {}).items() if k in selected}
        if filtered_ops:
            filtered_ac[idx] = {**ac, 'order_parameters': filtered_ops}
    
    return {
        'cutoff': constraints_data.get('cutoff', 3.5),
        'element_filter': constraints_data.get('element_filter', []),
        'atom_constraints': filtered_ac,
        'metadata': {'order_parameter_types': selected, 'filtered': True}
    }


# =============================================================================
# CIF ANALYSIS HELPERS
# =============================================================================

def get_stoichiometry_from_cif(cif_path: str) -> Dict[str, Any]:
    """
    Read CIF file and extract actual stoichiometry and cell information.
    
    Returns dict with:
        - species: list of unique elements
        - stoichiometry: list of counts for each species
        - composition: dict {element: count}
        - cell_type: 'cubic', 'tetragonal', 'orthorhombic', 'hexagonal', 'monoclinic', 'triclinic'
        - cell_params: dict with a, b, c, alpha, beta, gamma
        - is_cubic: bool
        - formula: string like "Si375O750"
    """
    try:
        atoms = read(cif_path)
        symbols = atoms.get_chemical_symbols()
        
        # Get composition
        unique_symbols = sorted(set(symbols))
        composition = {s: symbols.count(s) for s in unique_symbols}
        stoichiometry = [composition[s] for s in unique_symbols]
        
        # Get cell parameters
        cell = atoms.get_cell()
        a, b, c = cell.lengths()
        alpha, beta, gamma = cell.angles()
        
        cell_params = {
            'a': a, 'b': b, 'c': c,
            'alpha': alpha, 'beta': beta, 'gamma': gamma
        }
        
        # Determine cell type
        tol_length = 0.01  # Tolerance for comparing lengths (relative)
        tol_angle = 0.5    # Tolerance for angles (degrees)
        
        a_eq_b = abs(a - b) / max(a, b) < tol_length
        b_eq_c = abs(b - c) / max(b, c) < tol_length
        a_eq_c = abs(a - c) / max(a, c) < tol_length
        all_eq = a_eq_b and b_eq_c
        
        alpha_90 = abs(alpha - 90) < tol_angle
        beta_90 = abs(beta - 90) < tol_angle
        gamma_90 = abs(gamma - 90) < tol_angle
        gamma_120 = abs(gamma - 120) < tol_angle
        all_90 = alpha_90 and beta_90 and gamma_90
        
        if all_eq and all_90:
            cell_type = 'cubic'
        elif a_eq_b and all_90:
            cell_type = 'tetragonal'
        elif all_90 and not all_eq:
            cell_type = 'orthorhombic'
        elif a_eq_b and alpha_90 and beta_90 and gamma_120:
            cell_type = 'hexagonal'
        elif alpha_90 and gamma_90 and not beta_90:
            cell_type = 'monoclinic'
        else:
            cell_type = 'triclinic'
        
        # Build formula string
        formula_parts = [f"{s}{composition[s]}" for s in unique_symbols]
        formula = "".join(formula_parts)
        
        return {
            'species': unique_symbols,
            'stoichiometry': stoichiometry,
            'composition': composition,
            'cell_type': cell_type,
            'cell_params': cell_params,
            'is_cubic': cell_type == 'cubic',
            'formula': formula,
            'n_atoms': len(atoms),
            'volume': atoms.get_volume()
        }
    except Exception as e:
        print(f"WARNING: Could not analyze CIF file: {e}")
        return None


def print_cell_info(cell_info: Dict, cif_path: str):
    """Print formatted cell information."""
    print(f"  CIF file: {cif_path}")
    print(f"  Formula: {cell_info['formula']}")
    print(f"  Total atoms: {cell_info['n_atoms']}")
    print(f"  Composition: {cell_info['composition']}")
    print(f"  Cell type: {cell_info['cell_type'].upper()}")
    
    p = cell_info['cell_params']
    print(f"  Cell parameters:")
    print(f"    a = {p['a']:.4f} Å, b = {p['b']:.4f} Å, c = {p['c']:.4f} Å")
    print(f"    α = {p['alpha']:.2f}°, β = {p['beta']:.2f}°, γ = {p['gamma']:.2f}°")
    print(f"  Volume: {cell_info['volume']:.2f} Å³")
    
    if not cell_info['is_cubic']:
        print(f"  ⚠ WARNING: Non-cubic cell detected ({cell_info['cell_type']})")
        print(f"    This may require special handling in the optimization.")


# =============================================================================
# STATE WRAPPER
# =============================================================================

class StateWrapper:
    def __init__(self, original_state):
        self.__dict__.update(original_state.__dict__)
        self.system_idx = None
        self.n_systems = None


def setup_state_wrapper(state, atoms_list, device):
    wrapped = StateWrapper(state)
    wrapped.n_systems = torch.tensor(len(atoms_list))
    atoms_per_system = torch.tensor([len(a) for a in atoms_list], device=device)
    wrapped.system_idx = torch.repeat_interleave(
        torch.arange(len(atoms_list), device=device), atoms_per_system
    )
    wrapped.atomic_numbers = torch.tensor(
        [chemical_symbols.index(a.symbol) for a in atoms_list[0]], 
        dtype=torch.int64, device=device
    )
    return wrapped


# =============================================================================
# MAIN
# =============================================================================

@hydra.main(config_path=str(MODELS_PROJECT_ROOT / "configs"), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    
    run_timestamp = int(time.time())
    run_name = f"run_{run_timestamp}"
    
    # Get target type from config
    target_type = OmegaConf.select(cfg, 'target', default='S_Q')
    if target_type not in TARGET_CONFIG:
        print(f"WARNING: Invalid target '{target_type}', using S_Q")
        target_type = 'S_Q'
    
    target_cfg = TARGET_CONFIG[target_type]
    
    print(f"\n{'=' * 70}")
    print(f"  TorchDisorder Training v1")
    print(f"{'=' * 70}")
    print(f"  Target Spectrum: {target_cfg['name']} [{target_cfg['symbol']}]")
    print(f"{'=' * 70}\n")
    
    # W&B setup
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(
        project="torchdisorder-optimization",
        config=wandb_config,
        name=f"{run_name}_{target_type}",
        tags=["structure_optimization", target_type]
    )

    dtype = torch.float32
    accelerator = OmegaConf.select(cfg, 'accelerator', default='cuda')
    device = torch.device(accelerator)

    # ==========================================
    # LOAD TARGET DATA
    # ==========================================
    print(f"\n{'=' * 70}")
    print("  TARGET DATA CONFIGURATION")
    print(f"{'=' * 70}")
    
    # Extract stride parameters from config
    data_cfg = cfg.data.data if hasattr(cfg.data, 'data') else cfg.data
    stride_q = OmegaConf.select(data_cfg, 'stride_q', default=1)
    stride_r = OmegaConf.select(data_cfg, 'stride_r', default=1)
    
    if stride_q > 1:
        print(f"  Using stride_q={stride_q} for Q-space subsampling")
    if stride_r > 1:
        print(f"  Using stride_r={stride_r} for r-space subsampling")
    
    # Build complete config dict, merging nested data section with top-level parameters
    rdf_config = to_dict(data_cfg)
    
    # Add top-level parameters that might not be in nested section
    top_level_params = ['r_min', 'r_max', 'q_min', 'q_max', 'n_r_bins', 'kernel_width']
    for param in top_level_params:
        if param not in rdf_config:
            value = OmegaConf.select(cfg.data, param, default=None)
            if value is not None:
                rdf_config[param] = value
    
    rdf_data = TargetRDFData.from_dict(
        rdf_config, 
        device=accelerator,
        stride_q=stride_q,
        stride_r=stride_r
    )
    
    # Print configuration paths
    data_cfg = cfg.data.data
    print("\n  Data Files:")
    print(f"    F(Q)/S(Q) path: {OmegaConf.select(data_cfg, 's_of_q_path', default='null')}")
    print(f"    T(r) path: {OmegaConf.select(data_cfg, 't_of_r_path', default='null')}")
    
    print("\n  Loaded Data:")
    
    # Check Q-space data F(Q)/S(Q)
    if rdf_data.has_q_data():
        q = rdf_data.q_bins.cpu().numpy()
        S_Q = rdf_data.S_Q_target.cpu().numpy()
        F_Q = rdf_data.F_q_target.cpu().numpy()
        original_fmt = "F(Q)" if rdf_data.input_was_F_Q() else "S(Q)"
        print(f"    ✓ Q-space data (original format: {original_fmt}):")
        print(f"        Bins: {len(q)}")
        print(f"        Q range: [{q.min():.2f}, {q.max():.2f}] Å⁻¹")
        print(f"        S(Q) range: [{S_Q.min():.3f}, {S_Q.max():.3f}]")
        print(f"        F(Q) range: [{F_Q.min():.3f}, {F_Q.max():.3f}]")
    else:
        print(f"    ✗ F(Q)/S(Q) data: Not loaded")
    
    # Check T(r) data
    if rdf_data.has_r_data():
        r = rdf_data.r_bins.cpu().numpy()
        T_r = rdf_data.T_r_target.cpu().numpy()
        print(f"    ✓ T(r) data:")
        print(f"        Bins: {len(r)}")
        print(f"        r range: [{r.min():.2f}, {r.max():.2f}] Å")
        print(f"        T range: [{T_r.min():.3f}, {T_r.max():.3f}]")
    else:
        print(f"    ✗ T(r) data: Not loaded")
    
    # Verify target type has data
    if target_type == 'S_Q' and not rdf_data.has_q_data():
        print(f"\n  ⚠️  WARNING: Target S(Q) selected but no Q-space data loaded!")
    elif target_type == 'T_r' and not rdf_data.has_r_data():
        print(f"\n  ⚠️  WARNING: Target T(r) selected but no T(r) data loaded!")
    
    print(f"{'=' * 70}\n")

    # Output directories
    plots_dir = Path(OmegaConf.select(cfg, 'output.plots_dir', default='outputs/plots'))
    trajectory_path = OmegaConf.select(cfg, 'output.trajectory_path', default='outputs/trajectory')
    write_trajectory = OmegaConf.select(cfg, 'output.write_trajectory', default=True)
    
    # Create plotter
    plotter = SpectraPlotter(output_dir=plots_dir, target_type=target_type)
    
    # Plot and log experimental spectrum
    print("Plotting experimental spectrum...")
    data_files = {
        'F_q_target_file': str(data_cfg.get('f_of_q_path', 'N/A')),
        'T_r_target_file': str(data_cfg.get('t_of_r_path', 'N/A')),
    }
    plotter.plot_experimental_spectrum(rdf_data, data_files)

    # ==========================================
    # STRUCTURE GENERATION
    # ==========================================
    print(f"\n{'=' * 70}")
    print("  STRUCTURE GENERATION")
    print(f"{'=' * 70}")
    
    struct_cfg = cfg.structure
    init_type = struct_cfg.get('init', 'unknown')
    print(f"  Initialization type: {init_type}")
    
    cif_info = None
    if init_type == 'cif':
        cif_path = struct_cfg.get('cif_path', None)
        if cif_path and Path(cif_path).exists():
            # Analyze CIF file
            cif_info = get_stoichiometry_from_cif(cif_path)
            if cif_info:
                print_cell_info(cif_info, cif_path)
                
                # Override config stoichiometry with actual CIF values
                print(f"\n  Auto-detected stoichiometry from CIF:")
                print(f"    Species: {cif_info['species']}")
                print(f"    Counts: {cif_info['stoichiometry']}")
        else:
            print(f"  CIF file: {cif_path} (not found)")
    elif init_type == 'random':
        print(f"  Species: {struct_cfg.get('species', 'N/A')}")
        print(f"  Stoichiometry: {struct_cfg.get('stoichiometry', 'N/A')}")
        print(f"  Box length: {struct_cfg.get('box_length', 'N/A')} Å")
        print(f"  Target density: {struct_cfg.get('target_density', 'N/A')} g/cm³")
    
    atoms = generate_atoms_from_config(cfg.structure)
    atoms_list = [atoms]
    
    # Print structure info
    symbols = atoms.get_chemical_symbols()
    unique_symbols = sorted(set(symbols))
    composition = {s: symbols.count(s) for s in unique_symbols}
    cell = atoms.get_cell()
    volume = atoms.get_volume()
    
    # Get cell parameters for any cell type
    a, b, c = cell.lengths()
    alpha, beta, gamma = cell.angles()
    
    print(f"\n  Generated Structure:")
    print(f"    Total atoms: {len(atoms)}")
    print(f"    Composition: {composition}")
    print(f"    Cell vectors:")
    print(f"      a = {a:.4f} Å, b = {b:.4f} Å, c = {c:.4f} Å")
    print(f"      α = {alpha:.2f}°, β = {beta:.2f}°, γ = {gamma:.2f}°")
    print(f"    Volume: {volume:.2f} Å³")
    print(f"    Density: {sum(atoms.get_masses()) / volume * 1.66054:.4f} g/cm³")
    
    # Warn about non-cubic cells
    is_cubic = (abs(a - b)/max(a,b) < 0.01 and abs(b - c)/max(b,c) < 0.01 and 
                abs(alpha - 90) < 0.5 and abs(beta - 90) < 0.5 and abs(gamma - 90) < 0.5)
    if not is_cubic:
        cell_type = cif_info['cell_type'] if cif_info else 'non-cubic'
        print(f"\n  ⚠ WARNING: {cell_type.upper()} cell detected")
        print(f"    The cell is NOT cubic. Ensure your model handles this correctly.")
    
    print(f"{'=' * 70}\n")
    
    state = atoms_to_state(atoms_list, device=accelerator, dtype=dtype)
    state.positions.requires_grad_(True)
    state.cell.requires_grad_(True)
    base_sim_state = setup_state_wrapper(state, atoms_list, device)

    # Prepare model config from data config
    # Note: Hydra merges data config - try multiple possible locations
    neutron_lengths = None
    xray_params = None
    
    # Try to find neutron_scattering_lengths in various locations
    # Hydra loads data config under cfg.data, so check there first
    search_paths_neutron = [
        'data.neutron_scattering_lengths',  # In data yaml at top level
        'neutron_scattering_lengths',       # In main config
        'data.data.neutron_scattering_lengths',  # If nested under data.data
    ]
    for path in search_paths_neutron:
        val = OmegaConf.select(cfg, path, default=None)
        if val is not None:
            converted = to_dict(val)
            if converted and len(converted) > 0:
                neutron_lengths = converted
                print(f"  Found neutron_scattering_lengths at: {path}")
                break
    
    # Try to find xray_form_factor_params
    search_paths_xray = [
        'data.xray_form_factor_params',
        'xray_form_factor_params',
        'data.data.xray_form_factor_params',
    ]
    for path in search_paths_xray:
        val = OmegaConf.select(cfg, path, default=None)
        if val is not None:
            converted = to_dict(val)
            if converted and len(converted) > 0:
                xray_params = converted
                print(f"  Found xray_form_factor_params at: {path}")
                break
    
    # Try to find kernel_width
    kernel_width = OmegaConf.select(cfg, 'data.kernel_width', default=None)
    if kernel_width is None:
        kernel_width = OmegaConf.select(cfg, 'kernel_width', default=None)
    if kernel_width is None:
        kernel_width = OmegaConf.select(cfg, 'data.data.kernel_width', default=0.03)
    
    # Test forward pass
    print(f"\n{'=' * 70}")
    print("  MODEL CONFIGURATION")
    print(f"{'=' * 70}")
    print(f"  kernel_width: {kernel_width}")
    print(f"  neutron_scattering_lengths: {neutron_lengths}")
    print(f"  xray_form_factor_params: {list(xray_params.keys()) if xray_params else 'None'}")
    print(f"  unique_symbols: {unique_symbols}")
    
    # Validate neutron scattering lengths - required for neutron calculations
    if not neutron_lengths:
        print(f"\n  WARNING: neutron_scattering_lengths not found in config!")
        print(f"    Available top-level keys: {list(cfg.keys())}")
        
        # Try to build from known values for common elements
        default_neutron_lengths = {
            'Si': 4.1491, 'O': 5.803, 'Ge': 8.185,
            'P': 5.13, 'S': 2.847, 'Li': -1.90,
            'Na': 3.63, 'Fe': 9.45, 'N': 9.36, 'Cl': 9.577,
            'Ta': 6.91, 'K': 3.67, 'Ca': 4.70, 'Mg': 5.375,
        }
        neutron_lengths = {s: default_neutron_lengths.get(s, 5.0) for s in unique_symbols}
        print(f"    Using default values: {neutron_lengths}")
    
    # Try to find scattering_type (default to neutron for backward compatibility)
    scattering_type = OmegaConf.select(cfg, 'data.scattering_type', default=None)
    if scattering_type is None:
        scattering_type = OmegaConf.select(cfg, 'scattering_type', default='neutron')
    print(f"  scattering_type: {scattering_type}")
    
    model_config = {
        'kernel_width': kernel_width,
        'neutron_scattering_lengths': neutron_lengths,
        'xray_form_factor_params': xray_params if xray_params else {},
        'scattering_type': scattering_type,
    }
    
    xrd_model = XRDModel(
        symbols=unique_symbols,
        config=model_config,
        r_bins=rdf_data.r_bins,
        q_bins=rdf_data.q_bins,
        rdf_data=rdf_data,
        device=accelerator,
    )

    try:
        results = xrd_model(base_sim_state)
        print(f"  Model forward pass: OK")
        print(f"  Available outputs: {list(results.keys())}")
        
        # Print spectrum info
        for key, val in results.items():
            if val is not None:
                print(f"    {key}: shape={val.shape}, range=[{val.min():.4f}, {val.max():.4f}]")
    except Exception as e:
        print(f"  ERROR: {e}")
        raise
    print(f"{'=' * 70}\n")

    # ==========================================
    # LOSS FUNCTION
    # ==========================================
    
    cooper_loss = CooperLoss(target_data=rdf_data, target_type=target_type, device=accelerator)
    print(f"CooperLoss initialized with target: {target_type}")
    
    def loss_fn(desc: dict) -> dict:
        return cooper_loss(desc)

    # ==========================================
    # CONSTRAINT CONFIGURATION
    # ==========================================
    print(f"\n{'=' * 70}")
    print("  CONSTRAINT CONFIGURATION")
    print(f"{'=' * 70}")
    
    constraints_enabled = OmegaConf.select(cfg, 'constraints.enabled', default=True)
    constraints_use_types = OmegaConf.select(cfg, 'constraints.use_types', default='all')
    if hasattr(constraints_use_types, '__iter__') and not isinstance(constraints_use_types, str):
        constraints_use_types = list(constraints_use_types)
    
    print(f"  Constraints enabled: {constraints_enabled}")
    print(f"  Use types: {constraints_use_types}")
    
    json_path = OmegaConf.select(cfg, 'data.json_path', default=None)
    
    use_constraints = False
    cooper_problem = None
    
    if json_path and constraints_enabled:
        json_path = Path(json_path)
        print(f"  Constraints file: {json_path}")
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                constraints_data = json.load(f)
            
            # Show original constraint info
            orig_n_atoms = len(constraints_data.get('atom_constraints', {}))
            orig_op_types = constraints_data.get('metadata', {}).get('order_parameter_types', [])
            cutoff = constraints_data.get('cutoff', 'N/A')
            
            print(f"  Original constraints: {orig_n_atoms} atoms, types: {orig_op_types}")
            print(f"  Cutoff: {cutoff} Å")
            
            # Filter constraints
            filtered = filter_constraints(constraints_data, constraints_use_types, constraints_enabled)
            n_atoms = len(filtered.get('atom_constraints', {}))
            op_types = filtered.get('metadata', {}).get('order_parameter_types', [])
            
            print(f"  After filtering: {n_atoms} atoms, types: {op_types}")
            
            if n_atoms > 0:
                # Save filtered constraints
                filtered_path = plots_dir / "filtered_constraints.json"
                filtered_path.parent.mkdir(parents=True, exist_ok=True)
                with open(filtered_path, 'w') as f:
                    json.dump(filtered, f, indent=2)
                print(f"  Filtered constraints saved: {filtered_path}")
                
                # Get penalty configuration
                penalty_cfg = OmegaConf.select(cfg, 'penalty', default=None)
                if penalty_cfg is not None:
                    penalty_config = OmegaConf.to_container(penalty_cfg)
                    print(f"  Penalty config: init={penalty_config.get('init', 10.0)}, "
                          f"growth={penalty_config.get('growth_rate', 1.5)}")
                else:
                    # Fallback to legacy penalty_rho
                    penalty_config = OmegaConf.select(cfg, 'penalty_rho', default=10.0)
                    print(f"  Penalty rho: {penalty_config}")
                
                constraint_warmup_steps = int(OmegaConf.select(cfg, 'stability.constraint_warmup_steps', default=0))
                
                cooper_problem = StructureFactorCMPWithConstraints(
                    model=xrd_model,
                    base_state=base_sim_state,
                    target_vec=rdf_data,
                    constraints_file=str(filtered_path),
                    loss_fn=loss_fn,
                    target_kind=target_type,
                    device=str(device),
                    penalty_config=penalty_config,
                    constraint_warmup_steps=constraint_warmup_steps
                )
                use_constraints = True
            else:
                print("  WARNING: No constraints after filtering!")
        else:
            print(f"  WARNING: Constraints file not found: {json_path}")
    else:
        print("  No constraints file specified or constraints disabled")
    
    if not use_constraints:
        print("\n  >>> Running UNCONSTRAINED optimization <<<")
    
    print(f"{'=' * 70}\n")

    # ==========================================
    # OPTIMIZERS
    # ==========================================
    
    primal_params = [base_sim_state.positions]
    if OmegaConf.select(cfg, 'optimize_cell', default=False):
        base_sim_state.cell.requires_grad_(True)
        primal_params.append(base_sim_state.cell)

    lr_primal = OmegaConf.select(cfg, 'lr.primal', default=1e-3)
    lr_dual = OmegaConf.select(cfg, 'lr.dual', default=1e-2)

    primal_optimizer = torch.optim.Adam(primal_params, lr=lr_primal)
    dual_optimizer = None
    cooper_opt = None
    
    if use_constraints:
        dual_params = []
        for ci in cooper_problem.constraint_dict.values():
            dual_params.extend(ci['constraint'].multiplier.parameters())
        dual_optimizer = torch.optim.SGD(dual_params, lr=lr_dual, maximize=True)
        cooper_opt = cooper.optim.AlternatingDualPrimalOptimizer(
            primal_optimizers=primal_optimizer,
            dual_optimizers=dual_optimizer,
            cmp=cooper_problem
        )

    # ==========================================
    # TRAINING STATE
    # ==========================================
    
    max_steps = int(OmegaConf.select(cfg, 'max_steps', default=50000))
    checkpoint_interval = OmegaConf.select(cfg, 'checkpoint_interval', default=10000)
    plot_interval = OmegaConf.select(cfg, 'output.plot_interval', default=1000)
    
    initial_loss = None
    lr_reduced = False
    loss = None
    cmp_state = None
    step = 0
    current_reduction = 0.0
    pred_spectrum = None
    
    def get_prediction(misc_or_results):
        """Extract prediction tensor for the target type."""
        if isinstance(misc_or_results, dict):
            # Try target type first, then S_Q, then Y
            if target_type in misc_or_results:
                return misc_or_results[target_type]
            if 'S_Q' in misc_or_results:
                return misc_or_results['S_Q']
            if 'Y' in misc_or_results:
                return misc_or_results['Y']
        return None

    def save_final_results():
        """Save final results."""
        try:
            output_dir = Path(trajectory_path).parent / "final_results"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save structure
            final_atoms = state_to_atoms(base_sim_state)
            for atoms_obj in final_atoms:
                write(str(output_dir / "final_structure.xyz"), atoms_obj, format="xyz")
                write(str(output_dir / "final_structure.cif"), atoms_obj, format="cif")
                plotter.plot_structure_3d(atoms_obj, title="Final Structure", prefix="final_")

            # Save state
            torch.save({
                "positions": base_sim_state.positions.detach().cpu(),
                "cell": base_sim_state.cell.detach().cpu(),
                "step": step,
                "loss": loss.item() if loss is not None else None,
                "target_type": target_type,
            }, str(output_dir / "final_state.pt"))

            # Plot final spectrum
            if pred_spectrum is not None:
                paths = plotter.plot_target_spectrum(rdf_data, pred_spectrum, step=step, prefix="final_")
                if paths:
                    wandb.log({f"final/{target_cfg['symbol']}": wandb.Image(str(paths['png']))})
            
            # Plot loss history
            loss_paths = plotter.plot_loss_history(prefix="final_")
            if loss_paths:
                wandb.log({"final/loss_history": wandb.Image(str(loss_paths['png']))})

            print(f"\nResults saved to {output_dir}")

        except Exception as e:
            print(f"Error saving results: {e}")
            import traceback
            traceback.print_exc()

    def signal_handler(sig, frame):
        print("\n\nInterrupt! Saving results...")
        save_final_results()
        wandb.finish()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # ==========================================
    # STABILITY SETTINGS
    # ==========================================
    grad_clip_norm = OmegaConf.select(cfg, 'stability.grad_clip_norm', default=None)
    max_displacement = OmegaConf.select(cfg, 'stability.max_displacement', default=None)
    constraint_warmup_steps = int(OmegaConf.select(cfg, 'stability.constraint_warmup_steps', default=0))

    print(f"\n{'=' * 70}")
    print(f"  STARTING OPTIMIZATION")
    print(f"{'=' * 70}")
    print(f"  Max steps: {max_steps}")
    print(f"  Target: {target_cfg['symbol']}")
    print(f"  Constraints: {'ON' if use_constraints else 'OFF'}")
    print(f"  Plot interval: {plot_interval}")
    print(f"  Checkpoint interval: {checkpoint_interval}")
    if grad_clip_norm is not None:
        print(f"  Gradient clipping: norm ≤ {grad_clip_norm}")
    if max_displacement is not None:
        print(f"  Position clamping: max {max_displacement} Å/step")
    if constraint_warmup_steps > 0:
        print(f"  Constraint warmup: {constraint_warmup_steps} steps")
    print(f"{'=' * 70}\n")

    # Initialize tracking variables before loop (in case of early exit)
    loss_val = float('nan')
    current_reduction = 0.0

    # ==========================================
    # GRADIENT SAFETY HOOK
    # ==========================================
    # Register a gradient hook on positions to sanitize NaN/Inf gradients
    # BEFORE the optimizer applies them. This is critical because Cooper's
    # roll() calls backward() + step() internally, so we can't clip
    # gradients between those two operations — the hook intercepts them.
    def _safe_grad_hook(grad):
        """Replace NaN/Inf with 0 and clip gradient norm."""
        # 1. Sanitize NaN/Inf
        bad_mask = ~torch.isfinite(grad)
        if bad_mask.any():
            n_bad = bad_mask.any(dim=-1).sum().item()
            grad = torch.where(bad_mask, torch.zeros_like(grad), grad)
            if step < 5 or step % 100 == 0:
                print(f"  ⚠ Sanitized {n_bad} atoms with NaN/Inf gradients at step {step}")
        # 2. Clip global norm
        if grad_clip_norm is not None:
            norm = grad.norm()
            if norm > grad_clip_norm:
                grad = grad * (grad_clip_norm / norm)
        return grad

    base_sim_state.positions.register_hook(_safe_grad_hook)
    print("  ✓ Gradient safety hook registered on positions")

    try:
        for step in range(max_steps):
            
            # Check for NaN in positions before optimization step
            if torch.isnan(base_sim_state.positions).any():
                print(f"\n⚠️ NaN detected in positions at step {step}!")
                print("This usually means the loss was NaN in a previous step.")
                print("Stopping training to prevent further corruption.")
                break
            
            if use_constraints:
                # Save positions before step (for displacement clamping)
                pos_before = base_sim_state.positions.detach().clone()
                
                # Constrained optimization with Cooper
                try:
                    roll_out = cooper_opt.roll(
                        compute_cmp_state_kwargs={
                            "positions": base_sim_state.positions,
                            "cell": base_sim_state.cell,
                            "step": step
                        }
                    )
                except RuntimeError as e:
                    print(f"\n⚠️ Error in Cooper roll at step {step}: {e}")
                    print("This often happens when positions become NaN or invalid.")
                    break
                
                # --- Gradient clipping (applied retroactively via position clamping) ---
                # Cooper's roll() already called backward() + step(), so we can't clip
                # gradients before the step. Instead we clamp the displacement.
                
                # --- Position displacement clamping ---
                # Also recovers from NaN positions (restores to pre-step values)
                if max_displacement is not None:
                    with torch.no_grad():
                        # If positions are NaN after roll, restore to pre-step
                        nan_mask = torch.isnan(base_sim_state.positions).any(dim=-1)
                        if nan_mask.any():
                            n_nan = nan_mask.sum().item()
                            print(f"  ⚠ Recovering {n_nan} NaN atoms → restoring pre-step positions")
                            base_sim_state.positions[nan_mask] = pos_before[nan_mask]
                        
                        displacement = base_sim_state.positions - pos_before
                        disp_norm = displacement.norm(dim=-1, keepdim=True)  # per-atom
                        max_disp_tensor = torch.tensor(max_displacement, device=device, dtype=disp_norm.dtype)
                        scale = torch.clamp(max_disp_tensor / (disp_norm + 1e-10), max=1.0)
                        base_sim_state.positions.copy_(pos_before + displacement * scale)
                    
                cmp_state = roll_out.cmp_state
                misc = cmp_state.misc if cmp_state.misc is not None else {}
                
                # Get loss from misc (Cooper stores it there)
                loss = misc.get('chi2_loss', None)
                if loss is None:
                    loss = cmp_state.loss
                if loss is None:
                    # Fallback: compute loss manually
                    results = xrd_model(base_sim_state)
                    loss_dict = loss_fn(results)
                    loss = loss_dict['total_loss']
                    pred_spectrum = get_prediction(results)
                else:
                    pred_spectrum = get_prediction(misc)
                
                chi2_loss = loss

                # Aggregate violations
                all_v = []
                for cs in cmp_state.observed_constraints.values():
                    if cs.violation is not None:
                        all_v.append(cs.violation.reshape(-1))
                violations = torch.cat(all_v) if all_v else torch.zeros(1, device=device)
                
                avg_viol = violations.mean().item()
                max_viol = violations.max().item()
                num_viol = (violations > 0).sum().item()
            else:
                # Unconstrained optimization
                primal_optimizer.zero_grad()
                results = xrd_model(base_sim_state)
                loss_dict = loss_fn(results)
                loss = loss_dict['total_loss']
                chi2_loss = loss_dict.get('chi2_loss', loss)
                loss.backward()
                
                # Gradient clipping for unconstrained path
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(primal_params, grad_clip_norm)
                
                primal_optimizer.step()
                
                pred_spectrum = get_prediction(results)
                avg_viol, max_viol, num_viol = 0.0, 0.0, 0
                violations = torch.zeros(1, device=device)

            # Ensure loss is valid
            if loss is None:
                print(f"WARNING: Loss is None at step {step}, computing manually...")
                results = xrd_model(base_sim_state)
                loss_dict = loss_fn(results)
                loss = loss_dict['total_loss']
                pred_spectrum = get_prediction(results)

            # Track progress
            loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
            
            if initial_loss is None:
                initial_loss = loss_val
                print(f"Initial loss: {initial_loss:.6f}")

            current_reduction = ((initial_loss - loss_val) / initial_loss) * 100 if initial_loss else 0.0

            # LR reduction
            if not lr_reduced and current_reduction >= 97.5:
                for pg in primal_optimizer.param_groups:
                    pg["lr"] = OmegaConf.select(cfg, 'lr.primal_reduced', default=1e-4)
                if dual_optimizer:
                    for pg in dual_optimizer.param_groups:
                        pg["lr"] = OmegaConf.select(cfg, 'lr.dual_reduced', default=1e-3)
                lr_reduced = True
                print(f"\nLR reduced at step {step} ({current_reduction:.1f}% reduction)")

            # Update history
            chi2_val = chi2_loss.item() if hasattr(chi2_loss, 'item') else float(chi2_loss) if chi2_loss is not None else loss_val
            plotter.update_history(step, {
                'total_loss': loss_val,
                'chi2_loss': chi2_val,
                'avg_violation': avg_viol,
                'max_violation': max_viol,
                'num_violated': num_viol,
            })

            # Print progress
            if step % 100 == 0 or step < 10:
                viol_str = f", Viol: {avg_viol:.4f}/{max_viol:.4f} ({num_viol})" if use_constraints else ""
                print(f"Step {step}: Loss={loss_val:.6f} ({current_reduction:.1f}%){viol_str}")

            # =========================================================
            # WANDB LOGGING - All metrics
            # =========================================================
            log_dict = {
                # Main loss metrics
                "train/total_loss": loss_val,
                "train/chi2_loss": chi2_val,
                "train/loss_reduction_pct": current_reduction,
                
                # Target-specific loss (for easy filtering)
                f"loss/{target_type}": chi2_val,
            }
            
            # Constraint violation metrics
            if use_constraints:
                log_dict.update({
                    "constraints/avg_violation": avg_viol,
                    "constraints/max_violation": max_viol,
                    "constraints/num_violated": num_viol,
                    "constraints/violation_sum": violations.sum().item() if violations is not None else 0.0,
                })
            
            # Log learning rates
            log_dict["lr/primal"] = primal_optimizer.param_groups[0]["lr"]
            if dual_optimizer:
                log_dict["lr/dual"] = dual_optimizer.param_groups[0]["lr"]
            
            wandb.log(log_dict, step=step)

            # Periodic plotting and wandb spectrum logging
            if step > 0 and step % plot_interval == 0:
                if pred_spectrum is not None:
                    # Save PNG/PDF plot
                    paths = plotter.plot_target_spectrum(rdf_data, pred_spectrum, step=step, prefix=f"step_{step}_")
                    if paths:
                        wandb.log({f"plots/{target_cfg['symbol']}": wandb.Image(str(paths['png']))}, step=step)
                    
                    # Log live spectrum comparison to wandb
                    plotter.log_spectrum_to_wandb(rdf_data, pred_spectrum, step)

            # Checkpointing
            if step > 0 and step % checkpoint_interval == 0:
                ckpt_dir = Path(trajectory_path).parent / "checkpoints" / f"step_{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                atoms_curr = state_to_atoms(base_sim_state)
                for a in atoms_curr:
                    write(str(ckpt_dir / f"structure_{step}.cif"), a, format="cif")
                print(f"Checkpoint saved: {ckpt_dir}")

            # Trajectory
            if write_trajectory and step % 1000 == 0:
                traj_path = Path(trajectory_path)
                traj_path.mkdir(parents=True, exist_ok=True)
                atoms_curr = state_to_atoms(base_sim_state)
                for a in atoms_curr:
                    write(str(traj_path / "trajectory.xyz"), a, format="xyz", append=(step > 0))

    except KeyboardInterrupt:
        print("\nInterrupted!")
        save_final_results()
        wandb.finish()
        sys.exit(0)

    # ==========================================
    # FINALIZATION
    # ==========================================
    
    print(f"\n{'=' * 70}")
    print(f"  OPTIMIZATION COMPLETED")
    print(f"{'=' * 70}")
    print(f"  Final loss: {loss_val:.6f}")
    print(f"  Total reduction: {current_reduction:.2f}%")
    print(f"  Target: {target_cfg['symbol']}")
    print(f"{'=' * 70}")
    
    # Create HTML animation of spectrum evolution
    print("\nCreating spectrum evolution animation...")
    animation_path = plotter.create_animation_html(prefix="final_")
    if animation_path:
        print(f"Animation saved: {animation_path}")
        # Log animation file to wandb
        wandb.log({"spectrum_animation": wandb.Html(open(str(animation_path)).read())})
    
    save_final_results()
    wandb.finish()


if __name__ == "__main__":
    main()
