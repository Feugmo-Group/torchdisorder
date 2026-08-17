"""
torchdisorder.analysis
======================

Post-training structure analysis tools.

Exports
-------
StructureAnalyzer
    High-level facade: wraps StructureDescriptors + plot_all + viz3d_all.
load_structure
    Load an ASE Atoms object from a CIF, XYZ, or .pt state file.
plot_all
    Run all matplotlib analyses for a run directory and save plots.
viz3d_all
    Run all PyVista 3-D renders for a run directory and save PNGs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import ase
import ase.io
import torch

from .descriptors import StructureDescriptors
from . import plots as _plots
from . import viz3d as _viz3d

__all__ = [
    "StructureAnalyzer",
    "load_structure",
    "plot_all",
    "viz3d_all",
    "StructureDescriptors",
]


# ---------------------------------------------------------------------------
# load_structure
# ---------------------------------------------------------------------------

def load_structure(path: str | Path) -> ase.Atoms:
    """
    Load an atomic structure from a file.

    Supported formats
    -----------------
    - ``*.cif``      → ASE CIF reader
    - ``*.xyz``      → ASE XYZ reader
    - ``*.pt``       → dict with keys ``positions``, ``cell``; builds an Atoms
                       object (requires atom species cannot be inferred; raises
                       ValueError unless the .pt file also has an ``elements``
                       or ``symbols`` key).

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    ase.Atoms
    """
    path = Path(path)
    if path.suffix in (".cif", ".xyz", ".extxyz"):
        return ase.io.read(str(path))
    if path.suffix == ".pt":
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        positions = state["positions"]
        if isinstance(positions, torch.Tensor):
            positions = positions.numpy()
        cell = state.get("cell", None)
        if isinstance(cell, torch.Tensor):
            cell = cell.squeeze(0).numpy()
        elements = state.get("elements", state.get("symbols", None))
        if elements is None:
            raise ValueError(
                "Cannot infer atom species from .pt file: "
                "key 'elements' or 'symbols' not found."
            )
        atoms = ase.Atoms(
            symbols=elements,
            positions=positions,
            cell=cell,
            pbc=True,
        )
        return atoms
    # Fallback: let ASE guess
    return ase.io.read(str(path))


# ---------------------------------------------------------------------------
# plot_all
# ---------------------------------------------------------------------------

def plot_all(
    atoms: ase.Atoms,
    run_dir: str | Path,
    system_name: str,
    central: str,
    neighbor: str,
    cutoff: float,
    central_z: int,
    neighbor_z: Optional[int] = None,
    exp_data: Optional[dict] = None,
    device: str = "cpu",
) -> None:
    """
    Compute all descriptors and save all matplotlib plots to
    ``<run_dir>/analysis/plots/``.

    Parameters
    ----------
    atoms       : ASE Atoms
    run_dir     : path to training output directory
    system_name : short name for titles
    central     : element symbol of central atom (e.g. 'Ta')
    neighbor    : element symbol of neighbor atom (e.g. 'Cl')
    cutoff      : cutoff distance (Å)
    central_z   : atomic number of central element
    neighbor_z  : atomic number of neighbor element (None = all)
    exp_data    : optional dict with keys 'r','g_r','q','s_q' for experimental data
    device      : torch device for order-parameter calculations
    """
    run_dir = Path(run_dir)
    plot_dir = run_dir / "analysis" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    desc = StructureDescriptors(atoms, cutoff=cutoff, device=device)

    # RDF
    r, g_r = desc.rdf((central, neighbor), r_max=8.0, n_bins=200)
    r_exp = exp_data.get("r") if exp_data else None
    g_r_exp = exp_data.get("g_r") if exp_data else None
    _plots.plot_rdf(
        r, g_r,
        r_exp=r_exp, g_r_exp=g_r_exp,
        pair_label=f"{central}-{neighbor}",
        title=f"{system_name} g(r)",
        save_path=plot_dir / "rdf.png",
    )

    # Bond lengths
    _plots.plot_bond_length_distribution(
        desc.bond_length_distribution((central, neighbor), cutoff),
        pair_label=f"{central}-{neighbor}",
        save_path=plot_dir / "bond_lengths.png",
    )

    # Bond angles
    _plots.plot_bond_angle_distribution(
        desc.bond_angle_distribution(central, neighbor, cutoff),
        central_label=f"{neighbor}-{central}-{neighbor}",
        save_path=plot_dir / "bond_angles.png",
    )

    # CN distribution
    _plots.plot_cn_distribution(
        desc.coordination_numbers(central, neighbor, cutoff),
        element=central,
        save_path=plot_dir / "cn_distribution.png",
    )

    # Order parameters
    op_data = desc.order_params_per_atom(
        central_z=central_z,
        neighbor_z=neighbor_z,
        cutoff=cutoff,
        compute=["q4", "q6", "tet", "fis"],
    )

    for key, label in [("q4", "$q_4$"), ("q6", "$q_6$"), ("tet", "tet"), ("fis", "$F_{IS}$")]:
        vals = op_data.get(key, np.array([]))
        _plots.plot_order_param_histogram(
            vals, label=label,
            save_path=plot_dir / f"{key}_histogram.png",
        )

    # F_IS vs q4 scatter
    fis_vals = op_data.get("fis", np.array([]))
    q4_vals = op_data.get("q4", np.array([]))
    if len(fis_vals) > 0 and len(q4_vals) > 0:
        _plots.plot_fis_q4_scatter(
            fis_vals, q4_vals,
            save_path=plot_dir / "fis_q4_scatter.png",
        )

    # Convergence (if log exists)
    log_file = run_dir / "train.log"
    if log_file.exists():
        _plots.plot_convergence(log_file, save_path=plot_dir / "convergence.png")

    # Summary panel
    _plots.plot_summary_panel(
        desc,
        system_name=system_name,
        central=central,
        neighbor=neighbor,
        cutoff=cutoff,
        exp_data=exp_data,
        save_path=plot_dir / "summary_panel.png",
    )


# ---------------------------------------------------------------------------
# viz3d_all
# ---------------------------------------------------------------------------

def viz3d_all(
    atoms: ase.Atoms,
    run_dir: str | Path,
    system_name: str,
    central: str,
    neighbor: str,
    cutoff: float,
    central_z: int,
    neighbor_z: Optional[int] = None,
    device: str = "cpu",
) -> None:
    """
    Render all 3-D views and save PNGs to ``<run_dir>/analysis/3d/``.
    """
    run_dir = Path(run_dir)
    out_dir = run_dir / "analysis" / "3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    desc = StructureDescriptors(atoms, cutoff=cutoff, device=device)
    op_data = desc.order_params_per_atom(
        central_z=central_z,
        neighbor_z=neighbor_z,
        cutoff=cutoff,
        compute=["q4", "fis"],
    )
    fis = op_data.get("fis", np.array([]))
    q4 = op_data.get("q4", np.array([]))
    cn = desc.coordination_numbers(central, neighbor, cutoff).astype(float)

    _viz3d.save_all_views(atoms, fis, q4, cn, out_dir, system_name)

    # Polyhedra view
    _viz3d.viz_polyhedra(
        atoms, central=central, neighbor=neighbor, cutoff=cutoff,
        save_path=out_dir / f"{system_name}_polyhedra.png",
    )

    # F_IS map
    if len(fis) > 0:
        _viz3d.viz_fis_map(
            atoms, fis,
            save_path=out_dir / f"{system_name}_fis_map.png",
        )


# ---------------------------------------------------------------------------
# StructureAnalyzer — high-level facade
# ---------------------------------------------------------------------------

class StructureAnalyzer:
    """
    High-level post-training structure analysis facade.

    Example
    -------
    >>> analyzer = StructureAnalyzer(
    ...     run_dir="outputs/NaTaCl6_2026-07-16/11-44-06",
    ...     system="NaTaCl6",
    ...     central="Ta",
    ...     neighbor="Cl",
    ...     cutoff=2.6,
    ...     central_z=73,
    ...     neighbor_z=17,
    ... )
    >>> analyzer.run()
    >>> analyzer.print_summary()
    """

    def __init__(
        self,
        run_dir: str | Path,
        system: str,
        central: str,
        neighbor: str,
        cutoff: float,
        central_z: int,
        neighbor_z: Optional[int] = None,
        structure_file: Optional[str | Path] = None,
        device: str = "cpu",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.system = system
        self.central = central
        self.neighbor = neighbor
        self.cutoff = cutoff
        self.central_z = central_z
        self.neighbor_z = neighbor_z
        self.device = device

        # Load structure
        if structure_file is None:
            structure_file = self.run_dir / "final_results" / "final_structure.cif"
            if not structure_file.exists():
                structure_file = self.run_dir / "final_results" / "final_structure.xyz"
        self.atoms = load_structure(structure_file)
        self.desc = StructureDescriptors(
            self.atoms, cutoff=cutoff, device=device
        )

        # Cache computed properties
        self._op_data: Optional[dict[str, np.ndarray]] = None
        self._cn: Optional[np.ndarray] = None

    @property
    def op_data(self) -> dict[str, np.ndarray]:
        if self._op_data is None:
            self._op_data = self.desc.order_params_per_atom(
                central_z=self.central_z,
                neighbor_z=self.neighbor_z,
                cutoff=self.cutoff,
                compute=["q4", "q6", "tet", "fis"],
            )
        return self._op_data

    @property
    def cn(self) -> np.ndarray:
        if self._cn is None:
            self._cn = self.desc.coordination_numbers(
                self.central, self.neighbor, self.cutoff
            )
        return self._cn

    def run(self) -> None:
        """Run all analyses (plots + 3D renders)."""
        plot_all(
            self.atoms,
            self.run_dir,
            system_name=self.system,
            central=self.central,
            neighbor=self.neighbor,
            cutoff=self.cutoff,
            central_z=self.central_z,
            neighbor_z=self.neighbor_z,
            device=self.device,
        )
        try:
            viz3d_all(
                self.atoms,
                self.run_dir,
                system_name=self.system,
                central=self.central,
                neighbor=self.neighbor,
                cutoff=self.cutoff,
                central_z=self.central_z,
                neighbor_z=self.neighbor_z,
                device=self.device,
            )
        except Exception as exc:
            print(f"  [viz3d] skipped — {exc}")

    def print_summary(self) -> None:
        """Print a text summary table to stdout."""
        op = self.op_data
        cn = self.cn

        print(f"\n{'=' * 56}")
        print(f"  Structure Analysis Summary — {self.system}")
        print(f"{'=' * 56}")
        print(f"  N atoms      : {len(self.atoms)}")
        print(f"  Cell volume  : {self.atoms.get_volume():.2f} Å³")
        print()
        print(f"  {'Property':<20}  {'Mean':>8}  {'Std':>8}")
        print(f"  {'-' * 38}")
        for key in ("q4", "q6", "tet", "fis"):
            vals = op.get(key, np.array([]))
            if len(vals) > 0:
                print(
                    f"  {key:<20}  {vals.mean():>8.4f}  {vals.std():>8.4f}"
                )

        # CN distribution
        if len(cn) > 0:
            print()
            print("  Coordination Number Distribution:")
            unique, counts = np.unique(cn, return_counts=True)
            for u, c in zip(unique, counts):
                print(f"    CN={int(u):2d}  →  {c:4d} atoms  ({100 * c / len(cn):.1f}%)")

        # Bond length
        lengths = self.desc.bond_length_distribution(
            (self.central, self.neighbor), self.cutoff
        )
        if len(lengths) > 0:
            print()
            print(
                f"  {self.central}-{self.neighbor} bond length:  "
                f"{lengths.mean():.4f} ± {lengths.std():.4f} Å"
            )

        print(f"{'=' * 56}\n")
