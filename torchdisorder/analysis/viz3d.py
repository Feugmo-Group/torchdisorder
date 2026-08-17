"""
3-D PyVista Visualizations
==========================

Headless-safe (sets OFF_SCREEN before anything else).
All functions return the pv.Plotter so callers can chain or embed further.

Requires: pyvista, scipy
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import ase
import ase.neighborlist

# Headless mode must be set before any pyvista import on a display-less server
try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    try:
        pv.global_theme.allow_empty_mesh = True
    except AttributeError:
        pass
    _PV_AVAILABLE = True
except ImportError:
    _PV_AVAILABLE = False
    warnings.warn(
        "pyvista not found — viz3d functions will raise ImportError if called."
    )

__all__ = [
    "ELEMENT_COLORS",
    "ELEMENT_RADII",
    "atoms_to_polydata",
    "viz_colored_by",
    "viz_by_element",
    "viz_polyhedra",
    "viz_fis_map",
    "viz_voronoi",
    "make_bond_network",
    "save_all_views",
]

# ---------------------------------------------------------------------------
# Element colour / radii tables
# ---------------------------------------------------------------------------

ELEMENT_COLORS: dict[str, str] = {
    "H": "#FFFFFF",
    "Li": "#CC80FF",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "Na": "#AB5CF2",
    "Si": "#F0C8A0",
    "P": "#FF8000",
    "Cl": "#1FF01F",
    "Fe": "#E06633",
    "Ta": "#4DA6FF",
    "S": "#FFFF30",
    "Ge": "#668F8F",
}

ELEMENT_RADII: dict[str, float] = {
    "H": 0.31,
    "Li": 1.28,
    "N": 0.71,
    "O": 0.66,
    "Na": 1.66,
    "Si": 1.11,
    "P": 1.07,
    "Cl": 1.02,
    "Fe": 1.32,
    "Ta": 1.70,
    "S": 1.05,
    "Ge": 1.22,
}

_DEFAULT_COLOR = "#AAAAAA"
_DEFAULT_RADIUS = 0.77


def _pv_check() -> None:
    if not _PV_AVAILABLE:
        raise ImportError("pyvista is required for 3-D visualisations.")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def atoms_to_polydata(atoms: ase.Atoms) -> "pv.PolyData":
    """Convert an ASE Atoms object to a PyVista point cloud with element labels."""
    _pv_check()
    pts = atoms.positions.copy()
    cloud = pv.PolyData(pts)
    cloud["element"] = np.array(atoms.get_chemical_symbols())
    cloud["atomic_number"] = atoms.numbers.astype(np.int32)
    return cloud


def viz_colored_by(
    atoms: ase.Atoms,
    values: np.ndarray,
    label: str,
    cmap: str = "coolwarm",
    save_path: Optional[str | Path] = None,
    show: bool = False,
    window_size: tuple[int, int] = (1200, 900),
) -> "pv.Plotter":
    """
    3-D scatter of atoms, sized by covalent radius and coloured by ``values``.

    Parameters
    ----------
    atoms   : ASE Atoms
    values  : per-atom scalar array
    label   : colour-bar label
    cmap    : matplotlib colormap name
    """
    _pv_check()
    symbols = atoms.get_chemical_symbols()
    plotter = pv.Plotter(off_screen=not show, window_size=list(window_size))
    plotter.set_background("white")

    for i, (pos, sym) in enumerate(zip(atoms.positions, symbols)):
        radius = ELEMENT_RADII.get(sym, _DEFAULT_RADIUS) * 0.4
        sphere = pv.Sphere(radius=radius, center=pos)
        color_val = float(values[i]) if i < len(values) else 0.0
        sphere["value"] = np.full(sphere.n_points, color_val)
        plotter.add_mesh(sphere, scalars="value", cmap=cmap,
                         clim=[float(values.min()), float(values.max())],
                         show_scalar_bar=False)

    # Single scalar bar
    plotter.add_scalar_bar(label, vertical=True, height=0.6)
    plotter.camera_position = "iso"

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(p))

    if show:
        plotter.show()

    return plotter


def viz_by_element(
    atoms: ase.Atoms,
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> "pv.Plotter":
    """Colour atoms by element using ELEMENT_COLORS."""
    _pv_check()
    symbols = atoms.get_chemical_symbols()
    plotter = pv.Plotter(off_screen=not show, window_size=[1200, 900])
    plotter.set_background("white")

    legend_entries: list[list] = []
    seen = set()

    for pos, sym in zip(atoms.positions, symbols):
        radius = ELEMENT_RADII.get(sym, _DEFAULT_RADIUS) * 0.4
        color = ELEMENT_COLORS.get(sym, _DEFAULT_COLOR)
        sphere = pv.Sphere(radius=radius, center=pos)
        plotter.add_mesh(sphere, color=color, smooth_shading=True)
        if sym not in seen:
            legend_entries.append([sym, color])
            seen.add(sym)

    plotter.add_legend(legend_entries, bcolor="white", size=(0.15, 0.2))
    plotter.camera_position = "iso"

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(p))

    if show:
        plotter.show()

    return plotter


def viz_polyhedra(
    atoms: ase.Atoms,
    central: str,
    neighbor: str,
    cutoff: float,
    alpha: float = 0.4,
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> "pv.Plotter":
    """
    Draw coordination polyhedra for each central atom via ConvexHull.
    Overlay atom spheres on top.
    """
    _pv_check()
    from scipy.spatial import ConvexHull  # type: ignore

    symbols = np.array(atoms.get_chemical_symbols())
    idx_central = np.where(symbols == central)[0]

    nl = ase.neighborlist.NeighborList(
        [cutoff / 2.0] * len(atoms),
        self_interaction=False,
        bothways=True,
    )
    nl.update(atoms)

    plotter = pv.Plotter(off_screen=not show, window_size=[1200, 900])
    plotter.set_background("white")

    # Atom spheres first (small, for context)
    for pos, sym in zip(atoms.positions, symbols):
        radius = ELEMENT_RADII.get(sym, _DEFAULT_RADIUS) * 0.3
        color = ELEMENT_COLORS.get(sym, _DEFAULT_COLOR)
        sphere = pv.Sphere(radius=radius, center=pos)
        plotter.add_mesh(sphere, color=color, smooth_shading=True, opacity=0.9)

    # Polyhedra
    poly_color = ELEMENT_COLORS.get(neighbor, _DEFAULT_COLOR)
    for i in idx_central:
        indices, offsets = nl.get_neighbors(i)
        neigh_pos = []
        for j, offset in zip(indices, offsets):
            if symbols[j] != neighbor:
                continue
            dr = (
                atoms.positions[j]
                + offset @ atoms.cell.array
                - atoms.positions[i]
            )
            if 0 < float(np.linalg.norm(dr)) <= cutoff:
                neigh_pos.append(atoms.positions[j] + offset @ atoms.cell.array)

        if len(neigh_pos) < 4:
            continue
        pts = np.array(neigh_pos)
        try:
            hull = ConvexHull(pts)
            faces_flat = []
            for simplex in hull.simplices:
                faces_flat.extend([3, *simplex])
            mesh = pv.PolyData(pts, np.array(faces_flat))
            plotter.add_mesh(mesh, color=poly_color, opacity=alpha,
                             smooth_shading=True)
        except Exception:
            pass

    plotter.camera_position = "iso"

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(p))

    if show:
        plotter.show()

    return plotter


def viz_fis_map(
    atoms: ase.Atoms,
    fis_values: np.ndarray,
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> "pv.Plotter":
    """
    Atoms coloured by F_IS (blue=-1 to red=+1).
    Thin cylinder network for bonds within 1.2× mean bond length.
    """
    _pv_check()
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors

    symbols = atoms.get_chemical_symbols()
    plotter = pv.Plotter(off_screen=not show, window_size=[1200, 900])
    plotter.set_background("#1E1E1E")

    cmap = mcm.get_cmap("coolwarm")
    fis_min, fis_max = float(fis_values.min()), float(fis_values.max())
    fis_range = fis_max - fis_min or 1.0

    # Bond cylinders (very thin, grey)
    nl_wide = ase.neighborlist.NeighborList(
        [4.0] * len(atoms), self_interaction=False, bothways=False
    )
    nl_wide.update(atoms)
    bond_lengths: list[float] = []
    bond_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(atoms)):
        indices, offsets = nl_wide.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            dr = (
                atoms.positions[j]
                + offset @ atoms.cell.array
                - atoms.positions[i]
            )
            d = float(np.linalg.norm(dr))
            if d > 0:
                bond_lengths.append(d)
                bond_pairs.append((atoms.positions[i], atoms.positions[i] + dr))

    mean_bl = float(np.mean(bond_lengths)) if bond_lengths else 3.0
    cutoff_bond = 1.2 * mean_bl
    for (p0, p1), bl in zip(bond_pairs, bond_lengths):
        if bl <= cutoff_bond:
            cyl = pv.Cylinder(
                center=0.5 * (p0 + p1),
                direction=(p1 - p0),
                radius=0.04,
                height=float(np.linalg.norm(p1 - p0)),
            )
            plotter.add_mesh(cyl, color="gray", opacity=0.3)

    # Atom spheres coloured by FIS (full atoms array, map FIS only for central atoms)
    # We need a full-length fis array; zero for non-central atoms
    n_atoms = len(atoms)
    if len(fis_values) < n_atoms:
        # partial array — we just render the atoms where we have values
        # Build index arrays based on positions match to first len(fis_values)
        fis_full = np.zeros(n_atoms)
        fis_full[: len(fis_values)] = fis_values
    else:
        fis_full = fis_values[:n_atoms]

    for i, (pos, sym) in enumerate(zip(atoms.positions, symbols)):
        radius = ELEMENT_RADII.get(sym, _DEFAULT_RADIUS) * 0.4
        val = float(fis_full[i])
        norm_val = (val - fis_min) / fis_range
        rgba = cmap(norm_val)
        hex_color = mcolors.to_hex(rgba)
        sphere = pv.Sphere(radius=radius, center=pos)
        plotter.add_mesh(sphere, color=hex_color, smooth_shading=True)

    plotter.add_scalar_bar(
        r"F_IS", vertical=True, height=0.6,
        mapper=None,  # colour bar is approximate; real one would need a mesh scalar
    )
    plotter.camera_position = "iso"

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(p))

    if show:
        plotter.show()

    return plotter


def make_bond_network(
    atoms: ase.Atoms,
    pairs: list[tuple[str, str]],
    cutoffs: dict[tuple[str, str], float],
) -> list[tuple[int, int]]:
    """
    Return list of (i, j) bond pairs for specified element pairs and cutoffs.

    Parameters
    ----------
    atoms   : ASE Atoms
    pairs   : list of (elem_A, elem_B) pairs to include
    cutoffs : dict mapping (elem_A, elem_B) → cutoff distance
    """
    symbols = np.array(atoms.get_chemical_symbols())

    max_cutoff = max(cutoffs.values()) if cutoffs else 3.5
    nl = ase.neighborlist.NeighborList(
        [max_cutoff / 2.0] * len(atoms),
        self_interaction=False,
        bothways=False,
    )
    nl.update(atoms)

    bonds: list[tuple[int, int]] = []
    pair_set = set(pairs) | {(b, a) for a, b in pairs}

    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            pair = (symbols[i], symbols[j])
            if pair not in pair_set:
                continue
            key = pair if pair in cutoffs else (pair[1], pair[0])
            cut = cutoffs.get(key, max_cutoff)
            dr = (
                atoms.positions[j]
                + offset @ atoms.cell.array
                - atoms.positions[i]
            )
            if 0 < float(np.linalg.norm(dr)) <= cut:
                bonds.append((int(i), int(j)))

    return bonds


def save_all_views(
    atoms: ase.Atoms,
    fis: np.ndarray,
    q4: np.ndarray,
    cn: np.ndarray,
    output_dir: Path,
    system_name: str,
) -> None:
    """
    Save 4 PNG files:
      - by_element.png
      - by_fis.png
      - by_q4.png
      - by_cn.png
    """
    _pv_check()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    viz_by_element(atoms, save_path=output_dir / f"{system_name}_by_element.png")

    if len(fis) > 0:
        viz_colored_by(
            atoms, fis, label=r"F_IS", cmap="coolwarm",
            save_path=output_dir / f"{system_name}_by_fis.png",
        )
    if len(q4) > 0:
        viz_colored_by(
            atoms, q4, label=r"q4", cmap="viridis",
            save_path=output_dir / f"{system_name}_by_q4.png",
        )
    if len(cn) > 0:
        viz_colored_by(
            atoms, cn.astype(float), label="CN", cmap="plasma",
            save_path=output_dir / f"{system_name}_by_cn.png",
        )


# ---------------------------------------------------------------------------
# Voronoi tessellation (freud periodic + PyVista + VTK export)
# ---------------------------------------------------------------------------

def viz_voronoi(
    atoms: ase.Atoms,
    color_by: str = "volume",
    cmap: str = "coolwarm",
    alpha: float = 0.6,
    save_path: Optional[str | Path] = None,
    save_vtk: Optional[str | Path] = None,
    show: bool = False,
    window_size: tuple[int, int] = (1200, 900),
) -> "pv.Plotter":
    """
    Periodic Voronoi tessellation using freud, rendered with PyVista.

    Each Voronoi polyhedron is coloured by `color_by`:
      - 'volume'    — cell volume in Å³ (large → open packing → disorder)
      - 'nfaces'    — number of Voronoi faces (connectivity)
      - 'isoperimetric' — sphericity index 36π V²/A³ (1=sphere, <1=irregular)

    Optionally exports a VTK UnstructuredGrid for ParaView / Blender.

    Parameters
    ----------
    atoms        ASE Atoms (must have PBC cell)
    color_by     'volume', 'nfaces', or 'isoperimetric'
    save_path    PNG screenshot path
    save_vtk     VTK file path (.vtk or .vtu)
    show         open interactive window (requires a display)
    """
    _pv_check()
    try:
        import freud
        from freud.locality import Voronoi as FreudVoronoi
    except ImportError as e:
        raise ImportError("freud-analysis is required for Voronoi: pip install freud-analysis") from e
    from scipy.spatial import ConvexHull

    pos = atoms.positions.astype(np.float64)
    cell = np.array(atoms.cell, dtype=np.float64)

    # --- freud Voronoi (periodic) ---
    box = freud.box.Box.from_matrix(cell)
    # freud expects fractional coordinates
    frac = np.linalg.solve(cell.T, pos.T).T  # (N, 3) in [0, 1)
    frac = frac % 1.0
    pos_frac = frac

    voro = FreudVoronoi()
    voro.compute((box, pos_frac))

    # Build per-atom scalar values and polyhedra meshes
    scalars = []
    cell_meshes = []

    for i, polytope in enumerate(voro.polytopes):
        # polytope is (M, 3) array of vertex coords in freud box frame
        # Convert back to Cartesian
        verts_cart = polytope @ cell  # freud stores fractional-like coords

        try:
            hull = ConvexHull(verts_cart)
        except Exception:
            scalars.append(0.0)
            cell_meshes.append(None)
            continue

        vol = hull.volume
        area = hull.area
        nfaces = len(hull.simplices)
        iso = (36 * np.pi * vol ** 2) / (area ** 3 + 1e-30)

        if color_by == "volume":
            scalars.append(vol)
        elif color_by == "nfaces":
            scalars.append(float(nfaces))
        elif color_by == "isoperimetric":
            scalars.append(iso)
        else:
            scalars.append(vol)

        # Build pyvista surface mesh from convex hull faces
        faces = []
        for simplex in hull.simplices:
            faces += [3, simplex[0], simplex[1], simplex[2]]
        mesh = pv.PolyData(verts_cart, np.array(faces))
        cell_meshes.append(mesh)

    scalars = np.array(scalars, dtype=np.float32)
    vmin, vmax = scalars.min(), scalars.max()

    import matplotlib.cm as mplcm
    cm = mplcm.get_cmap(cmap)

    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl.set_background("white")

    # Collect all polyhedra into a single merged mesh for VTK export
    vtk_blocks = pv.MultiBlock()

    for i, (mesh, val) in enumerate(zip(cell_meshes, scalars)):
        if mesh is None:
            continue
        norm_val = (val - vmin) / (vmax - vmin + 1e-12)
        rgba = cm(norm_val)
        color = tuple(int(c * 255) for c in rgba[:3])
        pl.add_mesh(mesh, color=color, opacity=alpha, show_edges=True,
                    edge_color="gray", line_width=0.5)
        mesh["scalar"] = np.full(mesh.n_points, val)
        vtk_blocks.append(mesh)

    # Overlay atom spheres
    pts = pv.PolyData(pos.astype(np.float32))
    pts["element"] = np.array([atoms.get_chemical_symbols()], dtype=str).flatten()
    sphere_glyphs = pts.glyph(scale=False, geom=pv.Sphere(radius=0.3))
    pl.add_mesh(sphere_glyphs, color="black", opacity=0.8)

    label = {"volume": "Voronoi volume (Å³)",
             "nfaces": "Number of faces",
             "isoperimetric": "Isoperimetric ratio"}.get(color_by, color_by)
    pl.add_scalar_bar(title=label, color="black")
    pl.camera_position = "iso"
    pl.camera.zoom(0.9)

    if save_path:
        pl.screenshot(str(save_path))

    if save_vtk:
        merged = vtk_blocks.combine()
        merged.save(str(save_vtk))

    if show:
        pl.show()
    else:
        pl.close()

    return pl
