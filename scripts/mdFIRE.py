# md_fire_random.py
import numpy as np
import torch
import torch_sim as ts
from ase import Atoms

# Choose CPU/GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load an ML interatomic potential (MACE foundations)
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel
mace = mace_mp(model="small", return_raw_model=True)  # small/medium/large available
model = MaceModel(model=mace, device=device)

def random_structure(symbol="Cu", n=64, cell_a=14.0, dmin=2.2, seed=0):
    """Random positions in a periodic cubic box with minimum-image spacing >= dmin."""
    rng = np.random.default_rng(seed)
    a = cell_a
    cell = np.eye(3) * a
    pos = []
    for i in range(n):
        for _ in range(100000):
            cand = rng.random(3) * a
            ok = True
            for pj in pos:
                d = cand - pj
                d -= a * np.round(d / a)  # minimum image
                if np.linalg.norm(d) < dmin:
                    ok = False
                    break
            if ok:
                pos.append(cand)
                break
        else:
            raise RuntimeError(f"Could not place atom {i} with dmin={dmin}")
    return Atoms(symbol * n, positions=pos, cell=cell, pbc=True)

def main():
    # 1) Build a random structure
    atoms0 = random_structure(symbol="Cu", n=64, cell_a=14.0, dmin=2.2, seed=42)

    # 2) Short MD "shake" (NVT Langevin)
    md_state = ts.integrate(
        system=atoms0,
        model=model,
        integrator=ts.integrators.nvt_langevin,
        n_steps=500,          # 500 * 0.002 ps = 1 ps
        timestep=0.002,       # ps
        temperature=800.0,    # K
        gamma=1.0,            # 1/ps
        trajectory_reporter=dict(filenames=["random_md.h5md"], state_frequency=50),
    )

    # 3) FIRE relaxation (positions-only)
    relaxed = ts.optimize(
        system=md_state,
        model=model,
        optimizer=ts.frechet_cell_fire,    # use ts.frechet_cell_fire for variable-cell if model provides stress
        max_steps=5000,
        autobatcher=True,
    )

    # 4) Report results and write final structure
    final_atoms = relaxed.to_atoms()
    print(f"Final energy (eV): {float(relaxed.energy):.6f}")
    #print(f"Atoms: {len(final_atoms)}; cell (Å):\n{final_atoms.cell.array}")

    # Optional: write to files
    try:
        from ase.io import write
        write("final_structure.xyz", final_atoms)
        write("final_structure.traj", final_atoms)
        print("Wrote final_structure.xyz and .traj")
    except Exception:
        pass

if __name__ == "__main__":
    main()
