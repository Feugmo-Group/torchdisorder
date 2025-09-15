import numpy as np
import torch
import torch_sim as ts
from ase import Atoms
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel
import cooper
from mace.calculators.mace import MACECalculator  # ASE MACE calculator [3]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mace = mace_mp(model="small", return_raw_model=True)
model = MaceModel(model=mace, device=device)

def random_structure(symbol="Cu", n=64, cell_a=14.0, dmin=2.2, seed=0):
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

def ase_atoms_to_tensors(atoms):
    # If a trajectory (list/tuple) is passed, take the last frame
    atoms = atoms[-1] if isinstance(atoms, (list, tuple)) else atoms
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device, requires_grad=True)  # ASE getter [1]
    cell = torch.tensor(atoms.cell.array, dtype=torch.float32, device=device, requires_grad=False)     # ASE cell array [1]
    Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)                      # Atomic numbers [1]
    return pos, cell, Z

def mace_energy(pos, cell, Z):
    # Compute energy via ASE MACE calculator (avoids ts.System)
    atoms_tmp = Atoms(
        numbers=Z.detach().cpu().numpy(),
        positions=pos.detach().cpu().numpy(),
        cell=cell.detach().cpu().numpy(),
        pbc=True,
    )
    calc = MACECalculator(model="small", default_dtype="float32")  # aligned with MD dtype choice [3]
    atoms_tmp.calc = calc
    E = atoms_tmp.get_potential_energy()  # eV scalar float [1]
    return torch.tensor(E, device=device, dtype=torch.float32, requires_grad=True)

def min_image_dmin(pos, cell):
    inv_cell = torch.linalg.inv(cell)
    frac = pos @ inv_cell.T                      # (N,3)
    df = frac.unsqueeze(1) - frac.unsqueeze(0)  # (N,N,3)
    df = df - torch.round(df)                   # minimum image
    dr = df @ cell                              # back to Cartesian
    N = pos.shape                            # ensure integer for eye() [2]
    mask = torch.eye(N, device=pos.device).bool()
    dists = torch.linalg.norm(dr, dim=-1)
    dists = dists.masked_fill(mask, float("inf"))
    return dists.min()

class GeometryCMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self, Z, dmin_target=2.2, formulation="Lagrangian"):
        super().__init__()
        self.Z = Z
        self.dmin_target = dmin_target
        self.mult = cooper.multipliers.DenseMultiplier(num_constraints=1, device=device)
        formulation_type = getattr(cooper.formulations, formulation)  # "Lagrangian" or "AugmentedLagrangian" [4]
        self.constraint = cooper.Constraint(
            multiplier=self.mult,
            constraint_type=cooper.ConstraintType.INEQUALITY,
            formulation_type=formulation_type,
        )

    def compute_cmp_state(self, pos, cell):
        loss = mace_energy(pos, cell, self.Z)
        dmin_now = min_image_dmin(pos, cell)
        violation = torch.relu(self.dmin_target - dmin_now).unsqueeze(0)  # shape (1,)
        observed = {self.constraint: cooper.ConstraintState(violation=violation)}
        return cooper.CMPState(loss=loss, observed_constraints=observed, misc={"dmin": dmin_now})  # Cooper CMP API [4]

def cooper_relax(atoms, steps=2000, lr_pos=1e-2, lr_dual=1e-1, dmin_target=2.2, formulation="Lagrangian"):
    pos, cell, Z = ase_atoms_to_tensors(atoms)
    cmp = GeometryCMP(Z, dmin_target=dmin_target, formulation=formulation)

    prim_opt = torch.optim.Adam([pos], lr=lr_pos)
    dual_opt = torch.optim.SGD(cmp.dual_parameters(), lr=lr_dual, maximize=True)

    coop = cooper.optim.SimultaneousOptimizer(
        cmp=cmp,
        primal_optimizers=prim_opt,
        dual_optimizers=dual_opt,
    )  # Simultaneous primal–dual updates per docs [4]

    best_E = None
    best_pos = None
    for k in range(steps):
        roll = coop.roll(compute_cmp_state_kwargs={"pos": pos, "cell": cell})  # One Cooper step [4]
        with torch.no_grad():
            E = roll.cmp_state.loss.item()
            v = roll.cmp_state.observed_constraints[cmp.constraint].violation.item()
            dmin_now = roll.cmp_state.misc["dmin"].item()
            if best_E is None or E < best_E:
                best_E, best_pos = E, pos.detach().clone()
        if v < 1e-3:
            pass

    with torch.no_grad():
        if best_pos is not None:
            pos.copy_(best_pos)

    final_atoms = atoms[-1].copy() if isinstance(atoms, (list, tuple)) else atoms.copy()
    final_atoms.set_positions(pos.detach().cpu().numpy())
    return final_atoms, (best_E if best_E is not None else float("nan"))

def main():
    atoms0 = random_structure(symbol="Cu", n=64, cell_a=14.0, dmin=2.2, seed=42)

    md_state = ts.integrate(
        system=atoms0,
        model=model,
        integrator=ts.integrators.nvt_langevin,
        n_steps=500,
        timestep=0.002,
        temperature=800.0,
        gamma=1.0,
    )  # Torch-Sim integration returning an MD state convertible to frames [5][6]

    final_atoms, final_E = cooper_relax(md_state.to_atoms(), steps=2000, dmin_target=2.2, formulation="Lagrangian")
    print(f"Final energy (eV): {final_E:.6f}")

if __name__ == "__main__":
    main()
