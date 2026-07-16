"""
TorchSim Structural Order Parameters
------------------------------------

Fast structural descriptors for atomistic systems.

Compatible with TorchSim state objects.

Descriptors
-----------

cn         coordination number
cn_smooth  smooth coordination
q2         Steinhardt order parameter
q4         fast polynomial Steinhardt
q6         fast polynomial Steinhardt
qtet       tetrahedral order
qoct       octahedral order
bcc,fcc,hcp crystal similarity
rdf        radial distribution
sq         structure factor
"""

import math
import torch
import torch.nn as nn

import torch_sim as ts
from torch_sim.neighbors import torch_nl_linked_cell


class TorchSimOrderParameters(nn.Module):

    def __init__(
        self,
        cutoff: float = 3.5,
        device: str = "cpu",
        max_neighbors: int = 64,
        rdf_bins: int = 200,
        sq_bins: int = 200,
    ):
        super().__init__()

        self.cutoff = cutoff
        self.device = torch.device(device)
        self.max_neighbors = max_neighbors
        self.rdf_bins = rdf_bins
        self.sq_bins = sq_bins

        # descriptor registry
        self.registry = {
            "cn": self._op_cn,
            "cn_smooth": self._op_cn_smooth,
            "q2": self._op_q2,
            "q4": self._op_q4,
            "q6": self._op_q6,
            "qtet": self._op_qtet,
            "qoct": self._op_qoct,
            "bcc": self._op_bcc,
            "fcc": self._op_fcc,
            "hcp": self._op_hcp,
            "rdf": self._op_rdf,
            "sq": self._op_sq,
        }

    # ======================================================
    # Main call
    # ======================================================

    def forward(
        self,
        state,
        atom_indices,
        order_params,
        element_filter=None,
    ):

        neighbor_idx, neighbor_pos, mask = self._build_neighbors(
            state, atom_indices
        )

        vectors, distances, thetas, phis, unit_vec = self._compute_geometry(
            state.positions,
            atom_indices,
            neighbor_pos,
        )

        results = {}

        for op in order_params:

            if op not in self.registry:
                raise ValueError(
                    f"Unsupported order parameter: {op}\n"
                    f"Available: {list(self.registry.keys())}"
                )

            results[op] = self.registry[op](
                distances=distances,
                thetas=thetas,
                unit_vec=unit_vec,
                mask=mask,
            )

        return results

    # ======================================================
    # Neighbor list
    # ======================================================

    def _build_neighbors(self, state, atom_indices):

        pos = state.positions
        cell = state.cell
        pbc = state.pbc
        system_idx = state.system_idx

        cutoff = torch.tensor(self.cutoff, device=pos.device)

        # TorchSim neighbor list
        mapping, system_map, shifts = torch_nl_linked_cell(
            pos,
            cell,
            pbc,
            cutoff,
            system_idx,
        )

        center = mapping[0]
        neigh = mapping[1]

        n_atoms = pos.shape[0]

        # -----------------------------
        # sort neighbors by center atom
        # -----------------------------

        order = torch.argsort(center)
        center = center[order]
        neigh = neigh[order]

        # -----------------------------
        # compute neighbor offsets
        # -----------------------------

        unique, counts = torch.unique_consecutive(center, return_counts=True)

        offsets = torch.zeros(
            n_atoms + 1,
            dtype=torch.long,
            device=pos.device,
        )

        offsets[unique + 1] = counts
        offsets = torch.cumsum(offsets, dim=0)

        # -----------------------------
        # build padded neighbor matrix
        # -----------------------------

        neighbor_idx = torch.full(
            (n_atoms, self.max_neighbors),
            -1,
            dtype=torch.long,
            device=pos.device,
        )

        max_n = self.max_neighbors

        for i in range(n_atoms):

            start = offsets[i]
            end = offsets[i + 1]

            if start == end:
                continue

            nn = min(end - start, max_n)

            neighbor_idx[i, :nn] = neigh[start:start + nn]

        # restrict to requested atoms
        neighbor_idx = neighbor_idx[atom_indices]

        mask = neighbor_idx >= 0

        neighbor_pos = pos[neighbor_idx.clamp(min=0)]

        return neighbor_idx, neighbor_pos, mask

    # ======================================================
    # Geometry
    # ======================================================

    def _compute_geometry(self, positions, atom_indices, neighbor_pos):

        center = positions[atom_indices].unsqueeze(1)

        vec = neighbor_pos - center
        dist = torch.norm(vec, dim=2) + 1e-12

        unit = vec / dist.unsqueeze(-1)

        cos_theta = unit[:, :, 2]
        thetas = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
        phis = torch.atan2(unit[:, :, 1], unit[:, :, 0])

        return vec, dist, thetas, phis, unit

    # ======================================================
    # Coordination number
    # ======================================================

    def _op_cn(self, **kw):
        return kw["mask"].sum(dim=1).float()

    def _op_cn_smooth(self, **kw):
        r = kw["distances"]
        rc = self.cutoff
        return torch.sum(1 / (1 + (r / rc) ** 6), dim=1)

    # ======================================================
    # q2
    # ======================================================

    def _op_q2(self, **kw):

        thetas = kw["thetas"]
        mask = kw["mask"]

        cos_t = torch.cos(thetas)

        Y20 = 0.5 * math.sqrt(5 / math.pi) * (3 * cos_t**2 - 1)

        real = (Y20 * mask.float()).sum(dim=1)

        n = mask.sum(dim=1).float().clamp(min=1)

        return torch.sqrt(4 * math.pi * real**2 / (5 * n**2 + 1e-12))

    # ======================================================
    # Fast Steinhardt q4
    # ======================================================

    def _op_q4(self, **kw):

        u = kw["unit_vec"]
        mask = kw["mask"].float()

        x = u[:, :, 0]
        y = u[:, :, 1]
        z = u[:, :, 2]

        Y40 = 35*z**4 - 30*z**2 + 3
        Y41 = x*z*(7*z**2 - 3)
        Y42 = (x**2 - y**2)*(7*z**2 - 1)
        Y43 = x*z*(x**2 - 3*y**2)
        Y44 = x**4 - 6*x**2*y**2 + y**4

        q = Y40**2 + Y41**2 + Y42**2 + Y43**2 + Y44**2

        q = (q * mask).sum(dim=1)

        n = mask.sum(dim=1).clamp(min=1)

        return torch.sqrt(q / (n**2 + 1e-12))

    # ======================================================
    # Fast Steinhardt q6
    # ======================================================

    def _op_q6(self, **kw):

        u = kw["unit_vec"]
        mask = kw["mask"].float()

        x = u[:, :, 0]
        y = u[:, :, 1]
        z = u[:, :, 2]

        Y60 = 231*z**6 - 315*z**4 + 105*z**2 - 5
        Y61 = x*z*(33*z**4 - 30*z**2 + 5)
        Y62 = (x**2 - y**2)*(33*z**4 - 18*z**2 + 1)
        Y63 = x*z*(x**2 - 3*y**2)*(11*z**2 - 1)
        Y64 = (x**4 - 6*x**2*y**2 + y**4)*(11*z**2 - 1)
        Y65 = x*z*(x**4 - 10*x**2*y**2 + 5*y**4)
        Y66 = x**6 - 15*x**4*y**2 + 15*x**2*y**4 - y**6

        q = Y60**2 + Y61**2 + Y62**2 + Y63**2 + Y64**2 + Y65**2 + Y66**2

        q = (q * mask).sum(dim=1)

        n = mask.sum(dim=1).clamp(min=1)

        return torch.sqrt(q / (n**2 + 1e-12))

    # ======================================================
    # Tetrahedral order
    # ======================================================

    def _op_qtet(self, **kw):

        u = kw["unit_vec"]

        cos = torch.einsum("bij,bkj->bik", u, u)

        return 1 - (3/8) * torch.sum((cos + 1/3)**2, dim=(1,2))

    # ======================================================
    # Octahedral order
    # ======================================================

    def _op_qoct(self, **kw):

        u = kw["unit_vec"]

        cos = torch.einsum("bij,bkj->bik", u, u)

        return 1 - (3/8) * torch.sum(cos**2, dim=(1,2))

    # ======================================================
    # Crystal similarity
    # ======================================================

    def _op_bcc(self, **kw):

        u = kw["unit_vec"]

        dirs = torch.tensor([
            [1,1,1],[-1,1,1],[1,-1,1],[1,1,-1],
            [-1,-1,1],[-1,1,-1],[1,-1,-1],[-1,-1,-1]
        ], device=u.device) / math.sqrt(3)

        dot = torch.matmul(u, dirs.T)

        return torch.mean(dot**2, dim=(1,2))

    def _op_fcc(self, **kw):

        u = kw["unit_vec"]

        dirs = torch.tensor([
            [1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],
            [1,0,1],[1,0,-1],[-1,0,1],[-1,0,-1],
            [0,1,1],[0,1,-1],[0,-1,1],[0,-1,-1]
        ], device=u.device) / math.sqrt(2)

        dot = torch.matmul(u, dirs.T)

        return torch.mean(dot**2, dim=(1,2))

    def _op_hcp(self, **kw):

        u = kw["unit_vec"]

        dirs = torch.tensor([
            [1,0,0],[-1,0,0],
            [0,1,0],[0,-1,0],
            [0.5, math.sqrt(3)/2,0],
            [-0.5,-math.sqrt(3)/2,0]
        ], device=u.device)

        dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)

        dot = torch.matmul(u, dirs.T)

        return torch.mean(dot**2, dim=(1,2))

    # ======================================================
    # RDF
    # ======================================================

    def _op_rdf(self, **kw):

        r = kw["distances"]

        hist = torch.histc(
            r.flatten(),
            bins=self.rdf_bins,
            min=0,
            max=self.cutoff,
        )

        return hist

    # ======================================================
    # Structure factor
    # ======================================================

    def _op_sq(self, **kw):

        r = kw["distances"]

        q = torch.linspace(
            0.1,
            20,
            self.sq_bins,
            device=r.device
        )

        qr = q.unsqueeze(0) * r.flatten().unsqueeze(1)

        sq = torch.mean(torch.sin(qr) / (qr + 1e-8), dim=0)

        return sq
if __name__ == "__main__":
    import torch
    import torch_sim as ts
    import matplotlib.pyplot as plt

    from ase.build import bulk, make_supercell
    from ase.spacegroup import crystal

    #from order_params import TorchSimOrderParameters

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # Build SiO2 crystal (alpha quartz)
    # ------------------------------------------------------------

    atoms = crystal(
        symbols=["Si", "O"],
        basis=[
            (0.4697, 0.0000, 0.0000),
            (0.4133, 0.2672, 0.1188),
        ],
        spacegroup=152,
        cellpar=[4.913, 4.913, 5.405, 90, 90, 120],
    )

    atoms = atoms.repeat((3, 3, 3))

    # ------------------------------------------------------------
    # Initialize TorchSim state
    # ------------------------------------------------------------

    state = ts.initialize_state(
        atoms,
        device=device,
        dtype=torch.float32,
    )

    symbols = atoms.get_chemical_symbols()

    # ------------------------------------------------------------
    # Select atoms
    # ------------------------------------------------------------

    atom_indices = torch.arange(len(atoms), device=device)

    # ------------------------------------------------------------
    # Create descriptor model
    # ------------------------------------------------------------

    descriptor_model = TorchSimOrderParameters(
        cutoff=3.0,
        device=device,
        max_neighbors=32
    )

    # ------------------------------------------------------------
    # Compute order parameters
    # ------------------------------------------------------------

    ops = descriptor_model(
        state,
        atom_indices,
        order_params=[
            "cn",
            "q4",
            "q6",
            "qtet",
            "qoct",
            "rdf",
        ],
    )

    # ------------------------------------------------------------
    # Print averages
    # ------------------------------------------------------------

    print("Average CN:", ops["cn"].mean().item())
    print("Average q4:", ops["q4"].mean().item())
    print("Average q6:", ops["q6"].mean().item())
    print("Average qtet:", ops["qtet"].mean().item())
    print("Average qoct:", ops["qoct"].mean().item())

    # ------------------------------------------------------------
    # Plot RDF
    # ------------------------------------------------------------

    rdf = ops["rdf"].cpu()

    plt.figure()
    plt.plot(rdf)
    plt.title("RDF (SiO2)")
    plt.xlabel("bin")
    plt.ylabel("g(r)")
    plt.show()