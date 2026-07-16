import torch
import torch_sim as ts
from torch_sim.neighbors import torch_nl_linked_cell


class SpectrumCalculator:
    """
    TorchSim structural descriptor calculator.

    Computes:
        - Radial distribution function (RDF)
        - Partial RDF
        - Coordination number
        - Angular distribution function (ADF)
        - Structure factor S(Q)

    Fully Torch-based (no NumPy) and GPU compatible.
    """

    def __init__(self, cutoff=6.0, sigma_r=0.05, sigma_theta=0.05):

        self.cutoff = cutoff
        self.sigma_r = sigma_r
        self.sigma_theta = sigma_theta
        self.sqrt_2pi = torch.sqrt(torch.tensor(2.0 * torch.pi))

    # --------------------------------------------------------
    # Neighbor list
    # --------------------------------------------------------

    def build_neighbor_list(self, state):

        mapping, system_mapping, shifts = torch_nl_linked_cell(
            state.positions,
            state.cell,
            state.pbc,
            torch.tensor(self.cutoff, device=state.positions.device),
            state.system_idx,
        )

        return mapping, shifts

    # --------------------------------------------------------
    # Pair distances
    # --------------------------------------------------------

    def pair_distances(self, state, mapping, shifts):

        rij = state.positions[mapping[0]] - state.positions[mapping[1]]

        if shifts is not None:
            rij = rij - shifts

        dist = torch.sqrt((rij ** 2).sum(dim=1) + 1e-12)

        return dist, rij

    # --------------------------------------------------------
    # RDF
    # --------------------------------------------------------

    def compute_rdf(self, state, r_bins):

        mapping, shifts = self.build_neighbor_list(state)

        r_ij, _ = self.pair_distances(state, mapping, shifts)

        sigma = self.sigma_r

        # gauss = torch.exp(
        #     -0.5 * ((r_bins[None, :] - r_ij[:, None]) / sigma) ** 2
        # ) / (sigma * torch.sqrt(2 * torch.pi))

        x = (r_bins[None, :] - r_ij[:, None]) / sigma

        gauss = torch.exp(-0.5 * x ** 2) / (sigma * self.sqrt_2pi)

        g_r = gauss.sum(dim=0)

        g_r = gauss.sum(dim=0)

        volume = torch.det(state.cell[0]).abs()

        n_atoms = state.positions.shape[0]

        density = n_atoms / volume

        g_r = g_r / (density * 4 * torch.pi * r_bins ** 2 * n_atoms)

        return g_r

    # --------------------------------------------------------
    # Partial RDF
    # --------------------------------------------------------

    def compute_partial_rdf(self, state, r_bins):

        mapping, shifts = self.build_neighbor_list(state)

        r_ij, _ = self.pair_distances(state, mapping, shifts)

        Z = state.atomic_numbers

        pair_dict = {}

        for idx in range(len(r_ij)):

            i = mapping[0][idx]
            j = mapping[1][idx]

            Zi = int(Z[i])
            Zj = int(Z[j])

            key = tuple(sorted((Zi, Zj)))

            pair_dict.setdefault(key, []).append(r_ij[idx])

        partial_rdfs = {}

        sigma = self.sigma_r

        for key, distances in pair_dict.items():

            distances = torch.stack(distances)

            # gauss = torch.exp(
            #     -0.5 * ((r_bins[None, :] - distances[:, None]) / sigma) ** 2
            # ) / (sigma * torch.sqrt(2 * torch.pi))

            x = (r_bins[None, :] - distances[:, None]) / sigma

            gauss = torch.exp(-0.5 * x ** 2) / (sigma * self.sqrt_2pi)

            g = gauss.sum(dim=0)

            partial_rdfs[key] = g

        return partial_rdfs

    # --------------------------------------------------------
    # Coordination number
    # --------------------------------------------------------

    def compute_coordination_number(self, g_r, r_bins, density):

        dr = r_bins[1] - r_bins[0]

        coord = torch.cumsum(
            4 * torch.pi * density * g_r * r_bins ** 2 * dr,
            dim=0,
        )

        return coord

    # --------------------------------------------------------
    # Angular Distribution Function
    # --------------------------------------------------------

    def compute_adf(self, state, theta_bins):

        mapping, shifts = self.build_neighbor_list(state)

        _, rij = self.pair_distances(state, mapping, shifts)

        centers = mapping[0]

        neighbors = {}

        for idx, center in enumerate(centers):
            neighbors.setdefault(center.item(), []).append(idx)

        angles = []

        for center, idxs in neighbors.items():

            if len(idxs) < 2:
                continue

            vecs = rij[idxs]

            for a in range(len(vecs)):
                for b in range(a + 1, len(vecs)):

                    v1 = vecs[a]
                    v2 = vecs[b]

                    cos_theta = torch.dot(v1, v2) / (
                        torch.norm(v1) * torch.norm(v2) + 1e-12
                    )

                    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

                    angles.append(theta)

        if len(angles) == 0:
            return torch.zeros_like(theta_bins)

        angles = torch.stack(angles)

        sigma = self.sigma_theta

        # gauss = torch.exp(
        #     -0.5 * ((theta_bins[None, :] - angles[:, None]) / sigma) ** 2
        # ) / (sigma * torch.sqrt(2 * torch.pi))

        x = (theta_bins[None, :] - angles[:, None]) / sigma

        gauss = torch.exp(-0.5 * x ** 2) / (sigma * self.sqrt_2pi)

        adf = gauss.sum(dim=0)

        return adf

    # --------------------------------------------------------
    # Structure factor
    # --------------------------------------------------------

    def compute_structure_factor(self, r_bins, g_r, q_bins, density):

        dr = r_bins[1] - r_bins[0]

        r = r_bins[None, :]
        q = q_bins[:, None]

        integrand = (
            (g_r - 1)
            * r_bins ** 2
            * torch.sin(q * r) / (q * r + 1e-12)
        )

        S_q = 1 + 4 * torch.pi * density * torch.sum(integrand * dr, dim=1)

        return S_q

# import torch
# import torch_sim as ts
# import matplotlib.pyplot as plt
# from ase.build import bulk
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype = torch.float32
#
# #from rdf import SpectrumCalculator
#
# atoms = bulk("Si", "diamond", a=5.43, cubic=True)
#
# state = ts.initialize_state(atoms, device=device, dtype=dtype)
#
# calc = SpectrumCalculator(cutoff=6.0)
#
# r_bins = torch.linspace(0.1, 6.0, 200)
# theta_bins = torch.linspace(0, torch.pi, 180)
# q_bins = torch.linspace(0.1, 15.0, 200)
#
# rdf = calc.compute_rdf(state, r_bins)
#
# volume = torch.det(state.cell[0]).abs()
# density = state.positions.shape[0] / volume
#
# coord = calc.compute_coordination_number(rdf, r_bins, density)
#
# adf = calc.compute_adf(state, theta_bins)
#
# Sq = calc.compute_structure_factor(r_bins, rdf, q_bins, density)

import torch
import torch_sim as ts
import matplotlib.pyplot as plt
from ase.build import bulk

#from rdf import SpectrumCalculator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

atoms = bulk("Si", "diamond", a=5.43, cubic=True)

state = ts.initialize_state(atoms, device=device, dtype=dtype)

calc = SpectrumCalculator(cutoff=6.0)

r_bins = torch.linspace(0.1, 6.0, 200, device=device)
theta_bins = torch.linspace(0, torch.pi, 180, device=device)
q_bins = torch.linspace(0.1, 15.0, 200, device=device)

# ---- Compute descriptors ----

rdf = calc.compute_rdf(state, r_bins)

volume = torch.det(state.cell[0]).abs()
density = state.positions.shape[0] / volume

coord = calc.compute_coordination_number(rdf, r_bins, density)

adf = calc.compute_adf(state, theta_bins)

Sq = calc.compute_structure_factor(r_bins, rdf, q_bins, density)

# ---- Move to CPU for plotting ----

r = r_bins.cpu().detach().numpy()
rdf_np = rdf.cpu().detach().numpy()
coord_np = coord.cpu().detach().numpy()

theta = theta_bins.cpu().detach().numpy()
adf_np = adf.cpu().detach().numpy()

q = q_bins.cpu().detach().numpy()
sq_np = Sq.cpu().detach().numpy()

# ---- Plot RDF ----

plt.figure()
plt.plot(r, rdf_np)
plt.xlabel("r (Å)")
plt.ylabel("g(r)")
plt.title("Radial Distribution Function")
plt.grid()
plt.show()

# ---- Plot Coordination Number ----

plt.figure()
plt.plot(r, coord_np)
plt.xlabel("r (Å)")
plt.ylabel("N(r)")
plt.title("Coordination Number")
plt.grid()
plt.show()

# ---- Plot Angular Distribution ----

plt.figure()
plt.plot(theta, adf_np)
plt.xlabel("θ (rad)")
plt.ylabel("P(θ)")
plt.title("Angular Distribution Function")
plt.grid()
plt.show()

# ---- Plot Structure Factor ----

plt.figure()
plt.plot(q, sq_np)
plt.xlabel("Q (Å⁻¹)")
plt.ylabel("S(Q)")
plt.title("Structure Factor")
plt.grid()
plt.show()