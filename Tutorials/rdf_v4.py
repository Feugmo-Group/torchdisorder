import torch
import torch_sim as ts
from torch_sim.neighbors import torch_nl_linked_cell


class StructuralDescriptors:
    """
    GPU-optimized structural descriptor calculator.

    Computes:
        g(r)   : radial distribution function
        adf    : angle distribution function
        G(r)   : neutron weighted RDF
        T(r)   : total correlation function
        S(Q)   : structure factor
        CN(r)  : coordination number
    """

    def __init__(
        self,
        cutoff: float = 8.0,
        sigma: float = 0.05,
        scattering_lengths=None,
        device=None,
        dtype=torch.float32,
    ):

        self.cutoff = cutoff
        self.sigma = sigma

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.dtype = dtype

        self.sqrt_2pi = torch.sqrt(torch.tensor(2.0 * torch.pi, device=self.device))

        if scattering_lengths is None:
            self.scattering_lengths = {"Si": 4.1491}
        else:
            self.scattering_lengths = scattering_lengths

    # ----------------------------------------------------
    # Neighbor list
    # ----------------------------------------------------

    def _neighbor_distances(self, state):

        mapping, system_mapping, shifts = torch_nl_linked_cell(
            state.positions,
            state.cell,
            state.pbc,
            torch.tensor(self.cutoff, device=self.device),
            state.system_idx,
        )

        i = mapping[0]
        j = mapping[1]

        rij = state.positions[i] - state.positions[j]

        if shifts is not None:
            rij = rij - shifts

        dist = torch.sqrt((rij ** 2).sum(dim=1) + 1e-12)

        return dist

    # ----------------------------------------------------
    # RDF
    # ----------------------------------------------------

    def compute_rdf(self, state, r_bins):

        dist = self._neighbor_distances(state)

        x = (r_bins[None, :] - dist[:, None]) / self.sigma

        gauss = torch.exp(-0.5 * x ** 2) / (self.sigma * self.sqrt_2pi)

        hist = gauss.sum(dim=0)

        volume = torch.abs(torch.det(state.cell[0]))

        N = state.positions.shape[0]

        rho = N / volume

        g_r = hist / (4 * torch.pi * r_bins ** 2 * rho * N)

        return g_r

    # ----------------------------------------------------
    # Coordination number
    # ----------------------------------------------------

    def compute_coordination_number(self, g_r, r_bins, rho):

        integrand = 4 * torch.pi * r_bins ** 2 * rho * g_r

        cn = torch.cumsum(integrand * (r_bins[1] - r_bins[0]), dim=0)

        return cn

    # ----------------------------------------------------
    # Angle distribution function
    # ----------------------------------------------------

    def compute_adf(self, state, theta_bins):

        mapping, system_mapping, shifts = torch_nl_linked_cell(
            state.positions,
            state.cell,
            state.pbc,
            torch.tensor(self.cutoff, device=self.device),
            state.system_idx,
        )

        i = mapping[0]
        j = mapping[1]

        rij = state.positions[j] - state.positions[i]

        if shifts is not None:
            rij = rij - shifts

        dist = torch.sqrt((rij ** 2).sum(dim=1) + 1e-12)

        mask = dist < self.cutoff

        rij = rij[mask]
        i = i[mask]

        theta_list = []

        unique_i = torch.unique(i)

        for center in unique_i:

            neigh = rij[i == center]

            if neigh.shape[0] < 2:
                continue

            neigh = neigh / torch.norm(neigh, dim=1, keepdim=True)

            cos_theta = torch.matmul(neigh, neigh.T)

            idx = torch.triu_indices(neigh.shape[0], neigh.shape[0], offset=1)

            cos_theta = cos_theta[idx[0], idx[1]]

            theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

            theta_list.append(theta)

        if len(theta_list) == 0:
            return torch.zeros_like(theta_bins)

        theta_vals = torch.cat(theta_list)

        x = (theta_bins[None, :] - theta_vals[:, None]) / self.sigma

        gauss = torch.exp(-0.5 * x ** 2) / (self.sigma * self.sqrt_2pi)

        adf = gauss.sum(dim=0)

        adf = adf / torch.trapz(adf, theta_bins)

        return adf

    # ----------------------------------------------------
    # Neutron G(r)
    # ----------------------------------------------------

    def compute_G_r(self, g_r, symbols):

        N = len(symbols)

        unique = set(symbols)

        frac = {s: symbols.count(s) / N for s in unique}

        b_mean = sum(frac[s] * self.scattering_lengths[s] for s in frac)

        G_r = (g_r - 1.0) * b_mean ** 2

        return G_r

    # ----------------------------------------------------
    # Total correlation
    # ----------------------------------------------------

    def compute_T_r(self, G_r, r_bins, rho):

        T_r = 4 * torch.pi * r_bins * rho * G_r

        return T_r

    # ----------------------------------------------------
    # Structure factor
    # ----------------------------------------------------

    def compute_structure_factor(self, G_r, r_bins, q_bins, rho):

        r = r_bins[None, :]
        q = q_bins[:, None]

        integrand = r * G_r * torch.sin(q * r) / (q * r + 1e-12)

        S_q = 1 + 4 * torch.pi * rho * torch.trapz(integrand, r_bins, dim=1)

        return S_q

if __name__ == "__main__":
    import torch
    import torch_sim as ts
    import matplotlib.pyplot as plt

    from ase.build import bulk

    #from rdf import StructuralDescriptors


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    atoms = bulk("Si", "diamond", a=5.43, cubic=True)

    state = ts.initialize_state(
        atoms,
        device=device,
        dtype=torch.float32,
    )

    symbols = atoms.get_chemical_symbols()

    calc = StructuralDescriptors(
        cutoff=6.0,
        sigma=0.05,
    )

    r_bins = torch.linspace(0.1, 6.0, 200, device=device)
    q_bins = torch.linspace(0.5, 15.0, 200, device=device)


    g_r = calc.compute_rdf(state, r_bins)

    volume = torch.abs(torch.det(state.cell[0]))

    rho = state.positions.shape[0] / volume


    theta_bins = torch.linspace(0, torch.pi, 180, device=device)

    adf = calc.compute_adf(state, theta_bins)

    cn = calc.compute_coordination_number(g_r, r_bins, rho)

    G_r = calc.compute_G_r(g_r, symbols)

    T_r = calc.compute_T_r(G_r, r_bins, rho)

    S_q = calc.compute_structure_factor(G_r, r_bins, q_bins, rho)






    plt.figure()
    plt.plot(r_bins.cpu(), g_r.cpu())
    plt.xlabel("r (Å)")
    plt.ylabel("g(r)")
    plt.title("Radial Distribution Function")

    plt.figure()
    plt.plot(theta_bins.cpu(), adf.cpu())
    plt.xlabel("Angle (rad)")
    plt.ylabel("P(θ)")
    plt.title("Angle Distribution Function")
    plt.show()

    plt.figure()
    plt.plot(r_bins.cpu(), cn.cpu())
    plt.xlabel("r (Å)")
    plt.ylabel("Coordination Number")

    plt.figure()
    plt.plot(r_bins.cpu(), T_r.cpu())
    plt.xlabel("r (Å)")
    plt.ylabel("T(r)")

    plt.figure()
    plt.plot(q_bins.cpu(), S_q.cpu())
    plt.xlabel("Q (Å⁻¹)")
    plt.ylabel("S(Q)")

    plt.show()