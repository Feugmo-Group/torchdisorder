import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import logging
import torch
from typing import Callable, Optional, Dict, Tuple
import torch.nn as nn
from torchdisorder.common.utils import MODELS_PROJECT_ROOT
from torchdisorder.common.target_rdf import TargetRDFData
from torchdisorder.model.rdf import SpectrumCalculator
from torchdisorder.model.generator import generate_atoms_from_config
from torchdisorder.model.loss import AugLagLoss, AugLagHyper
from torchdisorder.engine.trainer import run_optimization
from torchdisorder.model.xrd import XRDModel
import numpy as np
from torchdisorder.engine.optimizer import aug_lag
import torch_sim as ts
from   torch_sim.io import atoms_to_state, state_to_atoms
from ase.data import chemical_symbols
from ase import Atoms
from ase.io import write, Trajectory
from torch_sim.state import DeformGradMixin, SimState
from torch_sim.optimizers import fire
from torch_sim.models.mace import MaceModel, MaceUrls
from mace.calculators.foundations_models import mace_mp
from pathlib import Path
from torchdisorder.engine.unconstrained import AugmentedLagrangian,AugLagNN
from torchdisorder.common.utils import OrderParameter
from ase import Atoms
from torchdisorder.viz.plotting import init_live_total_correlation, init_live_total_scattering, update_live_plot
from torchdisorder.viz.plotting import plot_total_correlation, plot_total_scattering
from torchdisorder.common.utils import write_trajectories
from torch_sim.state import DeformGradMixin, SimState
import plotly.graph_objects as go
from ipywidgets import interact
from torchdisorder.model.loss import ChiSquaredObjective, ConstraintChiSquared
from IPython.display import display
from plotly.subplots import make_subplots
try:
    import wandb  # soft‑import
except ModuleNotFoundError:  # pragma: no cover
    wandb = None  # type: ignore


import logging

try:
    import wandb  # soft‑import
except ModuleNotFoundError:  # pragma: no cover
    wandb = None  # type: ignore

logger = logging.getLogger(__name__)

@hydra.main(config_path=str(MODELS_PROJECT_ROOT / "configs"), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    #logger.info("Loaded Hydra Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Run optimization
    dtype = torch.float32
    device = torch.device(cfg.accelerator)
    # Load experimental RDF/SF targets
    rdf_data = TargetRDFData.from_dict(cfg.data.data, device=cfg.accelerator)



    if cfg.output.save_plots:
        plot_dir = Path(cfg.output.plots_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(cfg.output.plots_dir) / "T_r_initial_plots.html"
    fig_T_r, trace_T_r = init_live_total_correlation(r_bins=rdf_data.r_bins, T_target=rdf_data.T_r_target, out_path=filename)
    filename = Path(cfg.output.plots_dir) / "S_Q_initial_plots.html"
    fig_S_Q, trace_S_Q = init_live_total_scattering(q_bins=rdf_data.q_bins, F_target=rdf_data.F_q_target, out_path=filename)

    #display(fig_T_r)
    #display(fig_S_Q)
    # Show widgets (if using Jupyter)



    # Then in each update step:
    # update_live_plot(trace_T_r, state["T_r"], save_path=plot_dir, fig=fig_T_r, step=step)
    # update_live_plot(trace_S_Q, state["S_Q"], save_path=plot_dir, fig=fig_S_Q, step=step)

    # Initialize RDF/spectrum calculator
    spec_calc = SpectrumCalculator.from_config_dict(cfg.data)

    # Generate initial structure
    atoms = generate_atoms_from_config(cfg.structure)
    # -- initial SimState + FIRE ---------------------------------------------
    atoms_list = [atoms]
    state =  atoms_to_state(atoms_list, device=cfg.accelerator, dtype=dtype)
    state.positions.requires_grad_(True)
    state.cell.requires_grad_(True)
    print(state.cell)
    class StateWrapper:
        def __init__(self, original_state):
            self.__dict__.update(original_state.__dict__)
            self.system_idx = None  # or whatever you want
            self.n_systems = None  # or whatever you want

    # Usage

    state = StateWrapper(state)
    # state.system_idx = torch.arange(len(atoms_list), device=device)
    state.n_systems = torch.tensor(len(atoms_list))
    # Create system indices using repeat_interleave
    atoms_per_system = torch.tensor([len(a) for a in atoms_list], device=device)
    state.system_idx = torch.repeat_interleave(
        torch.arange(len(atoms_list), device=device), atoms_per_system
    )



    # Convert atomic numbers to tenso



    state.atomic_numbers = torch.tensor([chemical_symbols.index(a.symbol) for a in atoms_list[0]], dtype=torch.int64, device=device)



    xdr_model = XRDModel(spectrum_calc=spec_calc,
                         rdf_data=rdf_data,
                         dtype= dtype,
                         device=cfg.accelerator ,
                         )



    results = xdr_model(state)


    xxdr_model = XRDModel(spectrum_calc=spec_calc,
                         rdf_data=rdf_data,
                         dtype=dtype,
                         device=cfg.accelerator,
                         )

    objective = ChiSquaredObjective(xxdr_model)

    def constraint_tetrahedral(order_model: nn.Module, threshold: float = 0.5) -> Callable:
        def constraint_fn(state) -> torch.Tensor:
            values = order_model(state)["q_tet"]  # shape: [n_atoms]

            return values - threshold  # want values < threshold ⇒ values - threshold ≤ 0

        return constraint_fn

    def get_constraint_indices(state, target_symbol: str, device=None):
        """
        Find local atom indices where atomic symbol == target_symbol within a single system.

        Args:
            state: AugLagState with attributes:
                - atomic_numbers (torch.Tensor)
            target_symbol: str, chemical symbol to match, e.g. "W"
            device: torch.device (optional)

        Returns:
            torch.Tensor: 1D tensor of local indices of atoms matching target_symbol
        """
        from ase.data import chemical_symbols

        atomic_numbers_np = state.atomic_numbers.detach().cpu().numpy()
        symbols = [chemical_symbols[z] for z in atomic_numbers_np]

        device = device or state.atomic_numbers.device

        indices = [i for i, s in enumerate(symbols) if s == target_symbol]
        return torch.tensor(indices, device=device, dtype=torch.long)

    def init_lam_sigma(state, constraint_indices, init_lam_val=1.0, init_sigma_val=1.0, device=None,
                       dtype=torch.float32):
        """
        Initialize lam, sigma, eta, and a constraint mask for the Augmented Lagrangian optimizer.

        Args:
            state: AugLagState with `atomic_numbers` and `system_idx`.
            constraint_indices: global 1D LongTensor of atom indices under constraint.
            init_lam_val: initial value for lambda.
            init_sigma_val: initial value for sigma.
            device: torch device.
            dtype: tensor dtype.

        Returns:
            lam (Tensor): shape [n_atoms], initialized lam.
            sigma (Tensor): shape [n_atoms], initialized sigma.
            eta (Tensor): shape [n_atoms], computed as lam / sigma safely.
            mask (BoolTensor): shape [n_atoms], True where atoms are constrained.
        """
        device = device or state.atomic_numbers.device
        n_atoms = state.atomic_numbers.shape[0]

        lam = torch.zeros(n_atoms, dtype=dtype, device=device)
        sigma = torch.zeros_like(lam)
        mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)

        if constraint_indices.numel() > 0:
            lam[constraint_indices] = init_lam_val
            sigma[constraint_indices] = init_sigma_val
            mask[constraint_indices] = True

        eta = torch.zeros_like(lam)
        eta[mask] = lam[mask] / sigma[mask]

        return lam, sigma, eta, mask

    # Usage example (assuming 'state' is a single system AugLagState):

    constraint_indices = get_constraint_indices(state, target_symbol="Si")
    lam, sigma,eta, mask = init_lam_sigma(state, constraint_indices, init_lam_val=1.0, init_sigma_val=1.0)


    order_model = OrderParameter( central=cfg.data.central,
                         neighbour=cfg.data.neighbour,
                         cutoff=torch.tensor(cfg.data.cutoff),
                         compute_q_tet=True,
                         dtype=dtype,
                         device=cfg.accelerator).to(device)


    # Build the constraint

    # constraint_tet = constraint_q(order_model, "q_tet", 0.3)
    # constraint_oct = constraint_q(order_model, "q_octahedral", 0.6)
    #
    # def constraint_q(order_model: nn.Module, key: str, threshold: float):
    #     def constraint_fn(state) -> torch.Tensor:
    #         return order_model(state)[key] - threshold
    #
    #     return constraint_fn
    #
    # constraints = [constraint_tet, constraint_oct]


    constraint_fn = constraint_tetrahedral(order_model, threshold=0.5)

    # Wrap in a list if needed
    constraints = [(constraint_fn, False)]  # False means "less than or equal to"


    # aug_init = AugmentedLagrangian( objective=ChiSquaredObjective,
    #                                 model=xdr_model,
    #                                 constraints_eq=[],
    #                                 constraints_ineq=constraints,
    #                                 lam=[0.0],
    #                                 sigma=[1.0],
    #                                 method="BFGS",
    #                                 )

    aug_init = AugLagNN(objective=ChiSquaredObjective,
                                   model=xdr_model,
                                   constraints_eq=[],
                                   constraints_ineq=constraints,
                                   lam=lam,
                                   sigma=sigma,
                                    eta=eta,
                        mask=mask,
                                   method="BFGS",
                                   )


    state = aug_init(state)



    state = aug_init.step(state)

    print(state)
    # rresults = objective(state)
    # print(rresults)

    #rdf_data = TargetRDFData.from_dict(cfg.data.data, device=cfg.accelerator)

    # fig_T_r, trace_T_r = init_live_total_correlation(...)
    # fig_S_Q, trace_S_Q = init_live_total_scattering(...)
    # display(fig_T_r)
    # display(fig_S_Q)




    # Initialize unit cell gradient descent optimizer
    # init_fn, update_fn = aug_lag(model=xdr_model,loss_fn=AugLagLoss(),
    #                              device=cfg.accelerator,
    #                              optimize_cell=cfg.trainer.optimize_cell,
    #                              lr=cfg.trainer.lr,
    #                              scheduler=None,  # Add scheduler if needed
    #                              lag_loss=None,  # Add AugLagLoss if needed
    #                              dtype=dtype)




    # # # Load Augmented Lagrangian hyperparameters
    # hyper = AugLagHyper.from_yaml("../configs/trainer/AugLag.yaml")
    # hyper.tol = float(hyper.tol)
    # print("Type of self.hyper.tol:", float(hyper.tol))
    # loss_fn = AugLagLoss(rdf_data, hyper, device=device).to(device)
    #
    # loss_dict = loss_fn(results)
    # init_fn, update_fn = aug_lag(model=xdr_model,loss_fn=loss_fn,
    #                              lag_loss=loss_fn,  # Add AugLagLoss if needed
    #                              )
    # #
    # state = init_fn(state)
    # steps = 500
    #
    # for step in range(steps):
    #     state = update_fn(state)
    #     aux = state.diagnostics  # type: ignore[attr-defined]



if __name__ == "__main__":
    main()