import sys, pathlib
from typing import Any, Dict
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
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
from torchdisorder.engine.trainer import run_optimization
import numpy as np
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
from torchdisorder.common.utils import OrderParameter
from ase import Atoms
from torchdisorder.model.xrd import XRDModel
from torchdisorder.viz.plotting import init_live_total_correlation, init_live_total_scattering, update_live_plot
from torchdisorder.viz.plotting import plot_total_correlation, plot_total_scattering
from torchdisorder.common.utils import write_trajectories
from torch_sim.state import DeformGradMixin, SimState
import plotly.graph_objects as go
from ipywidgets import interact
from torchdisorder.model.loss import ChiSquaredObjective, ConstraintChiSquared
from IPython.display import display
from plotly.subplots import make_subplots
from torchdisorder.engine.optimizer import StructureFactorCMP


@hydra.main(config_path=str(MODELS_PROJECT_ROOT / "configs"), config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("data.root_dir:", cfg.data.root_dir)
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

    # fig_T_r.show(renderer="browser")
    # fig_S_Q.show(renderer="browser")
    # Show widgets (if using Jupyter)



    # Then in each update step:
    # update_live_plot(trace_T_r, state["T_r"], save_path=plot_dir, fig=fig_T_r, step=step)
    # update_live_plot(trace_S_Q, state["S_Q"], save_path=plot_dir, fig=fig_S_Q, step=step)

    # Initialize RDF/spectrum calculator
    spec_calc = SpectrumCalculator.from_config_dict(cfg.data)

    # Generate initial structure
    atoms = generate_atoms_from_config(cfg.structure)
    atoms_list = [atoms]
    state =  atoms_to_state(atoms_list, device=cfg.accelerator, dtype=dtype)
    print(state)
    state.positions.requires_grad_(True)
    state.cell.requires_grad_(True)

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




    state.atomic_numbers = torch.tensor([chemical_symbols.index(a.symbol) for a in atoms_list[0]], dtype=torch.int64, device=device)
    print(state.atomic_numbers)

    xdr_model = XRDModel(spectrum_calc=spec_calc,
                         rdf_data=rdf_data,
                         dtype= dtype,
                         device=cfg.accelerator,
                         )


    results = xdr_model(state)
    print(results)

#     objective = ChiSquaredObjective(xdr_model)
#     cooper_optimizer = StructureFactorCMP(
#     model=xdr_model,
#     base_state=base_sim_state,
#     target_vec=target_structure_vector,
#     target_kind="S_Q",  # or "F_Q"
#     q_bins=q_values,
#     chi_squared_fn=objective
# )


if __name__ == "__main__":
    main()

