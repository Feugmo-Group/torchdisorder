import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import logging
import torch
from torchdisorder.common.utils import MODELS_PROJECT_ROOT
from torchdisorder.common.target_rdf import TargetRDFData
from torchdisorder.model.rdf import SpectrumCalculator
from torchdisorder.model.generator import generate_atoms_from_config
from torchdisorder.model.loss import AugLagLoss, AugLagHyper
from torchdisorder.engine.trainer import run_optimization
from torchdisorder.model.xrd import XRDModel
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
from ase import Atoms
from torchdisorder.viz.plotting import init_live_total_correlation, init_live_total_scattering, update_live_plot
from torchdisorder.viz.plotting import plot_total_correlation, plot_total_scattering
from torchdisorder.common.utils import write_trajectories
from torch_sim.state import DeformGradMixin, SimState
import plotly.graph_objects as go
from ipywidgets import interact

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




    xdr_model = XRDModel(spectrum_calc=spec_calc,
                         rdf_data=rdf_data,
                         central=cfg.data.central,
                         neighbour=cfg.data.neighbour,
                         cutoff=torch.tensor(cfg.data.cutoff),
                         compute_descriptors= True,
                         compute_q_tet = True,
                         dtype= dtype,
                         device=cfg.accelerator ,
                         )



    results = xdr_model(state)


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

    # # Load Augmented Lagrangian hyperparameters
    hyper = AugLagHyper.from_yaml("../configs/trainer/AugLag.yaml")
    hyper.tol = float(hyper.tol)
    print("Type of self.hyper.tol:", float(hyper.tol))
    loss_fn = AugLagLoss(rdf_data, hyper, device=device).to(device)

    loss_dict = loss_fn(results)
    init_fn, update_fn = aug_lag(model=xdr_model,loss_fn=loss_fn,
                                 lag_loss=loss_fn,  # Add AugLagLoss if needed
                                 )
    #
    state = init_fn(state)
    #


    # -- MACE model for fast relaxations -------------------------------------
    raw_mace = mace_mp(MaceUrls.mace_mpa_medium, return_raw_model=True, default_dtype=dtype, device=device)
    mace_model = MaceModel(raw_mace, device=device, compute_forces=True, dtype=dtype)
    mace_init, mace_step = fire(model=mace_model)

    logger.info("— Optimising with AugLagLoss + FIRE —")

    steps =  500
    mace_relax_every =  0
    mace_relax_steps =  5
    use_wandb =  False
    log_every =  5

    # def write_trajectories(atoms, traj_path: str, lammps_path: str):
    #     # Write ASE trajectory file (.traj)
    #     with Trajectory(traj_path, mode='w') as traj:
    #         traj.write(atoms)
    #
    #     # Write LAMMPS trajectory file (.lammpstrj)
    #     write(lammps_path, atoms, format='lammps-dump-text')

    def write_trajectories(atoms: Atoms, traj_path: str, xdatcar_path: str):
        # Write ASE trajectory file (.traj) for internal use
        write(traj_path, atoms, format="traj", append=True)

        # Write to VASP XDATCAR format
        write(xdatcar_path, atoms, format="vasp-xdatcar", append=True)

    for step in range(steps):
        state = update_fn(state)
        aux = state.diagnostics  # type: ignore[attr-defined]



        # Optionally save trajectory
        #trajectory_writer: Optional[Trajectory] = None,  # NEW
        if cfg["output"]["write_trajectory"]:
            #traj = Trajectory(cfg["output"]["trajectory_path"], mode='w')
            # Ensure output directory exists
            traj_path = Path(cfg.output.trajectory_path)
            traj_path.mkdir(parents=True, exist_ok=True)
            #traj_path.parent.mkdir(parents=True, exist_ok=True)
            # Ensure trajectory path is a directory


            # Create ASE Trajectory file
            ase_path = traj_path / "trajectory.traj"
            xdatcar_path = traj_path / "XDATCAR"
            # traj = Trajectory(str(ase_path), mode='w')
            # trajectory_writer = traj

            atms = state_to_atoms(state)[0]
            # atms = Atoms(
            #     numbers=new_state.atomic_numbers.cpu().numpy(),
            #     positions=new_state.positions.cpu().numpy(),
            #     masses=new_state.masses.cpu().numpy(),
            #     cell=new_state.cell.cpu().numpy(),
            #     pbc=new_state.pbc,
            # )
            #trajectory_writer.write(atms)

            write_trajectories(atms, str(ase_path), str(xdatcar_path))

        # Optionally initialize W&B
        if cfg.wandb.mode != "disabled":
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.run_name,
                mode=cfg.wandb.mode,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        # ---- logging --------------------------------------------------------
        if step % log_every == 0:
            logger.info(
                f"step {step:4d} | loss = {aux['loss'].item():.6f} | "
                f"χ²_corr={aux['chi2_corr'].item():.3e} | "
                f"χ²_scatt={aux['chi2_scatt'].item():.3e} | q_loss={aux['q_loss'].item():.3e} | "
                f"scale_q={aux['scale_q'].item():.3e} | scale_scatt={aux['scale_scatt'].item():.3e} | "
                f"rho={aux['rho'].item():.3e} | lambda_corr={aux['lambda_corr'].item():.3e} | "

            )

        if use_wandb and wandb is not None:
            if step % cfg.wandb.log_every == 0:
                wandb.log({
                    "step": step,
                    "loss": aux["loss"].item(),
                    "chi2_corr": aux["chi2_corr"].item(),
                    "chi2_scatt": aux["chi2_scatt"].item(),
                    "q_loss": aux["q_loss"].item(),
                    "scale_q": aux["scale_q"].item(),
                    "scale_scatt": aux["scale_scatt"].item(),
                    "rho": aux["rho"].item(),
                    "lambda_corr": aux["lambda_corr"].item(),
                })


                # wandb.log({
                #     "T_r_plot": wandb.Image(fig_T_r),
                #     "S_Q_plot": wandb.Image(fig_S_Q)
                # })


        if cfg.output.save_plots:
            plot_dir = Path(cfg.output.plots_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)

            # trace_T_r.y = desc["T_r"].cpu().numpy()
            # trace_S_Q.y = desc["S_Q"].cpu().numpy()
        # if step % save_plot == 0:
        #     # Compute RDF + S(Q) + Save plots
        #     plot_total_correlation(
        #         r_bins=desc["r_bins"],
        #         T_computed=desc["T_r"],
        #         T_target=target_data.T_r_target,
        #         out_path=plot_dir / f"T_r_step{step:04d}.html",
        #     )
        #     plot_total_scattering(
        #         q_bins=desc["q_bins"],
        #         F_computed=desc["S_Q"],
        #         F_target=target_data.F_q_target,
        #         out_path=plot_dir / f"S_Q_step{step:04d}.html",
        #     )

        # plot_losses_plotly(logged, "outputs/descriptor_logs")
        # plot_energy_plotly(logged, "outputs/descriptor_logs")

        # out_path = Path(out_path)
        # out_path.parent.mkdir(parents=True, exist_ok=True)
        # fig.write_html(out_path.as_posix())

        # ---- interleaved MACE relaxation -----------------------------------
        if mace_relax_every > 0 and step % mace_relax_every == 0:
            logger.info(f"[MACE] Relax block at outer step {step}")
            s_mace = mace_init(state)
            for sub in range(mace_relax_steps):
                s_mace = mace_step(s_mace)
                if sub % 3 == 0:
                    logger.info(f"    MACE sub‑step {sub:02d} | E = {s_mace.energy[0].item():.6f} eV")
            # safe copy‑back without breaking autograd
            with torch.no_grad():
                state.positions.copy_(s_mace.positions.detach())
                state.cell.copy_(s_mace.cell.detach())

            # with torch.no_grad():
            #     state = ts.io.atoms_to_state([ts.io.state_to_atoms(s_mace)[0]], device=device, dtype=dtype)
            #     state.positions.requires_grad_(True)
            #     state.cell.requires_grad_(True)
            #     state = fire_init(state)






    #
    # # Dynamically load trainer loss object
    # loss_fn = hydra.utils.instantiate(cfg.trainer)
    # logger.info(f"Using trainer: {loss_fn.__class__.__name__}")
    #
    # # Optionally initialize W&B
    # if cfg.wandb.mode != "disabled":
    #     wandb.init(
    #         project=cfg.wandb.project,
    #         entity=cfg.wandb.entity,
    #         name=cfg.wandb.run_name,
    #         mode=cfg.wandb.mode,
    #         config=OmegaConf.to_container(cfg, resolve=True),
    #     )


    # final_state= run_optimization(cfg,
    #     state=state,
    #     #atoms=atoms,
    #     spectrum_calc=spec_calc,
    #     rdf_data=rdf_data,
    #     hyper=hyper,
    #     device=cfg.accelerator,
    #     steps=cfg.experiment.n_steps,
    #     mace_relax_every=cfg.trainer.mace_relax_every,
    #     mace_relax_steps=cfg.trainer.mace_relax_steps,
    #     use_wandb=(cfg.wandb.mode != "disabled"),
    #     log_every=cfg.experiment.log_every,
    #     dtype=dtype,
    # )


    # -- final structure back to ASE -----------------------------------------
    final_atoms = ts.io.state_to_atoms(state)[0]
    atoms.set_positions(final_atoms.positions)
    atoms.set_cell(final_atoms.cell)

if __name__ == "__main__":
    main()

 #
 # # Construct loss function
 #    loss_fn = AugLagLoss(
 #        spectrum_calc=spec_calc,
 #        rdf_data=rdf_data,
 #        hyper=hyper,
 #        device=cfg.accelerator,
 #    )
 #run_optimization(atoms=atoms, loss_fn=loss_fn, device=cfg.accelerator, steps=cfg.trainer.steps)
    
#python -m torchdisorder.cli optimizer.lr=5e-4 system.n_atoms=1000

# python script/run_sio2.py wandb=online
# python script/run_sio2.py wandb=disabled
