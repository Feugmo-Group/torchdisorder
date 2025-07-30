# ---------------------------------------------------------------------------
# viz/plotting.py (Plotly version)
# ---------------------------------------------------------------------------
# Generates interactive HTML files so the user can zoom & inspect curves.

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import torch
import plotly.graph_objects as go
from pathlib import Path
from typing import Union

import plotly.graph_objects as go
from pathlib import Path
from typing import Union
from typing import Union, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
def plot_partial_rdfs(r_bins,
                      rdf_dict,
                      out_path: Union[str, Path] = None,
                    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
                      ) -> None:
    """Plot partial RDFs and save to the specified path.

    Parameters:
    - r_bins: torch.Tensor
    - rdf_dict: dict of (species_i, species_j) -> torch.Tensor
    - out_path: output path (e.g., .html or .png)
    """
    fig = make_subplots(rows=1, cols=1)
    for (a, b), g_r in rdf_dict.items():
        fig.add_trace(go.Scatter(x=r_bins.cpu().numpy(), y=g_r.cpu().numpy(), mode="lines", name=f"{a}-{b}"))

    fig.update_layout(
        title="Partial RDFs",
        xaxis_title="Radial distance r (Å)",
        yaxis_title="Partial pair distribution g(r)",
    )

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    #out_path = Path(out_path)
    #out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path.as_posix())
    #fig.write_html(out_path.with_suffix(".html"))
    #fig.write_image(out_path.with_suffix(".png"))
    if wandb_run:
        wandb_run.log({"Partial RDFs": wandb.Image(fig)})




def plot_losses_plotly(logged_losses: dict,
                       out_path: Union[str, Path]= None,
                       wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
                       ) -> None:
    """
    Plot all loss components using Plotly and save to an HTML file.

    Parameters:
    - logged_losses: dict with keys like 'loss', 'q_loss', etc.
    - out_path: path prefix where plot_loss.html will be saved
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()
    for key, values in logged_losses.items():
        if "energy" in key.lower():
            continue  # skip energy-related keys
        fig.add_trace(go.Scatter(y=values, mode="lines", name=key))

    fig.update_layout(
        title="Loss Component Convergence",
        xaxis_title="Iteration",
        yaxis_title="Loss Value (log scale)",
        yaxis_type="log",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99)
    )
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path.with_name(out_path.stem + "_loss.html"))

    if wandb_run:
        wandb_run.log({"loss_convergence": wandb.Image(fig)})

    return fig

def plot_energy_plotly(logged_losses: dict,
                       out_path: Union[str, Path]= None,
                       wandb_run: Optional[wandb.sdk.wandb_run.Run] = None) -> None:
    """
    Plot energy evolution using Plotly and save to an HTML file.

    Parameters:
    - logged_losses: dict with a key like 'energy'
    - out_path: path prefix where plot_energy.html will be saved
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    energy_key = next((k for k in logged_losses if "energy" in k.lower()), None)
    if energy_key is None:
        print("No energy-related key found in logged_losses.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=logged_losses[energy_key], mode="lines", name=energy_key))

    fig.update_layout(
        title="Energy Evolution",
        xaxis_title="Iteration",
        yaxis_title="Energy",
        template="plotly_white"
    )
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path.with_name(out_path.stem + "_energy.html"))
    if wandb_run:
        wandb_run.log({"Energy_Evolution": wandb.Image(fig)})


def plot_total_correlation(
    r_bins: torch.Tensor,
    T_computed: torch.Tensor,
    T_target: torch.Tensor,
    out_path: Union[str, Path]= None,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r_bins.cpu().numpy(), y=T_computed.cpu().numpy(), mode="lines", name="Computed"))
    fig.add_trace(
        go.Scatter(
            x=r_bins.cpu().numpy(),
            y=T_target.cpu().numpy(),
            mode="lines",
            name="Target",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title="Total Correlation T(r)",
        xaxis_title="Radial distance r (Å)",
        yaxis_title="T(r)",
    )

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path.as_posix())

    if wandb_run:
        wandb_run.log({"T_r": wandb.Image(fig)})


def plot_total_scattering(
    q_bins: torch.Tensor,
    F_computed: torch.Tensor,
    F_target: torch.Tensor,
    out_path: Union[str, Path]= None,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q_bins.cpu().numpy(), y=F_computed.cpu().numpy(), mode="lines", name="Computed"))
    fig.add_trace(
        go.Scatter(
            x=q_bins.cpu().numpy(),
            y=F_target.cpu().numpy(),
            mode="lines",
            name="Target",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title="Total Structure Factor S(Q)",
        xaxis_title="Scattering vector Q (Å⁻¹)",
        yaxis_title="S(Q)",
    )
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_path.as_posix())

    if wandb_run:
        wandb_run.log({"S_Q": wandb.Image(fig)})


def init_live_total_correlation(r_bins, T_target, out_path=None):
    fig = go.FigureWidget()
    fig.add_scatter(x=r_bins, y=T_target, name="Target", line=dict(dash='dash'))
    trace = fig.add_scatter(x=r_bins, y=T_target * 0, name="Computed", line=dict())
    fig.update_layout(title="Total Correlation Function T(r)",
                      xaxis_title="r (Å)", yaxis_title="T(r)", height=400)
    if out_path:
        fig.write_html(str(out_path))  # Save interactive HTML
    #fig.show()  # Show plot interactively
    return fig, trace

def init_live_total_scattering(q_bins, F_target, out_path=None):
    fig = go.FigureWidget()
    fig.add_scatter(x=q_bins, y=F_target, name="Target", line=dict(dash='dash'))
    trace = fig.add_scatter(x=q_bins, y=F_target * 0, name="Computed", line=dict())
    fig.update_layout(title="Total Scattering S(Q)",
                      xaxis_title="Q (Å⁻¹)", yaxis_title="S(Q)", height=400)
    if out_path:
        fig.write_html(str(out_path))  # Save interactive HTML
    #fig.show()  # Show plot interactively
    return fig, trace




def update_live_plot(trace, new_y, save_path: Path | None = None, fig: go.FigureWidget | None = None,
                     step: int | None = None):
    with trace.batch_update():
        trace.y = new_y

    if save_path is not None and fig is not None and step is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        filename = save_path / f"{fig.layout.title.text.replace(' ', '_')}_step{step:04d}.html"
        fig.write_html(str(filename))




def window(sequence):
    smax, _ = torch.max(sequence, axis=0)
    smin, _ = torch.min(sequence, axis=0)
    Max = 1.1 * max(max(smax), abs(min(smin)))
    shape = smax.shape[0]
    return Max, shape

def plot_surface(func, sequence, info=None):
    Max, shape = window(sequence)

    if shape == 1:
        x = torch.linspace(-Max, Max, 250)
        y = func(x)
        y_iter = func(sequence)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x.numpy(), y=y.numpy(), mode='lines', name='Function'))
        fig.add_trace(go.Scatter(x=sequence.numpy(), y=y_iter.numpy(), mode='markers+lines',
                                 name='Path', marker=dict(color='red', symbol='star')))

        fig.update_layout(
            title="1D Surface",
            xaxis_title="x",
            yaxis_title="y",
            height=600,
            width=600
        )
        return fig

    else:
        x = torch.linspace(-Max, Max, 250)
        y = torch.linspace(-Max, Max, 250)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        Z = func([X, Y])
        iter_x, iter_y = sequence[:, 0], sequence[:, 1]
        iter_z = func([iter_x, iter_y])

        fig = go.Figure(data=[
            go.Surface(z=Z.numpy(), x=X.numpy(), y=Y.numpy(), colorscale='Jet', opacity=0.5, name='Surface'),
            go.Scatter3d(
                x=iter_x.numpy(),
                y=iter_y.numpy(),
                z=iter_z.numpy(),
                mode='markers+lines',
                marker=dict(size=4, color='red', symbol='circle'),
                name='Path'
            )
        ])

        title = ""
        if info is not None:
            key, values = list(info.items())[0]
            if key == 'al':
                title = f"$\\lambda$={values[0].item()}, $\\sigma$={values[1].item()}"
            elif key == 'pe':
                title = f"$\\sigma$={values[0].item()}"
            elif key == 'ba':
                title = f"$\\mu$={values[0].item()}"

        fig.update_layout(
            title=title or "3D Surface",
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z'
            ),
            height=700,
            width=700
        )
        return fig

def plot_contour(func, sequence, info=None):
    Max, _ = window(sequence)

    x = torch.linspace(-Max, Max, 250)
    y = torch.linspace(-Max, Max, 250)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    Z = func([X, Y])
    iter_x, iter_y = sequence[:, 0], sequence[:, 1]

    fig = go.Figure()

    fig.add_trace(go.Contour(
        z=Z.numpy(),
        x=x.numpy(),
        y=y.numpy(),
        colorscale='Jet',
        contours=dict(showlines=False),
        name='Function'
    ))

    fig.add_trace(go.Scatter(
        x=iter_x.numpy(),
        y=iter_y.numpy(),
        mode='markers+lines',
        marker=dict(color='red', symbol='star'),
        name='Path'
    ))

    title = ""
    if info is not None:
        key, values = list(info.items())[0]
        if key == 'al':
            title = f"$\\lambda$={values[0].item()}, $\\sigma$={values[1].item()}"
        elif key == 'pe':
            title = f"$\\sigma$={values[0].item()}"
        elif key == 'ba':
            title = f"$\\mu$={values[0].item()}"

    fig.update_layout(
        title=title or "Contour Plot",
        xaxis_title='x',
        yaxis_title='y',
        height=700,
        width=700
    )
    return fig


# fig = plot_surface(func, sequence, info)
# fig.show()  # Interactive in Jupyter or browser
#
# fig2 = plot_contour(func, sequence, info)
# fig2.write_html("contour.html")  # Save as HTML
