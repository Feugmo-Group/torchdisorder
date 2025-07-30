# torchdisorder

Torch-based modeling for disordered materials using PyTorch autograd to match target RDFs from XRD or neutron diffraction.

## Features

- Differentiable radial distribution function (RDF) loss
- Automatic config management with Hydra
- Experiment tracking with Weights & Biases
- Compatible with CUDA/GPU acceleration

## Usage

Run an experiment:

```bash
python -m torchdisorder.cli
```

Override parameters:

```bash
python -m torchdisorder.cli experiment.n_steps=10000 optimizer.lr=5e-4
```


```
amorphgen/
│
├── src/
│   └── amorphgen/
│       ├── __init__.py
│       ├── cli.py               # hydra entrypoint
│       ├── data/
│       │   ├── target_rdf.py    # loads & interpolates experimental RDF
│       │   └── utils.py
│       ├── model/
│       │   ├── generator.py     # positions -> (torch.tensor) coords
│       │   ├── rdf.py           # differentiable RDF calculator
│       │   └── loss.py
│       ├── engine/
│       │   ├── trainer.py       # main optimisation loop
│       │   └── callbacks.py     # W&B + other hooks
│       └── viz/
│           └── plot_rdf.py
│
├── configs/                     # hydra default config tree
│   ├── config.yaml
│   ├── experiment/
│   ├── optimizer/
│   ├── system/
│   └── wandb/
│
├── tests/
│
├── pyproject.toml               # poetry / setuptools
└── README.md
```