# Retired run scripts

These drove the LiPS glass structures now in
`data/crystal-structures/retired/` — every one of which contains overlapping atoms —
so each script here launches a refinement against an invalid seed. They are kept for
provenance: if you need to trace what a `run_lips_*` or `slurm_lips_*` result came
from, it is here.

`scripts/slurm_lips_glass.sh` supersedes all of them. It regenerates the glass by
melt-quench first rather than assuming a valid seed exists, and it is chemistry-aware
about the melt temperature (LiPS runs at 1500 K, not the 4000 K used for the oxides,
because Li₂S–P₂S₅ melts near 900–1100 K and would dissociate).

Do not resurrect one of these without first checking that its structure passes a
zero `< 1.2 Å` pair count.
