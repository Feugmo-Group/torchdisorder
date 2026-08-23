# `lips70_glass_lips25_1200.cif` — provenance

Amorphous Li₇P₃S₁₁ (70Li₂S·30P₂S₅), generated here by MLIP melt-quench. Unlike
the GeO₂ glass in this directory, it is the unmodified output of a simulation —
no hand repair.

672 atoms, ρ = 1.850 g/cm³, min contact 1.980 Å.

## How it was made

`scripts/slurm_lips_glass.sh` with `LABEL=lips70`, using the **LiPS-25**
potential rather than a foundation model — a graph-pes MACE checkpoint
(`model_mace_cutoff_6_1.pt`) from
[github.com/nfragapane/lips-25](https://github.com/nfragapane/lips-25), loaded
on cuda/float32. Foundation models reduce P(V) to P(IV) in this system; the
system-specific potential does not.

Schedule: melt 1200 K for 30 ps → quench 1200 → 300 K over 40 ps (22.5 K/ps) →
anneal 300 K for 5 ps → FIRE relaxation.

## Verified properties

```
scripts/assess_glass.py data/crystal-structures/lips70_glass_lips25_1200.cif \
    --system LiPS --verbose
```

| quantity | value |
|---|---|
| disorder | std 0.489 = **1.57×** noise floor 0.312 (ceiling 0.624) |
| max g(r) | 3.55 (limit 8.0) |
| P–P pairs | 0 |
| S–S bonds (polysulfide) | 0 |
| P-free sulfur | 0 |
| speciation (diagnostic) | PS₄ 46%, P₂S₇ 46%, P₂S₆ 0%, other 8% |

Both gates pass. The chemistry is completely clean, which is the part foundation
models could not deliver.

The 1.57× disorder ratio reads as comfortable, not marginal — see the note in
`glass_quality.py` on why the raw std of 0.489 against a flat 0.5 limit was
misread as a 2% squeak before the noise floor was reported on passing runs.

## Why 1200 K works here and not for Li₃PS₄

The same schedule at the same temperature fails for lips75 (Li₃PS₄): its P–P
sublattice plateaus at std ≈ 0.60–0.70 *during* the 1200 K hold and never goes
lower, so the melt is never achieved and no quench rate rescues it. Li₇P₃S₁₁ is
the better glass former of the two, and 1200 K is enough for it. Do not carry
this temperature over to other Li–P–S compositions without checking the melt
trace.
