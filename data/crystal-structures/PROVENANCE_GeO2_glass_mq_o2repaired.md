# `GeO2_glass_mq_o2repaired.cif` — provenance

An amorphous GeO₂ model generated here by MLIP melt-quench, **then repaired by
hand**. It is not the unmodified output of a simulation, and it must not be
described as one. Read this file before citing it.

1125 atoms (375 Ge, 750 O), cell 26.300 × 26.300 × 29.798 Å, γ = 120°.

## How it was made

1. Melt-quench from `c-GeO2.cif` with MACE-MPA-0, via
   `scripts/slurm_geo2_remelt.sh` (ρ = 3.65 g/cm³, cutoff 2.4 Å, two-stage melt
   with a superheat stage and a slow quench arm). The exact `MELT_T` / `QUENCH`
   for this replicate are in the SLURM log on yemba, not reproduced here.
   Output: `GeO2_mq_hot_slowq_r1_REJECTED.cif` — a good network carrying **one
   trapped O₂ molecule**, and so rejected on the chemistry gate.
2. Repair with `scripts/repair_o2_defect.py`, strategy B:

   ```
   python scripts/repair_o2_defect.py \
       --input  data/crystal-structures/GeO2_mq_hot_slowq_r1_REJECTED.cif \
       --output data/crystal-structures/GeO2_glass_mq_o2repaired.cif \
       --central Ge --ligand O --system GeO2
   ```

## Why the repair is defensible

The O₂ was removed only after testing whether it wanted to exist. Two
independent interventions — separating the two oxygens in place, and separating
them while parking one on the open face of the under-coordinated Ge — were each
followed by a FIRE relaxation. The molecule **did not re-form** under either.
So it was kinetically trapped during the quench, not energetically preferred,
which is what licenses removing it. Had it re-formed, the honest conclusion
would have been that MACE-MPA-0 cannot make clean a-GeO₂, and no amount of
process tuning would have fixed that.

The naive fix would have failed. The molecule sat in a void bonded to zero Ge,
while the only under-coordinated Ge was **12.5 Å away** — the network had
already absorbed the missing pair by over-coordinating ten Ge to CN = 5. Pulling
the O₂ apart in place leaves both oxygens with nothing to bond to. Parking one
on that distant open face is what worked, and it cleared the CN = 3 site as well.

`repair_o2_defect.py` refuses to touch a structure carrying more than two
molecules — many O₂ means the melt was wrong and the answer is a different melt,
not surgery — and exits non-zero unless the result passes `assess_glass`.

## Verified properties

Measured here, against the published NNP model in `geo2_glass_nnp.cif`
(Kasamatsu et al., J. Chem. Phys. **161**, 204103 (2024); see
`PROVENANCE_geo2_glass_nnp.md`):

| quantity | this file | published NNP |
|---|---|---|
| atoms | 1125 | 3240 |
| Ge-sublattice disorder | 1.86× noise floor | 2.38× |
| max g(r) | 5.25 | 5.12 |
| ⟨CN⟩ Ge–O | 4.032 | 4.021 |
| O–Ge–O | 109.24 ± 8.07° | 109.22 ± 8.18° |
| Ge–O–Ge | 128.63° | 128.48° |
| O₂ molecules | 0 | 0 |

Both gates of `assess_glass` pass. Reproduce with:

```
python scripts/assess_glass.py data/crystal-structures/GeO2_glass_mq_o2repaired.cif \
    --system GeO2 --verbose
```

## Reproducibility of the underlying melt

Across four replicates at identical settings the Ge sublattice landed at
1.64–1.81× its noise floor — better than the published NNP model — while the O₂
count came out 1, 2, 2, 3. The disorder is reproducible; the oxygen defect is
what varies, and it never reached zero. That is the gap this repair closes.
