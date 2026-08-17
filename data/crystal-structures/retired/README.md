# Retired structures

Every file here contains atoms closer together than any chemical bond. They are kept
rather than deleted so the failure stays inspectable, and so any result that cites
them can be traced back. **Do not use them as seeds.** Refinement cannot repair a
broken seed — it inherits the overlaps, and `validate_structure` in `scripts/train.py`
now refuses to start from one.

Verify with a `< 1.2 Å` pair count, which must be zero:

```python
from ase.io import read
from ase.neighborlist import neighbor_list
d = neighbor_list("d", read("some.cif"), 1.2)
print(len(d) // 2, "overlapping pairs")   # must be 0
```

## What is here and why

Measured 2026-08-15. `shortest` is the closest interatomic distance in the cell; a
real Li–S bond is ≈ 2.4 Å and P–S ≈ 2.0 Å, so 0.066 Å is ~35× too short. This is a
packing failure in the glass generator, not strain.

| structure | atoms | pairs < 1.2 Å | shortest |
|---|---|---|---|
| `glass_70Li2S_noLi.cif` | 3224 | 140 | 0.066 Å |
| `glass_70Li2S_withLi.cif` | 5016 | 210 | 0.066 Å |
| `glass_75Li2S_withLi.cif` | 5837 | 1172 | 0.069 Å |
| `glass_67Li2S_small_noLi.cif` | 401 | 22 | 0.094 Å |
| `glass_67Li2S_small_withLi.cif` | 625 | 30 | 0.094 Å |
| `glass_70Li2S_small_noLi.cif` | 400 | 19 | 0.094 Å |
| `glass_70Li2S_small_withLi.cif` | 624 | 30 | 0.094 Å |
| `glass_75Li2S.cif` | 8204 | 1657 | 0.102 Å |
| `glass_67Li2S.cif` | 5054 | 241 | 0.102 Å |
| `glass_70Li2S.cif` | 5065 | 242 | 0.119 Å |
| `sio2_glass.cif` | 1125 | 181 | 0.185 Å |
| `glass_75Li2S_small_withLi.cif` | 366 | 77 | 0.219 Å |
| `glass_75Li2S_noLi.cif` | 3533 | 133 | 0.220 Å |
| `glass_67Li2S_noLi.cif` | 3227 | 124 | 0.273 Å |
| `glass_67Li2S_withLi.cif` | 5019 | 180 | 0.273 Å |
| `glass_67Li2S_small.cif` | 907 | 44 | 0.517 Å |
| `glass_75Li2S_small_noLi.cif` | 222 | 7 | 0.705 Å |
| `glass_75Li2S_small.cif` | 499 | 12 | 1.020 Å |
| `glass_70Li2S_small.cif` | 283 | 20 | 1.020 Å |

The `_small` variants fail too, so there was never a clean subset to fall back on.
The crystal inputs they were built from (`Li7P3S11.cif`, `c-SiO2.cif`) are fine — only
the generated glasses are broken, which points at the unit-placement step in
`torchdisorder/constraints/lps_generator.py`.

`sio2_glass.cif` is the seed behind the withdrawn F_IS crystal-vs-glass result
(⟨CN⟩ = 3.02, no first-shell peak). The corrected comparison uses the published GAP
model instead. Its refined descendants are in `outputs/retired/`.

## Replacements

- **SiO₂** — `../SiO2_mq_hot.cif`, validated against the published GAP model on all
  five order parameters. See the README section "Building a Glass Seed by Melt-Quench".
- **GeO₂** — `../GeO2_mq.cif`, likewise validated against the published NNP model.
  Carries 11 O–O pairs near 1.24 Å (see the caveat in the main README).
- **LiPS** — **not yet available.** The melt-quench regeneration from job 1207 fixed
  the overlaps (min d ≈ 1.97–2.00 Å) but left P under-coordinated, because that melt
  never reached its setpoint either. Jobs 1211/1215 are the rerun with γ = 1.0. Until
  one of those passes, there is no blessed LiPS glass seed.

## Configs left dangling

Eighteen configs still name files that now live here, and will fail to load:

```
configs/data/LiPS.yaml            configs/structure/LiPS.yaml
configs/data/LiPS_67.yaml         configs/structure/LiPS_67_{noLi,withLi}{,_small}.yaml
configs/data/LiPS_70.yaml         configs/structure/LiPS_70_{noLi,withLi}{,_small}.yaml
configs/data/LiPS_75.yaml         configs/structure/LiPS_75_{noLi,withLi}{,_small}.yaml
configs/structure/silica.yaml
```

This is deliberate. They cannot be repointed yet: the melt-quench route produces one
structure per composition, with no `_noLi` / `_withLi` / `_small` variants, and the
LiPS seeds are not validated regardless. A config that fails loudly is better than one
that silently trains on overlapping atoms. `configs/structure/silica.yaml` can be
repointed at `SiO2_mq_hot.cif` as soon as someone confirms the density it expects.
