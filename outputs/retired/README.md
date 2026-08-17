# Retired runs

**The run data has been deleted; this note is what remains.** Every result listed
below was void, so the directories were removed on 2026-08-16 to reclaim ~1.1 GB.
What matters is *why* they were void — that is recorded here so a number from one of
these runs can still be traced and dismissed if it turns up in a draft or a slide.

Each run was void for one of two reasons, and in some cases both.

## 1. Wrong F(Q) convention

Anything refined against `target=F_Q` using an ISIS data file was minimising a units
mismatch, not a physical disagreement. `compute_neutron_F_Q` returns the PDFgetX
reduced structure factor Q·[S(Q) − 1], while the ISIS Oxide Glass Data archive
publishes ⟨b⟩²[S(Q) − 1] in barn — no factor of Q. The optimizer bought agreement with
that artefact by wrecking real structure, driving Si–O from 1.619 Å to 1.72–1.79 Å and
Si–O–Si from 141° to 120–128° while χ² fell 93–98 %.

Fixed by `OutputType.F_Q_ISIS`. Point any config whose target came from ISIS at
`F_Q_ISIS`, then rerun — these runs are worth repeating, not merely discarding.

Affected: `sio2_constrained_2026-08-13`, `sio2_constraints_active_2026-08-13`,
`sio2_overlap_fix_test_2026-08-13`, `sio2_penalty_fixed_2026-08-13`,
`sio2_speedtest_2026-08-13`.

## 2. Broken seed

`SiO2_mace_debug_2026-07-16` and `SiO2_2026-07-16` were seeded from
`sio2_glass.cif`, which already had ⟨CN⟩ = 3.02 and Si–O contacts down to 0.185 Å
before any optimization. Checkpoints sit at ⟨CN⟩ ≈ 3.0 from step 200 through step
2000: the refinement never repaired what it inherited, and `force_weight=1e-5` every
50 steps could not have.

`SiO2_mace_debug_2026-07-16/13-55-58/final_results/final_structure.cif` is the source
of the withdrawn **F_IS(a-SiO₂) = +0.049, ΔF_IS = 0.380** and the claim that F_IS is
"~4× more sensitive to disorder than q4/q6". Both are retracted. The corrected
comparison against the published GAP model gives **ΔF_IS = +0.025**, with q4 shifting
~4.3× *more* than F_IS — the reverse of the original claim. The crystal-side value
(−0.331) was always sound.

## 3. Superseded by the v6 rewrite

Also deleted, from the older code:

- `scripts/outputs/` and `scripts/wandb/` — 38 runs and 13 output directories from
  February 2026, written into `scripts/` because training was launched from that
  directory. Both paths were already in `.gitignore`; they were never meant to be
  kept. All predate the v6 optimizer and the F(Q) fix.
- `outputs/{Fe2O3,Fe2O3_mace,Fe2O3_fis_feedback,ferrihydrite_baseline,ferrihydrite_mace,
  LiPON,NaTaCl6}_2026-07-16` — July 2026 refinements on the v5 path. The LiPON and
  NaTaCl₆ runs are void for an additional reason: their F(Q) data has known
  normalization problems, so there was nothing sound to fit in the first place.
- `wandb/` — the local run cache for the above.

## Replacements

Use `data/crystal-structures/SiO2_mq_hot.cif` as the seed and `F_Q_ISIS` as the
target. See the main README, "Building a Glass Seed by Melt-Quench".

The surviving figures in `outputs/` are the two that are still valid:
`melt_quench_validation.{png,pdf}` (the crystal → glass validation) and
`sio2_FQ_target_vs_computed.png` (the diagnostic that identified the F(Q) convention
mismatch).
