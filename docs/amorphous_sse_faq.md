# Amorphous solid electrolytes — FAQ

Answers to the questions that cost us time between June and August 2026, written
so they are not rediscovered. Every number was measured with the tooling in this
repo. Where something is believed but not measured, it says so.

The destination is **mixed-anion** amorphous solid electrolytes. SiO₂, GeO₂ and
Li–P–S are calibration systems chosen because published references exist for
them; the questions below are sorted by what generalises rather than by material.

For the procedure itself see `docs/amorphous_electrolyte_protocol.md`. This file
is the companion: what goes wrong, and how to tell.

---

## Contents

- [Judging a structure](#judging-a-structure)
- [Running the melt-quench](#running-the-melt-quench)
- [Chemistry and mixed anions](#chemistry-and-mixed-anions)
- [Potentials](#potentials)
- [Refinement against scattering data](#refinement-against-scattering-data)
- [What we still do not know](#what-we-still-do-not-know)

---

## Judging a structure

### My structure has ⟨CN⟩ = 4.00 with zero defects. Is it a glass?

Almost certainly not. **A crystal that never melted scores a perfect coordination
number**, because the crystal already contains the motif you are looking for.
On 2026-08-17 this single blind spot reported three unmelted GeO₂ crystals as
successes in one day.

DFT BOMD of a-Li₃PS₄ gives PS₄ : P₂S₆ : P₂S₇ ≈ 6 : 2 : 1, which works out to
⟨CN_P⟩ = 3.67. A real glass has defects; 4.00 is the under-melted answer.

> Perfection is the warning sign, not the goal.

### Then what does tell a glass from a crystal?

Central-atom sublattice g(r), judged **beyond 6 Å**. Long-range order is what
melting destroys. First-shell peaks survive in both phases and discriminate
nothing, which is exactly why coordination number fails.

Both gates are required and they are independent: chemistry (is it the right
material?) and disorder (did it actually melt?). Passing one says nothing about
the other. `assess_glass` runs both, but only if you pass `--system`; without it
the verdict rests on coordination alone and an unmelted crystal walks through.

### My std is 0.504 against a limit of 0.5. Is that a marginal failure?

No, and this one nearly cost a real result. **Raw std is not comparable between
chemistries**, because a g(r) built from few centres is noisy. An oxide
sublattice has 375–1728 central atoms; a 672-atom Li₇P₃S₁₁ cell has 96 P.

Measure the counting-noise floor by randomising positions in the same cell and
quote the **excess over that floor**:

| structure | N centres | std | noise | ratio | truth |
|---|---|---|---|---|---|
| sio2_glass_gap | 1728 | 0.134 | 0.039 | 3.47 | glass |
| SiO2_mq_hot_mpa | 375 | 0.143 | 0.079 | 1.80 | glass |
| lips70 (LiPS-25) | 96 | 0.504 | 0.309 | **1.63** | disordered |
| c-SiO2 | 375 | 2.303 | 0.072 | 31.97 | crystal |

That 0.504 is almost entirely shot noise. Its excess over the floor is lower
than every published glass reference in the table. Crystals sit 14–32× over
their own floor and are rescued by neither test.

The floor is measured by randomising, so ratios move a little between runs and
between tool versions: `assess_glass` currently reports lips70 at 0.489 / 1.57×
against the 0.504 / 1.63× recorded above. Read the ratio to two significant
figures, not as an exact reproduction.

### Can I compare my ratio against the one in a published paper?

**Not unless both are measured at the same cell size.** Noise falls as 1/√N
while a glass holds roughly constant structural signal, so the same material
scores higher in a bigger cell. Published a-Li₃PS₄ reads **4.15× at 1111 P and
1.36× at ~100 P** — same structure, same tool, ratio three times apart.

This is easy to do by accident. An early claim here that our GeO₂ "beat the
published NNP model, 1.86× vs 2.38×" was exactly this mistake, and it reverses
once both are measured at matched size. Carve the reference down first:

```bash
poetry run python scripts/calibrate_disorder_gate.py \
    --reference data/crystal-structures/sio2_glass_gap.cif \
    --like      data/crystal-structures/SiO2_mq_hot_mpa.cif \
    --species Si
```

It reports a bracket rather than a number, because carving cuts genuine
long-range correlation at the seam and therefore understates a real glass.

### My cell is small and the disorder gate passed. Should I trust it?

Check that the gate had a beyond-6 Å window at all. Crystalline Li₃PS₄ is 32
atoms in a 6 Å cell, which gives **one histogram bin whose std is exactly 0.000**
and sails under any ceiling. The gate is written to fail closed on this, but the
underlying trap is general: half your shortest cell length must exceed 6 Å or the
measurement is not defined.

Elongated cells bite the same way for a subtler reason. A seed measuring
18.5 × 24.2 × 39.2 Å clamps the r-window to 9.23 Å, so its std is measured over a
*different, narrower* range than a compact cell's and the two numbers are not
comparable. Build seeds compact.

### Should I use `--expected-cn 4`?

Only if 4 is the *glass* number, and usually it is not. That crystal-derived
default rejected two good runs, one of them by 0.0025. Where no glass speciation
is published, skip the coordination check rather than run it against a guess.

---

## Running the melt-quench

### What melt temperature?

**~2.0 × Tm, not Tm.** A perfect crystal in periodic boundaries has no
nucleation sites, so it superheats well past its thermodynamic melting point
before it becomes mechanically unstable. On a 30 ps MD timescale the
superheating limit is what governs whether anything melts.

| System | Tm (K) | Working melt T | Ratio |
|---|---|---|---|
| SiO₂ | 1986 | 4000 | 2.0× |
| GeO₂ | 1388 | 2800 | 2.0× |

**The ratio is the transferable number.** Setting melt T near Tm produced three
"successes" that were solid crystal.

Too hot decomposes the material, and that is a real failure rather than a
potential artefact: GeO₂ inherited silica's 4000 K by accident, which is 2.9× its
Tm and deep into GeO₂ → GeO + ½O₂. It produced 11 O₂ molecules. The potential was
correctly describing decomposition of a material held far too hot.

### My temperature explodes to 10¹⁶ K in fifty steps.

`torch_sim`'s time unit is not femtoseconds. Its internal unit is
√(amu·Å²/eV) ≈ 10.18 fs, and `MetalUnits.time` is one picosecond expressed in it.
Passing `dt=1.0` means one **picosecond** — a 1000× overshoot. Convert explicitly.

### The melt looks cold and the structure came out wrong.

Check the temperature trace before blaming anything else. Velocities initialise
at the target and equipartition immediately halves it, so the thermostat supplies
the rest. **At γ = 0.1 ps⁻¹ that takes ~10 ps**, so a 20 ps melt spends most of
its life cold.

A silica run that produced q4 = 0.297 against a published 0.142 had only reached
3730 K of its 4000 K setpoint. Nothing was wrong with the potential. Use
γ = 1.0 ps⁻¹, which reaches setpoint in 2–3 ps.

> If T is not at setpoint within the first ~10% of the melt, the run is invalid.

### Can I just rattle a crystal instead?

No, and this is quantitative rather than a matter of taste. On c-SiO₂ at glass
density, a 0.3 Å rattle leaves only 44% of Si four-fold with atoms 0.46 Å apart,
while the second-shell count stays at the crystalline 4.0. On the disorder
metric it moves std from 2.303 to 1.709, about a tenth of the way to the glass
value of 0.134, while wrecking coordination on the way.

**Displacement degrades a crystal without disordering it.**

### Hot enough to melt ruins the chemistry, cool enough to keep it never melts. Now what?

This is the central difficulty for both GeO₂ and Li–P–S, and for GeO₂ it was
mapped completely: melting completes only between 2700 and 2800 K, and 2800 K is
exactly where oxygen starts coming off. **The window is not narrow, it is closed.**

The proposed escape is a two-stage melt — superheat briefly to destroy the
crystal, then hold below the degradation threshold so defects heal, then quench.
The reasoning is that chemical damage accrues with *time* at high T while the
crystal breaks up *quickly*.

**It has not yet produced an accepted glass**, and this answer is the honest
version rather than the hopeful one. Three tests so far:

| test | outcome |
|---|---|
| GeO₂ 10 ps superheat @ 2900 K → 2400 K hold (job 1252) | rejected, still under-melted at 4.04× floor, 1 O₂ |
| GeO₂ 20 ps superheat (job 1254) | melted (1.51×) but 4 O₂ |
| Li₃PS₄ superheat scan, 1361–1736 K | made disorder monotonically **worse** |

The 20 ps GeO₂ run is the informative one: it shows the superheat can finish the
melt, and that the O₂ came with it. Treat the method as open, not as a recipe.

### Anything about how to structure the job itself?

Three rules bought with real losses:

- **Write the structure before judging it.** A 1.5-hour LiPS-25 run completed its
  physics and then died on `ModuleNotFoundError` reaching for the validator,
  losing everything. The dynamics is the expensive, irreproducible half; a bug in
  the cheap, repeatable half must never destroy it.
- **Reject loudly.** Rejected structures get renamed `*_REJECTED.cif` and the job
  exits non-zero. Every invalid structure that got reused downstream had looked
  like a clean `COMPLETED` in the queue history.
- **Filenames are not evidence.** `geo2_glass.cif` measured byte-identically to
  `c-GeO2.cif`. Audit the directory before trusting anything as a reference.

---

## Chemistry and mixed anions

### How does the chemistry gate decide what is acceptable?

By enumerating the units the chemistry legitimately forms and gating on the
**residual** — what does not classify. Two intensive fractions carry the verdict:
orphan ligands (limit 1%) and unclassified central atoms (limit 4%).

Absolute rules survive only for species that are illegitimate at *any*
concentration: O₂ in an oxide, S–S polysulfide in a thiophosphate.

### Why not just list the forbidden contacts?

Two reasons, both measured.

**A forbidden contact can be a legitimate unit.** The old absolute P–P rule
rejected both published PCCP references, at 121 and 91 pairs, because it could
not tell a genuine P₂S₆ unit from P(V) reduction. Same bond, same length,
opposite meaning. Topology is what separates them: P₂S₆ has three sulfur plus one
phosphorus, reduction leaves undercoordinated P alongside free sulfur.

**Counts are extensive.** The same model shows 2 P–P at 62 P and 121 at 1111 P.
Any absolute threshold silently becomes stricter as the cell grows.

A recognized-unit list also fails safe in a way a forbidden list cannot: anything
unforeseen lands in the residual instead of passing unexamined.

### Does the chemistry gate work for mixed anions?

By construction, yes: units are matched on **topology alone** — ligand count,
bridging-ligand count, homonuclear-neighbour count — and the match ignores which
species the ligands are. PS₃F²⁻ has the topology of PS₄ and needs no new rule,
provided you declare F as a ligand.

**This is tested synthetically, not on a real mixed-anion structure.** The unit
test constructs a PS₃F case and confirms it classifies as PS₄ with
`ligands=("S","F")` and falls to the residual with `ligands=("S",)`. Running a
genuine mixed-anion melt-quench through the gate is the next validation step and
has not been done.

### Both fractions passed. Am I clear?

Not necessarily, and there is a live example. `lips70_glass_mpa1200` clears the
speciation gate entirely — 0.28% orphan, 2.08% unclassified, zero P–P — and is
caught only by the absolute S–S rule, on a single polysulfide bond.

The complementarity is the point. A single O₂ among 750 oxygen is 0.27% orphan
oxygen, under any sane ceiling, and still a broken oxide. Fractions catch
pervasive damage; absolute rules catch the one defect that invalidates the
material regardless of how rare it is.

### Where did the thresholds come from?

Observed gaps, not fitting. Orphan-ligand fraction: positives at 0.00%, negatives
at 2.67–53%, limit set at 1%. Unclassified: positives at 0.00–2.08%, negatives at
5.07–100%, limit at 4%.

The unclassified gap is narrow and the limit should be read with that in mind. At
96 P, one further broken atom moves the fraction by 1.04%, so the nearest
rejection (`lips75_glass_sh1800x1` at 1.04% / 4.17%) sits within one atom of the
line. That is a resolution limit of small cells, not a claim of precision.

---

## Potentials

### The structure came out wrong. Is the potential to blame?

Assume not, until you have exonerated the settings. Most apparent potential
failures here were ours. Silica's "bad potential" was a cold thermostat; GeO₂'s
"bad potential" was a melt temperature copied from silica.

Confirm all of: melt T ≈ 2 × Tm, γ = 1.0 with the trace actually reaching
setpoint, dt genuinely in fs, melt ≥ 30 ps, seed free of bad contacts, and the
**glass** density rather than the crystal's.

### When is it legitimately the potential?

When a specific chemical failure survives correct settings. For Li–P–S this
condition is now met, and it took a year: LiPS-25 genuinely melts the crystal and
gives 78% PS₄ against a DFT ~67%, while every universal potential either left it
crystalline or wrecked it. What remains is a polysulfide residue that is not a
settings artefact.

Universal potentials degrade this chemistry as a class, not just one model. At
1500 K, ⟨CN_P⟩ for Li₃PS₄ is 3.722 under MACE-MPA-0 and 3.944 under MatterSim-5M.
MatterSim is better across all three compositions and still short of 4.0.

### Should I fine-tune?

The evidence points in different directions depending on what you mean.

- **Fine-tuning a foundation model: evidence against.** The LiPS-25 authors found
  fine-tuned foundation models *worse than zero-shot in the liquid regime*,
  catastrophic forgetting exactly where melt-quench operates.
- **From-scratch domain training: evidence for.** LiPS-25 is trained from scratch
  on the Li₂S–P₂S₅ tie-line including ~1750 amorphous melt-quench cells, and it
  solved a melting problem three universal potentials could not. Domain coverage,
  not model size, decided it.
- **Domain training is not sufficient either.** LiPS-25 still produces
  polysulfide, reproducibly, across two compositions.

If you invest, the highest-value form is targeted active learning on the specific
failure: harvest the frames where S–S bonds form, label with DFT, train a delta
on the domain model, re-run the identical melt-quench and check the S–S count.
That is a project, not an afternoon, and the cost is the DFT labels.

### My polysulfide problem — potential or settings?

Check temperature first. For LiPS-25 the S–S count goes **12 → 2 → 0 as T falls
1500 → 1300 → 1200 K**, with disorder essentially unchanged. That is a settings
answer wearing a chemistry costume, and it is why fine-tuning stayed deferred.

### Can I fine-tune until the gate passes?

No. Two of our thresholds were wrong (the std ceiling and the CN target) and the
right fix in both cases was to correct the criterion on evidence, not to bend the
model toward it. Tuning a model against a gate you have not calibrated optimises
the gate, not the physics.

---

## Refinement against scattering data

### My χ² is excellent. Is the structure good?

Not by itself. **Fitting a 1-D scattering curve against 3N coordinates is
underdetermined.** An audit of 35 archived runs found 25 of them, 71%, with atoms
overlapping — invisible in the loss curve throughout.

Always validate on quantities you did not optimise. That is the same lesson the
melt-quench section states, arriving from the other direction.

### My run completed and the results are nonsense.

Check the atom ordering. Constraints are keyed by atom **index**, and that index
is not a property of the CIF alone: pymatgen expands symmetry operations and
groups sites by label, ASE reads the literal `atom_site` order. For `c-SiO2.cif`
the two disagree on 510 of 1125 sites.

Generators now write a fingerprint into `metadata.atom_order` and the trainer
refuses to start on a mismatch, so this fails loudly. Older constraints files
carry no fingerprint and print that the check was skipped — regenerate them.

### Is a matching element count enough to confirm the pairing?

No. An element check catches Si constraints landing on oxygen but is blind to a
permutation **among the Si themselves**: every index still points at a Si, and
every constraint still lands on the wrong one. Permuting all 375 Si in a
generator's own output leaves the element check reporting zero problems, while
the fingerprint flags 64 of 64 sampled sites.

### A structure in the repo looks usable. Can I seed from it?

Check `data/crystal-structures/retired/` first. Everything there contains atoms
closer than any chemical bond, down to 0.066 Å against a real Li–S bond of 2.4 Å,
and refinement cannot repair a broken seed — it inherits the overlaps. Configs
pointing at retired seeds carry a header saying so.

---

## What we still do not know

Kept explicit so nobody mistakes a gap for a settled answer.

- **The gate has never seen a real mixed-anion structure.** Topology matching is
  argued and unit-tested; it is not validated on the systems it exists for.
- **The two-stage melt has not produced an accepted glass** in three attempts.
- **WWW bond switching is unvalidated.** The acceptance bug that made it accept
  every move while rewiring nothing is fixed, but no run has yet shown sustained
  rewiring at a plausible acceptance rate.
- **Li₄P₂S₇ is untested and has no seed in the repo.** It is the strongest
  remaining prediction the route makes for Li–P–S: built entirely from bridging
  P₂S₇⁴⁻, the motif that distinguishes Li₇P₃S₁₁ (glasses) from Li₃PS₄ (does not).
- **Li₃PS₄ is a calibrated negative.** Ten runs, best 2.49× against a 1.36–1.60×
  target, cause understood. Note it is made experimentally by ball-milling rather
  than melt-quenching, so this may be the simulation reproducing real physics.

---

## See also

- `docs/amorphous_electrolyte_protocol.md` — the procedure
- `torchdisorder/common/glass_quality.py` — the gates, with calibration data
- `scripts/assess_glass.py` — audit structures already on disk
- `scripts/calibrate_disorder_gate.py` — carve a reference to your cell size
