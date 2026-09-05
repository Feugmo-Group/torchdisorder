# Building amorphous solid-electrolyte models — a working protocol

A recipe for producing a *defensible* amorphous structure of a solid electrolyte
(or any network glass) from a crystalline starting point, written from what went
wrong on SiO₂, GeO₂ and Li–P–S between June and August 2026.

Every number here was measured with the tooling in this repo. Claims that are
**not** yet verified are marked as such — the protocol is only useful if it
distinguishes what we know from what we hope.

---

## 0. The one lesson that generalises

**A structure that reproduces your target is not thereby validated.**

This bites twice, in two different ways, and both cost us months:

- **In refinement.** χ² against a 1-D scattering curve is underdetermined against
  3N coordinates. An audit of 35 archived runs found 25 (71 %) with atoms
  overlapping — invisible in the loss curve.
- **In melt-quench.** Coordination number cannot distinguish a glass from a
  crystal that never melted. On 2026-08-17 that single blind spot reported three
  unmelted GeO₂ crystals as successes in one day.

The second is the more insidious, because the failure *looks better than success*:
if your crystal already contains the target motif — isolated PS₄ in Li₃PS₄,
corner-sharing GeO₄ in GeO₂ — then a structure that never melted scores a
**perfect** ⟨CN⟩ = 4.00 with zero defects. A real glass has defects. DFT BOMD of
a-Li₃PS₄ gives PS₄ : P₂S₆ : P₂S₇ ≈ 6 : 2 : 1, so "~100 % PS₄" means under-melted.

> **Perfection is the warning sign, not the goal.**

Always validate on quantities you did **not** optimise or target.

---

## 1. Choose the route

| Route | When | Notes |
|---|---|---|
| **Published model** | One exists for your material | Always beats a general potential on its own system. Check first. |
| **Melt-quench** | No published model | The physics does the work; ring statistics come out right without consulting scattering data. **Default choice.** |
| **WWW bond switching** | No potential available | Uses the objective as its acceptance criterion, so a compromised objective accepts bad topology. |
| ~~Rattling a crystal~~ | **Never** | See below. |

**Rattling does not work, quantitatively.** On c-SiO₂ at glass density, a 0.3 Å
rattle leaves only 44 % of Si four-fold with atoms 0.46 Å apart, while the
second-shell count stays at the crystalline 4.0. Measured on the disorder metric:
rattling moves std from 2.303 (crystal) only to 1.709, about a tenth of the way
to the glass value of 0.134 — while wrecking coordination on the way.
**Displacement degrades a crystal without disordering it.**

---

## 2. Set the melt temperature from the superheating limit, not Tm

This is the single most common parameter error, and it is subtle in both
directions.

A perfect crystal in periodic boundaries has **no nucleation sites**, so it
superheats to roughly **1.7–2.0 × Tm** before mechanical instability. On a 30 ps
MD timescale, the superheating limit — not the thermodynamic melting point — is
what governs whether anything melts.

| System | Tm (K) | Working melt T | Ratio |
|---|---|---|---|
| SiO₂ | 1986 | 4000 | 2.0× |
| GeO₂ | 1388 | 2800 | 2.0× |

**The ratio is the transferable number.** Setting melt T near Tm produced three
"successes" that were solid crystal (GeO₂ at 2000/2200/2400 K).

But too hot decomposes the material. GeO₂ inherited silica's 4000 K by accident,
which is 2.9× its Tm — deep into GeO₂ → GeO + ½O₂. That produced 11 O₂ molecules.
The potential was correctly describing real decomposition of a material held far
too hot.

**Start at 2.0 × Tm.** Then verify melting actually happened (§5).

---

## 3. Thermostat and integrator traps

Both of these silently invalidate runs while producing plausible-looking output.

- **`torch_sim`'s time unit is not femtoseconds.** Its internal unit is
  √(amu·Å²/eV) ≈ 10.18 fs, and `MetalUnits.time` is one picosecond expressed in
  it. Passing `dt=1.0` means one *picosecond* — a 1000× overshoot that sends the
  temperature to ~10¹⁶ K within fifty steps. Convert explicitly.
- **γ = 0.1 ps⁻¹ is too weak.** Velocities initialise at the target temperature
  and equipartition immediately halves it, so the thermostat must supply the
  rest. At γ = 0.1 that takes ~10 ps, so a 20 ps melt spends most of its life
  cold. A silica run that produced q4 = 0.297 against a published 0.142 had only
  reached 3730 K of its 4000 K setpoint. Nothing was wrong with the potential.
  **Use γ = 1.0 ps⁻¹** (reaches setpoint in 2–3 ps).

> **Read the temperature trace before drawing any conclusion.** If T is not at
> setpoint within the first ~10 % of the melt, the run is invalid.

---

## 4. When no single temperature works: the two-stage melt

Both GeO₂ and Li–P–S hit the same wall:

> Hot enough to melt → the potential degrades the chemistry.
> Cool enough to preserve chemistry → it never melts.

For GeO₂ this was mapped out completely (`dr = 0.05`, Ge–Ge sublattice):

| T (K) | max g | std(r > 6 Å) | reading |
|---|---|---|---|
| crystal | 26.30 | 2.152 | — |
| 2000–2400 | ~12 | ~1.4 | barely touched |
| 2500 | 13.78 | 0.888 | partially melted |
| 2600 | 13.19 | 0.844 | partially melted |
| 2700 | 9.83 | 0.576 | closer, still not glass |
| **2800** | **4.80** | **0.125** | **glass — but 7 O₂** |
| published NNP | 5.12 | 0.130 | glass |

Melting is a smooth progression that completes only between 2700 and 2800 K —
and 2800 K is exactly where oxygen starts coming off. **The window is not narrow,
it is closed.**

A corroborating diagnostic: the O–Ge–O angular spread was 4.31° / 4.60° / 5.46°
at 2500 / 2600 / 2700 K against the published glass's **8.18°**, narrowing
monotonically as T fell. Angular spread is a useful second read when ⟨CN⟩ is
uninformative (it sat at 4.008 for all three).

**The escape:** chemical damage accrues with *time* at high T, while the crystal
breaks up *quickly*. So superheat briefly to destroy the crystal, then hold below
the degradation threshold to let defects heal, then quench. Once molten the
system stays liquid at the lower hold temperature — 2400 K is still well above
GeO₂'s 1388 K Tm; it failed only to *initiate* melting, for want of nucleation
sites.

```bash
python scripts/build_glass_melt_quench.py ... \
    --superheat-temp 2900 --superheat-steps 10000 \
    --melt-temp 2400 --melt-steps 30000
```

> ⚠️ **Tested, and it has not yet produced an accepted glass.** Three runs:
>
> | test | outcome |
> |---|---|
> | GeO₂ 10 ps superheat @ 2900 K → 2400 K hold (job 1252) | rejected — still under-melted, 4.04× floor, 1 O₂ |
> | GeO₂ 20 ps superheat (job 1254) | melted (1.51×) but 4 O₂ |
> | Li₃PS₄ superheat scan, 1361–1736 K | disorder monotonically **worse** |
>
> The 20 ps GeO₂ run is the informative one: the superheat *can* finish the melt,
> and the O₂ came with it, so the time-separation argument is not yet doing the
> work it was proposed to do. Treat this as an open method, not a recipe.

---

## 5. Validation: two independent gates, both required

Implemented in `torchdisorder/common/glass_quality.py`; run via `--system` on the
builder, or `scripts/assess_glass.py` for structures already on disk.

### Gate A — chemistry (speciation)

Gate A asks whether the network is chemically intact. It runs in two halves,
and a structure must clear both.

**A1 — recognized units, judged by intensive fractions.** Enumerate the units a
chemistry legitimately forms, classify every central atom by local topology, and
gate on what is left over:

| Measure | Limit | Meaning |
|---|---|---|
| orphan-ligand fraction | 1 % | ligand atoms no central atom claims — the network shed them |
| unclassified-central fraction | 4 % | central atoms in no recognized unit |

Classification uses ligand count, how many of those ligands bridge to a second
central atom, and how many homonuclear neighbours — never the ligand *species*.
Recognized units are `PS4`, `P2S7`, `PS3-chain`, `P2S6` for Li–P–S, and `MO4`,
`MO5` for the oxides.

The *distribution* over those units is reported but never gated: published
models of a-Li₃PS₄ range from 58 % to 90 % PS₄ by phosphorus, so no threshold on
it is defensible. Only the residual is judged. How a network divides itself
among valid anions is a property of the material and the quench rate; a central
atom belonging to nothing is a defect either way.

**A2 — absolute rules, for species illegitimate at any concentration.**

| System | Forbidden | Meaning |
|---|---|---|
| SiO₂ / GeO₂ | O–O < 1.35 Å | molecular O₂ (1.21 Å) — thermal reduction |
| Li–P–S | S–S < 2.4 Å | polysulfide — sulfide oxidised. Clean PS₄ has S···S edges at 3.3 Å |

Both halves are needed. A single O₂ among 750 O is 0.27 % orphan oxygen, under
any sane fraction ceiling, and is still a broken oxide — A1 cannot catch it. A1
catches network damage that no enumerated contact anticipates, with numbers that
do not move when the cell size does.

**Why this replaced a forbidden-contact list.** Two rules were retired:

- `P-P pairs` **rejected the published literature.** The PCCP a-Li₃PS₄ model
  failed on 121 P–P pairs and a-Li₇P₃S₁₁ on 91, when those bonds are the genuine
  P₂S₆ units the material is known to contain — every P₂S₆ has one by
  construction. The rule counted a real unit and a reduced phosphorus
  identically because they are the same bond at the same length; the difference
  is the rest of the coordination shell, which is topology. Both references now
  pass, and `tests/test_glass_quality.py` pins that so it cannot regress.
- `P-free sulfur` was the right failure mode expressed as the wrong kind of
  number. Counts are *extensive* — the same model shows 2 P–P at 62 P and 121 at
  1111 P — so any absolute threshold is a statement about cell size. It is now
  the orphan-ligand fraction.

The `--tolerate "P-P pairs=N"` allowance that Li–P–S callers passed is therefore
gone, and needs no replacement: a real P₂S₆ now costs nothing and a reduced P
still fails, so there is no budget to guess at. An allowance naming a rule that
no longer exists is reported as a warning rather than silently ignored.

**Why topology, for mixed anions.** In Li₃PS₄₋ₓOₓ the mixed PS₃O / PS₂O₂ / PSO₃
tetrahedra *are* the material; an O and an S on the same P sit ~2.7 Å apart as a
legitimate edge, and no distance cutoff separates that from phase separation.
Topology does — PS₃F²⁻ has the topology of PS₄ and matches the same entry once F
is named a ligand, with no new rule. Enumerating what is *recognized* also fails
safe where enumerating what is *forbidden* does not: an unforeseen species lands
in `other` and counts against the structure rather than passing unnoticed.

**Calibration.** Measured across every structure on hand:

| | orphan ligand | unclassified |
|---|---|---|
| both published PCCP references | 0.00 % | 0.00–0.08 % |
| our four accepted glasses | 0.00 % | 0.09–2.08 % |
| GeO₂_mq (20 free O) | 2.67 % | 5.07 % |
| retired invalid LiPS glasses (18 files) | 40–53 % | 83–100 % |

Both limits sit in the observed gaps. The narrower gap is the unclassified one,
and the margin above lips70's 2.08 % is about two of its 96 P — a Li–P–S cell
that dilute is granular here, since one further broken P moves the fraction by
1.04 %. That is a limit on resolution, not a bias; prefer a larger cell when the
verdict is close.

**The reduced MACE runs.** All 24 Li–P–S melt-quench outputs were pulled off
remote scratch and re-measured. The three MACE-MPA runs that motivated this work
land exactly where the old gate put them:

| run | orphan S | unclassified | S–S | verdict |
|---|---|---|---|---|
| lips70_glass_mpa | 11.08 % | 29.17 % | 26 | chemistry FAIL |
| lips67_glass_mpa | 9.52 % | 31.25 % | 28 | chemistry FAIL |
| lips75_glass_mpa | 9.03 % | 13.89 % | 19 | chemistry FAIL |
| lips70_glass_lips25_1200 | 0.00 % | 2.08 % | 0 | **GLASS** |

**No structure moved from rejected to accepted.** Eight did have their chemistry
verdict relax while still failing on disorder, and every one was rejected by the
old rule over 1–4 orphaned sulfurs in a cell of ~384 — 0.26–0.93 %, which is the
extensive-count problem the fractions exist to fix.

One run is worth reading as a live case for keeping both halves of Gate A:
`lips70_glass_mpa1200` has 0 P–P pairs and a single orphaned S (0.28 %, under the
1 % limit) with 2.08 % unclassified — it clears A1 outright and is caught only by
the absolute S–S rule, on one polysulfide bond. The nearest call in the other
direction is `lips75_glass_sh1800x1`, which fails A1 on both counts at 1.04 % and
4.17 %, just past each limit.

### Gate B — disorder (did it actually melt?)

Central-atom sublattice g(r), judged **beyond 6 Å**. Long-range order is what
melting destroys; first-shell peaks survive in both phases and discriminate
nothing.

**Raw std is not comparable between chemistries.** An oxide sublattice has
375–1728 central atoms; a 672-atom Li₇P₃S₁₁ cell has only 96 P, and a g(r) built
from 96 centres is far noisier. Measure the counting-noise floor by randomising
positions in the same cell:

| structure | N_a | std | noise | ratio | truth |
|---|---|---|---|---|---|
| sio2_glass_gap | 1728 | 0.134 | 0.039 | 3.47 | glass |
| SiO2_mq_hot_mpa | 375 | 0.143 | 0.079 | 1.80 | glass |
| geo2_glass_nnp | 1080 | 0.130 | 0.049 | 2.66 | glass |
| lips70 (LiPS-25) | 96 | 0.504 | 0.309 | **1.63** | disordered |
| Li3PS4_gamma | 108 | 3.930 | 0.285 | 13.79 | crystal |
| c-SiO2 | 375 | 2.303 | 0.072 | 31.97 | crystal |

lips70's std of 0.504 looked like a marginal crystal against a flat 0.5 ceiling.
It is almost entirely shot noise — its excess over the floor is *lower than every
published glass reference*.

**Criterion:** `std < max(0.5, 2 × noise_floor)`. Both terms are load-bearing —
a pure ratio test fails large glasses, since noise falls as 1/N_a while a glass
holds std ≈ 0.13, so the same silica scores 1.80 at 375 Si and 3.47 at 1728 Si.
Crystals are rescued by neither, sitting 14–32× over their own floor.

**The ratio is not comparable between cell sizes either — calibrate before you
judge.** Noise falls as 1/√N_a while a glass holds roughly constant structural
signal, so the *same* material scores higher in a bigger cell. The published
a-Li₃PS₄ reads **4.15× at 1111 P and 1.36× at ~100 P**. Comparing a candidate
against a published reference measured in its own much larger cell compares
nothing, and it is easy to do by accident — an early claim here that our GeO₂
"beat the published NNP model, 1.86× vs 2.38×" was exactly this mistake, and it
reverses once both are measured at the same size.

So for any new chemistry, carve a published model down to the cell size you
actually generate and measure *that*:

```bash
poetry run python scripts/calibrate_disorder_gate.py \
    --reference data/crystal-structures/sio2_glass_gap.cif \
    --like      data/crystal-structures/SiO2_mq_hot_mpa.cif \
    --species Si
```

(The two `ref_a*_pccp.cif` Li-P-S references used for the Li rows below are
**not** in the repo — their licence is unresolved, so they must be re-derived
from the published supplementary data. See the note in `.gitignore`. The oxide
references *do* ship, so the command above runs on a fresh clone.)

It reports a bracket, because carving has a seam that scrambles genuine
long-range correlation and therefore *understates* a real glass (a-Li₃PS₄:
signal 0.344 full-cell, 0.256 carved). Lower bound = carved; upper bound =
full-cell signal recombined with the carved noise floor.

| chemistry | reference model | real-glass band **at our cell size** | ours | standing |
|---|---|---|---|---|
| SiO₂ | GAP (Erhard 2022) | 1.54–1.93× | 1.80× | inside |
| GeO₂ | NNP (Kasamatsu 2024) | 1.37–1.73× | 1.86× | just above |
| Li₇P₃S₁₁ | PCCP class2 | 1.29–1.45× | 1.57× | just above |
| Li₃PS₄ | PCCP class2 | 1.36–1.60× | 2.49× best of 10 | **fails** |

Read "just above" as *at the ordered edge of what a real glass looks like*, not
as a failure — the sub-box spread is ±0.10–0.18 and our side is a single
measurement. Note also that the Li-P-S references are fixed-bond-topology force
field models, which cannot recrystallise and so may be more disordered than a
real glass; the oxide references carry no such caveat.

**Fail closed.** A cell too small to have a beyond-6 Å range must FAIL, not pass:
crystalline Li₃PS₄ is 32 atoms in a 6 Å cell, giving one histogram bin whose std
is exactly 0.000 — which sails under any ceiling.

### Gate C — coordination, using the *glass* target

`--expected-cn 4` is a crystal number. It rejected two good runs (lips75 at
3.848, lips70 at 3.771 — the first by 0.0025).

For a-Li₃PS₄, the 6:2:1 speciation gives PS₄ 6 P at CN 4, P₂S₆ 4 P at CN 3, P₂S₇
2 P at CN 4 → **⟨CN⟩ = 44/12 = 3.67**. A structure at 4.00 is the under-melted one.

Where no glass speciation is published, **skip the check** rather than run it
against a guess, and let Gates A and B carry the verdict.

---

## 6. Operational rules that saved (or cost) real work

- **Write the structure before judging it.** A 1.5-hour LiPS-25 run completed its
  physics, then died on `ModuleNotFoundError` reaching for the validator, losing
  everything. The dynamics is the expensive, irreproducible half; a bug in the
  cheap, repeatable half must never destroy it.
- **Fail fast on dependencies.** Import everything the run will need in the first
  second, not the eighty-ninth minute.
- **Keep validators dependency-light.** Each MLIP needs its own conda env
  (mace-torch pins e3nn==0.4.4; MatterSim/SevenNet need ≥0.6), and those envs
  carry only what the dynamics needs. Anything reachable from the melt-quench
  script must import only numpy, ASE and torch_sim.
- **Reject loudly.** Rejected structures get renamed (`*_REJECTED.cif`) and the
  job exits non-zero. Every invalid structure that got reused downstream looked
  like a clean `COMPLETED` in the queue history.
- **Monitor chemistry during the run.** The melt-vs-decomposition dilemma is a
  race, and you cannot see which is winning from one measurement on the final
  structure. Log defect counts every N steps.
- **Filenames are not evidence.** `geo2_glass.cif` measures byte-identically to
  `c-GeO2.cif`. Audit the directory (`scripts/assess_glass.py`) before trusting
  anything as a reference.

---

## 7. The potential: when to suspect it, and whether to fine-tune

### Default rule: exonerate the settings first

Most apparent "potential failures" were ours. Before touching the model, confirm:
melt T is ~2 × Tm; γ = 1.0 and the T trace reaches setpoint; dt is really
femtoseconds; the melt is long enough (≥ 30 ps); the seed has no bad contacts;
density is the *glass* density. Silica's "bad potential" was a cold thermostat.
GeO₂'s "bad potential" was a temperature copied from silica.

### When the settings are exonerated

For Li–P–S this condition is **now satisfied**, and it took a year to reach:
LiPS-25 genuinely melts the crystal (disorder gate passes, 78 % PS₄ close to the
DFT ~67 %) while every universal potential either left it crystalline or wrecked
it. What remains is a chemistry residue — polysulfide — that is not a settings
artefact. **That is the legitimate trigger for looking at the model.**

### Evidence on fine-tuning specifically

Mixed, and the direction matters:

- **Fine-tuning a foundation model: evidence against.** The LiPS-25 authors found
  fine-tuned foundation models were **worse than zero-shot in the liquid regime**
  — catastrophic forgetting exactly where melt-quench operates. Do not start here.
  This is why `models/fine-tuned/` in that repo is *not* what we use.
- **From-scratch domain training: evidence for.** LiPS-25 is trained from scratch
  on the Li₂S–P₂S₅ tie-line, including ~1750 melt-quench cells generated to be
  amorphous. It solved a melting problem that three universal potentials could
  not. Domain coverage, not model size, was the deciding factor.
- **But domain training is not sufficient.** LiPS-25 still produces polysulfide,
  reproducibly, across two compositions. Its training set evidently does not
  constrain S–S bonding in the high-T liquid well enough.

### Recommendation

If we invest in a potential, the highest-value form is **targeted active learning
on the specific failure**, not general fine-tuning:

1. Harvest configurations from our own failed runs — the frames where S–S bonds
   form (we now log exactly when, via the live monitor).
2. Label those with DFT (single-point, plus short AIMD around the transition).
3. Train a **delta on the domain model** (LiPS-25 style, from scratch on the
   augmented set) rather than fine-tuning a universal foundation model.
4. Re-run the identical melt-quench and check the S–S count specifically.

This is a genuine project, not an afternoon: the cost is the DFT labels. Do it
only if (a) the two-stage melt fails to suppress polysulfide, and (b) an
amorphous Li–P–S model is on the critical path for the science. If a published
model exists for the target composition, use it instead — that remains the
cheapest good answer.

**Do not fine-tune to make a validation gate pass.** Two of our thresholds were
wrong this session (the std ceiling and the CN target) and the right fix was to
correct the criterion on evidence, not to bend the model toward it.

---

## 8. Failure atlas

| Symptom | Likely cause | Check / fix |
|---|---|---|
| ⟨CN⟩ perfect, zero defects | never melted | disorder gate; raise T toward 2 × Tm |
| Molecular O₂ present | melt too hot for this oxide | lower T; two-stage melt |
| S–S bonds ~2.05 Å | sulfide oxidised to polysulfide | potential/domain coverage |
| P–P bonds + free S | P(V) → P(IV) reduction | universal potential; use domain model |
| T never reaches setpoint | γ too weak | γ = 1.0 ps⁻¹ |
| T explodes in ~50 steps | dt interpreted as ps | convert via `MetalUnits.time` |
| Atoms 0.2–0.9 Å apart | refinement overfitting χ² | `validate_structure` overlap gate |
| ⟨CN⟩ climbs with cutoff | no resolved first shell | plateau test; check `bond_cutoff` |
| std ≈ 0.5 on a dilute sublattice | counting noise, not order | noise-aware ceiling |

---

## 9. Checklist

**Before running**
- [ ] Seed has zero contacts below the covalent floor
- [ ] Cell large enough that half its shortest length exceeds 6 Å (else the
      disorder gate cannot judge it)
- [ ] Density set to the **glass** density
- [ ] Melt T ≈ 2.0 × Tm; γ = 1.0 ps⁻¹; dt genuinely fs
- [ ] `--system` passed, so the gates actually run

**After running**
- [ ] Temperature trace reached setpoint early in the melt
- [ ] Chemistry gate: forbidden species counted, not averaged
- [ ] Disorder gate: std vs its own noise floor
- [ ] Coordination judged against the glass target, not the crystal's
- [ ] Compared against a published model where one exists
- [ ] Report **both** gates and both numbers — never a bare PASS

---

## Current status (2026-08-24)

All disorder figures below are judged against that chemistry's own reference band
at matched cell size, per Gate B.

| System | Status |
|---|---|
| **SiO₂** | ✅ Done. `SiO2_mq_hot_mpa.cif` — 1.80×, inside the 1.54–1.93× band. The strongest result. |
| **GeO₂** | ✅ Done. `GeO2_glass_mq_o2repaired.cif` — 1.86× vs a 1.37–1.73× band. **Hand-repaired**, not raw simulation output: see `PROVENANCE_GeO2_glass_mq_o2repaired.md` and `scripts/repair_o2_defect.py`. Melt-quench reliably leaves 1–3 trapped O₂; the repair is licensed only because the molecule does not re-form under relaxation. |
| **Li₇P₃S₁₁** | ✅ Done. `lips70_glass_lips25_1200.cif` — 1.57× vs a 1.29–1.45× band. Unmodified melt-quench output, LiPS-25 at 1200 K. |
| **Li₃PS₄** | ❌ Not reachable by this route. 10 runs; best 2.49× against a 1.36–1.60× target. Cause understood: the 1200 K liquid never disorders (melt-stage std plateaus at 0.60–0.70 and stays), and hotter melts cost chemistry — S–S bonds appear in under 1 ps above ~1600 K. A superheat scan at 1361–1736 K made it *worse*, monotonically. |
| **Li₄P₂S₇** | Untouched; its seed is separately broken. |

**The Li₃PS₄ negative result is calibrated, not merely repeated.** The gate was
the prime suspect and was cleared: a genuine a-Li₃PS₄ at the same cell size
scores 1.36–1.60×, so the rejections are real. Note 75Li₂S·25P₂S₅ is made
experimentally by ball-milling rather than melt-quenching — a poor glass former
at that composition — so this may be the simulation reproducing real physics.

## See also

- `torchdisorder/common/glass_quality.py` — the gates, with the calibration data
- `torchdisorder/common/validation.py` — overlap and first-shell checks
- `scripts/assess_glass.py` — audit structures already on disk
- `scripts/build_glass_melt_quench.py` — the builder
- `scripts/compare_to_literature.py` — score against a published model
