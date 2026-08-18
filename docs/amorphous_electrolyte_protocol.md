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

> ⚠️ **Unverified.** As of 2026-08-18 this is implemented and argument-tested but
> the first real run (job 1252) is still in flight. Do not cite it as a working
> method until that reports.

---

## 5. Validation: two independent gates, both required

Implemented in `torchdisorder/common/glass_quality.py`; run via `--system` on the
builder, or `scripts/assess_glass.py` for structures already on disk.

### Gate A — chemistry (species that should not exist)

**Count them explicitly; never average.** Seven free O₂ in a 3000-atom cell move
⟨CN⟩ by less than 0.01.

| System | Forbidden | Meaning |
|---|---|---|
| SiO₂ / GeO₂ | O–O < 1.35 Å | molecular O₂ (1.21 Å) — thermal reduction |
| Li–P–S | P–P < 2.8 Å | P(V) → P(IV) reduction (but real P₂S₆ has one — allow a budget) |
| Li–P–S | **S–S < 2.4 Å** | **polysulfide** — sulfide oxidised. Clean PS₄ has S···S edges at 3.3 Å |
| Li–P–S | S with no P within 2.8 Å | sulfur left the network |

The S–S rule was added late and mattered: LiPS-25 output showed 12 S–S bonds at
2.03–2.07 Å accounting for 16 of its 18 "P-free sulfurs". The older rule was
detecting the *consequence* while naming the wrong cause. **Name the mechanism.**

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

## Current status (2026-08-18)

| System | Status |
|---|---|
| **SiO₂** | ✅ Done. `SiO2_mq_hot_mpa.cif` (4.54 / 0.143 vs published GAP 4.61 / 0.134) |
| **GeO₂** | ⏳ Single-temperature route exhausted; two-stage melt in flight (job 1252) |
| **Li–P–S** | ⏳ Melting solved by LiPS-25; blocked on polysulfide across both compositions |

## See also

- `torchdisorder/common/glass_quality.py` — the gates, with the calibration data
- `torchdisorder/common/validation.py` — overlap and first-shell checks
- `scripts/assess_glass.py` — audit structures already on disk
- `scripts/build_glass_melt_quench.py` — the builder
- `scripts/compare_to_literature.py` — score against a published model
