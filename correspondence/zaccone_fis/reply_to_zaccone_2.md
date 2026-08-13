# Reply to Alessio Zaccone (#2) — sanity checks completed

**To:** Alessio Zaccone
**From:** Conrard Tetsassi Feugmo (giresse.feugmo@gmail.com)
**Subject:** Re: TorchDisorder — F_IS implementation and potential collaboration

**Status:** DRAFT — not sent.

**Suggested attachment:** `scripts/validate_fis_tetrahedron.py` (the 9-test suite, self-contained,
NumPy only — he can re-run every number below himself).

---

Dear Alessio,

Thank you for such a careful and generous reply — it was far more than a correction, and
it sent us back to the code productively. I have now run every check you suggested. The
short version is that you were right on all counts, one of your suggestions turned out to
fix a real bug rather than a cosmetic one, and one of the numbers in my previous email has
to be withdrawn. Details below.

**1. The negative crystalline value is not a bug — your derivation reproduces exactly.**

An ideal regular tetrahedron gives F_IS = −0.333333, deviating from −1/3 by 3.9 × 10⁻¹⁶ in
our implementation. It is invariant to both scale and rigid rotation, as it must be. Your
explanation of the mechanism — the numerator being a squared vector sum with cross terms
while the denominator is a sum of squares, so that F_IS is not bounded to [0,1] — is
precisely what the code does.

Isolating a single SiO₄ unit from the crystal and computing F_IS on it alone gives
−0.3309 ± 0.0015 (a single representative tetrahedron: −0.3331). So the −0.331 we reported
comes entirely from local tetrahedral geometry, exactly as you predicted; the +0.0025
offset from −1/3 is the genuine α-quartz distortion, nothing more.

I also verified that our environments are clean: 375 of 375 Si centres, Si-centred Si–O
only, ⟨CN⟩ = 4.000 exactly, Si–O = 1.6141 ± 0.0020 Å. The O-centred environments (CN = 2)
are not mixed in — that had been a real worry of mine after reading your note, since they
would have diluted the distribution.

**2. The polymorph is α-quartz, and your point about it stands.**

Confirmed: 5×5×5 supercell, primitive a = 4.9150 Å, c = 5.4313 Å (α-quartz reference
4.9134, 5.4052). Your observation that α-quartz is itself non-centrosymmetric is important
and I had glossed over it. We will not describe the crystal → glass change as a "loss of
inversion symmetry" — both phases are non-centrosymmetric. What changes is the coherent
tetrahedral addition in the affine force field, which is a different and more specific
statement.

Your progressive-distortion mechanism is also reproduced cleanly. Perturbing the ideal
tetrahedron with Gaussian noise of width σ gives a monotonic loss of coherence:

| σ (Å) | 0.00 | 0.10 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|
| F_IS | −0.3333 | −0.3173 | −0.2665 | −0.1889 | −0.0026 |

**3. Your combination proposal fixes a real 3× error, not just an aesthetic one.**

This is the part I most wanted to tell you about. You expected the two ways of combining
the three shear planes — averaging the per-plane ratios versus summing numerators and
denominators first — to agree by symmetry. For a generic orientation they do, and for the
ideal tetrahedron they agree exactly (both −1/3).

But consider a *rotation about a single axis*, which is the situation for any crystal with
a symmetry axis along a cell vector. Two of the three shear planes then stay degenerate:
n^μ n^ν vanishes identically for every bond, so their denominators are zero and they carry
no information. Our implementation zero-filled those planes and averaged, returning
(1 + 0 + 0)/3 = **+1/3 for a perfectly centrosymmetric octahedron**. Your sum-then-ratio
form correctly returns **+1**, because the empty planes contribute nothing to either sum
instead of contributing a spurious zero to the mean.

That is a 3× systematic error, and it lands precisely on the octahedral systems we are
targeting next (Fe₂O₃, and the Li₂HfCl₆₋ₓFₓ halide electrolytes), which we build from CIFs
in the crystallographic frame. It also matters more for us than it would in a post-hoc
analysis, because we use F_IS as a differentiable optimization objective — a wrong target
value does not just misreport the structure, it actively steers the refinement to the
wrong place. We have adopted your formulation.

(A related edge case, for completeness: an *exactly* axis-aligned octahedron degenerates in
all three planes and is silently reported as F_IS = 0 under either scheme. We now flag
that explicitly rather than returning a number.)

**4. A correction I owe you: the glass result has to be withdrawn.**

You were right to caution against my "F_IS is ~4× more sensitive than q4/q6" framing, and
it turns out the problem is more basic than the one you raised. On re-examining the
amorphous SiO₂ structure those numbers came from, it is not physical:

| | c-SiO₂ | a-SiO₂ (used) |
|---|---|---|
| ⟨CN⟩ | 4.000 | 2.98 (only 29% at CN = 4) |
| min Si–O | 1.612 Å | **0.342 Å** |
| Si–O pairs < 1.4 Å | 0 | **240** |

There is no first-shell peak at all — the Si–O distances are near-uniform from 1.2 to
3.0 Å and the coordination number rises monotonically with cutoff, with no plateau at 4.
Atoms are overlapping. The structure reproduces the experimental F(Q) while being
physically meaningless: the classic underdetermined-fit failure mode.

Tracing where it came from was instructive. It is not that our machine-learning potential
regularization was switched off — it was on. The fault is upstream: the *starting*
structure we seed the refinement from was already broken (⟨CN⟩ = 3.02, 298 contacts below
1.4 Å), and the refinement never repaired it. Checkpointing through the run shows the
coordination flat at ⟨CN⟩ ≈ 3.0 from the first saved step to the last. A weak force
penalty applied every fiftieth step cannot undo a 0.2 Å overlap, and our force clipping
(5 eV/Å) was in fact removing precisely the enormous restoring forces that would have
pushed those atoms apart. So the regularization was not just too weak — it was clipped
into irrelevance exactly where it was needed.

The lesson we are taking from this is that agreement with F(Q) is close to worthless as a
validation criterion on its own, and that structural health checks belong *in* the
refinement loop rather than after it.

So the reported ΔF_IS = 0.380 conflated three different things — genuine tetrahedral
distortion, a coordination change from 4 to 3, and unphysical short contacts.

**5. The corrected numbers — and they reverse the conclusion.**

Rather than ask you to wait, we redid the comparison properly. We replaced our glass with
a published, independently generated model (Erhard, Rohrer, Albe & Deringer, npj Comput.
Mater. **8**, 90 (2022); the GAP melt-quench configuration from their Zenodo archive),
which we verified before use: 5184 atoms, ρ = 2.194 g/cm³, ⟨CN⟩ = 4.000 over a cutoff
plateau, Si–O = 1.619 ± 0.033 Å, O–Si–O = 109.4 ± 5.4°, and Si–O–Si = 140.8 ± 14.4°
against the experimental ~144°. Only 5 coordination defects in 1728 Si.

In the course of that we also found a genuine bug in our own neighbour-list construction:
a cell-convention mismatch (row- versus column-vector) meant that on a hexagonal cell the
search ran on a transposed lattice, reporting ⟨CN⟩ = 4.32 for α-quartz where the true
value is exactly 4.000. It was silently corrupting *every* order parameter. The tell was
that our crystalline q4 and q6 had a non-zero spread, which for a perfect crystal is
impossible — with the bug fixed they are exactly zero, as they must be.

With a physical glass and a correct neighbour list:

| | F_IS | q4 | q6 | tet |
|---|---|---|---|---|
| c-SiO₂ | −0.3255 ± 0.0046 | +0.2498 ± 0 | +0.0457 ± 0 | +0.9971 ± 0 |
| a-SiO₂ | −0.3005 ± 0.0550 | +0.1416 ± 0.0950 | +0.1372 ± 0.1008 | +0.9180 ± 0.0581 |
| **Δ** | **+0.025** | **−0.108** | **+0.092** | −0.079 |

So ΔF_IS is **+0.025**, not +0.380 — a factor of fifteen smaller. And the sensitivity
claim inverts completely: F_IS shifts *less* than q4 by a factor of about four, where I
had told you it shifted more by a factor of about four. Your caution was not merely
warranted, it was understated, and I am sorry to have put the original figure in front of
you.

The physical reading now seems clear in hindsight: both α-quartz and a-SiO₂ are built from
near-ideal SiO₄ tetrahedra, and F_IS is dominated by that local tetrahedral geometry. It
therefore *barely moves* across the transition — exactly your point that the reference
value belongs to the coordination geometry, not to crystallinity. What actually changes on
melting is the inter-tetrahedral arrangement, which q4 sees and F_IS largely does not.

What survives, and what I would now put the weight on, is the result you singled out:
F_IS and q4 are essentially independent axes carrying complementary information. That
holds regardless of absolute scale — and the corrected numbers arguably strengthen it,
since the two are now clearly responding to different things rather than to the same
disorder with different gains.

I should also correct a smaller slip in my last email: the crystal contains **375 Si
atoms** (1125 atoms in total), not 1125 Si.

What we can offer in return is a differentiable implementation with your corrected
combination scheme, validated against the analytic limits, which can be used as an
optimization target rather than only as a descriptor. If that is useful to your group,
it is yours — and we will of course acknowledge your contribution fully.

Thank you again for engaging so substantively. It saved us from publishing a wrong number,
which is a debt I am glad to owe.

With best wishes,

Conrard Tetsassi Feugmo
Feugmo Research Group

---

## Notes for Conrard before sending

- Everything above is reproduced by `poetry run python scripts/validate_fis_tetrahedron.py`.
- **Verify his current affiliation** (Milan vs Göttingen) before any follow-up about the
  spring 2027 visit — deliberately left out of this email, since this one is a correction
  and it is better not to mix the two.
- The corrected numbers are now *in* the email (§5), so it no longer depends on a pending
  re-refinement — it can go as soon as you are happy with the tone.
- Decide how much of the neighbour-list bug you want to volunteer. I have left it in
  because it explains why the crystalline q4/q6 spreads in the first email were non-zero,
  which he may well have noticed; omitting it would leave that unexplained.
- Consider whether to offer the corrected `sum_then_ratio` implementation to his group
  explicitly, since §5 now ends on the independence result rather than on the code.
