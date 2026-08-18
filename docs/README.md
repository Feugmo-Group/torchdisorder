# Documentation

## `workflow.tex` / `workflow.pdf`

A single-page diagram of what the refinement actually optimises, with the
governing equations and the config key controlling each stage.

```bash
pdflatex -output-directory=docs docs/workflow.tex
```

It exists because the objective alone does not determine a structure. Fitting a
one-dimensional scattering curve against `3N` coordinates is underdetermined, so
`chi^2` on its own admits configurations with overlapping atoms — an audit of 35
archived runs found 25 that ended that way, with nothing in the loss curve to
show for it. The diagram makes the constraint terms and the validation gates as
visible as the objective, so a reader can trace any number in a run back to the
switch that produced it.

The three `penalty.*` balancing strategies are laid out side by side with their
formulae; see `configs/config.yaml` for the defaults and precedence.

## `amorphous_electrolyte_protocol.md`

The working recipe for producing a defensible amorphous structure from a
crystalline seed: route selection, melt temperature from the superheating limit
rather than Tm, thermostat traps, the two-stage melt, and the validation gates
that separate a glass from a crystal that never melted.

It exists because the same three mistakes kept recurring across SiO2, GeO2 and
Li-P-S — trusting coordination number as a glass test, setting melt temperature
from Tm, and judging a glass by a crystal's acceptance criteria. Every number in
it was measured with the tooling here; claims that are not yet verified are
marked as such.

It also records where a potential is genuinely the suspect, and the evidence for
and against fine-tuning one.
