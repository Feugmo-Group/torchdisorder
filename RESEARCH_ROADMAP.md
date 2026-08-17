# TorchDisorder — Research Roadmap

Post-publication directions for applying differentiable structure refinement + F_IS to real experimental systems and experimental collaborations.

---

## What we can offer experimentalists

> "Give us your PDF/S(Q) beamtime data → we return an atomistic structure, F_IS distribution, MACE energetics, and convergence certificate — in hours."

TorchDisorder's differentiable, constraint-driven approach has two advantages over classical RMC:
1. **Gradients** — the optimizer knows which atomic moves improve the fit, rather than sampling randomly.
2. **Joint objectives** — chi² scattering + F_IS + MACE energy are combined automatically via adaptive aggregators (`relobralo`, `ema`, `soft_adapt`, …).

---

## Battery / solid electrolyte directions

### 1. LiPON — fast win (data already in repo)

**Data on disk:** `data/xrd_measurements/li2.99_po3.38_n0.41/F_of_Q.csv`, `T_of_r.csv`  
**CIF on disk:** `data/crystal-structures/LiPON_defected.cif`

LiPON is the only amorphous solid electrolyte in commercial production today (Apple Watch, hearing aids, medical implants). The N-for-O substitution that boosts conductivity from ~10⁻⁷ to ~10⁻⁶ S/cm is structurally unresolved — no gradient-based refinement has been published.

```bash
python scripts/train.py data=lipon structure=lipon target=F_Q \
  fis.target=0.02 fis.weight=2.0 fis.cutoff=1.8 fis.central_z=15
```

**Deliverable:** First gradient-based atomistic model of LiPON; F_IS map of N-coordinated P sites vs. O-only sites.

---

### 2. Li₃PS₄ glass (data already in repo)

**Data on disk:** `data/xrd_measurements/Li3PS4/`, `data/xrd_measurements/Li3PS4_gamma/`  
**CIF on disk:** `data/crystal-structures/Li3PS4_beta.cif`, `Li3PS4_gamma.cif`  
**Constraints on disk:** `data/json/glass_67Li2S_constraints.json` (reusable PS₄ environment)

The β→γ phase transition is linked to conductivity; the glass is amorphous but retains PS₄ tetrahedra. We have both crystal references and glass XRD data — ideal for a crystal-vs-glass F_IS comparison identical to what we did for SiO₂.

**Key question:** Does F_IS of the PS₄ tetrahedron predict Li-ion hopping barriers (computable with MACE NEB)?

---

### 3. Li₆PS₅Cl argyrodite glass

No data in repo yet — must obtain.

Argyrodite is the highest-conductivity sulfide electrolyte in production (Samsung SDI 400 Wh/kg cells). The disordered phase has higher conductivity than the crystal for reasons not understood at the atomic level. Neutron PDF is ideal (S/Cl differ in scattering length). Extend existing LPS constraint generator for Cl site disorder.

**Collaboration target:** Groups with NOMAD/ISIS/ORNL beamtime on battery electrolytes.

---

### 4. In-situ / operando PDF tracking during lithiation

Beamtime at APS 11-ID-B or NSLS-II gives PDF every 30 s during cycling. Re-run TorchDisorder at each time step using the prior structure as a warm start (gradient checkpoint restarts).

**Output:** A time-resolved movie of structural evolution during charge/discharge — a genuine first for gradient-based methods.

---

## Non-battery directions

### 5. Amorphous silicon — photovoltaics / neuromorphic (data in repo)

**Data on disk:** `data/xrd_measurements/Si/annealed-S-of-Q.csv`, `annealed-J-of-R.csv`  
**CIF on disk:** `data/crystal-structures/c-Si.cif`

Hydrogenated amorphous silicon (a-Si:H) is the active layer in thin-film solar cells and is being explored for neuromorphic computing (memristors). The degree of local tetrahedral order (q4, F_IS) directly controls the bandgap and carrier mobility. We already have the scattering data; the constraint generator for tetrahedral Si is analogous to SiO₂.

**Unique angle:** F_IS can serve as a proxy for the dangling-bond density without needing EPR measurements. A paper correlating TorchDisorder-refined F_IS with measured carrier lifetime would be immediately useful to the photovoltaics community.

---

### 6. ε-Fe₂O₃ polymorphs — magnetism and catalysis (data in repo)

**Data on disk:** `data/xrd_measurements/Fe2O3/F_of_Q.csv`, `T_of_r.csv`  
**CIFs on disk:** `data/crystal-structures/epsilon_Fe2O3_iso_1.cif` through `iso_5.cif`, `my_fe2o3.cif`

The ε-Fe₂O₃ polymorph is a room-temperature multiferroic and has the highest known coercive field of any simple oxide (~20 kOe). Five isostructural variants with different local Fe environments are already in the repo as separate CIFs. TorchDisorder could:

1. Refine the amorphous Fe₂O₃ scattering data.
2. Compute F_IS per Fe site → link inversion symmetry breaking to ferroelectric distortion.
3. With MACE forces, estimate local stress tensor heterogeneity.

**Collaboration target:** Groups doing Mössbauer spectroscopy or neutron diffraction on Fe₂O₃ nanoparticles (magnetic storage, heterogeneous catalysis for CO₂ reduction).

---

### 7. NaTaCl₆ — halide perovskite and radiation detector (data in repo)

**Data on disk:** `data/xrd_measurements/NaTaCl6/F_of_Q.csv`, `g_r.csv`, `T_of_r.csv`  
**CIF on disk:** `data/crystal-structures/NaTaCl6.cif`

Halide double perovskites are candidates for non-toxic photovoltaics and γ-ray detectors. NaTaCl₆ is an elpasolite-type compound where local TaCl₆ octahedral distortions control the bandgap. Differentiable refinement of the octahedral tilt pattern (analogous to our tetrahedral order) is a clean extension of the existing framework.

**Key order parameter to add:** Octahedral tilt angle distribution (similar to q4 but for MX₆ coordination shells).

---

### 8. GeO₂ glass — optical fiber / photonics (published baseline)

**Data on disk:** `data/xrd_measurements/GeO2/F_of_Q.csv`, `T_of_r.csv`  
**Already validated** in the published paper as a benchmark system.

GeO₂ is the standard dopant glass in silica optical fibers (refractive index tuning). Under pressure, GeO₂ undergoes a coordination change from 4-fold to 6-fold Ge. TorchDisorder is uniquely suited to track this transition:

- Start from the published GeO₂ amorphous structure.
- Apply an isotropic strain to the simulation cell (differentiable).
- Run refinement at each strain step against simulated or experimental high-pressure PDF.

**Collaboration:** Groups with diamond anvil cell + synchrotron PDF (ESRF ID15, APS HPCAT).

---

## Cross-cutting capability extensions

### Multi-modal refinement: NMR + PDF

Almost every experimentalist doing glasses also collects ²⁹Si, ³¹P, ²⁷Al, or ⁷Li NMR. NMR gives CN distributions directly. Adding an NMR constraint term is one differentiable loss term:

```python
nmr_loss = F.mse_loss(predicted_cn_histogram, nmr_cn_histogram)
```

Wire into `CooperLoss` alongside chi² and F_IS; let `relobralo` balance the three. This makes TorchDisorder a genuine multi-modal tool and opens collaborations with any solid-state NMR group.

### MACE energy as a third loss term

Current limitation acknowledged in the paper: no interatomic potential enforcement. Adding:

```python
mace_energy_loss = (energy_per_atom - target_energy_per_atom) ** 2
```

as a third term in `CooperLoss` prevents unphysical short contacts and allows joint scattering + thermodynamic refinement. The adaptive aggregators already handle N=3 loss terms.

### Uncertainty quantification via ensemble refinement

Run TorchDisorder from K different random starting structures → K different converged models. The spread of F_IS / CN / bond-angle distributions across the ensemble quantifies structural uncertainty — directly useful to experimentalists who want error bars on derived properties.

---

## Surprising applications you may not have considered

### 9. Phase-change memory — Ge₂Sb₂Te₅ (GST)

**Industry size:** $10B+ (Intel Optane, Sony Blu-ray, embedded automotive NVM)

GST switches between amorphous ("0") and crystalline ("1") states in nanoseconds. The amorphous phase structure — specifically the local Ge coordination change from octahedral to tetrahedral — is the entire storage mechanism, yet it is structurally poorly understood because the transition happens too fast for conventional diffraction. Published neutron PDF data exists (Salmon group, Bath). TorchDisorder is the only differentiable refinement tool that could pin down the local Ge environment continuously along the crystallization pathway.

**F_IS angle:** The crystal has near-centrosymmetric Ge sites (F_IS ≈ −0.3); the amorphous phase breaks this (F_IS > 0). The jump in F_IS between the two phases quantifies the structural "distance" from the crystallization nucleus — a metric no one has computed from experimental data.

**Collaboration:** Groups at ESRF or ISIS with fast-acquisition PDF setups (< 1 s per pattern) doing laser-pump PDF experiments.

---

### 10. Ferrihydrite — environmental contamination (arsenic in drinking water)

**Human impact:** ~220 million people exposed to unsafe arsenic levels from ferrihydrite-bound As in groundwater.

Ferrihydrite (nominally Fe₁₀O₁₄(OH)₂) is the most structurally debated mineral in geochemistry — its exact local Fe coordination has been argued for 30 years. It is also the primary scavenger of arsenic, lead, and uranium in natural waters. The adsorption capacity depends directly on the surface Fe site geometry, which is controlled by the bulk amorphous structure.

TorchDisorder can refine neutron/XRD PDF of synthetic ferrihydrite with different As loadings, directly connecting local Fe order to adsorption capacity. F_IS of the FeO₆/FeO₄ mixed coordination shell has never been computed.

**Data availability:** Michel et al. (2007) *Science* data; Hiemstra group (Wageningen) neutron data — publicly available.

---

### 11. Nuclear waste glass — borosilicate and phosphate

**Funding source:** US DOE spends ~$1B/year on Hanford Site cleanup. EU Horizon covers analogous programs.

Borosilicate glass is the universal matrix for immobilizing high-level nuclear waste (Cs, Sr, Ba, Ln, actinides). Leaching resistance — the rate at which radioactive ions escape into groundwater over 10,000 years — depends on local B/Si coordination and the degree of network connectivity. Regulatory agencies (NRC, Euratom) are moving toward requiring atomistic structural characterization.

TorchDisorder can refine non-radioactive surrogate glasses (using Ba for Ra, Ce for Pu) from existing neutron PDF datasets. F_IS of the glass network nodes (B, Si, Al) would be the first mechanistic link between local symmetry and durability.

**Collaboration target:** Pacific Northwest National Laboratory (PNNL), Savannah River National Laboratory (SRNL), CEA Marcoule.

---

### 12. Metallic glasses — structural materials and biomedical implants

**Applications:** Apple Watch Series 7+ case (Liquidmetal/Heraeus), surgical scalpels, golf club faces, bone screws.

Zr-Cu-Ni-Al and Zr-Ti-Cu-Ni-Be metallic glasses have extraordinary strength-to-weight ratios. Their mechanical properties (yield strength, fracture toughness) are determined by the local icosahedral short-range order (ISRO) — which is exactly what F_IS and q4 were designed to capture.

Existing TorchDisorder BOO metrics (tet, q4) generalize directly to icosahedral order (q6). The published neutron/XRD PDF for Zr₅₂.₅Cu₁₇.₉Ni₁₄.₆Al₁₀Ti₅ is available from Johnson group (Caltech) datasets.

**Novel claim:** First differentiable structural refinement of a metallic glass that outputs a per-atom F_IS map → local regions of high/low symmetry breaking → shear transformation zones (STZs), the microscopic mechanism of plastic deformation.

---

### 13. Amorphous drug solids — pharmaceutical bioavailability

**Industry size:** ~40% of drug candidates are poorly water-soluble. Converting to amorphous solid dispersions (ASDs) is the dominant formulation strategy (e.g., Kaletra/ritonavir, Sporanox/itraconazole).

The FDA increasingly requires structural characterization of ASDs. Existing tools (PXRD, DSC, ssNMR) cannot provide atomistic models. TorchDisorder's PDF refinement — if applied to small-molecule organic glasses — would be the first gradient-based atomistic model of an amorphous drug.

**Challenges:** Organic molecules require new constraint generators (molecular graph → bond-length/angle constraints). However, the differentiable framework is identical; only the constraint JSON format changes. MACE-OFF (organic MACE potential) can replace MACE-MP for energy enforcement.

**Collaboration:** Any pharma formulation group with synchrotron PDF access (Pfizer, AstraZeneca, Merck all have internal beamtime allocations at NSLS-II and Diamond).

---

### 14. Amorphous oxide semiconductors — displays (a-IGZO)

**Industry size:** Every OLED/LCD display manufactured since 2012 uses a-IGZO (In-Ga-Zn-O) thin-film transistors — ~1 billion devices/year.

Electron mobility in a-IGZO (10–40 cm²/Vs vs. ~1 for a-Si) arises from the percolation of In 5s orbitals through the disordered network. Local In coordination (4-fold vs. 5-fold vs. 6-fold) is the structural determinant of mobility, but has never been resolved from a differentiable atomistic model.

TorchDisorder would be the first tool to output per-In-atom F_IS in a refined a-IGZO structure, directly linking local inversion symmetry breaking to the mobility percolation threshold.

**Data:** Published neutron PDF from Takagi group (Tokyo); synchrotron PDF from APS sector 11.

---

### 15. ZBLAN fluoride glass — space manufacturing and medical lasers

**Context:** ZBLAN (ZrF₄-BaF₂-LaF₃-AlF₃-NaF) is the only known glass with optical transmission from UV to mid-IR (0.2–7 µm). It cannot be made bubble-free on Earth — NASA manufactures it on the ISS. It is used to deliver CO₂ laser energy for surgical procedures (ophthalmology, ENT, urology).

The crystallization problem (bubbles/inclusions) that forces space manufacturing is a direct consequence of local ZrF₈ polyhedra clustering — which TorchDisorder could refine from existing PDF data and characterize with F_IS. Understanding why certain local environments nucleate crystals would guide Earth-based manufacturing or identify stabilizing dopant strategies.

**Collaboration:** NASA Glenn Research Center, IRflex Corporation, University of Bordeaux (Moizan group).

---

## Priority matrix

| System | Data available | Time to paper | Domain | Surprising angle |
|---|---|---|---|---|
| LiPON | In repo | 1–2 months | Battery | N-for-O mechanism |
| Amorphous Si | In repo | 1–2 months | Solar / neuromorphic | F_IS → dangling bond proxy |
| Li₃PS₄ glass | In repo | 1–2 months | Battery | β/γ CN comparison |
| ε-Fe₂O₃ | In repo | 2–3 months | Catalysis / magnetism | Multiferroic + F_IS per site |
| NaTaCl₆ | In repo | 2–3 months | Photovoltaics / detectors | Octahedral tilt OP |
| Ferrihydrite | Public datasets | 2–3 months | Environment / water | F_IS → As adsorption site |
| Phase-change GST | Public datasets | 3–4 months | Memory (Intel Optane) | Crystal ↔ amorphous F_IS jump |
| Metallic glasses | Public datasets | 3–4 months | Structural / biomedical | STZ → F_IS map |
| GeO₂ pressure | In repo (baseline) | 3–4 months | Photonics | High-pressure coord change |
| Nuclear waste glass | DOE/CEA data | 3–5 months | Nuclear remediation | Durability ↔ F_IS of network nodes |
| a-IGZO display glass | Public datasets | 4–5 months | Electronics (displays) | In coordination → mobility |
| Li₆PS₅Cl argyrodite | Need beamtime | 3–5 months | Battery | Disorder-conductivity link |
| Amorphous drugs (ASD) | Pharma partner | 4–6 months | Pharmaceutical | First atomistic model of ASD |
| ZBLAN fluoride glass | Published PDF | 4–6 months | Space / medical lasers | Crystallization nucleation sites |
| Operando PDF | Need beamtime | 6–12 months | Battery | Time-resolved structure movie |

**Start here:** LiPON and amorphous Si — both have data in the repo today, both map directly onto real applications, and both generate papers without needing a beamtime proposal.

**Highest surprise factor:** Ferrihydrite (environmental/humanitarian impact), phase-change GST (largest industry you're not thinking about), and amorphous drugs (pharma has money and no tools).

---

## Data sources — what to download and from where

### Already in repo (run immediately)

| System | File path | Format |
|---|---|---|
| LiPON | `data/xrd_measurements/li2.99_po3.38_n0.41/` | F(Q), T(r) CSV |
| Li₃PS₄ | `data/xrd_measurements/Li3PS4/` | F(Q), S(Q), g(r) CSV |
| Li₃PS₄ γ | `data/xrd_measurements/Li3PS4_gamma/` | F(Q), T(r) CSV |
| Amorphous Si | `data/xrd_measurements/Si/` | S(Q), J(r) CSV + .dat |
| ε-Fe₂O₃ | `data/xrd_measurements/Fe2O3/` | F(Q), T(r) CSV |
| NaTaCl₆ | `data/xrd_measurements/NaTaCl6/` | F(Q), g(r), T(r) CSV |
| NaLiI | `data/xrd_measurements/NaiLiI/` | F(Q), T(r) CSV — needs CIF from ICSD/COD |
| GeO₂ | `data/xrd_measurements/GeO2/` | F(Q), T(r) CSV |
| SiO₂ | `data/xrd_measurements/SiO2/` | F(Q), T(r) CSV |

---

### Downloadable from published paper supplementary data

**Ferrihydrite**
- Michel, F. M. et al. "The Structure of Ferrihydrite, a Nanocrystalline Material." *Science* 316, 1726 (2007). DOI: 10.1126/science.1122249
- Supplementary material contains the X-ray and neutron PDF (G(r)) as table files.
- Also: Hiemstra, T. et al. *Geochim. Cosmochim. Acta* (2013) — additional PDF data in SI.

**Phase-change memory GST (Ge₂Sb₂Te₅)**
- Kohara, S. et al. "Structural basis for the fast phase change of Ge₂Sb₂Te₅." *Appl. Phys. Lett.* 89, 201910 (2006). DOI: 10.1063/1.2388012 — SI contains S(Q) and G(r).
- Salmon, P. S. et al. *J. Non-Cryst. Solids* 353 (2007) — neutron G(r) of amorphous GST in figures (digitizable with WebPlotDigitizer).
- Caravati, S. et al. *Appl. Phys. Lett.* 91, 171906 (2007) — comparison of DFT MD with experimental PDF.

**Metallic glasses (Zr-Cu-Ni-Al)**
- Sheng, H. W. et al. "Atomic packing and short-to-medium-range order in metallic glasses." *Nature* 439, 419 (2006). DOI: 10.1038/nature04421 — SI has g(r) tables for Zr₅₂.₅Cu₁₇.₉Ni₁₄.₆Al₁₀Ti₅.
- Ma, D. et al. *Nature Mater.* 8, 30 (2009) — additional PDF data for binary Zr-Cu.
- Cheng, Y. Q. & Ma, E. *Prog. Mater. Sci.* 56, 379 (2011) — review with compiled g(r) datasets.

**a-IGZO (amorphous In-Ga-Zn-O)**
- Nomura, K. et al. "Room-temperature fabrication of transparent flexible thin-film transistors using amorphous oxide semiconductors." *Nature* 432, 488 (2004). DOI: 10.1038/nature03090 — structural data in follow-up papers.
- Ide, K. et al. "Functions of crystalline and amorphous phases in In-Ga-Zn-O semiconductors." *Adv. Mater.* 26, 7484 (2014) — neutron PDF in supplementary.
- Nomura, K. et al. *Phys. Rev. B* 75, 035212 (2007) — detailed structural analysis with G(r).

**ZBLAN fluoride glass**
- Royle, M. et al. "X-ray diffraction studies and structural modelling of fluorozirconate glasses." *J. Non-Cryst. Solids* 233, 101 (1998) — g(r) data.
- Moizan, V. et al. *Opt. Express* (2008) — structural data for photonics-grade ZBLAN.
- Lucas group (Université de Rennes) papers in J. Non-Cryst. Solids 1995–2010 are the primary source.

**Li₆PS₅Cl argyrodite (some published data)**
- Kraft, M. A. et al. "Influence of Lattice Polarizability on the Ionic Conductivity in the Lithium Superionic Argyrodites." *JACS* 140, 16330 (2018). DOI: 10.1021/jacs.8b10282 — neutron diffraction patterns in SI (Rietveld, not PDF — but G(r) is in follow-up work).
- Banik, A. et al. *ACS Energy Lett.* 4, 2404 (2019) — PDF data for disordered argyrodite.
- Best route: email M. A. Kraft (Stuttgart) or T. Famprikis (Grenoble) directly. Both groups are responsive and have amorphous argyrodite PDF that is not yet published.

---

### Public databases (searchable, free registration)

| Database | URL | What's there | Format |
|---|---|---|---|
| **ICDD PDF-4+** | icdd.com/pdf-4 | Diffraction patterns for 900k+ phases | Paid, most universities subscribe |
| **Crystallography Open Database (COD)** | crystallography.net/cod | Crystal CIF files only (no amorphous PDF) | Free CIF download |
| **Materials Project** | materialsproject.org | DFT structures + computed S(Q) | Free, API available |
| **AFLOW** | aflow.org | DFT structures, some MD trajectories | Free, REST API |
| **NOMAD Repository** | nomad-lab.eu | DFT/MD data deposits from published papers | Free, searchable by DOI |
| **ICSD (Inorganic Crystal Structure Database)** | icsd.fiz-karlsruhe.de | Crystal CIFs only | Paid, most universities subscribe |
| **PNNL Glass Properties Database** | pnl.gov/glass | Composition + property data for 1000s of glasses | Free registration — no raw PDF |
| **ILL Data Portal** | ill.eu/users/scientific-groups/data-portal | Neutron data from ILL experiments | Free after embargo (usually 3 yr) |
| **ISIS ICAT** | data.isis.stfc.ac.uk | Neutron data from ISIS experiments | Free after embargo |
| **ORNL SNS** | neutrons.ornl.gov | Neutron PDF from NOMAD beamline | Some open, registration required |
| **Zenodo / Figshare** | zenodo.org, figshare.com | Author-deposited datasets from papers | Free, search by material name |

**Practical tip for any system:** Search Zenodo for "pair distribution function + material name" — authors who deposit data at Zenodo do so voluntarily and the data is immediately usable in CSV/dat format without reformatting.

---

### Data that needs direct author contact

| System | Who to contact | Why they'll respond |
|---|---|---|
| Amorphous drugs (ASD) | Taylor group (Purdue, L. Taylor); Schuth group (MPI Mülheim) | Academic, publish openly, interested in structural tools |
| Li₆PS₅Cl amorphous | Famprikis group (ILL Grenoble); Kraft group (Stuttgart) | Battery researchers always want more structural tools |
| Nuclear waste glass (raw PDF) | PNNL (J. Vienna, A. Kruger); CEA Marcoule (S. Gin) | DOE-funded, often open to academic collaboration |
| ZBLAN glass | Lucas group (Rennes); Moizan (CNRS) | Niche field, would welcome new computational approach |

For each of these: a 3-sentence cold email explaining TorchDisorder + offering co-authorship on the refinement paper has a very high hit rate. These groups collect data and struggle to extract full atomistic models from it.
