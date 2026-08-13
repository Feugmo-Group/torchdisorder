# Reply to Alessio Zaccone — F_IS implementation results

**To:** Alessio Zaccone  
**From:** Conrard Tetsassi Feugmo (giresse.feugmo@gmail.com)  
**Subject:** Re: TorchDisorder — F_IS implementation and potential collaboration  

**Attachments to send:**
- `fis_vs_boo_comparison.pdf` — 3-panel histogram comparison (F_IS, q4, q6) — **main result figure**
- `sio2_fis_vs_q4.png` — F_IS vs q4 scatter (crystal vs glass, independence of axes)
- `fis_q4_scatter.png` — F_IS vs q4 with structural quadrant labels

---

Dear Alessio,

Thank you very much for sending the tailored code — it was very clear and straightforward to integrate. We have now implemented F_IS as a fully differentiable order parameter within TorchDisorder and obtained our first results, which I am happy to share with you.

**Implementation.** We integrated F_IS as a differentiable term in our optimization framework. Beyond using it as a post-hoc structural descriptor, we can now include it directly as a training objective — steering atomic positions toward a target mean F_IS during the scattering optimization. This opens a route to jointly fitting experimental diffraction data and physical symmetry constraints on the local coordination environments, which we think is particularly powerful for materials where the local symmetry type is known from theory or experiment.

**Results on a-SiO₂.** As a first test, we computed F_IS on an amorphous SiO₂ structure optimized against experimental X-ray F(Q) data (1125 Si atoms, cutoff 2.2 Å, variable-R weighting). The results confirm your paper's finding very clearly (see attached figures):

- F_IS shifts **Δ = 0.380** between c-SiO₂ (−0.331) and a-SiO₂ (+0.049)
- q4 shifts only Δ = 0.089 and q6 shifts Δ = 0.087 over the same transition — **F_IS is ~4× more sensitive**
- The glass F_IS distribution (σ = 0.235) is **twice as wide** as q4 (σ = 0.121), resolving far greater structural heterogeneity within the glass itself
- As the scatter plot shows, F_IS and q4 are essentially **orthogonal descriptors** — confirming they capture independent aspects of local structure

We are also applying this to Fe₂O₃ (octahedral Fe, target F_IS ≈ +0.5–0.7) and Li₂HfCl₆ halide solid electrolytes (octahedral Hf, relevant to ion transport).

**Potential areas of collaboration.** Beyond the F_IS work, I have been following your group's recent results on topological defects in glasses with great interest. The 2025 Nature Communications papers on hedgehog defects in 3D glasses, and the connection between defect density and plasticity, are directly relevant to the structures we generate. TorchDisorder produces amorphous structures constrained simultaneously by experimental diffraction data and local order parameters — which could provide an interesting test bed for computing topological defect distributions in experimentally-calibrated glasses, rather than purely simulated ones. I would be very curious to know whether you think the F_IS distributions we observe carry information about defect density in your framework.

More broadly, our differentiable optimization approach could in principle be extended to target other descriptors from your work — for instance, to impose constraints on vibrational mode density or elastic heterogeneity as proxies, if tractable.

**Possible visit.** I plan to be in Europe in spring 2027 and would be very happy to visit your group in Milan or Göttingen to present our work and discuss these ideas in person, if that would be of interest to you. Please do let me know.

Thank you again for your generosity in sharing the code and for your stimulating papers. We will of course acknowledge your contribution fully in any publication that comes from this work.

I look forward to staying in touch.

With best wishes,

Conrard Tetsassi Feugmo  
Feugmo Research Group
