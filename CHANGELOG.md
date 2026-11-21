# Changelog

## 2.2.1 — 2026-11-20

- Corrective archive release of the verified v2.2.0 geometric-control source, outputs and reproducibility scripts.
- Uses a new Git tag because the previously created `v2.2.0` tag points to the same commit as `v2.0.0`.
- No scientific model changes or newly computed numerical results are introduced by this packaging correction.
- Release source and submission metadata now identify version 2.2.1.

## 2.2.0 — 2026-11-20

- Added `ModelConfig.arrival_mode` (`dual`, `supply`, `capture`) to place the near-surface arrival field in cell supply, capture or both; `dual` is the default and reproduces 2.0.0 exactly.
- Added `BladeGeometry.row_phase` for the offset of alternating blade columns; the default 0.5 reproduces 2.0.0 exactly.
- Added `analysis_controls.py`, `tables_controls.py` and `figures_controls.py` with arrival-formulation, spacing-sweep, independent-coefficient, coverage-matched, relief-factorial, row-phase and grid-refinement experiments. Outputs are in `results_v2/controls/`.
- Corrected the first author's affiliation.

## 2.0.0 — 2026-11-19

Major model and repository revision for the Journal of the Royal Society Interface manuscript.

- Replaced the spike-based surface model with staggered arrays of tapered, imbricated blades and tip-associated pits.
- Added cell-footprint contact energetics, appendage engagement, species-dependent near-surface arrival, capacity-limited compartment dynamics and the effective EPS field.
- Added the 40-reconstruction production ensemble, grid/time-step/domain verification, mechanism knockouts, design maps, material-adhesion sensitivity, long-time simulations, and geometric and mechanistic Saltelli/Sobol analyses.
- Added the manuscript and supplementary LaTeX sources, compiled PDFs, figures, numerical tables and simulation-level outputs.
- Corrected the reproducibility documentation, improved the Sobol figure layout and repaired the micrometre symbol in the compiled PDFs.
- Archived the previous implementation under `legacy/`.
- Replaced the old v1.0.0 software metadata. A new Zenodo version DOI must be added after the v2.0.0 GitHub release is archived.

## 1.0.0 — archived predecessor

The previous Zenodo release describes the earlier spike-based hierarchical model. It is retained in Git history and is not the implementation used by the current manuscript.
