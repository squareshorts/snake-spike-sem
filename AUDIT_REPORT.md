# JRSI v2.2.1 release verification

**Date:** 20 November 2026
**Release version:** 2.2.1
**Manuscript:** *Lateral blade packing predicts bacterial retention on reconstructed ball python scales*

## Scope and provenance

This release retains the v2.0.0 model, ensemble, numerical verification, sensitivity analyses and published-observation comparisons. The verified v2.2.0 local analysis, now archived in v2.2.1, adds structural arrival-formulation tests, independently varied species coefficients, coverage-matched spacing comparisons, a three-dimensional geometry factorial, a fixed-density row-phase control, and finer-grid numerical checks. The underlying experimental observations remain those reported in the cited literature; no new bacterial experiments were performed.

The additional analysis scripts are `analysis_controls.py`, `tables_controls.py` and `figures_controls.py`. Saved numerical outputs and figures are in `results_v2/controls/`. The main model's default arrival mode and row phase preserve the v2.0.0 behaviour.

## Results supported by the saved outputs

- **Coverage-matched spacing:** all 144 matched contrasts are positive across the tested species, arrival formulations, pit states and surface seeds. The comparison controls projected blade coverage, but does not hold feature density and blade width fixed.
- **Three-dimensional geometric factorial:** all 144 paired contrasts are positive across the sampled tip heights, pit depths and nuisance geometries. They represent a prescribed sensitivity design, not 144 independent biological replicates.
- **Independently varied species parameters:** *E. coli* exceeds *S. aureus* in predicted suppression in 148/160 tested coefficient scenarios; the ordering can reverse.
- **Fixed-density row staggering:** the mean change is below 0.002 log units. The model does not show a distinctive benefit of a staggered arrangement at fixed feature density.
- **Matched-coverage grid check:** all 24 contrasts remain positive at both 0.10 and 0.07 micrometre resolution. The largest change in an individual contrast is below 0.06 log units.
- **Crowded geometries:** individual predicted log reductions can change by approximately 0.2 under finer-grid comparison. High-suppression values in extreme tall/crowded configurations do not have established pointwise convergence.

## Reproducibility checks and limits

The Windows project virtual environment is Python 3.10.11. Following installation of the release-safeguard patch, `python -m pytest -q` returned **15 passed**. Nine original tests and six additional regression checks cover model and package consistency, including saved geometry-control outputs and safe release-script defaults. These automated tests check saved numerical results and selected model behaviour; they do **not** certify a new uninterrupted end-to-end run of every simulation stage.

An independent external audit reported exact agreement for arrival-formulation, spacing, coverage-matched, row-phase and matched-grid summaries, with renewed random nuisance draws giving comparable but nonidentical factorial and species-coefficient sensitivity summaries. The archive includes the explicit recorded outputs used by the manuscript.

## Release and submission status

The earlier public `v2.2.0` Git tag pointed to the exact commit also tagged `v2.0.0`, so it did not archive the newly computed geometry controls. The associated Zenodo DOI `10.5281/zenodo.24408778` should not be cited for the present analysis. Version `v2.2.1` is the corrective release of the locally verified code and output tables. Its own Zenodo DOI must be minted from the correct GitHub tag and then entered into the manuscript and submission metadata. The publication script is preview-only by default and uses a 127-path whitelist.

The 15 passing tests were run on the corrected local repository on 20 November 2026; no new end-to-end simulation run is claimed in this corrective release.

## Author checks

The authors should confirm the affiliations, CRediT roles, funding, competing interests and the precise description of AI-assisted development, code checking and language editing. Scientific conclusions remain conditional on the model and published observations. The original keratin-versus-polystyrene assay does not isolate the causal contribution of surface topography.
