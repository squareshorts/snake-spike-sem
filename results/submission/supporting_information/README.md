# Supporting Information Package

This directory provides the reproducibility package requested for the submission bundle. All files are generated from `python -m snake_model.pipeline` and are intended to be submitted alongside the manuscript.

## Included files

- `si_surface_priors.csv`: truncated-normal priors and hard bounds used to sample candidate surfaces.
- `si_descriptor_acceptance.csv`: descriptor targets, tolerances, and the ensemble acceptance cutoff (`loss <= 12`).
- `si_uq_bounds.csv`: Latin-hypercube uncertainty bounds used for the global sensitivity analysis.
- `si_calibration_objective.json`: calibration targets, raw means/spread, and the objective used to choose `obs_gain` and `obs_bias`.
- `si_figure_manifest.csv`: figure-generation provenance, data dependencies, and scale information.

## Surface-generation algorithm

1. Generate an anisotropic hexagonal centroid lattice across a 36 x 36 um domain with Gaussian centroid jitter.
2. Sample spike height, tip radius, base radius, and sharpness from the priors in `si_surface_priors.csv`.
3. Reconstruct a 2.5D height field by taking the pointwise maximum across all spikes and adding low-amplitude correlated latent roughness.
4. Compute ensemble descriptors (density, nearest-neighbour statistics, anisotropy, pair-correlation trace, height statistics, tip/base radii, sharpness, occupied area fraction).
5. Accept surfaces whose descriptor loss against the Peroutka-derived reference target remains below 12.0.

## Calibration objective

The observation map is calibrated only on the reported E. coli and S. aureus remaining fractions. The optimisation minimises squared error between observed and predicted stable fractions, adds a weak penalty for large prediction spread across accepted surfaces, and regularises excessively steep observation gains.

## Sensitivity ranges

- Phase-diagram pitch values: 3.8, 4.4, 5.0, 5.6, 6.2 um.
- Phase-diagram appendage scales: 0.6, 0.9, 1.2, 1.5, 1.8.
- Global-sensitivity parameters: tip_radius_mu, base_radius_mu, sharpness_mu, compliance_scale, appendage_length_scale, eps_diffusivity, capture_rate.

## Benchmark summary

| Species | Smooth | Sharklet-like | Snake ensemble |
| --- | --- | --- | --- |
| E. coli | 0.999 | 0.706 | 0.121 +/- 0.022 |
| S. aureus | 0.985 | 0.791 | 0.218 +/- 0.055 |
| P. aeruginosa | 1.000 | 0.864 | 0.136 +/- 0.030 |

## Device-envelope robustness summary

| Species | Min suppression vs smooth | Max suppression vs smooth | Combined-envelope stable fraction |
| --- | --- | --- | --- |
| E. coli | 87.2% | 89.1% | 0.108 |
| S. aureus | 75.1% | 81.4% | 0.179 |
| P. aeruginosa | 84.9% | 88.0% | 0.120 |
