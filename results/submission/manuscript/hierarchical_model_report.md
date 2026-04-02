# Hierarchical Snake-Scale Colonisation Report

Observation gain: 60.781
Observation bias: 0.038
Calibration SSE: 0.00018

## Full hierarchical means
- E. coli: 0.121 +/- 0.022
- P. aeruginosa: 0.136 +/- 0.030
- S. aureus: 0.218 +/- 0.055

## Orthogonal validation layer
An external Sharklet-like micropattern benchmark (2 um feature width, 2 um spacing, 3 um relief, 4-16 um motif lengths) is now evaluated without recalibrating the observation map.

| Species | Smooth | Sharklet-like | Snake ensemble |
| --- | --- | --- | --- |
| E. coli | 0.999 | 0.706 | 0.121 +/- 0.022 |
| S. aureus | 0.985 | 0.791 | 0.218 +/- 0.055 |
| P. aeruginosa | 1.000 | 0.864 | 0.136 +/- 0.030 |

## Device-envelope robustness
A protein-conditioned, low-shear, compliant silicone-device envelope is now evaluated as an application-relevant computational robustness layer.

| Species | Suppression vs smooth across scenarios | Combined-envelope stable fraction |
| --- | --- | --- |
| E. coli | 87.2% to 89.1% | 0.108 |
| S. aureus | 75.1% to 81.4% | 0.179 |
| P. aeruginosa | 84.9% to 88.0% | 0.120 |

## Supporting information package
- `supporting_information/README.md`: SI manifest and algorithm summary.
- `supporting_information/si_surface_priors.csv`: sampling priors and hard bounds.
- `supporting_information/si_descriptor_acceptance.csv`: descriptor weights/tolerances and acceptance threshold.
- `supporting_information/si_uq_bounds.csv`: uncertainty-analysis ranges.
- `supporting_information/si_calibration_objective.json`: calibration targets and optimisation details.
- `supporting_information/si_figure_manifest.csv`: figure provenance and scale notes.

## Figure-scale note
Representative surface panels now report x/y axes in um and include a 10 um scale bar; each panel spans 36 x 36 um laterally.
