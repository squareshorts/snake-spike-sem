# Submission Strengthening Addendum

        This addendum implements the reviewer-facing strengthening items directly in the submission bundle.

        ## 1. Orthogonal validation beyond the Peroutka-derived reconstruction

        We added an external positive-control benchmark based on the Sharklet-style anti-biofilm micropattern reported for silicone medical-device surfaces (2 um feature width, 2 um spacing, 3 um relief, 4-16 um repeating motif lengths). The observation map was not recalibrated for this analysis.

        | Species | Smooth | Sharklet-like | Snake ensemble |
| --- | --- | --- | --- |
| E. coli | 0.999 | 0.706 | 0.121 +/- 0.022 |
| S. aureus | 0.985 | 0.791 | 0.218 +/- 0.055 |
| P. aeruginosa | 1.000 | 0.864 | 0.136 +/- 0.030 |

        This creates an orthogonal validation layer that is no longer closed within the Peroutka-derived topography family.

        ## 2. Application-relevant boundary conditions

        We added a device-envelope robustness analysis motivated by protein-conditioned, low-shear silicone-device deployment. The scenarios include protein conditioning, low-shear loading, compliant polymer contact, fabrication rounding, and their combined envelope.

        | Species | Suppression vs smooth across scenarios | Combined-envelope stable fraction |
| --- | --- | --- |
| E. coli | 87.2% to 89.1% | 0.108 |
| S. aureus | 75.1% to 81.4% | 0.179 |
| P. aeruginosa | 84.9% to 88.0% | 0.120 |

        These outputs are written to `data/deployment_boundary_conditions.csv`, `data/deployment_boundary_condition_summary.csv`, and `figures/figure_device_boundary_conditions.png`.

        ## 3. Supporting information package

        The submission bundle now includes a real SI directory under `supporting_information/` with priors, descriptor tolerances, calibration details, sensitivity bounds, and figure-generation metadata.

        ## 4. Explicit scale information for image panels

        `figure_surface_ensemble.png` now includes x/y axes in um plus a 10 um scale bar. Caption language can now state explicitly that each admissible reconstructed surface panel spans 36 x 36 um.
