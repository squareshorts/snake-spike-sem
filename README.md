# snake-spike-sem

Hierarchical modelling pipeline for evaluating how reconstructed *Python regius* scale topography suppresses early bacterial colonisation.

## Repository layout

- `snake_model/`: package source for the hierarchical model and compatibility entrypoint.
- `scripts/`: convenience runners.
- `results/submission/`: committed figures, tables, and report files used for the current manuscript-facing submission bundle.
- `legacy/`: archived helper scripts retained for provenance but not used by the current pipeline.

## Dependencies

Install the Python requirements with:

```bash
python -m pip install -r requirements.txt
```

## Run the model

Default run:

```bash
python -m snake_model.pipeline
```

Example with explicit settings:

```bash
python -m snake_model.pipeline --ensemble-size 8 --uq-samples 18 --dx 0.6
```

This writes a fresh analysis bundle under `outputs/`.

## Submission artifacts

The current manuscript-facing outputs are committed under `results/submission/`, including:

- `results/submission/data/hierarchical_model_summary.json`
- `results/submission/data/hierarchical_ensemble_summary.csv`
- `results/submission/data/hierarchical_mechanism_ablation.csv`
- `results/submission/data/hierarchical_global_sensitivity.csv`
- `results/submission/data/hierarchical_phase_diagram.csv`
- `results/submission/data/ensemble_size_convergence.csv`
- `results/submission/figures/figure_surface_ensemble.png`
- `results/submission/figures/figure_surface_losses.png`
- `results/submission/figures/figure_mechanism_ablation.png`
- `results/submission/figures/figure_phase_diagram.png`

## Notes

The original reduced support-connectivity model is retained in the codebase as an explicit limiting case of the hierarchical model, and the current README reflects the submission-oriented workflow rather than the earlier exploratory runs.
