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

## Zenodo release checklist

1. Log in to Zenodo and connect your GitHub account.
2. In Zenodo, open the GitHub integration page, click `Sync now`, and enable `squareshorts/snake-spike-sem`.
3. After the repository is enabled, create a GitHub release from tag `v1.0.0`.
4. Wait for Zenodo to ingest the release and mint the version DOI.
5. Copy the Zenodo DOI badge back into this README and into the manuscript data-availability statement.

The repository already includes both `CITATION.cff` and `.zenodo.json`. Per Zenodo's current documentation, Zenodo will use `.zenodo.json` for GitHub release archiving metadata, while GitHub will use `CITATION.cff` to display a preferred citation.

## Notes

The original reduced support-connectivity model is retained in the codebase as an explicit limiting case of the hierarchical model, and the current README reflects the submission-oriented workflow rather than the earlier exploratory runs.
