# Lateral blade packing and bacterial retention on ball python scales

**Software release:** `2.2.1`
**Manuscript:** *Lateral blade packing predicts bacterial retention on reconstructed ball python scales*
**Target journal:** Journal of the Royal Society Interface

This repository contains the blade-array model, analysis pipeline, numerical outputs, manuscript source and electronic supplementary material for the study of species-dependent bacterial suppression on *Python regius* scales.

The current release implements tapered, imbricated blades with tip-associated pits, cell-scale contact energetics, near-surface arrival, capacity-limited compartment dynamics and an effective extracellular-polymeric-substance field. Each patterned surface is compared with a flat reference under the same species coefficients. The analysis includes a 40-reconstruction production ensemble, numerical verification, mechanism ablations, geometric design maps, material-adhesion sensitivity, time-course analysis, geometric and mechanistic Sobol designs, alternative arrival formulations, independently varied species coefficients and geometric control experiments.

## Repository layout

- `snake_model/` — active blade-array model and species parameter definitions.
- `analysis_v2.py` — simulation stages and numerical experiments.
- `stats_v2.py` — bootstrap endpoint mapping, prediction intervals and Sobol summaries.
- `figures_v2.py` — manuscript and supplementary figure generation.
- `make_tables.py` — LaTeX tables and numerical macros generated from saved outputs.
- `analysis_controls.py` — arrival-formulation, spacing-sweep, independent-coefficient and geometric control experiments.
- `tables_controls.py`, `figures_controls.py` — tables, macros and figures for the control experiments.
- `results_v2/controls/` — simulation-level CSV outputs, tables and figures of the control experiments.
- `results_v2/data/` — simulation-level CSV outputs and summary JSON.
- `results_v2/tables/` — derived statistical tables.
- `results_v2/figures/` — figure source outputs in PDF and PNG formats.
- `manuscript/` — manuscript, supplementary-material and cover-letter sources and compiled PDFs.
- `legacy/` — archived code from the earlier spike-based model; it is not used by the current workflow.
- `tests/` — model smoke tests and submission-bundle consistency checks.

## Environment

Python 3.10 or later is supported. Install the dependencies from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

The simulation scripts use NumPy, SciPy, pandas and Matplotlib. The LaTeX source can be compiled with XeLaTeX; the compiled PDFs are included in `manuscript/`.

## Reproduce the analysis

Run the full pipeline from the repository root. The default worker count is two; set `NPROC` to adjust it.

**Windows PowerShell**

```powershell
$env:NPROC = "2"
.\run_all.ps1
```

**Linux, macOS or Git Bash**

```bash
NPROC=2 ./run_all.sh
```

The pipeline runs the following stages in order:

1. `ensemble` — 40 independent blade-array reconstructions at the production grid, plus time-course and Sharklet comparisons.
2. `numerics` — grid, time-step and periodic-domain checks on four blade arrays and the Sharklet AF-type cell.
3. `knockouts` — one-at-a-time removal of arrival, slope, appendage, steric, compliance and EPS mechanisms.
4. `design` — tip-height/base-width and adjacent-tip-spacing design maps using three common lattice seeds.
5. `chemistry` — flat-control adhesion perturbations.
6. `longtime` — extension of the compartment model to 72 model-time units.
7. `gsa_geometry` — 64-point Saltelli base design for eleven geometric inputs (832 geometry evaluations).
8. `gsa_mechanism` — 64-point Saltelli base design for twelve coefficient multipliers (896 coefficient settings).

The pipeline then runs `stats_v2.py`, `figures_v2.py` and `make_tables.py`. The latter regenerates `manuscript/numbers.tex` and all `manuscript/tab_*.tex` files from the saved outputs. The geometric Sobol analysis samples uniformly over each parameter's bounded support; it is a sensitivity analysis over that design space, not a variance decomposition under the truncated-normal production priors.

For a single stage, use:

```powershell
python analysis_v2.py --stage ensemble
python stats_v2.py
python figures_v2.py
python make_tables.py
```

Run the lightweight validation tests with:

```powershell
python -m pytest
```

To rebuild the three PDFs locally, run `build_manuscripts.ps1` (Windows) or `./build_manuscripts.sh` (Unix-like systems). The source can also be compiled in Overleaf with `manuscript/manuscript.tex` as the main manuscript file.

## Control experiments

`analysis_controls.py` takes one stage name and writes its CSV to `results_v2/controls/`:

```bash
for s in structural spacing coverage factorial phase indep matched_grid grid; do python analysis_controls.py $s; done
python tables_controls.py
python figures_controls.py
```

| Stage | Design |
|---|---|
| `structural` | 40 production reconstructions under dual-pathway, supply-only and capture-only arrival |
| `spacing` | six adjacent-tip spacings (3.4–5.9 µm), four seeds, three arrival formulations |
| `coverage` | spacings 3.8, 4.4 and 5.2 µm at matched projected blade coverage (0.35 with pits, 0.46 without), eight seeds |
| `factorial` | spacing × tip height × pit depth over eight nuisance-geometry draws (144 surfaces) |
| `phase` | row offset 0, 0.25 and 0.5 at fixed density, with and without pits, eight seeds |
| `indep` | 160 geometries with species-specific ±20% coefficient, β and control-adhesion multipliers (Δx = 0.15 µm) |
| `matched_grid` | coverage-matched contrast recomputed at Δx = 0.07 µm |
| `grid` | close-packed and wide arrays at Δx = 0.15, 0.10, 0.07 and 0.05 µm |

`ModelConfig.arrival_mode` and `BladeGeometry.row_phase` default to the 2.0.0 behaviour (`"dual"`, `0.5`), so all 2.0.0 outputs are unchanged.

## Data provenance

No new bacterial experiments were performed for this study. The 48 h *E. coli* and *S. aureus* observations and the SEM-derived geometric measurements were digitised or estimated from Peroutka et al. (2026), cited in the manuscript. The external Sharklet AF comparison uses published short-term measurements from May et al. (2014) and Xu et al. (2017). The extraction values, analysis definitions, model parameters and simulation outputs are retained in the source files and `results_v2/`.

See [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) for the observation sources and assumptions, and [`SUBMISSION_METADATA.md`](SUBMISSION_METADATA.md) for the text required by the JRSI submission system.

## Citation and archival release

The complete geometric-control analyses are included in the GitHub release [`v2.2.1`](https://github.com/squareshorts/snake-spike-sem/releases/tag/v2.2.1). Use the version-specific Zenodo DOI minted from this tag for the journal submission. Earlier archived releases, including `10.5281/zenodo.24378494` (version 2.0.0) and `10.5281/zenodo.24408778` (tagged source identical to version 2.0.0), do not contain the present geometric-control analyses.

## License

The software and accompanying reproducibility materials are released under the MIT License. See [`LICENSE`](LICENSE).
