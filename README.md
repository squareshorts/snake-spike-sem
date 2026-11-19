# Lateral packing of imbricated blades and bacterial suppression

**Software release:** `2.0.0`  
**Manuscript:** *Lateral packing of imbricated blades governs species-dependent bacterial suppression on ball python scales*  
**Target journal:** Journal of the Royal Society Interface

This repository contains the blade-array model, analysis pipeline, numerical outputs, manuscript source and electronic supplementary material for the study of species-dependent bacterial suppression on *Python regius* scales.

The current release implements tapered, imbricated blades with tip-associated pits, cell-scale contact energetics, near-surface arrival, capacity-limited compartment dynamics and an effective extracellular-polymeric-substance field. Each patterned surface is compared with a flat reference under the same species coefficients. The analysis includes a 40-reconstruction production ensemble, numerical verification, mechanism ablations, geometric design maps, material-adhesion sensitivity, time-course analysis, and geometric and mechanistic Sobol designs.

## Repository layout

- `snake_model/` — active blade-array model and species parameter definitions.
- `analysis_v2.py` — simulation stages and numerical experiments.
- `stats_v2.py` — bootstrap endpoint mapping, prediction intervals and Sobol summaries.
- `figures_v2.py` — manuscript and supplementary figure generation.
- `make_tables.py` — LaTeX tables and numerical macros generated from saved outputs.
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

## Install this release into the existing Windows repository

The supplied release ZIP contains `INSTALL_JRSI_V2.ps1` and `PUBLISH_GITHUB.ps1`. Extract it to a separate directory; do not extract it over the existing repository. In PowerShell, adjust the download path if necessary:

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\snake-spike-sem-v2.0.0.zip" `
  -DestinationPath "C:\work\snake_model_v2_package" -Force

& "C:\work\snake_model_v2_package\snake-spike-sem-v2.0.0\INSTALL_JRSI_V2.ps1" `
  -TargetRoot "C:\work\snake_model"
```

The installer verifies that the existing repository is `squareshorts/snake-spike-sem`, is on `main`, and matches `origin/main` before moving any files. It preserves the `.git` directory and backs up the previous working tree to a sibling `snake_model.pre_jrsi_<timestamp>` directory. Install the Python dependencies, then run `python -m pytest -q` from `C:\work\snake_model`.

When the tests pass, run `./PUBLISH_GITHUB.ps1` from the repository root. The script repeats the synchronization checks and tests, shows the pending Git changes, and requires the exact confirmation `PUBLISH` before committing, pushing `main`, and creating and pushing tag `v2.0.0`. If GitHub CLI (`gh`) is installed, it can also create the GitHub Release. Otherwise, create the release from the tag on GitHub. Zenodo archiving and the new version DOI are subsequent steps; the v1.0.0 DOI is not valid for this model.

## Data provenance

No new bacterial experiments were performed for this study. The 48 h *E. coli* and *S. aureus* observations and the SEM-derived geometric measurements were digitised or estimated from Peroutka et al. (2026), cited in the manuscript. The external Sharklet AF comparison uses published short-term measurements from May et al. (2014) and Xu et al. (2017). The extraction values, analysis definitions, model parameters and simulation outputs are retained in the source files and `results_v2/`.

See [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) for the observation sources and assumptions, and [`SUBMISSION_METADATA.md`](SUBMISSION_METADATA.md) for the text required by the JRSI submission system.

## Citation and archival release

The active code and manuscript package are version `2.0.0`. The earlier GitHub/Zenodo `1.0.0` release describes a spike-based precursor and is not the implementation used in this manuscript. Its DOI, `10.5281/zenodo.19316098`, must not be cited as the archive for the present blade-array model.

After the `v2.0.0` GitHub release is created, enable or synchronise the repository in Zenodo and use the newly minted version DOI in the submission system and in `CITATION.cff`. The version-specific DOI is intentionally not guessed or copied from the earlier release.

## License

The software and accompanying reproducibility materials are released under the MIT License. See [`LICENSE`](LICENSE).
