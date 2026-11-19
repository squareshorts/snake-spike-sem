# JRSI submission and reproducibility audit

**Audit date:** 19 November 2026  
**Package version:** 2.0.0  
**Manuscript:** *Lateral packing of imbricated blades governs species-dependent bacterial suppression on ball python scales*

## Scope

The supplied local archive was inspected at source-code, simulation-output, statistical-summary, manuscript-source and compiled-PDF levels. The active package was reorganised around the blade-array implementation. The earlier spike-based model was moved to `legacy/` and is no longer imported by the active pipeline.

## Reproducibility checks

The following analyses were rerun from the refactored source and compared with the supplied CSV outputs. All numeric columns matched exactly at the saved precision.

| Analysis | Rows/evaluations | Result |
|---|---:|---|
| Production ensemble | 120 | Exact match |
| Time-course subset | 36 | Exact match |
| Sharklet comparison | 3 | Exact match |
| Grid/domain verification | 108 | Exact match |
| Sharklet grid verification | 15 | Exact match |
| Geometric Sobol design | 2,496 species-level evaluations | Exact match |

The saved output bundle also passed structural checks for the mechanism knockouts (252 rows), design maps (486), material-adhesion analysis (120), long-time simulations (24), and mechanistic Sobol design (5,376). Statistical summaries, manuscript tables, numerical macros and figures were regenerated from the saved outputs.

Eight automated tests passed. They cover species parameter integrity, finite blade and Sharklet surface generation, flat-reference normalisation, finite species-specific ratios, the subunit-occupancy condition, expected output row counts, numerical-macro precision and the matched P. aeruginosa mechanistic endpoint refit.

The full eight-stage analysis pipeline was not certified as a single uninterrupted end-to-end run in this audit. The production ensemble, numerical verification and geometric Sobol stages were independently rerun; the other stage outputs were checked structurally and through the regenerated statistical summaries.

## Scientific and numerical corrections made

1. The abstract now limits the grid-refinement claim to the four blade-array tests actually evaluated. Their maximum change is 0.06 log10 units; the Sharklet AF-type cell changes by 0.15.
2. The maximum time-step effect is reported as 0.0015 log10 units, consistent with the saved maximum of 0.001483.
3. The geometric Sobol analysis is described as uniform sampling over bounded parameter supports. It is not presented as a variance decomposition under the truncated-normal production priors.
4. The 12-reconstruction time-course analysis and eight-reconstruction extension to 72 model-time units are distinguished.
5. The 128 A/B settings of the Saltelli design are described as parameter settings, not independent posterior draws.
6. The P. aeruginosa mechanistic interval was recalculated by refitting the shared endpoint exponent within each of the 128 matched A/B coefficient settings and redrawing the two observed reductions over 200,000 Monte Carlo draws. The resulting interval is 0.87–1.00, distinct from the 0.88–0.99 baseline endpoint interval.
7. The supplementary model specification now defines the annular appendage descriptor, logistic gate widths, nonlinear EPS step-size scale, EPS field units, curvature smoothing scale and shared parameter units.
8. The EPS operator is described as implemented. It is a yield-gated nonlinear diffusion term without an outer divergence operator; it is not described as a conservative flux.
9. The retained-activity symbol is separated from the arrival weighting, the finite-cell pitch-rounding bound is given for both lattice directions, and the supplementary numerical-verification table is kept together on one page.
10. The discussion was revised to connect the geometric mechanism, species contrast, endpoint mapping, and material-chemistry limitation without compressed or ambiguous transitions.
11. The micro sign in micrometre units was rendering incorrectly in the XeLaTeX PDFs. The source now declares an explicit `\textmu`-based SI unit; the rebuilt PDFs display and extract `µm` correctly.
12. Figure 6 was enlarged to improve legibility and use of the manuscript page.

## Manuscript and submission package

- Main manuscript, supplementary material and cover letter were rebuilt from the LaTeX sources with XeLaTeX.
- The main manuscript retains the funding statement. Ethics, data accessibility, competing-interest and author-contribution declarations are supplied separately in `SUBMISSION_METADATA.md`, consistent with the journal's submission workflow.
- The AI-use statement in `SUBMISSION_METADATA.md` names Claude and ChatGPT and describes their code and language contributions. The authors must confirm its accuracy before submission.
- The cover letter includes a concise statement of journal fit and the public code repository URL.
- The final source scan found no TODO, TBD, INSERT HERE or similar manuscript placeholders.

## Repository and release state

The active repository layout is prepared for a major version change to `v2.0.0`. The previous v1.0.0 release and its DOI `10.5281/zenodo.19316098` refer to the spike-based predecessor, not this manuscript. The new `.zenodo.json` and `CITATION.cff` identify version 2.0.0 and intentionally omit a new DOI until Zenodo archives the new GitHub release.

The GitHub remote has not been updated from this environment. Use `INSTALL_JRSI_V2.ps1` to install the prepared tree while preserving the existing `.git` directory and backing up the previous working files. Then run `PUBLISH_GITHUB.ps1` to review, commit and push `main`, create and push tag `v2.0.0`, and optionally create the GitHub release through `gh`.

## Items requiring author confirmation

1. **Bianca Cruz Neves's affiliation:** the current manuscript and v2 metadata list Federal University of Pará. The previous Zenodo metadata listed the Institute of Chemistry, Federal University of Rio de Janeiro. Confirm the correct current affiliation and make the manuscript, CFF and Zenodo metadata agree.
2. **AI-use declaration:** confirm that the description of Claude and ChatGPT assistance is complete and accurate for both authors.
3. **CRediT roles:** confirm the roles listed in `SUBMISSION_METADATA.md` before entering them in ScholarOne.
4. **Zenodo DOI:** create the GitHub release, allow Zenodo to archive it, and add the newly minted v2.0.0 DOI to the data-accessibility statement and citation metadata. Do not reuse the v1.0.0 DOI.
