"""Release regression checks for saved geometry-control output in the corrected v2.2.1 release.

These are dataset checks, not independent reruns of the numerical solver.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results_v2' / 'controls'


def read_control(name):
    df = pd.read_csv(DATA / f'{name}.csv')
    assert np.isfinite(df['Y']).all()
    assert (df['Y'] > 0).all()
    return df.assign(L=-np.log10(df['Y']))


def test_matched_coverage_reduction_and_area_matching():
    d = read_control('coverage')
    assert len(d) == 432
    expected_coverage = np.where(d['pit'], 0.35, 0.46)
    assert np.max(np.abs(d['blade_fraction'] - expected_coverage)) < 0.0002
    assert d['base_width'].between(1.6, 3.4).all()
    p = d.pivot(index=['pit', 'seed', 'mode', 'species'], columns='pitch_a', values='L')
    delta = p[3.8] - p[5.2]
    assert len(delta) == 144
    assert np.isfinite(delta).all()
    assert (delta > 0).all()


def test_three_dimensional_factorial_paired_direction():
    d = read_control('factorial')
    assert len(d) == 432
    p = d.pivot(index=['nuis', 'tip_height', 'pit_depth', 'species'], columns='pitch_a', values='L')
    delta = p[3.8] - p[5.0]
    assert len(delta) == 144
    assert (delta > 0).all()
    for species in ['E. coli', 'S. aureus', 'P. aeruginosa']:
        ds = delta.xs(species, level='species')
        assert len(ds) == 48
        assert 0.5 < ds.mean() < 1.0


def test_independent_species_sensitivity_not_forced():
    d = read_control('indep')
    assert len(d) == 480
    p = d.pivot(index='scenario', columns='species', values='L')
    assert len(p) == 160
    assert (p['E. coli'] > p['S. aureus']).sum() == 148
    assert (p > 0).all().all()


def test_row_phase_mean_is_small_not_identically_zero():
    d = read_control('phase')
    assert len(d) == 144
    p = d.pivot(index=['pit', 'seed', 'species'], columns='phase', values='L')
    delta = p[0.5] - p[0.0]
    assert len(delta) == 48
    assert abs(delta.mean()) < 0.002
    assert delta.abs().max() < 0.012
    assert delta.abs().max() > 0.010


def test_matched_spacing_contrast_survives_grid_refinement():
    d = read_control('matched_grid')
    assert len(d) == 96
    p = d.pivot(index=['pit', 'seed', 'species', 'dx', 'mode'], columns='pitch_a', values='L')
    delta = p[3.8] - p[5.2]
    assert len(delta) == 48
    assert (delta > 0).all()
    q = delta.unstack('dx')
    assert q.shape[0] == 24
    assert (q[0.07] > 0).all()
    assert (q[0.10] > 0).all()
    assert (q[0.07] - q[0.10]).abs().max() < 0.06


def test_release_scripts_are_safe_and_version_consistent():
    publish = (ROOT / 'PUBLISH_GITHUB.ps1').read_text(encoding='utf-8')
    installer = (ROOT / 'INSTALL_JRSI_V2.ps1').read_text(encoding='utf-8')
    runner = (ROOT / 'run_all.ps1').read_text(encoding='utf-8')
    controls = (ROOT / 'analysis_controls.py').read_text(encoding='utf-8')
    assert "'v2.2.1'" in publish
    assert '& git -C $RepoRoot add -A' not in publish
    assert 'pathspec-from-file' in publish
    assert 'PREVIEW ONLY' in publish
    assert 'test_geometry_regression.py' in publish
    assert '[switch]$Install' in installer
    assert 'PREVIEW ONLY' in installer
    assert 'Move-Item' not in installer
    assert '.venv\\Scripts\\python.exe' in runner
    assert 'if __name__ == "__main__":' in controls
