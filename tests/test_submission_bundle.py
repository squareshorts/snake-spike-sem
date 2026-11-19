import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results_v2" / "data"


def test_saved_simulation_tables_have_expected_evaluation_counts():
    expected_rows = {
        "ensemble.csv": 120,
        "time_course.csv": 36,
        "sharklet.csv": 3,
        "numerics.csv": 108,
        "numerics_sharklet.csv": 15,
        "knockouts.csv": 252,
        "design.csv": 486,
        "chemistry.csv": 120,
        "longtime.csv": 24,
        "gsa_geometry.csv": 2496,
        "gsa_mechanism.csv": 5376,
    }
    for name, n_rows in expected_rows.items():
        frame = pd.read_csv(DATA / name)
        assert len(frame) == n_rows, name


def test_summary_reports_40_surfaces_and_subunit_occupancy():
    summary = json.loads((ROOT / "results_v2" / "summary.json").read_text(encoding="utf-8"))
    assert summary["ensemble_n"] == 40
    assert summary["max_occupancy"] < 1.0
    assert summary["max_occupancy"] > 0.0


def test_generated_time_step_precision_does_not_understate_saved_maximum():
    summary = json.loads((ROOT / "results_v2" / "summary.json").read_text(encoding="utf-8"))
    maximum = max(
        summary["numerics"]["max_abs_dL_time_0.1"],
        summary["numerics"]["max_abs_dL_time_0.025"],
    )
    numbers = (ROOT / "manuscript" / "numbers.tex").read_text(encoding="utf-8")
    import re
    match = re.search(r"\\newcommand\{\\NumTime\}\{([^}]+)\}", numbers)
    assert match is not None
    assert float(match.group(1)) >= maximum
    assert float(match.group(1)) - maximum < 0.0001


def test_pseudomonas_mechanistic_interval_uses_matched_endpoint_refits():
    summary = json.loads((ROOT / "results_v2" / "summary.json").read_text(encoding="utf-8"))
    mech = summary["mechanistic_uncertainty"]
    assert mech["pred_Pa_mech_n_coeff_settings"] == 128
    assert mech["pred_Pa_mech_n_mc"] == 200_000
    lo, hi = mech["pred_Pa_mech_ci"]
    assert 0.85 < lo < 0.88
    assert 0.99 < hi < 1.02
    numbers = (ROOT / "manuscript" / "numbers.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\PredPaMechCI}{0.87--1.00}" in numbers
