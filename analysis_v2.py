#!/usr/bin/env python3
"""Analyses for the revised manuscript (model v2).

Usage: python analysis_v2.py --stage <name> [--outdir results_v2]
Stages: numerics, ensemble, knockouts, design, chemistry, longtime, gsa_geometry,
gsa_mechanism, all. Each stage writes its CSV files to results_v2/data. The geometric
Sobol design samples each input uniformly over its bounded support; the production
ensemble uses the parameter-specific priors declared in GEOM_PRIOR.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from snake_model.model import (SPECIES, BladeGeometry, FlatCache, ModelConfig, build_blade_surface,
                            build_sharklet_af, ratio)

OUT = Path("results_v2")
NPROC = int(os.environ.get("NPROC", "2"))
PROD_DX = 0.1
SCREEN_DX = 0.2
SP_NAMES = list(SPECIES)

# Geometric prior.  Measured quantities (Peroutka et al. 2026) enter as normal priors with
# the reported SD; unmeasured quantities enter as wide uniform priors.
GEOM_PRIOR = {
    "pitch_f": ("normal", 4.6, 0.2, 4.0, 5.2),
    "pitch_a": ("normal", 4.4, 0.6, 3.2, 6.0),
    "length": ("normal", 9.0, 0.7, 7.0, 11.0),
    "base_width": ("uniform", 1.6, 3.4),
    "taper": ("uniform", 0.7, 1.5),
    "tip_height": ("uniform", 0.5, 3.0),
    "pit_a": ("uniform", 0.9, 1.7),
    "pit_b": ("uniform", 0.5, 0.85),
    "pit_depth": ("uniform", 0.5, 2.5),
    "jitter": ("uniform", 0.05, 0.25),
    "rough_amp": ("uniform", 0.02, 0.15),
}


def sample_geometry(rng, idx):
    kw = {}
    for k, spec in GEOM_PRIOR.items():
        if spec[0] == "normal":
            _, mu, sd, lo, hi = spec
            v = rng.normal(mu, sd)
            while not lo <= v <= hi:
                v = rng.normal(mu, sd)
        else:
            v = rng.uniform(spec[1], spec[2])
        kw[k] = float(v)
    kw["lattice_seed"] = int(rng.integers(1, 2**31 - 1))
    kw["rough_seed"] = int(rng.integers(1, 2**31 - 1))
    return BladeGeometry(**kw)


# ----------------------------------------------------------------------------- workers
_CACHE = None


def _cache():
    global _CACHE
    if _CACHE is None:
        _CACHE = FlatCache()
    return _CACHE


def _eval_geom(args):
    tag, g, cfg, sp_names, sp_overrides = args
    z, info = build_blade_surface(g, cfg)
    rows = []
    for name in sp_names:
        sp = replace(SPECIES[name], **sp_overrides.get(name, {})) if sp_overrides else SPECIES[name]
        r = ratio(z, sp, cfg, _cache())
        rows.append({**tag, "species": name, **{f"g_{k}": v for k, v in asdict(g).items()}, 
                     "occupied": info["occupied"], "pit_fraction": info["pit_fraction"], "relief": info["relief"], **r})
    return rows


def _eval_surface(args):
    tag, z, cfg, sp_list = args
    rows = []
    for name, sp in sp_list:
        r = ratio(z, sp, cfg, _cache())
        rows.append({**tag, "species": name, **r})
    return rows


def run_pool(fn, jobs):
    t = time.time()
    with Pool(NPROC) as pool:
        out = pool.map(fn, jobs, chunksize=1)
    print(f"  {len(jobs)} jobs in {time.time() - t:.0f}s", flush=True)
    return pd.DataFrame([r for rows in out for r in rows])


# ------------------------------------------------------------------------------ stages
def ensemble_geometries(n=40, seed=2026):
    rng = np.random.default_rng(seed)
    return [sample_geometry(rng, i) for i in range(n)]


def stage_numerics(out):
    geoms = [BladeGeometry()] + ensemble_geometries(3, seed=77)
    jobs = []
    for gi, g in enumerate(geoms):
        for dx in (0.3, 0.2, 0.15, 0.1, 0.07):
            jobs.append(({"test": "grid", "surface": gi, "dx": dx, "h": 0.05, "domain": 36.0},
                         g, ModelConfig(dx=dx, record_times=(1.0, 2.0, 4.5)), SP_NAMES, None))
        for h in (0.1, 0.025):
            jobs.append(({"test": "time", "surface": gi, "dx": 0.15, "h": h, "domain": 36.0},
                         g, ModelConfig(dx=0.15, h_state=h, record_times=(1.0, 2.0, 4.5)), SP_NAMES, None))
        for dom in (24.0, 48.0):
            jobs.append(({"test": "domain", "surface": gi, "dx": 0.15, "h": 0.05, "domain": dom},
                         g, ModelConfig(dx=0.15, domain_x=dom, domain_y=dom, record_times=(1.0, 2.0, 4.5)), SP_NAMES, None))
    jobs.sort(key=lambda j: j[2].dx)  # expensive jobs first finish last; fine
    df = run_pool(_eval_geom, jobs)
    df.to_csv(out / "numerics.csv", index=False)
    # Sharklet at several grids
    sj = []
    for dx in (0.3, 0.2, 0.15, 0.1, 0.07):
        cfg = ModelConfig(dx=dx, record_times=(1.0, 2.0, 4.5))
        sj.append(({"test": "grid", "dx": dx}, build_sharklet_af(cfg), cfg, list(SPECIES.items())))
    run_pool(_eval_surface, sj).to_csv(out / "numerics_sharklet.csv", index=False)


def stage_ensemble(out, n=40):
    geoms = ensemble_geometries(n)
    cfg = ModelConfig(dx=PROD_DX, record_times=(1.0, 2.0, 4.5))
    jobs = [({"surface": i}, g, cfg, SP_NAMES, None) for i, g in enumerate(geoms)]
    run_pool(_eval_geom, jobs).to_csv(out / "ensemble.csv", index=False)
    cfg2 = ModelConfig(dx=PROD_DX, record_times=(0.5, 1.0, 2.0, 4.5, 9.0))
    z = build_sharklet_af(cfg2)
    run_pool(_eval_surface, [({"surface": "sharklet_af"}, z, cfg2, list(SPECIES.items()))]).to_csv(out / "sharklet.csv", index=False)
    # time-robustness on a subset of the ensemble
    cfgt = ModelConfig(dx=0.15, record_times=(0.5, 1.0, 2.0, 4.5, 9.0))
    jobs = [({"surface": i}, g, cfgt, SP_NAMES, None) for i, g in enumerate(geoms[:12])]
    run_pool(_eval_geom, jobs).to_csv(out / "time_course.csv", index=False)


KNOCKOUTS = {
    "full": {},
    "no_appendages": {"use_appendages": False},
    "no_compliance": {"use_compliance": False},
    "no_slope": {"use_slope": False},
    "no_steric": {"use_steric": False},
    "no_transport": {"use_transport": False},
    "no_dynamic_eps": {"use_dynamic_eps": False},
}


def stage_knockouts(out, n=12):
    geoms = ensemble_geometries(40)[:n]
    jobs = []
    for name, upd in KNOCKOUTS.items():
        cfg = replace(ModelConfig(dx=0.15, record_times=(1.0, 2.0, 4.5)), **upd)
        jobs += [({"surface": i, "variant": name}, g, cfg, SP_NAMES, None) for i, g in enumerate(geoms)]
    run_pool(_eval_geom, jobs).to_csv(out / "knockouts.csv", index=False)


def stage_design(out, seeds=(11, 22, 33)):
    cfg = ModelConfig(dx=SCREEN_DX, record_times=(1.0, 2.0, 4.5))
    jobs = []
    for h in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        for b in (1.4, 1.8, 2.2, 2.6, 3.0, 3.4):
            for s in seeds:  # common random numbers: identical seeds at every design point
                g = BladeGeometry(tip_height=h, base_width=b, lattice_seed=s, rough_seed=s + 1)
                jobs.append(({"tip_height": h, "base": b, "seed": s}, g, cfg, SP_NAMES, None))
    for pa in (3.4, 3.9, 4.4, 4.9, 5.4, 5.9):
        for h in (0.5, 1.5, 3.0):
            for s in seeds:
                g = BladeGeometry(pitch_a=pa, tip_height=h, lattice_seed=s, rough_seed=s + 1)
                jobs.append(({"pitch_a": pa, "tip_height": h, "seed": s, "series": "pitch"}, g, cfg, SP_NAMES, None))
    run_pool(_eval_geom, jobs).to_csv(out / "design.csv", index=False)


def saltelli(d, n, seed):
    """Saltelli A, B and AB_i matrices from a scrambled Sobol' sequence on [0,1]^d."""
    s = qmc.Sobol(2 * d, scramble=True, seed=seed).random(n)
    A, B = s[:, :d], s[:, d:]
    AB = [np.where(np.arange(d) == i, B, A) for i in range(d)]
    return A, B, AB


GSA_GEOM = {k: (v[3], v[4]) if v[0] == "normal" else (v[1], v[2]) for k, v in GEOM_PRIOR.items()}


def stage_gsa_geometry(out, n=64):
    keys = list(GSA_GEOM)
    A, B, AB = saltelli(len(keys), n, seed=5)
    cfg = ModelConfig(dx=SCREEN_DX, record_times=(1.0, 2.0, 4.5))
    jobs = []
    for block, M in [("A", A), ("B", B)] + [(f"AB{i}", m) for i, m in enumerate(AB)]:
        for r, u in enumerate(M):
            v = {k: lo + (hi - lo) * x for (k, (lo, hi)), x in zip(GSA_GEOM.items(), u)}
            g = BladeGeometry(lattice_seed=101, rough_seed=102, **v)
            jobs.append(({"block": block, "row": r}, g, cfg, SP_NAMES, None))
    run_pool(_eval_geom, jobs).to_csv(out / "gsa_geometry.csv", index=False)


# Mechanistic multipliers (log-uniform on [1/2, 2]); each multiplies the coefficient for
# all three species, so that prescribed species differences are preserved.
GSA_MECH = {
    "body_adhesion": ["body_adhesion"], "adhesion_range": ["adhesion_range"], "compliance": ["compliance"],
    "steric_penalty": ["steric_penalty"], "slope_penalty": ["slope_penalty"],
    "appendage": ["appendage_strength", "appendage_binding"],
    "transport_penalty": ["transport_slope_penalty", "transport_relief_penalty"],
    "detach_rate": ["detach_rate"], "anchor_rate": ["anchor_rate"],
    "eps_kinetics": ["eps_activation_rate", "maturation_rate", "eps_production"],
    "capture_rate": "cfg", "beta": "cfg",
}


def _mech_job(args):
    tag, mult, dx = args
    cfg = ModelConfig(dx=dx, record_times=(1.0, 2.0, 4.5))
    cfg = replace(cfg, capture_rate=cfg.capture_rate * mult["capture_rate"], beta=cfg.beta * mult["beta"])
    sp_list = []
    for name, sp in SPECIES.items():
        upd = {}
        for k, fields in GSA_MECH.items():
            if fields == "cfg":
                continue
            for f in fields:
                upd[f] = getattr(sp, f) * mult[k]
        sp_list.append((name, replace(sp, **upd)))
    z_snake, _ = build_blade_surface(BladeGeometry(lattice_seed=101, rough_seed=102), cfg)
    z_shark = build_sharklet_af(cfg)
    rows = []
    for surf, z in (("snake", z_snake), ("sharklet", z_shark)):
        for name, sp in sp_list:
            r = ratio(z, sp, cfg, _cache())
            rows.append({**tag, "surface": surf, "species": name, **{f"m_{k}": v for k, v in mult.items()}, **r})
    return rows


def stage_gsa_mechanism(out, n=64, dx=SCREEN_DX):
    keys = list(GSA_MECH)
    A, B, AB = saltelli(len(keys), n, seed=9)
    jobs = []
    for block, M in [("A", A), ("B", B)] + [(f"AB{i}", m) for i, m in enumerate(AB)]:
        for r, u in enumerate(M):
            mult = {k: float(2.0 ** (2 * x - 1)) for k, x in zip(keys, u)}
            jobs.append(({"block": block, "row": r}, mult, dx))
    run_pool(_mech_job, jobs).to_csv(out / "gsa_mechanism.csv", index=False)


def stage_chemistry(out):
    """Material-chemistry sensitivity.  The assay compares keratin scales with polystyrene.
    The control surface is simulated with body adhesion scaled by f, while attachment
    propensities on both surfaces are referenced to the keratin flat-surface energy."""
    from snake_model.model import build_flat_surface, flat_reference_energy, simulate
    geoms = ensemble_geometries(40)[:8]
    cfg = ModelConfig(dx=0.15, record_times=(1.0, 2.0, 4.5))
    rows = []
    zf = build_flat_surface(cfg)
    flat = {}
    for name, sp in SPECIES.items():
        fe = flat_reference_energy(sp, cfg)
        for f in (0.8, 0.9, 1.0, 1.1, 1.25):
            flat[(name, f)] = simulate(zf, replace(sp, body_adhesion=sp.body_adhesion * f), cfg, fe)["B"]
    for i, g in enumerate(geoms):
        z, _ = build_blade_surface(g, cfg)
        for name, sp in SPECIES.items():
            bp = simulate(z, sp, cfg, flat_reference_energy(sp, cfg))["B"]
            for f in (0.8, 0.9, 1.0, 1.1, 1.25):
                rows.append({"surface": i, "species": name, "flat_adhesion_factor": f, "Y": bp / flat[(name, f)]})
    pd.DataFrame(rows).to_csv(out / "chemistry.csv", index=False)


def stage_longtime(out, n=8):
    """Long-horizon behaviour of the ratio (the model has no growth term)."""
    geoms = ensemble_geometries(40)[:n]
    cfg = ModelConfig(dx=0.15, record_times=(1.0, 4.5, 9.0, 18.0, 36.0, 72.0))
    jobs = [({"surface": i}, g, cfg, SP_NAMES, None) for i, g in enumerate(geoms)]
    run_pool(_eval_geom, jobs).to_csv(out / "longtime.csv", index=False)


STAGES = {"longtime": stage_longtime, "numerics": stage_numerics, "ensemble": stage_ensemble, "knockouts": stage_knockouts,
          "design": stage_design, "gsa_geometry": stage_gsa_geometry, "gsa_mechanism": stage_gsa_mechanism,
          "chemistry": stage_chemistry}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES) + ["all"])
    ap.add_argument("--outdir", type=Path, default=OUT)
    a = ap.parse_args()
    out = a.outdir / "data"
    out.mkdir(parents=True, exist_ok=True)
    for name in (STAGES if a.stage == "all" else [a.stage]):
        print(f"[{name}]", flush=True)
        STAGES[name](out)


if __name__ == "__main__":
    main()
