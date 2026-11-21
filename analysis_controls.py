"""Arrival-formulation, spacing, independent-coefficient and geometric control experiments."""
import sys
from dataclasses import replace
from multiprocessing import Pool, freeze_support
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_v2 as A
from snake_model.model import SPECIES, BladeGeometry, FlatCache, ModelConfig, build_blade_surface, ratio

OUT = Path("results_v2/controls")
MODES = ("dual", "supply", "capture")
_C = None


def work(args):
    global _C
    if _C is None:
        _C = FlatCache()
    tag, g, modes, dx = args
    base = ModelConfig(dx=dx, record_times=(1.0, 2.0, 4.5))
    z, info = build_blade_surface(g, base)
    rows = []
    for m in modes:
        cfg = replace(base, arrival_mode=m)
        for n, sp in SPECIES.items():
            rows.append({**tag, "mode": m, "species": n, "Y": ratio(z, sp, cfg, _C)["Y"],
                         "blade_fraction": info["blade_fraction"], "base_width": g.base_width})
    return rows


def run(jobs, name):
    out = []
    with Pool(2) as p:
        for k, r in enumerate(p.imap_unordered(work, jobs, chunksize=1)):
            out.append(r)
            if k % 10 == 0:
                print(name, k, len(jobs), flush=True)
    df = pd.DataFrame([r for rr in out for r in rr])
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(name, len(df), flush=True)


def match_width(g, target, dx=0.1, lo=1.6, hi=3.4, tol=1e-4):
    cfg = ModelConfig(dx=dx)
    f = lambda w: build_blade_surface(replace(g, base_width=w), cfg)[1]["blade_fraction"]
    flo, fhi = f(lo), f(hi)
    if not flo <= target <= fhi:
        return None
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm - target) < tol:
            return mid
        lo, hi = (mid, hi) if fm < target else (lo, mid)
    return 0.5 * (lo + hi)


def _mw(args):
    return args[0], match_width(*args[1:])



# Top-level worker required for Windows spawn multiprocessing.
def indep_worker(args):
    from snake_model.model import build_flat_surface, flat_reference_energy, simulate
    k, g, mult = args
    cfg0 = ModelConfig(dx=0.15, record_times=(1.0, 2.0, 4.5))
    z, _ = build_blade_surface(g, cfg0)
    zf = build_flat_surface(cfg0)
    rows = []
    for n, sp0 in SPECIES.items():
        m = dict(mult[n]); beta = m.pop("beta"); fadh = m.pop("flat_adhesion")
        cfg = replace(cfg0, beta=cfg0.beta * beta)
        sp = replace(sp0, **{f: getattr(sp0, f) * v for f, v in m.items()})
        fe = flat_reference_energy(sp, cfg)
        bp = simulate(z, sp, cfg, fe)["B"]
        bf = simulate(zf, replace(sp, body_adhesion=sp.body_adhesion * fadh), cfg, fe)["B"]
        rows.append({"scenario": k, "species": n, "Y": bp / bf, "beta_mult": beta, "flat_adh_mult": fadh})
    return rows


def main(argv=None):
    """Run one analysis stage, without spawning processes during import."""
    args = sys.argv[1:] if argv is None else argv
    allowed = ("structural", "spacing", "coverage", "factorial", "phase", "indep", "grid", "matched_grid")
    if len(args) != 1 or args[0] not in allowed:
        raise SystemExit("Usage: python analysis_controls.py <" + "|".join(allowed) + ">")
    OUT.mkdir(parents=True, exist_ok=True)
    stage = args[0]
    if stage == "structural":
        run([({"surface": i}, g, MODES, 0.1) for i, g in enumerate(A.ensemble_geometries(40))], "structural")
    elif stage == "spacing":
        jobs = [({"pitch_a": pa, "seed": s}, BladeGeometry(pitch_a=pa, tip_height=1.5, lattice_seed=s, rough_seed=s + 1), MODES, 0.1)
                for pa in (3.4, 3.9, 4.4, 4.9, 5.4, 5.9) for s in (11, 22, 33, 44)]
        run(jobs, "spacing")
    elif stage == "coverage":
        specs = []
        for pit, target in ((True, 0.35), (False, 0.46)):
            for pa in (3.8, 4.4, 5.2):
                for s in (11, 22, 33, 44, 55, 66, 77, 88):
                    g = BladeGeometry(pitch_a=pa, lattice_seed=s, rough_seed=s + 1, pit_depth=1.5 if pit else 0.0)
                    specs.append(((pit, pa, s, g), g, target))
        with Pool(2) as p:
            widths = p.map(_mw, specs, chunksize=1)
        jobs = []
        for (pit, pa, s, g), w in widths:
            if w is None:
                print("infeasible", pit, pa, s, flush=True)
                continue
            jobs.append(({"pit": pit, "pitch_a": pa, "seed": s}, replace(g, base_width=w), MODES, 0.1))
        run(jobs, "coverage")
    elif stage == "factorial":
        rng = np.random.default_rng(777)
        nuis = [A.sample_geometry(rng, k) for k in range(8)]
        jobs = []
        for k, g0 in enumerate(nuis):
            for pa in (3.8, 4.4, 5.0):
                for h in (0.5, 1.5, 3.0):
                    for d in (0.5, 2.5):
                        jobs.append(({"nuis": k, "pitch_a": pa, "tip_height": h, "pit_depth": d},
                                     replace(g0, pitch_a=pa, tip_height=h, pit_depth=d), ("dual",), 0.1))
        run(jobs, "factorial")
    elif stage == "phase":
        jobs = [({"phase": ph, "pit": pit, "seed": s},
                 BladeGeometry(row_phase=ph, lattice_seed=s, rough_seed=s + 1, pit_depth=1.5 if pit else 0.0), ("dual",), 0.1)
                for ph in (0.0, 0.25, 0.5) for pit in (True, False) for s in (11, 22, 33, 44, 55, 66, 77, 88)]
        run(jobs, "phase")
    elif stage == "indep":
        SP_FIELDS = ["body_adhesion", "adhesion_range", "transport_slope_penalty", "transport_relief_penalty",
                     "steric_penalty", "appendage_strength", "detach_rate", "anchor_rate", "eps_activation_rate"]
        rng = np.random.default_rng(20261120)
        jobs = []
        for k in range(160):
            g = A.sample_geometry(rng, k)
            mult = {n: {f: float(rng.uniform(0.8, 1.2)) for f in SP_FIELDS + ["beta", "flat_adhesion"]} for n in SPECIES}
            jobs.append((k, g, mult))

        with Pool(2) as p:
            out = [r for rr in p.imap_unordered(indep_worker, jobs, chunksize=2) for r in rr]
        pd.DataFrame(out).to_csv(OUT / "indep.csv", index=False)
        print("indep", len(out), flush=True)
    elif stage == "grid":
        G = {"close_nominal": BladeGeometry(pitch_a=3.4, lattice_seed=11, rough_seed=12),
             "close_crowded_tall": BladeGeometry(pitch_a=3.4, tip_height=3.0, base_width=3.2, lattice_seed=11, rough_seed=12),
             "close_short_narrow": BladeGeometry(pitch_a=3.4, tip_height=0.5, base_width=1.6, lattice_seed=11, rough_seed=12),
             "wide_nominal": BladeGeometry(pitch_a=5.9, lattice_seed=11, rough_seed=12)}
        jobs = [({"geometry": k, "dx": dx}, g, ("dual",), dx) for k, g in G.items() for dx in (0.15, 0.10, 0.07)]
        jobs += [({"geometry": k, "dx": 0.05}, G[k], ("dual",), 0.05) for k in ("close_nominal", "close_crowded_tall")]
        run(jobs[::-1], "grid")
    elif stage == "matched_grid":
        cov = pd.read_csv(OUT / "coverage.csv").drop_duplicates(["pit", "pitch_a", "seed"])
        jobs = []
        for _, r in cov[cov.seed.isin([11, 22, 33, 44]) & cov.pitch_a.isin([3.8, 5.2])].iterrows():
            g = BladeGeometry(pitch_a=r.pitch_a, lattice_seed=int(r.seed), rough_seed=int(r.seed) + 1,
                              pit_depth=1.5 if r.pit else 0.0, base_width=float(r.base_width))
            for dx in (0.10, 0.07):
                jobs.append(({"pit": bool(r.pit), "pitch_a": r.pitch_a, "seed": int(r.seed), "dx": dx}, g, ("dual",), dx))
        run(jobs, "matched_grid")


if __name__ == "__main__":
    freeze_support()
    main()
