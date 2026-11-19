#!/usr/bin/env python3
"""Statistics for the revised manuscript.  Reads results_v2/data/*.csv, writes
results_v2/summary.json and results_v2/tables/*.csv."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("results_v2/data")
T = Path("results_v2/tables")
T.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261118)
SP = ["E. coli", "S. aureus", "P. aeruginosa"]

# Peroutka et al. (2026), Fig. 5, digitised (48 h, scale pair vs smooth polystyrene, three
# biological replicates).  Panel (a): CFU on scales as a percentage of polystyrene, mean +/- SD.
# Panel (b): log10 CFU/cm2, mean +/- SD (no error bar is drawn for polystyrene).
FIG5A = {"E. coli": (0.118, 0.013), "S. aureus": (0.224, 0.083)}
FIG5B = {"E. coli": (6.340, 5.547, 0.198), "S. aureus": (6.623, 5.901, 0.269)}
N_REP = 3
LR_OBS = {k: -np.log10(m) for k, (m, sd) in FIG5A.items()}
SE_OBS = {k: sd / (m * np.log(10) * np.sqrt(N_REP)) for k, (m, sd) in FIG5A.items()}
LR_OBS_B = {k: ref - pr for k, (ref, pr, sd) in FIG5B.items()}
SE_OBS_B = {k: np.sqrt(2) * sd / np.sqrt(N_REP) for k, (ref, pr, sd) in FIG5B.items()}  # polystyrene SD taken equal to scale SD
SIGMA_EXP = None
SIGMA_SENS = ("fig5a", "fig5b")

# Short-term attachment on Sharklet AF-type silicone versus smooth silicone.
SHARKLET_OBS = [
    {"study": "May et al. 2014", "species": "E. coli", "strain": "ATCC 700336", "LR": 3.51 - 1.21},
    {"study": "May et al. 2014", "species": "S. aureus", "strain": "MRSA ATCC 700698", "LR": 5.83 - 3.72},
    {"study": "May et al. 2014", "species": "P. aeruginosa", "strain": "ATCC 9027", "LR": 2.78 - 1.22},
    {"study": "May et al. 2014", "species": "P. aeruginosa", "strain": "ATCC 10197", "LR": 3.62 - 2.26},
    {"study": "May et al. 2014", "species": "P. aeruginosa", "strain": "PA14 dbifA", "LR": 3.96 - 1.72},
    {"study": "Xu et al. 2017", "species": "S. aureus", "strain": "ATCC 6538", "LR": -np.log10(1 - 0.923)},
    {"study": "Xu et al. 2017", "species": "P. aeruginosa", "strain": "ATCC 10197", "LR": -np.log10(1 - 0.999)},
]


def L_of(y):
    return -np.log10(np.mean(y))


def boot_L(df, col="Y", B=4000):
    """Bootstrap over reconstructed surfaces; returns dict species -> array of L."""
    piv = df.pivot(index="surface", columns="species", values=col)[SP]
    n = len(piv)
    idx = RNG.integers(0, n, size=(B, n))
    return {s: -np.log10(piv[s].to_numpy()[idx].mean(axis=1)) for s in SP}, piv


def ci(x, q=(2.5, 97.5)):
    lo, hi = np.percentile(x, q)
    return float(lo), float(hi)


def endpoint_fit(Lb, reading="fig5a"):
    """Weighted least squares through the origin, LR_obs = alpha * L, inside a Monte Carlo
    over geometry bootstrap and experimental error."""
    lr0, se = (LR_OBS, SE_OBS) if reading == "fig5a" else (LR_OBS_B, SE_OBS_B)
    B = len(Lb["E. coli"])
    lr = {s: lr0[s] + se[s] * RNG.standard_normal(B) for s in lr0}
    w = {s: 1 / se[s] ** 2 for s in lr0}
    num = sum(w[s] * lr[s] * Lb[s] for s in lr0)
    den = sum(w[s] * Lb[s] ** 2 for s in lr0)
    a = num / den
    a_ec = lr["E. coli"] / Lb["E. coli"]
    a_sa = lr["S. aureus"] / Lb["S. aureus"]
    out = {"alpha": a, "pred_Pa": a * Lb["P. aeruginosa"],
           "loso_pred_Sa_from_Ec": a_ec * Lb["S. aureus"], "loso_pred_Ec_from_Sa": a_sa * Lb["E. coli"],
           "lr0": lr0, "se": se}
    out["loso_pp_Sa"] = out["loso_pred_Sa_from_Ec"] + se["S. aureus"] * RNG.standard_normal(B)
    out["loso_pp_Ec"] = out["loso_pred_Ec_from_Sa"] + se["E. coli"] * RNG.standard_normal(B)
    rho_obs = lr["S. aureus"] / lr["E. coli"]
    out["rho_obs"] = rho_obs
    return out


def sobol_indices(df, keys_prefix, out_col, group_cols, B=1000):
    """Saltelli/Jansen estimators with bootstrap CIs. df has block (A, B, AB<i>) and row."""
    res = []
    for gkey, g in df.groupby(group_cols):
        fA = g[g.block == "A"].sort_values("row")[out_col].to_numpy()
        fB = g[g.block == "B"].sort_values("row")[out_col].to_numpy()
        n = len(fA)
        d = len(keys_prefix)
        fAB = [g[g.block == f"AB{i}"].sort_values("row")[out_col].to_numpy() for i in range(d)]
        bidx = RNG.integers(0, n, size=(B, n))
        for i, k in enumerate(keys_prefix):
            def est(ix):
                a, b, ab = fA[ix], fB[ix], fAB[i][ix]
                V = np.var(np.concatenate([a, b]), ddof=1)
                return np.mean(b * (ab - a)) / V, 0.5 * np.mean((a - ab) ** 2) / V
            s1, st = est(np.arange(n))
            bs = np.array([est(ix) for ix in bidx])
            res.append({**dict(zip(group_cols, gkey if isinstance(gkey, tuple) else (gkey,))), "parameter": k,
                        "S1": s1, "S1_lo": np.percentile(bs[:, 0], 2.5), "S1_hi": np.percentile(bs[:, 0], 97.5),
                        "ST": st, "ST_lo": np.percentile(bs[:, 1], 2.5), "ST_hi": np.percentile(bs[:, 1], 97.5)})
    return pd.DataFrame(res)


def main():
    S = {"LR_obs": LR_OBS, "SE_obs": SE_OBS}
    # ------------------------------------------------------------------ ensemble
    ens = pd.read_csv(D / "ensemble.csv")
    S["ensemble_n"] = int(ens.surface.nunique())
    S["max_occupancy"] = float(ens.occ_max.max())
    Lb, piv = boot_L(ens)
    rows = []
    for s in SP:
        y = piv[s].to_numpy()
        Ls = -np.log10(y)
        rows.append({"species": s, "Y_ensemble": float(y.mean()), "L": L_of(y), "L_lo": ci(Lb[s])[0], "L_hi": ci(Lb[s])[1],
                     "L_surface_median": float(np.median(Ls)), "L_surface_min": float(Ls.min()), "L_surface_max": float(Ls.max()),
                     "L_surface_sd": float(Ls.std(ddof=1))})
    tab_ens = pd.DataFrame(rows)
    tab_ens.to_csv(T / "ensemble_L.csv", index=False)
    S["ensemble"] = tab_ens.set_index("species").to_dict(orient="index")
    # running ensemble mean (in a fixed order) for convergence with N
    run = []
    for s in SP:
        y = piv[s].to_numpy()
        for n in range(4, len(y) + 1, 4):
            run.append({"species": s, "N": n, "L": L_of(y[:n])})
    pd.DataFrame(run).to_csv(T / "ensemble_running.csv", index=False)
    # half-split agreement
    half = len(piv) // 2
    S["half_split_L"] = {s: [L_of(piv[s].to_numpy()[:half]), L_of(piv[s].to_numpy()[half:])] for s in SP}

    # Zero-parameter species contrast (independent of any monotone power-law endpoint map)
    rho = Lb["S. aureus"] / Lb["E. coli"]
    S["obs"] = {"fig5a": {"LR": LR_OBS, "SE": SE_OBS, "Y": {k: v[0] for k, v in FIG5A.items()}, "Y_sd": {k: v[1] for k, v in FIG5A.items()}},
                "fig5b": {"LR": LR_OBS_B, "SE": SE_OBS_B}}
    S["contrast"] = {"rho_model": float(S["ensemble"]["S. aureus"]["L"] / S["ensemble"]["E. coli"]["L"]), "rho_model_ci": ci(rho),
                     "p_model_order_correct": float(np.mean(Lb["S. aureus"] < Lb["E. coli"]))}
    for rd, (lr0, se) in {"fig5a": (LR_OBS, SE_OBS), "fig5b": (LR_OBS_B, SE_OBS_B)}.items():
        mc = (lr0["S. aureus"] + se["S. aureus"] * RNG.standard_normal(20000)) / (lr0["E. coli"] + se["E. coli"] * RNG.standard_normal(20000))
        S["contrast"][f"rho_obs_{rd}"] = float(lr0["S. aureus"] / lr0["E. coli"])
        S["contrast"][f"rho_obs_ci_{rd}"] = ci(mc)
        S["contrast"][f"model_pct_{rd}"] = float(np.mean(mc < S["contrast"]["rho_model"]))
    # zero-parameter magnitude (alpha = 1): model L against observed LR
    S["alpha1"] = {s_: {"L": S["ensemble"][s_]["L"], "L_ci": [S["ensemble"][s_]["L_lo"], S["ensemble"][s_]["L_hi"]],
                        "LR_a": LR_OBS[s_], "LR_b": LR_OBS_B[s_]} for s_ in LR_OBS}

    # ------------------------------------------------------------- endpoint exponent
    S["endpoint"] = {}
    for rd in SIGMA_SENS:
        fit = endpoint_fit(Lb, rd)
        S["endpoint"][rd] = {
            "alpha": float(np.median(fit["alpha"])), "alpha_ci": ci(fit["alpha"]),
            "pred_LR_Pa": float(np.median(fit["pred_Pa"])), "pred_LR_Pa_ci": ci(fit["pred_Pa"]),
            "pred_remaining_Pa": float(10 ** -np.median(fit["pred_Pa"])),
            "pred_remaining_Pa_ci": [float(10 ** -c) for c in ci(fit["pred_Pa"])[::-1]],
            "loso_Sa_from_Ec": float(np.median(fit["loso_pred_Sa_from_Ec"])), "loso_Sa_pp_ci": ci(fit["loso_pp_Sa"]),
            "loso_Ec_from_Sa": float(np.median(fit["loso_pred_Ec_from_Sa"])), "loso_Ec_pp_ci": ci(fit["loso_pp_Ec"]),
            "loso_Sa_obs_pct": float(np.mean(fit["loso_pp_Sa"] < fit["lr0"]["S. aureus"])),
            "loso_Ec_obs_pct": float(np.mean(fit["loso_pp_Ec"] < fit["lr0"]["E. coli"])),
        }
    a_hat = S["endpoint"]["fig5a"]["alpha"]

    # ------------------------------------------------------------------- sharklet
    sh = pd.read_csv(D / "sharklet.csv").set_index("species")
    obs = pd.DataFrame(SHARKLET_OBS)
    obs.to_csv(T / "sharklet_observed.csv", index=False)
    srows = []
    for s in SP:
        r = sh.loc[s]
        srows.append({"species": s, "L_t0.5": -np.log10(r["Y_t0.5"]), "L_t1": -np.log10(r["Y_t1"]), "L_t2": -np.log10(r["Y_t2"]),
                      "L_t4.5": -np.log10(r["Y_t4.5"]), "L_t9": -np.log10(r["Y_t9"]),
                      "pred_alpha1": -np.log10(r["Y_t1"]), "pred_alpha48h": a_hat * -np.log10(r["Y_t4.5"]),
                      "obs_min": obs[obs.species == s].LR.min(), "obs_max": obs[obs.species == s].LR.max()})
    tab_sh = pd.DataFrame(srows)
    tab_sh.to_csv(T / "sharklet_pred.csv", index=False)
    S["sharklet"] = tab_sh.set_index("species").to_dict(orient="index")
    S["sharklet_obs_range"] = [float(obs.LR.min()), float(obs.LR.max())]

    # ------------------------------------------------------------- time course
    tc = pd.read_csv(D / "time_course.csv")
    tcr = []
    for s in SP:
        g = tc[tc.species == s]
        for t in ("0.5", "1", "2", "4.5", "9"):
            tcr.append({"species": s, "t": float(t), "L": L_of(g[f"Y_t{t}"])})
    tct = pd.DataFrame(tcr)
    tct.to_csv(T / "time_course_L.csv", index=False)
    S["time_course"] = {s: dict(zip(tct[tct.species == s].t, tct[tct.species == s].L)) for s in SP}
    # contrast vs time
    S["contrast_vs_time"] = {str(t): float(S["time_course"]["S. aureus"][t] / S["time_course"]["E. coli"][t]) for t in (0.5, 1.0, 2.0, 4.5, 9.0)}

    # ---------------------------------------------------------------- numerics
    nu = pd.read_csv(D / "numerics.csv")
    nu["L"] = -np.log10(nu.Y)
    g = nu[nu.test == "grid"]
    ref = g[g.dx == 0.07].set_index(["surface", "species"]).L
    prod = g[g.dx == 0.1].set_index(["surface", "species"]).L
    d01 = (prod - ref).abs()
    S["numerics"] = {"max_abs_dL_dx0.1_vs_0.07": float(d01.max()), "mean_abs_dL_dx0.1_vs_0.07": float(d01.mean())}
    for dx in (0.3, 0.2, 0.15):
        dd = (g[g.dx == dx].set_index(["surface", "species"]).L - ref).abs()
        S["numerics"][f"max_abs_dL_dx{dx}_vs_0.07"] = float(dd.max())
    base15 = g[g.dx == 0.15].set_index(["surface", "species"]).L
    for test, col, vals in (("time", "h", (0.1, 0.025)), ("domain", "domain", (24.0, 48.0))):
        for v in vals:
            dd = (nu[(nu.test == test) & (nu[col] == v)].set_index(["surface", "species"]).L - base15).abs()
            S["numerics"][f"max_abs_dL_{test}_{v}"] = float(dd.max())
    nsh = pd.read_csv(D / "numerics_sharklet.csv")
    nsh["L"] = -np.log10(nsh.Y)
    r07 = nsh[nsh.dx == 0.07].set_index("species").L
    r10 = nsh[nsh.dx == 0.1].set_index("species").L
    S["numerics"]["sharklet_max_abs_dL_dx0.1_vs_0.07"] = float((r10 - r07).abs().max())
    # multi-fidelity offset between screening (0.2) and production (0.1) grids
    off = (g[g.dx == 0.2].set_index(["surface", "species"]).L - prod).groupby("species").agg(["mean", "std"])
    S["numerics"]["screen_offset_dL_0.2_minus_0.1"] = off.to_dict(orient="index")

    # --------------------------------------------------------------- knockouts
    ko = pd.read_csv(D / "knockouts.csv")
    ko["L"] = -np.log10(ko.Y)
    full = ko[ko.variant == "full"].set_index(["surface", "species"]).L
    krows = []
    for v in ko.variant.unique():
        if v == "full":
            continue
        dd = full - ko[ko.variant == v].set_index(["surface", "species"]).L  # positive: mechanism adds suppression
        for s in SP:
            x = dd.xs(s, level="species").to_numpy()
            bs = x[RNG.integers(0, len(x), (4000, len(x)))].mean(axis=1)
            krows.append({"variant": v, "species": s, "dL_mean": float(x.mean()), "lo": ci(bs)[0], "hi": ci(bs)[1]})
    tko = pd.DataFrame(krows)
    tko.to_csv(T / "knockouts_dL.csv", index=False)
    S["knockouts"] = {f"{r.variant}|{r.species}": [r.dL_mean, r.lo, r.hi] for r in tko.itertuples()}

    # ------------------------------------------------------------------ design
    de = pd.read_csv(D / "design.csv")
    de["L"] = -np.log10(de.Y)
    if "series" not in de.columns:
        de["series"] = np.nan
    dmap = de[de.series.isna()]
    dm = dmap.groupby(["species", "tip_height", "base"]).agg(L=("L", "mean"), L_sd=("L", "std"), occ=("occupied", "mean")).reset_index()
    dm.to_csv(T / "design_map.csv", index=False)
    dp = de[de.series == "pitch"].groupby(["species", "pitch_a", "tip_height"]).agg(L=("L", "mean"), L_sd=("L", "std")).reset_index()
    dp.to_csv(T / "design_pitch.csv", index=False)
    S["design_seed_sd_median"] = float(dm.L_sd.median())
    S["design"] = {}
    for s_ in SP:
        g2 = dm[dm.species == s_]
        S["design"][s_] = {"L_min": float(g2.L.min()), "L_max": float(g2.L.max()),
                           "L_h05_b22": float(g2[(g2.tip_height == 0.5) & (g2.base == 2.2)].L.iloc[0]),
                           "L_h30_b22": float(g2[(g2.tip_height == 3.0) & (g2.base == 2.2)].L.iloc[0]),
                           "L_h15_b14": float(g2[(g2.tip_height == 1.5) & (g2.base == 1.4)].L.iloc[0]),
                           "L_h15_b34": float(g2[(g2.tip_height == 1.5) & (g2.base == 3.4)].L.iloc[0])}
        g3 = dp[(dp.species == s_) & (dp.tip_height == 1.5)]
        S["design"][s_]["L_pa34"] = float(g3[g3.pitch_a == 3.4].L.iloc[0])
        S["design"][s_]["L_pa59"] = float(g3[g3.pitch_a == 5.9].L.iloc[0])

    # -------------------------------------------------------------------- GSA
    gg = pd.read_csv(D / "gsa_geometry.csv")
    gg["L"] = -np.log10(gg.Y)
    from analysis_v2 import GSA_GEOM
    keys = list(GSA_GEOM)
    sg = sobol_indices(gg, keys, "L", ["species"])
    sg.to_csv(T / "sobol_geometry.csv", index=False)
    S["sobol_geometry"] = {f"{r.species}|{r.parameter}": [r.S1, r.ST] for r in sg.itertuples()}
    if (D / "gsa_mechanism.csv").exists():
        gm = pd.read_csv(D / "gsa_mechanism.csv")
        gm["L"] = -np.log10(gm.Y)
        mkeys = [c[2:] for c in gm.columns if c.startswith("m_")]
        sm = sobol_indices(gm, mkeys, "L", ["surface", "species"])
        # contrast output (alpha-free): L_Sa / L_Ec on the snake surface
        snake = gm[gm.surface == "snake"].pivot_table(index=["block", "row"], columns="species", values="L").reset_index()
        snake["rho"] = snake["S. aureus"] / snake["E. coli"]
        snake["surface"] = "snake"
        snake["species"] = "contrast"
        sm2 = sobol_indices(snake, mkeys, "rho", ["surface", "species"])
        smech = pd.concat([sm, sm2])
        smech.to_csv(T / "sobol_mechanism.csv", index=False)
        S["sobol_mechanism"] = {f"{r.surface}|{r.species}|{r.parameter}": [r.S1, r.ST] for r in smech.itertuples()}
        # Mechanistic-uncertainty summaries over the 128 A/B settings of the scrambled
        # Saltelli design (64 settings in each base matrix).
        ind = gm[gm.block.isin(["A", "B"])]
        pp = {}
        for (surf, s), g3 in ind.groupby(["surface", "species"]):
            pp[f"{surf}|{s}"] = {"median": float(g3.L.median()), "q05": float(g3.L.quantile(0.05)), "q95": float(g3.L.quantile(0.95)),
                                 "frac_L_pos": float((g3.L > 0).mean())}
        ind_snake = snake[snake.block.isin(["A", "B"])]
        pp["snake|contrast"] = {"median": float(ind_snake.rho.median()), "q05": float(ind_snake.rho.quantile(0.05)),
                                "q95": float(ind_snake.rho.quantile(0.95)), "frac_order_correct": float((ind_snake.rho < 1).mean())}
        pivA = ind.pivot_table(index=["block", "row"], columns=["surface", "species"], values="L")
        sh_minus_snake = {s: (pivA[("sharklet", s)] - pivA[("snake", s)]) for s in SP}
        pp["sharklet_minus_snake_frac_pos"] = {s: float((v > 0).mean()) for s, v in sh_minus_snake.items()}
        S["mechanistic_uncertainty"] = pp
        # does mechanistic uncertainty change species ordering on snake?
        pp_order = ind.pivot_table(index=["block", "row"], columns=["surface", "species"], values="L")
        # Refit the shared endpoint exponent within each matched A/B coefficient setting.
        # The three species share the same coefficient multipliers in each row, so this
        # preserves the joint mechanistic perturbation when predicting P. aeruginosa.
        mech = gm[(gm.surface == "snake") & gm.block.isin(["A", "B"])].pivot_table(
            index=["block", "row"], columns="species", values="L"
        ).reindex(columns=SP)
        n_mech_mc = 200_000
        mech_idx = RNG.integers(0, len(mech), size=n_mech_mc)
        lm = mech.iloc[mech_idx]
        lr_ec = LR_OBS["E. coli"] + SE_OBS["E. coli"] * RNG.standard_normal(n_mech_mc)
        lr_sa = LR_OBS["S. aureus"] + SE_OBS["S. aureus"] * RNG.standard_normal(n_mech_mc)
        w_ec, w_sa = 1 / SE_OBS["E. coli"] ** 2, 1 / SE_OBS["S. aureus"] ** 2
        alpha_mech = (
            w_ec * lr_ec * lm["E. coli"].to_numpy()
            + w_sa * lr_sa * lm["S. aureus"].to_numpy()
        ) / (
            w_ec * lm["E. coli"].to_numpy() ** 2
            + w_sa * lm["S. aureus"].to_numpy() ** 2
        )
        pred_pa_mech = alpha_mech * lm["P. aeruginosa"].to_numpy()
        S["mechanistic_uncertainty"]["pred_Pa_mech_ci"] = ci(pred_pa_mech)
        S["mechanistic_uncertainty"]["pred_Pa_mech_median"] = float(np.median(pred_pa_mech))
        S["mechanistic_uncertainty"]["pred_Pa_mech_alpha_median"] = float(np.median(alpha_mech))
        S["mechanistic_uncertainty"]["pred_Pa_mech_n_coeff_settings"] = int(len(mech))
        S["mechanistic_uncertainty"]["pred_Pa_mech_n_mc"] = int(n_mech_mc)
        S["mechanistic_uncertainty"]["frac_Pa_ge_Ec_snake"] = float((pp_order[("snake", "P. aeruginosa")] >= pp_order[("snake", "E. coli")]).mean())


    # -------------------------------------------------------------- chemistry
    ch = pd.read_csv(D / "chemistry.csv")
    chm = ch.groupby(["species", "flat_adhesion_factor"]).Y.mean().reset_index()
    chm["L"] = -np.log10(chm.Y)
    chm.to_csv(T / "chemistry_L.csv", index=False)
    S["chemistry"] = {f"{r.species}|{r.flat_adhesion_factor}": float(r.L) for r in chm.itertuples()}

    # ------------------------------------------------------------- long horizon
    lt = pd.read_csv(D / "longtime.csv")
    S["longtime"] = {}
    for t in ("1", "4.5", "9", "18", "36", "72"):
        Ls = {s_: L_of(lt[lt.species == s_][f"Y_t{t}"]) for s_ in SP}
        S["longtime"][t] = {**Ls, "contrast": Ls["S. aureus"] / Ls["E. coli"]}
    # endpoint by time matching: model time at which L_Ec equals the observed E. coli reduction
    ts = np.array([36.0, 72.0]); le = np.array([S["longtime"]["36"]["E. coli"], S["longtime"]["72"]["E. coli"]])
    fr = (le[0] - LR_OBS["E. coli"]) / (le[0] - le[1])
    S["time_match"] = {"t_star": float(np.exp(np.log(36) + fr * np.log(2))),
                       **{s_: float(S["longtime"]["36"][s_] - fr * (S["longtime"]["36"][s_] - S["longtime"]["72"][s_])) for s_ in SP}}
    # chemistry shifts relative to f = 1
    S["chemistry_shift"] = {s_: {str(f): S["chemistry"][f"{s_}|{f}"] - S["chemistry"][f"{s_}|1.0"] for f in (0.8, 0.9, 1.1, 1.25)} for s_ in SP}
    # numerics: relative error at the production grid
    gg2 = g.pivot_table(index=["surface", "species"], columns="dx", values="L")
    S["numerics"]["max_rel_dL_dx0.1_vs_0.07"] = float(((gg2[0.1] - gg2[0.07]).abs() / gg2[0.07]).max())
    dom = nu[nu.test == "domain"].set_index(["surface", "species", "domain"]).L
    S["numerics"]["domain48_abs_dL"] = float(max(abs(dom.loc[(si, sp_, 48.0)] - base15.loc[(si, sp_)]) for si in range(4) for sp_ in SP))
    Path("results_v2/summary.json").write_text(json.dumps(S, indent=2, default=float))
    print(json.dumps({k: S[k] for k in ("ensemble", "contrast", "endpoint", "sharklet", "numerics")}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
