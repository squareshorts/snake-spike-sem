#!/usr/bin/env python3
"""Figures for the revised manuscript (reads results_v2/data, results_v2/tables, summary.json)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import LightSource
from snake_model.model import SPECIES, BladeGeometry, ModelConfig, attachment_layers, build_blade_surface, build_sharklet_af, flat_reference_energy

D, T, F = Path("results_v2/data"), Path("results_v2/tables"), Path("results_v2/figures")
F.mkdir(parents=True, exist_ok=True)
S = json.loads(Path("results_v2/summary.json").read_text())
SP = ["E. coli", "S. aureus", "P. aeruginosa"]
COL = {"E. coli": "#2a78d6", "S. aureus": "#eb6834", "P. aeruginosa": "#1baf7a"}
MK = {"E. coli": "o", "S. aureus": "s", "P. aeruginosa": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 400, "savefig.bbox": "tight", "lines.linewidth": 1.5})


def it(s):
    return f"$\\it{{{s.split()[0]}}}$ $\\it{{{s.split()[1]}}}$"


def panel(ax, letter):
    t = ax.get_title()
    ax.set_title(f"({letter})  {t}", loc="left", fontsize=8.5)
    ax.set_title("")


def save(fig, name):
    fig.savefig(F / f"{name}.pdf")
    fig.savefig(F / f"{name}.png", dpi=300)
    plt.close(fig)


def fig_surfaces():
    cfg = ModelConfig(dx=0.1)
    g = BladeGeometry()
    z, _ = build_blade_surface(g, cfg)
    zs = build_sharklet_af(cfg)
    fig, axs = plt.subplots(1, 5, figsize=(7.2, 1.9), constrained_layout=True)
    ext = (0, z.shape[1] * cfg.dx, 0, z.shape[0] * cfg.dx)
    exts = (0, zs.shape[1] * cfg.dx, 0, zs.shape[0] * cfg.dx)
    ls = LightSource(azdeg=315, altdeg=40)
    nz = int(round(18 / cfg.dx))
    zz = z[:nz, :nz]
    axs[0].imshow(ls.shade(zz, plt.cm.Greys_r, vert_exag=1, blend_mode="soft"), origin="lower", extent=(0, 18, 0, 18))
    im = plt.cm.ScalarMappable(cmap="Greys_r", norm=plt.Normalize(z.min(), z.max()))
    axs[0].set_title("Blade array")
    fig.colorbar(im, ax=axs[0], shrink=0.75, label="height (µm)")
    im = axs[1].imshow(zs, origin="lower", extent=exts, cmap="Greys_r")
    axs[1].set_title("Sharklet")
    fig.colorbar(im, ax=axs[1], shrink=0.75, label="height (µm)")
    for ax, s in zip(axs[2:], SP):
        sp = SPECIES[s]
        lay = attachment_layers(z, sp, cfg, flat_reference_energy(sp, cfg))
        im = ax.imshow(np.log10(np.maximum(lay["attach"], 1e-4)), origin="lower", extent=ext, cmap="Blues", vmin=-4, vmax=0)
        ax.set_title(it(s))
    fig.colorbar(im, ax=axs[2:], location="bottom", shrink=0.6, aspect=40, label="log$_{10}$ attachment propensity relative to flat")
    for i, ax in enumerate(axs):
        ax.set_xticks([0, 20, 40] if i == 1 else ([0, 9, 18] if i == 0 else [0, 18, 36])); ax.set_yticks([0, 16, 32] if i == 1 else ([0, 9, 18] if i == 0 else [0, 18, 36]))
        ax.set_xlabel("x (µm)")
        if i == 0:
            ax.set_ylabel("y (µm)")
        panel(ax, "abcde"[i])
    save(fig, "fig1_surfaces")


def fig_numerics():
    nu = pd.read_csv(D / "numerics.csv")
    nu["L"] = -np.log10(nu.Y)
    sh = pd.read_csv(D / "numerics_sharklet.csv")
    sh["L"] = -np.log10(sh.Y)
    g = nu[nu.test == "grid"]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), constrained_layout=True)
    for s in SP:
        for k, (surf, gg) in enumerate(g[g.species == s].groupby("surface")):
            gg = gg.sort_values("dx")
            axs[0].plot(gg.dx, gg.L, marker=MK[s], ms=3.5, color=COL[s], alpha=0.85, lw=1.0, label=it(s) if k == 0 else None)
        ss = sh[sh.species == s].sort_values("dx")
        axs[1].plot(ss.dx, ss.L, marker=MK[s], ms=4, color=COL[s], label=it(s))
    for ax in axs[:2]:
        ax.axvline(0.1, color=GRID, lw=1, zorder=0)
        ax.set_xlabel("grid spacing Δx (µm)")
        ax.invert_xaxis()
    axs[0].set_ylabel("model log$_{10}$ reduction, $L$")
    axs[0].set_title("Blade arrays (4 geometries)")
    axs[1].set_title("Sharklet AF-type")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    base = g[g.dx == 0.15].set_index(["surface", "species"]).L
    rows = []
    for test, col, vals, lab in (("time", "h", (0.1, 0.025), "Δt = {}"), ("domain", "domain", (24.0, 48.0), "domain {:.0f} µm")):
        for v in vals:
            d = (nu[(nu.test == test) & (nu[col] == v)].set_index(["surface", "species"]).L - base)
            rows.append((lab.format(v), d.to_numpy()))
    d07 = (g[g.dx == 0.1].set_index(["surface", "species"]).L - g[g.dx == 0.07].set_index(["surface", "species"]).L)
    rows.insert(0, ("Δx 0.10 vs 0.07", d07.to_numpy()))
    for i, (lab, d) in enumerate(rows):
        axs[2].scatter(d, np.full_like(d, i) + np.linspace(-0.15, 0.15, len(d)), s=10, color=INK2)
    axs[2].axvline(0, color=GRID, lw=1, zorder=0)
    axs[2].set_yticks(range(len(rows))); axs[2].set_yticklabels([r[0] for r in rows])
    axs[2].set_xlabel("ΔL relative to reference")
    axs[2].set_title("Time step, domain, grid")
    for a, l in zip(axs, "abc"):
        panel(a, l)
    save(fig, "fig2_numerics")


def fig_ensemble():
    ens = pd.read_csv(D / "ensemble.csv")
    ens["L"] = -np.log10(ens.Y)
    tab = pd.read_csv(T / "ensemble_L.csv").set_index("species")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    rng = np.random.default_rng(1)
    for i, s in enumerate(SP):
        y = ens[ens.species == s].L
        axs[0].scatter(i + rng.uniform(-0.18, 0.18, len(y)), y, s=8, color=COL[s], alpha=0.55, marker=MK[s], lw=0)
        r = tab.loc[s]
        axs[0].errorbar(i + 0.32, r.L, yerr=[[r.L - r.L_lo], [r.L_hi - r.L]], fmt="D", color=INK, ms=4, capsize=2, lw=1)
    axs[0].set_xticks(range(3)); axs[0].set_xticklabels([it(s) for s in SP])
    axs[0].set_ylabel("model log$_{10}$ reduction, $L$")
    axs[0].set_title("Per surface and ensemble")
    # endpoint map
    e = S["endpoint"]["fig5a"]
    a = e["alpha"]
    xs = np.linspace(0, 2.6, 10)
    axs[1].fill_between(xs, e["alpha_ci"][0] * xs, e["alpha_ci"][1] * xs, color=GRID, alpha=0.6, lw=0)
    axs[1].plot(xs, a * xs, color=INK2, lw=1)
    obs = S["LR_obs"]
    for s in SP:
        r = tab.loc[s]
        if s in obs:
            axs[1].errorbar(r.L, obs[s], xerr=[[r.L - r.L_lo], [r.L_hi - r.L]], yerr=4.303 * S["SE_obs"][s], fmt=MK[s], color=COL[s], ms=5, capsize=2, label=f"{it(s)} (observed)")
        else:
            axs[1].errorbar(r.L, e["pred_LR_Pa"], xerr=[[r.L - r.L_lo], [r.L_hi - r.L]],
                            yerr=[[e["pred_LR_Pa"] - e["pred_LR_Pa_ci"][0]], [e["pred_LR_Pa_ci"][1] - e["pred_LR_Pa"]]],
                            fmt=MK[s], mfc="white", color=COL[s], ms=5, capsize=2, label=f"{it(s)} (predicted)")
    axs[1].plot([0, 2.6], [0, 2.6], ls=":", color=INK2, lw=0.8)
    axs[1].set_xlim(0, 2.0); axs[1].set_ylim(0, 1.6)
    axs[1].set_xlabel("model $L$ (ensemble)")
    axs[1].set_ylabel("48 h log$_{10}$ CFU reduction")
    axs[1].set_title(f"Endpoint exponent α = {a:.2f}")
    axs[1].legend(frameon=False, loc="upper left", fontsize=6.5)
    # geometry dependence
    for s in SP:
        g = ens[ens.species == s]
        axs[2].scatter(g.g_pitch_a, g.L, s=9, color=COL[s], marker=MK[s], alpha=0.7, lw=0, label=it(s))
    axs[2].set_xlabel("adjacent-tip spacing (µm)")
    axs[2].set_ylabel("$L$ (per surface)")
    axs[2].set_title("Dependence on lateral packing")
    axs[2].legend(frameon=False, fontsize=6.5)
    for ax, l in zip(axs, "abc"):
        panel(ax, l)
    save(fig, "fig3_ensemble_calibration")


def fig_sharklet():
    sh = pd.read_csv(D / "sharklet.csv").set_index("species")
    obs = pd.read_csv(T / "sharklet_observed.csv")
    ts = [0.5, 1, 2, 4.5, 9]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4), constrained_layout=True, gridspec_kw={"width_ratios": [1.1, 1]})
    for s in SP:
        axs[0].plot(ts, [-np.log10(sh.loc[s, f"Y_t{t:g}"]) for t in ts], marker=MK[s], color=COL[s], ms=4, label=it(s))
    tc = pd.read_csv(T / "time_course_L.csv")
    for s in SP:
        g = tc[tc.species == s]
        axs[0].plot(g.t, g.L, ls="--", color=COL[s], lw=1)
    axs[0].set_xscale("log"); axs[0].set_xticks(ts); axs[0].set_xticklabels([str(t) for t in ts])
    axs[0].set_xlabel("model time")
    axs[0].set_ylabel("model $L$")
    axs[0].set_title("Sharklet AF (solid), blades (dashed)")
    axs[0].legend(frameon=False, fontsize=6.5)
    rng = np.random.default_rng(3)
    for i, s in enumerate(SP):
        o = obs[obs.species == s]
        mk = {"May et al. 2014": "o", "Xu et al. 2017": "v"}
        for st, oo in o.groupby("study"):
            axs[1].scatter(np.full(len(oo), i - 0.12) + rng.uniform(-0.04, 0.04, len(oo)), oo.LR, color=INK2, marker=mk[st], s=16,
                           label=st if i == (0 if st.startswith("May") else 1) else None)
        p1 = -np.log10(sh.loc[s, "Y_t1"])
        p45 = -np.log10(sh.loc[s, "Y_t4.5"])
        axs[1].plot([i + 0.12, i + 0.12], [min(p1, p45), max(p1, p45)], color=COL[s], lw=3, solid_capstyle="round")
        axs[1].scatter(i + 0.12, p1, color=COL[s], marker=MK[s], s=22, zorder=3)
    axs[1].set_xticks(range(3)); axs[1].set_xticklabels([it(s) for s in SP])
    axs[1].set_ylabel("log$_{10}$ reduction vs smooth")
    axs[1].set_title("Short-term assays: observed vs model")
    axs[1].set_ylim(0, 3.3)
    axs[1].legend(frameon=False, fontsize=6.5, loc="upper right")
    for ax, l in zip(axs, "ab"):
        panel(ax, l)
    save(fig, "fig4_sharklet")


def fig_mechanism():
    ko = pd.read_csv(T / "knockouts_dL.csv")
    dm = pd.read_csv(T / "design_map.csv")
    order = ["no_transport", "no_slope", "no_appendages", "no_steric", "no_compliance", "no_dynamic_eps"]
    lab = {"no_transport": "arrival/transport", "no_slope": "slope cost", "no_appendages": "appendages",
           "no_steric": "steric cost", "no_compliance": "compliance", "no_dynamic_eps": "dynamic EPS"}
    fig = plt.figure(figsize=(7.2, 2.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1])
    ax0 = fig.add_subplot(gs[0])
    for j, s in enumerate(SP):
        k = ko[ko.species == s].set_index("variant").reindex(order)
        y = np.arange(len(order)) + (j - 1) * 0.25
        ax0.barh(y, k.dL_mean, height=0.22, color=COL[s], label=it(s))
        ax0.errorbar(k.dL_mean, y, xerr=[k.dL_mean - k.lo, k.hi - k.dL_mean], fmt="none", ecolor=INK2, lw=0.8, capsize=1.5)
    ax0.axvline(0, color=INK2, lw=0.8)
    ax0.set_yticks(range(len(order))); ax0.set_yticklabels([lab[o] for o in order])
    ax0.invert_yaxis()
    ax0.set_xlabel("ΔL contributed")
    ax0.set_title("Knockouts")
    ax0.legend(frameon=False, fontsize=6.5, loc="lower right")
    panel(ax0, "a")
    vmax = dm.L.quantile(0.98)
    for j, s in enumerate(SP):
        ax = fig.add_subplot(gs[j + 1])
        p = dm[dm.species == s].pivot(index="base", columns="tip_height", values="L")
        im = ax.imshow(p.values, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=vmax,
                       extent=(-0.5, len(p.columns) - 0.5, -0.5, len(p.index) - 0.5))
        cs = ax.contour(np.arange(len(p.columns)), np.arange(len(p.index)), p.values, levels=[0.5, 1.0, 1.5, 2.0], colors=INK2, linewidths=0.7)
        ax.clabel(cs, fmt="%.1f", fontsize=5.5)
        ax.set_xticks(range(len(p.columns))); ax.set_xticklabels([f"{v:g}" for v in p.columns], fontsize=6.5)
        ax.set_yticks(range(len(p.index))); ax.set_yticklabels([f"{v:.1f}" for v in p.index], fontsize=6.5)
        ax.set_xlabel("tip height (µm)")
        if j == 0:
            ax.set_ylabel("blade base width (µm)")
        ax.set_title(it(s))
        panel(ax, "bcd"[j])
    fig.colorbar(im, ax=fig.axes[1:], shrink=0.8, label="model $L$")
    save(fig, "fig5_mechanism_design")


def fig_sobol():
    sg = pd.read_csv(T / "sobol_geometry.csv")
    sm = pd.read_csv(T / "sobol_mechanism.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.0), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.25]})
    gk = list(sg[sg.species == "E. coli"].sort_values("ST", ascending=False).parameter)
    for j, s in enumerate(SP):
        k = sg[sg.species == s].set_index("parameter").reindex(gk)
        y = np.arange(len(gk)) + (j - 1) * 0.25
        axs[0].barh(y, k.ST, height=0.22, color=COL[s], label=it(s))
        axs[0].errorbar(k.ST, y, xerr=[np.clip(k.ST - k.ST_lo, 0, None), np.clip(k.ST_hi - k.ST, 0, None)], fmt="none", ecolor=INK2, lw=0.7)
    axs[0].set_yticks(range(len(gk))); axs[0].set_yticklabels([g.replace("_", " ") for g in gk]); axs[0].invert_yaxis()
    axs[0].set_xlabel("total-order Sobol index $S_T$")
    axs[0].set_title("Geometry → $L$ (blade array)")
    axs[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    snake = sm[sm.surface == "snake"]
    mk = list(snake[snake.species == "E. coli"].sort_values("ST", ascending=False).parameter)
    outs = [("E. coli", COL["E. coli"], "$L$ " + it("E. coli")), ("S. aureus", COL["S. aureus"], "$L$ " + it("S. aureus")),
            ("contrast", INK2, "contrast $L_{Sa}/L_{Ec}$")]
    for j, (s, c, l) in enumerate(outs):
        k = snake[snake.species == s].set_index("parameter").reindex(mk)
        y = np.arange(len(mk)) + (j - 1) * 0.25
        axs[1].barh(y, k.ST, height=0.22, color=c, label=l)
    axs[1].set_yticks(range(len(mk))); axs[1].set_yticklabels([m.replace("_", " ") for m in mk]); axs[1].invert_yaxis()
    axs[1].set_xlabel("total-order Sobol index $S_T$")
    axs[1].set_title("Mechanistic coefficients (×½–×2)")
    axs[1].legend(frameon=False, fontsize=6.5, loc="lower right")
    for ax, l in zip(axs, "ab"):
        panel(ax, l)
    save(fig, "fig6_sobol")


def fig_running():
    r = pd.read_csv(T / "ensemble_running.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.4), constrained_layout=True)
    for s in SP:
        g = r[r.species == s]
        ax.plot(g.N, g.L, marker=MK[s], ms=3.5, color=COL[s], label=it(s))
    ax.set_xlabel("number of surfaces $N$")
    ax.set_ylabel("ensemble $L$")
    ax.legend(frameon=False)
    save(fig, "figS1_running")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["surfaces", "numerics", "ensemble", "sharklet", "mechanism", "sobol", "running"]
    for w in which:
        globals()[f"fig_{w}"]()
        print("done", w)

