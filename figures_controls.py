"""Figures for the control experiments: main figures 7 and 8 and ESM figures S2-S7."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

O = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results_v2/controls")
F = Path(sys.argv[2]) if len(sys.argv) > 2 else O / "figures"
F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.labelcolor": "#222", "xtick.color": "#444",
                     "ytick.color": "#444", "pdf.fonttype": 42, "axes.titlesize": 8})
SP = ["E. coli", "S. aureus", "P. aeruginosa"]
COL = {"E. coli": "#2a78d6", "S. aureus": "#eb6834", "P. aeruginosa": "#1baf7a"}
MODES = [("dual", "Dual pathway"), ("supply", "Supply only"), ("capture", "Capture only")]
LS = {"dual": "-", "supply": "--", "capture": ":"}
GRID = dict(axis="y", color="#e6e6e6", lw=0.6)
RNG = np.random.default_rng(1)


def load(name):
    d = pd.read_csv(O / f"{name}.csv")
    return d.assign(L=-np.log10(d.Y))


def save(fig, name):
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(F / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def species_handles(extra=()):
    return [Line2D([], [], color=COL[s], lw=1.5, label=s) for s in SP] + list(extra)


def legend(fig, handles, ncol, y=-0.02):
    leg = fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=7, bbox_to_anchor=(0.5, y))
    for t in leg.get_texts():
        if t.get_text() in SP:
            t.set_fontstyle("italic")


def italic_species_ticks(ax, n=3):
    ax.set_xticks(range(n), [s.replace(" ", "\n") for s in SP], fontstyle="italic")


mode_handles = [Line2D([], [], color="#444", ls=LS[m], label=lab) for m, lab in MODES]

# ============================================================ figure 7
st, spc, ind = load("structural"), load("spacing"), load("indep")
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
for mi, (m, lab) in enumerate(MODES):
    for si, sp in enumerate(SP):
        y = st[(st["mode"] == m) & (st.species == sp)].Y.to_numpy()
        boot = -np.log10(y[RNG.integers(0, len(y), (4000, len(y)))].mean(1))
        x = mi + (si - 1) * 0.22
        ax[0].plot([x, x], np.percentile(boot, [2.5, 97.5]), color=COL[sp], lw=1)
        ax[0].plot(x, -np.log10(y.mean()), "o", ms=4.5, color=COL[sp])
ax[0].set_xticks(range(3), [lab.replace(" ", "\n") for _, lab in MODES])
ax[0].set_ylabel("Ensemble log reduction, $L$")
ax[0].set_title("(a) Arrival formulation, 40 surfaces", loc="left")
ax[0].grid(**GRID)
m_ = spc.groupby(["mode", "species", "pitch_a"]).Y.mean().unstack()
c = -np.log10(m_[3.4]) + np.log10(m_[5.9])
for mi, (m, _) in enumerate(MODES):
    for si, sp in enumerate(SP):
        ax[1].bar(mi + (si - 1) * 0.24, c[(m, sp)], width=0.2, color=COL[sp], lw=0)
ax[1].set_xticks(range(3), [lab.replace(" ", "\n") for _, lab in MODES])
ax[1].set_ylabel(r"$L(3.4)-L(5.9)$ (log)")
ax[1].set_title("(b) Lateral packing, 4 seeds", loc="left")
ax[1].grid(**GRID)
w = ind.pivot(index="scenario", columns="species", values="L")
inv = w["S. aureus"] > w["E. coli"]
lim = [0, max(w["E. coli"].max(), w["S. aureus"].max()) * 1.05]
ax[2].plot(lim, lim, color="#888", lw=0.7, ls="--")
ax[2].plot(w["E. coli"][~inv], w["S. aureus"][~inv], "o", ms=3, color="#2a78d6", alpha=0.7, mew=0,
           label=r"$L_{Ec}>L_{Sa}$" + f" ({(~inv).sum()})")
ax[2].plot(w["E. coli"][inv], w["S. aureus"][inv], "s", ms=3.5, mfc="white", mec="#eb6834", mew=1,
           label=r"$L_{Sa}>L_{Ec}$" + f" ({inv.sum()})")
ax[2].set_xlim(lim); ax[2].set_ylim(lim)
ax[2].set_xlabel(r"$L$, $\it{E.\ coli}$"); ax[2].set_ylabel(r"$L$, $\it{S.\ aureus}$")
ax[2].set_title(f"(c) Independent coefficients, {len(w)} scenarios", loc="left")
ax[2].legend(frameon=False, fontsize=7, loc="upper left")
legend(fig, species_handles(), 3)
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "fig7_robustness")

# ============================================================ figure 8
cov, fac, pha = load("coverage"), load("factorial"), load("phase")
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})
w = cov.pivot_table(index=["pit", "mode", "species", "seed"], columns="pitch_a", values="L")
w["dL"] = w[3.8] - w[5.2]
for mi, (m, _) in enumerate(MODES):
    for si, sp in enumerate(SP):
        for pit, mk in ((True, "o"), (False, "s")):
            v = w.loc[(pit, m, sp)].dL
            x = mi + (si - 1) * 0.22 + (-0.07 if pit else 0.07)
            ax[0].plot([x, x], [v.min(), v.max()], color=COL[sp], lw=1)
            ax[0].plot(x, v.mean(), mk, ms=4.5, color=COL[sp], mfc=COL[sp] if pit else "white", mew=1)
ax[0].set_xticks(range(3), [lab.replace(" ", "\n") for _, lab in MODES])
ax[0].set_ylim(0, 0.8)
ax[0].set_ylabel(r"$L(3.8)-L(5.2)$ (log)")
ax[0].set_title("(a) Matched projected coverage", loc="left")
ax[0].grid(**GRID)
fw = fac.pivot_table(index=["nuis", "tip_height", "pit_depth", "species"], columns="pitch_a", values="L")
fw["dL"] = fw[3.8] - fw[5.0]
for sp in SP:
    d = fw.xs(sp, level="species").dL
    for dp, ls in ((2.5, "-"), (0.5, "--")):
        mean = d.xs(dp, level="pit_depth").groupby(level="tip_height").mean()
        ax[1].plot(mean.index, mean.values, ls, color=COL[sp], lw=1.5, marker="o", ms=3)
ax[1].set_xticks([0.5, 1.5, 3.0])
ax[1].set_xlabel(r"Tip height ($\mu$m)")
ax[1].set_ylabel(r"$L(3.8)-L(5.0)$ (log)")
ax[1].set_ylim(0, None)
ax[1].set_title("(b) Unmeasured relief", loc="left")
ax[1].grid(**GRID)
pw = pha.pivot_table(index=["pit", "species", "seed"], columns="phase", values="L")
pw["dL"] = (pw[0.5] - pw[0.0]) * 1e3
for si, sp in enumerate(SP):
    for pit, mk in ((True, "o"), (False, "s")):
        v = pw.loc[(pit, sp)].dL
        x = si + (-0.12 if pit else 0.12)
        ax[2].plot([x, x], [v.min(), v.max()], color=COL[sp], lw=1)
        ax[2].plot(x, v.mean(), mk, ms=4.5, color=COL[sp], mfc=COL[sp] if pit else "white", mew=1)
ax[2].axhline(0, color="#888", lw=0.7)
italic_species_ticks(ax[2]); ax[2].set_xlim(-0.5, 2.5)
ax[2].set_ylabel(r"$L_\mathrm{staggered}-L_\mathrm{aligned}$ ($10^{-3}$ log)")
ax[2].set_title("(c) Row phase, fixed density", loc="left")
ax[2].grid(**GRID)
legend(fig, species_handles([Line2D([], [], ls="", marker="o", color="#444", label="Pits"),
                             Line2D([], [], ls="", marker="s", mfc="white", color="#444", label="No pits"),
                             Line2D([], [], ls="-", color="#444", label=r"Pit depth 2.5 $\mu$m"),
                             Line2D([], [], ls="--", color="#444", label=r"Pit depth 0.5 $\mu$m")]), 7)
fig.tight_layout(rect=(0, 0.07, 1, 1))
save(fig, "fig8_geometric_identifiability")

# ============================================================ S2 spacing sweep
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=False)
m_ = spc.groupby(["mode", "species", "pitch_a"]).Y.mean()
for a, sp in zip(ax, SP):
    for m, _ in MODES:
        s = -np.log10(m_.loc[(m, sp)])
        a.plot(s.index, s.values, LS[m], color=COL[sp], lw=1.5, marker="o", ms=3)
    a.set_title(sp, fontstyle="italic"); a.set_xlabel(r"Adjacent-tip spacing $p_a$ ($\mu$m)"); a.grid(**GRID)
ax[0].set_ylabel("Log reduction, $L$")
legend(fig, mode_handles, 3)
fig.tight_layout(rect=(0, 0.07, 1, 1))
save(fig, "figS2_spacing")

# ============================================================ S3 matched coverage
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.4))
g = cov.groupby(["pit", "mode", "species", "pitch_a"]).L.agg(["mean", "min", "max"])
for r, (pit, plab) in enumerate(((True, "pits"), (False, "no pits"))):
    for cidx, sp in enumerate(SP):
        a = ax[r, cidx]
        for m, _ in MODES:
            s = g.loc[(pit, m, sp)]
            a.fill_between(s.index, s["min"], s["max"], color=COL[sp], alpha=0.12, lw=0)
            a.plot(s.index, s["mean"], LS[m], color=COL[sp], lw=1.5, marker="o", ms=3)
        a.set_title(f"$\\it{{{sp.replace(' ', '~')}}}$, {plab}")
        a.set_xticks([3.8, 4.4, 5.2]); a.grid(**GRID)
        if r == 1:
            a.set_xlabel(r"Adjacent-tip spacing $p_a$ ($\mu$m)")
    ax[r, 0].set_ylabel("Log reduction, $L$")
legend(fig, mode_handles, 3)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "figS3_matched_coverage")

# ============================================================ S4 factorial heatmap
cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#104281"])
grid = {sp: fw.xs(sp, level="species").dL.groupby(level=["tip_height", "pit_depth"]).mean().unstack() for sp in SP}
vmax = max(g_.values.max() for g_ in grid.values())
fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.3), sharey=True)
for a, sp in zip(ax, SP):
    g_ = grid[sp]
    im = a.imshow(g_.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(g_.shape[0]):
        for j in range(g_.shape[1]):
            v = g_.values[i, j]
            a.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 * vmax else "#222")
    a.set_xticks(range(g_.shape[1]), [f"{x:g}" for x in g_.columns]); a.set_yticks(range(g_.shape[0]), [f"{x:g}" for x in g_.index])
    a.set_xlabel(r"Pit depth ($\mu$m)"); a.set_title(sp, fontstyle="italic")
    for s in a.spines.values():
        s.set_visible(False)
ax[0].set_ylabel(r"Tip height ($\mu$m)")
fig.tight_layout(rect=(0, 0, 0.9, 1))
cax = fig.add_axes([0.91, 0.25, 0.015, 0.6]); cb = fig.colorbar(im, cax=cax)
cb.set_label(r"$L(3.8)-L(5.0)$ (log)"); cb.outline.set_visible(False)
save(fig, "figS4_factorial_3d")

# ============================================================ S5 row phase
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.2))
g = pha.groupby(["pit", "species", "phase"]).L.agg(["mean", "std"])
for r, (pit, plab) in enumerate(((True, "pits"), (False, "no pits"))):
    for cidx, sp in enumerate(SP):
        a = ax[r, cidx]; s = g.loc[(pit, sp)]
        a.errorbar(s.index, s["mean"], yerr=s["std"], color=COL[sp], lw=1.5, marker="o", ms=3, capsize=2)
        a.set_title(f"$\\it{{{sp.replace(' ', '~')}}}$, {plab}"); a.set_xticks([0, 0.25, 0.5]); a.grid(**GRID)
        if r == 1:
            a.set_xlabel(r"Row offset $\delta$ (following pitches)")
    ax[r, 0].set_ylabel("Log reduction, $L$")
fig.tight_layout()
save(fig, "figS5_fixed_density_phase")

# ============================================================ S6 grid refinement
gr = load("grid").pivot_table(index=["geometry", "species"], columns="dx", values="L")
GN = [("close_nominal", "Close, nominal"), ("close_crowded_tall", "Close, crowded/tall"),
      ("close_short_narrow", "Close, short/narrow"), ("wide_nominal", "Wide, nominal")]
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.3))
for a, (k, lab) in zip(ax, GN):
    for sp in SP:
        s = gr.loc[(k, sp)].dropna()
        pos = {0.15: 0, 0.10: 1, 0.07: 2, 0.05: 3}
        a.plot([pos[round(v, 2)] for v in s.index], s.values, color=COL[sp], lw=1.5, marker="o", ms=3)
    a.axvline(1, color="#bbb", lw=0.7, ls=":")
    a.set_xlim(-0.3, 3.3); a.set_xticks(range(4), ["0.15", "0.10", "0.07", "0.05"]); a.set_title(lab)
    a.set_xlabel(r"Grid spacing $\Delta x$ ($\mu$m)"); a.grid(**GRID)
ax[0].set_ylabel("Log reduction, $L$")
legend(fig, species_handles(), 3, y=-0.04)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, "figS6_close_grid")

# ============================================================ S7 matched contrast refinement
mg = load("matched_grid").pivot_table(index=["pit", "species", "seed", "dx"], columns="pitch_a", values="L")
mg["dL"] = mg[3.8] - mg[5.2]
x = mg.dL.unstack("dx")
fig, ax = plt.subplots(1, 2, figsize=(5.0, 2.5), sharey=True)
for a, (pit, lab) in zip(ax, ((True, "Pits"), (False, "No pits"))):
    a.plot([0, 0.85], [0, 0.85], color="#888", lw=0.7, ls="--")
    for sp in SP:
        v = x.loc[(pit, sp)]
        a.plot(v[0.10], v[0.07], "o", ms=4, color=COL[sp])
    a.set_xlim(0, 0.85); a.set_ylim(0, 0.85); a.set_title(lab)
    a.set_xlabel(r"$\Delta L$ at $\Delta x=0.10\,\mu$m")
ax[0].set_ylabel(r"$\Delta L$ at $\Delta x=0.07\,\mu$m")
legend(fig, species_handles(), 3, y=-0.05)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, "figS7_matched_grid")
print("figures written to", F)
