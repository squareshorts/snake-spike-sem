"""Tables and numerical macros for the control experiments (results_v2/controls)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

O = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results_v2/controls")
T = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("manuscript")
T.mkdir(parents=True, exist_ok=True)
SP = ["E. coli", "S. aureus", "P. aeruginosa"]
AB = {"E. coli": "Ec", "S. aureus": "Sa", "P. aeruginosa": "Pa"}
TEX = {s: f"\\{AB[s]}{{}}" for s in SP}
MODES = [("dual", "Both pathways"), ("supply", "Supply only"), ("capture", "Capture only")]
RNG = np.random.default_rng(20261120)
mac = {}


def load(name):
    d = pd.read_csv(O / f"{name}.csv")
    return d.assign(L=-np.log10(d.Y))


def sg(x, n=2):
    return f"{x:+.{n}f}".replace("-", "\\ensuremath{-}")


def table(path, caption, label, cols, header, rows, size="small"):
    body = "\n".join(rows)
    Path(T / path).write_text(
        f"\\begin{{table}}[htbp]\\centering\\{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{cols}}}\\toprule\n{header}\\\\\\midrule\n{body}\n\\bottomrule\\end{{tabular}}\n\\end{{table}}\n")


# ---------------------------------------------------------------- arrival formulations
d = load("structural")
rows = []
for m, lab in MODES:
    cells, Ls = [], {}
    for sp in SP:
        y = d[(d["mode"] == m) & (d.species == sp)].sort_values("surface").Y.to_numpy()
        boot = -np.log10(y[RNG.integers(0, len(y), (10000, len(y)))].mean(1))
        Ls[sp] = -np.log10(y.mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        cells.append(f"{Ls[sp]:.2f} ({lo:.2f}--{hi:.2f})")
        mac[f"Arr{m.capitalize()}{AB[sp]}"] = f"{Ls[sp]:.2f}"
    mac[f"Arr{m.capitalize()}Ratio"] = f"{Ls['S. aureus'] / Ls['E. coli']:.2f}"
    rows.append(f"{lab} & " + " & ".join(cells) + f" & {mac[f'Arr{m.capitalize()}Ratio']}\\\\")
table("tab_structural.tex", "Production-grid sensitivity to the representation of arrival. Values are ensemble log reductions with geometric-bootstrap 95\\% intervals, and the ratio of \\Sa{} to \\Ec{} log reductions.",
      "tab:structural", "lcccc", "Model & \\Ec{} $L$ & \\Sa{} $L$ & \\Pa{} $L$ & $L_{Sa}/L_{Ec}$", rows)

# ---------------------------------------------------------------- spacing sweep
d = load("spacing")
m_ = d.groupby(["mode", "species", "pitch_a"]).Y.mean().unstack()
c = -np.log10(m_[3.4]) + np.log10(m_[5.9])
rows = [f"{lab} & " + " & ".join(f"{c[(m, sp)]:.2f}" for sp in SP) + "\\\\" for m, lab in MODES]
for m, _ in MODES:
    for sp in SP:
        mac[f"Spc{m.capitalize()}{AB[sp]}"] = f"{c[(m, sp)]:.2f}"
w = d.pivot_table(index=["mode", "species", "seed"], columns="pitch_a", values="L")
mac["SpcPos"], mac["SpcN"] = str(int(((w[3.4] - w[5.9]) > 0).sum())), str(len(w))
table("tab_spacing.tex", "Lateral-packing contrasts in model log reduction, $L(3.4\\,\\si{\\micrometre})-L(5.9\\,\\si{\\micrometre})$, from the mean remaining fraction over four common lattice seeds. Positive values indicate stronger suppression for closer packing.",
      "tab:spacing", "lccc", "Model & \\Ec{} & \\Sa{} & \\Pa{}", rows)

# ---------------------------------------------------------------- independent species coefficients
d = load("indep")
w = d.pivot(index="scenario", columns="species", values="L")
LR = {"E. coli": -np.log10(0.118), "S. aureus": -np.log10(0.224)}
SE = {"E. coli": 0.013 / (0.118 * np.log(10) * np.sqrt(3)), "S. aureus": 0.083 / (0.224 * np.log(10) * np.sqrt(3))}
we, ws = 1 / SE["E. coli"] ** 2, 1 / SE["S. aureus"] ** 2
alpha = (we * LR["E. coli"] * w["E. coli"] + ws * LR["S. aureus"] * w["S. aureus"]) / (we * w["E. coli"] ** 2 + ws * w["S. aureus"] ** 2)
q = lambda x: np.percentile(x, [5, 95])
n_ord = int((w["E. coli"] > w["S. aureus"]).sum())
mac.update(IndN=str(len(w)), IndOrd=str(n_ord), IndOrdPct=f"{100 * n_ord / len(w):.1f}",
           IndRatio="{:.2f}--{:.2f}".format(*q(w["S. aureus"] / w["E. coli"])),
           IndPa="{:.2f}--{:.2f}".format(*q(alpha * w["P. aeruginosa"])))
rows = [f"Nominal species ranking, \\Ec{{}} suppression $>$ \\Sa{{}} & {n_ord}/{len(w)} ({mac['IndOrdPct']}\\%)\\\\",
        f"$L_{{Sa}}/L_{{Ec}}$, 5th--95th percentile & {mac['IndRatio']}\\\\",
        "\\Ec{} model $L$, 5th--95th percentile & %.2f--%.2f\\\\" % tuple(q(w["E. coli"])),
        "\\Sa{} model $L$, 5th--95th percentile & %.2f--%.2f\\\\" % tuple(q(w["S. aureus"])),
        f"\\Pa{{}} mapped 48~h LR, 5th--95th percentile & {mac['IndPa']}\\\\"]
table("tab_independent.tex", f"Joint sensitivity scenarios with species-specific multipliers ($N={len(w)}$). Quantiles describe the distribution of prescribed model scenarios. The predicted \\Pa{{}} 48~h reduction refits the shared endpoint exponent in each scenario.",
      "tab:independent", "lc", "Model diagnostic & Value", rows)

# ---------------------------------------------------------------- coverage-matched spacing
d = load("coverage")
mac["GeomMaxCovErr"] = f"{max((d[d.pit].blade_fraction - 0.35).abs().max(), (d[~d.pit].blade_fraction - 0.46).abs().max()):.4f}"
w = d.pivot_table(index=["pit", "mode", "species", "seed"], columns="pitch_a", values="L")
w["dL"] = w[3.8] - w[5.2]
mac["GeomNposCoverage"], mac["GeomNtotalCoverage"] = str(int((w.dL > 0).sum())), str(len(w))
rows = []
for pit, plab in ((True, "Pits present"), (False, "Pits absent")):
    for m, lab in MODES:
        cells = []
        for sp in SP:
            v = w.loc[(pit, m, sp)].dL
            cells.append(f"{sg(v.mean())} [{sg(v.min())}, {sg(v.max())}]")
        rows.append(f"{plab} & {lab} & " + " & ".join(cells) + "\\\\")
for sp in SP:
    mac[f"Geom{AB[sp]}CovPit"] = f"{w.loc[(True, 'dual', sp)].dL.mean():.2f}"
    mac[f"Geom{AB[sp]}CovNoPit"] = f"{w.loc[(False, 'dual', sp)].dL.mean():.2f}"
table("tab_geom_coverage.tex", "Paired log-reduction contrast at matched projected blade coverage, $\\Delta L=L(p_a=3.8\\,\\si{\\micrometre})-L(p_a=5.2\\,\\si{\\micrometre})$. Entries are mean [minimum, maximum] over eight common surface seeds, with blade-base width adjusted within its prior range. Pit-bearing and pit-free arrays use target coverages of 0.35 and 0.46, respectively.",
      "tab:geomcov", "llccc", "Pit condition & Arrival formulation & \\Ec{} & \\Sa{} & \\Pa{}", rows, size="scriptsize")

# ---------------------------------------------------------------- relief factorial
d = load("factorial")
fw = d.pivot_table(index=["nuis", "tip_height", "pit_depth", "species"], columns="pitch_a", values="L")
fw["dL"] = fw[3.8] - fw[5.0]
rows = []
for sp in SP:
    for h in (0.5, 1.5, 3.0):
        v = fw.xs((h, sp), level=["tip_height", "species"]).dL
        rows.append(f"{TEX[sp]} & {h} & {sg(v.mean())} [{sg(v.min())}, {sg(v.max())}] & {(v > 0).sum()} / {len(v)} & 8\\\\")
    rows.append("\\addlinespace")
    mac[f"GeomFact{AB[sp]}"] = f"{fw.xs(sp, level='species').dL.mean():.2f}"
table("tab_geom_factorial.tex", "Effect of lateral pitch across unmeasured relief in the dual-pathway model. Paired $\\Delta L=L(3.8)-L(5.0)$ is calculated for eight nuisance-geometry draws at each tip height and pit depth. Entries are mean [range] over the sixteen combinations of draw and pit depth at each height.",
      "tab:geom3d", "llccc", "Species & Tip height (\\si{\\micrometre}) & Mean $\\Delta L$ [range] & Positive & Draws", rows[:-1])
e = fw.xs("E. coli", level="species").dL
mac.update(GeomFactEcLow=f"{e.xs(0.5, level='tip_height').mean():.2f}", GeomFactEcHigh=f"{e.xs(3.0, level='tip_height').mean():.2f}",
           GeomFactEcMin=f"{e.min():.2f}", GeomFactEcMax=f"{e.max():.2f}", GeomFactPos=str(int((fw.dL > 0).sum())), GeomFactN=str(len(fw)))

# ---------------------------------------------------------------- row phase
d = load("phase")
pw = d.pivot_table(index=["pit", "species", "seed"], columns="phase", values="L")
pw["dL"] = pw[0.5] - pw[0.0]
rows = []
for pit, plab in ((True, "Pits present"), (False, "Pits absent")):
    for sp in SP:
        v = pw.loc[(pit, sp)].dL * 1e3
        rows.append(f"{plab} & {TEX[sp]} & {sg(v.mean(), 1)} [{sg(v.min(), 1)}, {sg(v.max(), 1)}] & {(v > 0).sum()} / {len(v)}\\\\")
mac["GeomPhaseMaxMean"] = f"{pw.groupby(level=[0, 1]).dL.mean().abs().max():.4f}"
mac["GeomPhaseMaxAbs"] = f"{pw.dL.abs().max():.3f}"
table("tab_geom_phase.tex", "Fixed-density row-arrangement contrast, $L(\\delta=0.5)-L(\\delta=0)$, in units of $10^{-3}$ log; $\\delta$ is the offset of alternating rows in units of the following pitch. Entries are mean [minimum, maximum] across eight common surface seeds. Blade count, column separation, pitch and blade dimensions are fixed. Results use the dual-pathway formulation.",
      "tab:geomphase", "llcc", "Pit condition & Species & $\\Delta L$ [range] & Positive pairs", rows)

# ---------------------------------------------------------------- grid refinement
d = load("grid")
gp = d.pivot_table(index=["geometry", "species"], columns="dx", values="L")
GN = [("close_nominal", "Close, nominal"), ("close_crowded_tall", "Close, crowded/tall"),
      ("close_short_narrow", "Close, short/narrow"), ("wide_nominal", "Wide, nominal")]
rows = []
for k, lab in GN:
    for sp in SP:
        r = gp.loc[(k, sp)]
        l05 = f"{r[0.05]:.3f}" if 0.05 in r and np.isfinite(r[0.05]) else "--"
        rows.append(f"{lab} & {TEX[sp]} & {r[0.15]:.3f} & {r[0.10]:.3f} & {r[0.07]:.3f} & {l05} & {sg(r[0.10] - r[0.07], 3)}\\\\")
    rows.append("\\addlinespace")
table("tab_geom_grid.tex", "Grid checks in close-packed and comparison geometries (dual-pathway model). The last column is the signed difference between the production grid and the first refinement; the \\SI{0.05}{\\micrometre} extension was run for the two close-packed geometries with the largest differences.",
      "tab:geomgrid", "llccccc", "Geometry & Species & $L_{0.15}$ & $L_{0.10}$ & $L_{0.07}$ & $L_{0.05}$ & $L_{0.10}-L_{0.07}$", rows[:-1])
diff = (gp[0.10] - gp[0.07]).abs()
mac["GeomGridMax"] = f"{diff.max():.3f}"
mac["GeomGridShort"] = f"{diff.loc['close_short_narrow'].max():.3f}"
mac["GeomGridWide"] = f"{diff.loc['wide_nominal'].max():.3f}"
mac["GeomExtremeMaxAdditional"] = f"{(gp[0.05] - gp[0.07]).abs().max():.3f}"

d = load("matched_grid")
w = d.pivot_table(index=["pit", "species", "seed", "dx"], columns="pitch_a", values="L")
w["dL"] = w[3.8] - w[5.2]
x = w.dL.unstack("dx")
x["delta"] = (x[0.07] - x[0.10]).abs()
rows = []
for pit, plab in ((True, "Pits present"), (False, "Pits absent")):
    for sp in SP:
        v = x.loc[(pit, sp)]
        rows.append(f"{plab} & {TEX[sp]} & {sg(v[0.10].mean(), 3)} & {sg(v[0.07].mean(), 3)} & {v.delta.max():.3f} & {(v[0.07] > 0).sum()}/{len(v)}\\\\")
mac["GeomFinePos"], mac["GeomFineN"] = str(int((x[0.07] > 0).sum())), str(len(x))
mac["GeomFineMaxDeltaError"] = f"{x.delta.max():.3f}"
table("tab_geom_matched_grid.tex", "Finer-grid repetition of the coverage-matched contrast, $\\Delta L=L(3.8)-L(5.2)$. Widths were fixed at the values selected on the \\SI{0.10}{\\micrometre} grid and the same surfaces were re-evaluated at \\SI{0.07}{\\micrometre}. Entries are mean paired contrasts across four matched seeds, the largest absolute change in any paired contrast between resolutions, and the count of positive refined contrasts.",
      "tab:geomfine", "llcccc", "Pits & Species & Mean $\\Delta L_{0.10}$ & Mean $\\Delta L_{0.07}$ & Max $|\\delta(\\Delta L)|$ & Positive", rows)

(T / "geometry_numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
print(mac)
