#!/usr/bin/env python3
"""Write LaTeX macros (numbers.tex) and table snippets from results_v2 so that every
number in the manuscript is generated from the analysis outputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v2 import GEOM_PRIOR
from snake_model.model import SPECIES, ModelConfig

S = json.loads(Path("results_v2/summary.json").read_text())
T = Path("results_v2/tables")
M = Path("manuscript")
SP = ["E. coli", "S. aureus", "P. aeruginosa"]
AB = {"E. coli": "Ec", "S. aureus": "Sa", "P. aeruginosa": "Pa"}
mac = {}


def f(x, n=2):
    v = f"{x:.{n}f}"
    if v.startswith("-"):
        v = v.lstrip("-")
        return v if float(v) == 0 else "\\ensuremath{-}" + v
    return v


def ci(c, n=2):
    return f"{c[0]:.{n}f}--{c[1]:.{n}f}"


E = S["ensemble"]
mac["EnsN"] = str(S["ensemble_n"])
mac["MaxOcc"] = f(S["max_occupancy"], 2)
for s in SP:
    a = AB[s]
    mac[f"L{a}"] = f(E[s]["L"])
    mac[f"L{a}CI"] = f"{f(E[s]['L_lo'])}--{f(E[s]['L_hi'])}"
    mac[f"Lmed{a}"] = f(E[s]["L_surface_median"])
    mac[f"Lmin{a}"] = f(E[s]["L_surface_min"])
    mac[f"Lmax{a}"] = f(E[s]["L_surface_max"])
    mac[f"Half{a}"] = f"{f(S['half_split_L'][s][0])} and {f(S['half_split_L'][s][1])}"
C = S["contrast"]
mac["RhoModel"] = f(C["rho_model"])
mac["RhoModelCI"] = ci(C["rho_model_ci"])
for rd, k in (("fig5a", "A"), ("fig5b", "B")):
    mac[f"RhoObs{k}"] = f(C[f"rho_obs_{rd}"])
    mac[f"RhoObsCI{k}"] = ci(C[f"rho_obs_ci_{rd}"])
    mac[f"RhoPct{k}"] = f(100 * C[f"model_pct_{rd}"], 0)
O = S["obs"]
for s_ in ("E. coli", "S. aureus"):
    a_ = AB[s_]
    mac[f"Yobs{a_}"] = f(O["fig5a"]["Y"][s_], 3)
    mac[f"Ysd{a_}"] = f(O["fig5a"]["Y_sd"][s_], 3)
    mac[f"LRobs{a_}"] = f(O["fig5a"]["LR"][s_])
    mac[f"SEobs{a_}"] = f(O["fig5a"]["SE"][s_], 3)
    mac[f"LRobsB{a_}"] = f(O["fig5b"]["LR"][s_])
    mac[f"SEobsB{a_}"] = f(O["fig5b"]["SE"][s_], 2)
ct = S["contrast_vs_time"]
mac["RhoTimeMin"] = f(min(ct.values()))
mac["RhoTimeMax"] = f(max(ct.values()))
mac["RhoLong"] = f(S["longtime"]["72"]["contrast"])
for t, nm in (("1", "one"), ("36", "tt"), ("72", "sevtwo")):
    for s in SP:
        mac[f"Long{AB[s]}T{nm}"] = f(S["longtime"][t][s])
MU = S["mechanistic_uncertainty"]
mac["RhoPPmed"] = f(MU["snake|contrast"]["median"])
mac["RhoPPlo"] = f(MU["snake|contrast"]["q05"])
mac["RhoPPhi"] = f(MU["snake|contrast"]["q95"])
mac["RhoPPorder"] = f(100 * MU["snake|contrast"]["frac_order_correct"], 0)
mac["PaGeEc"] = f(100 * MU["frac_Pa_ge_Ec_snake"], 0)
for s in SP:
    mac[f"PPsnake{AB[s]}"] = f"{f(MU['snake|' + s]['q05'])}--{f(MU['snake|' + s]['q95'])}"
    mac[f"PPshark{AB[s]}"] = f"{f(MU['sharklet|' + s]['q05'])}--{f(MU['sharklet|' + s]['q95'])}"
mac["PredPaMechCI"] = ci(MU["pred_Pa_mech_ci"])
_fr = [100 * v for v in MU["sharklet_minus_snake_frac_pos"].values()]
mac["SharkMoreMin"] = f(min(_fr), 0)
mac["SharkMoreMax"] = f(max(_fr), 0)
mac["SharkBelowSnake"] = f(100 * (1 - np.mean(list(MU["sharklet_minus_snake_frac_pos"].values()))), 0)
for rd, k in (("fig5a", ""), ("fig5b", "B")):
    EP = S["endpoint"][rd]
    mac[f"Alpha{k}"] = f(EP["alpha"])
    mac[f"AlphaCI{k}"] = ci(EP["alpha_ci"])
    mac[f"PredPa{k}"] = f(EP["pred_LR_Pa"])
    mac[f"PredPaCI{k}"] = ci(EP["pred_LR_Pa_ci"])
    mac[f"PredPaRem{k}"] = f(EP["pred_remaining_Pa"], 3)
    mac[f"PredPaRemCI{k}"] = f"{f(EP['pred_remaining_Pa_ci'][0], 3)}--{f(EP['pred_remaining_Pa_ci'][1], 3)}"
    mac[f"PredPaPct{k}"] = f(100 * (1 - EP["pred_remaining_Pa"]), 0)
    mac[f"LosoSa{k}"] = f(EP["loso_Sa_from_Ec"])
    mac[f"LosoSaCI{k}"] = ci(EP["loso_Sa_pp_ci"])
    mac[f"LosoSaPct{k}"] = f(100 * EP["loso_Sa_obs_pct"], 0)
    mac[f"LosoEc{k}"] = f(EP["loso_Ec_from_Sa"])
    mac[f"LosoEcCI{k}"] = ci(EP["loso_Ec_pp_ci"])
    mac[f"LosoEcPct{k}"] = f(100 * EP["loso_Ec_obs_pct"], 0)
SH = S["sharklet"]
for s in SP:
    a = AB[s]
    mac[f"Shark{a}"] = f(SH[s]["pred_alpha1"])
    mac[f"SharkLate{a}"] = f(SH[s]["L_t4.5"])
    mac[f"SharkAlpha{a}"] = f(SH[s]["pred_alpha48h"])
    mac[f"SharkObs{a}"] = f(SH[s]["obs_min"]) if abs(SH[s]["obs_min"] - SH[s]["obs_max"]) < 1e-9 else f"{f(SH[s]['obs_min'])}--{f(SH[s]['obs_max'])}"
N = S["numerics"]
mac["NumMean"] = f(N["mean_abs_dL_dx0.1_vs_0.07"], 3)
mac["NumMax"] = f(N["max_abs_dL_dx0.1_vs_0.07"])
mac["NumRel"] = f(100 * N["max_rel_dL_dx0.1_vs_0.07"], 0)
mac["NumCoarse"] = f(N["max_abs_dL_dx0.3_vs_0.07"])
mac["NumScreen"] = f(N["max_abs_dL_dx0.2_vs_0.07"])
mac["NumTime"] = f(max(N["max_abs_dL_time_0.1"], N["max_abs_dL_time_0.025"]), 4)
mac["NumDomTwo"] = f(N["max_abs_dL_domain_24.0"])
mac["NumDomFour"] = f(N["domain48_abs_dL"])
mac["NumShark"] = f(N["sharklet_max_abs_dL_dx0.1_vs_0.07"])
ko = pd.read_csv(T / "knockouts_dL.csv").set_index(["variant", "species"])
for v, k in (("no_transport", "Tr"), ("no_slope", "Sl"), ("no_appendages", "Ap"), ("no_compliance", "Co"), ("no_dynamic_eps", "Eps"), ("no_steric", "St")):
    for s in SP:
        r = ko.loc[(v, s)]
        mac[f"KO{k}{AB[s]}"] = f(r.dL_mean)
        mac[f"KO{k}{AB[s]}CI"] = f"{f(r.lo)} to {f(r.hi)}" if r.lo < 0 else f"{f(r.lo)}--{f(r.hi)}"
mac["DesSeedSD"] = f(S["design_seed_sd_median"], 3)
DK = {"L_min": "Min", "L_max": "Max", "L_h05_b22": "LowTip", "L_h30_b22": "HighTip", "L_h15_b14": "Narrow",
      "L_h15_b34": "Wide", "L_pa34": "Close", "L_pa59": "Far"}
for s_ in SP:
    for k, v in S["design"][s_].items():
        mac["Des" + DK[k] + AB[s_]] = f(v)
sg = pd.read_csv(T / "sobol_geometry.csv").set_index(["species", "parameter"])
GK = {"pitch_f": "Pf", "pitch_a": "Pa", "length": "Len", "base_width": "Bw", "taper": "Tap", "tip_height": "Th",
      "pit_a": "PitA", "pit_b": "PitB", "pit_depth": "PitD", "jitter": "Jit", "rough_amp": "Rough"}
for p, k in GK.items():
    vals = [sg.loc[(s_, p)].ST for s_ in SP]
    mac[f"ST{k}"] = f"{f(min(vals))}--{f(max(vals))}" if f(min(vals)) != f(max(vals)) else f(min(vals))
mac["STMinor"] = f(max(sg.loc[(s_, p)].ST for s_ in SP for p in ("pit_a", "pit_b", "pit_depth", "length", "pitch_f", "jitter", "rough_amp", "taper")))
mac["KOCoMin"] = f(min(-ko.loc[("no_compliance", s_)].dL_mean for s_ in SP))
mac["KOCoMax"] = f(max(-ko.loc[("no_compliance", s_)].dL_mean for s_ in SP))
mac["KOApMax"] = f(max(-ko.loc[("no_appendages", s_)].dL_mean for s_ in SP))
mac["KOSlMin"] = f(min(ko.loc[("no_slope", s_)].dL_mean for s_ in SP))
mac["KOSlMax"] = f(max(ko.loc[("no_slope", s_)].dL_mean for s_ in SP))
sm = pd.read_csv(T / "sobol_mechanism.csv").set_index(["surface", "species", "parameter"])
for p, k in (("beta", "Beta"), ("body_adhesion", "Adh"), ("adhesion_range", "Rng"), ("transport_penalty", "Trp"), ("appendage", "App"), ("anchor_rate", "Anc"), ("eps_kinetics", "EpsK")):
    vals = [sm.loc[("snake", s, p)].ST for s in SP]
    mac[f"SM{k}"] = f"{f(min(vals))}--{f(max(vals))}"
    mac[f"SMc{k}"] = f(sm.loc[("snake", "contrast", p)].ST)
TM = S["time_match"]
mac["TmT"] = f(TM["t_star"], 0)
mac["TmPa"] = f(TM["P. aeruginosa"])
mac["TmSa"] = f(TM["S. aureus"])
CS = S["chemistry_shift"]
for s in SP:
    mac[f"ChemLow{AB[s]}"] = f(-CS[s]["0.8"])
    mac[f"ChemHigh{AB[s]}"] = f(CS[s]["1.25"])

with open(M / "numbers.tex", "w") as fh:
    fh.write("% Generated by make_tables.py from results_v2/summary.json -- do not edit by hand\n")
    for k, v in mac.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(mac), "macros")


# ------------------------------------------------------------------- tables
def it(s):
    g, sp = s.split()
    return f"\\textit{{{g}~{sp}}}"


rows = []
for s in SP:
    a = AB[s]
    if s == "P. aeruginosa":
        obs, pred = "--", f"{mac['PredPa']} ({mac['PredPaCI']})"
    else:
        obs = f"{mac[f'LRobs{a}']} $\\pm$ {mac[f'SEobs{a}']}"
        pred = f"{mac['LosoSa']} ({mac['LosoSaCI']})" if s == "S. aureus" else f"{mac['LosoEc']} ({mac['LosoEcCI']})"
    rows.append(f"{it(s)} & {mac[f'L{a}']} ({mac[f'L{a}CI']}) & {obs} & {pred} & {mac[f'Shark{a}']} & {mac[f'SharkObs{a}']} \\\\")
(M / "tab_main.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Model log reductions and the observations they predict. $L$ is the model $\log_{10}$ reduction relative to a flat surface of the same material, with a 95\% bootstrap interval over \EnsN{} blade-array reconstructions. Observed 48~h reductions on \textit{P.~regius} scales relative to polystyrene are means $\pm$ s.e.\ from three biological replicates \cite{Peroutka2026}. Predicted 48~h reductions for \textit{E.~coli} and \textit{S.~aureus} use the endpoint exponent fitted to the other species alone, with 95\% Monte Carlo predictive intervals that propagate geometry-bootstrap and observation uncertainty; the \textit{P.~aeruginosa} prediction uses the exponent fitted to both species. Sharklet AF reductions in short-term attachment assays on silicone \cite{May2014,Xu2017} are compared with $L$ at $t=1$ without any fitted parameter.}
\label{tab:main}
\small
\begin{tabular}{lccccc}
\toprule
 & \multicolumn{3}{c}{\textit{P.~regius} blade arrays, 48 h} & \multicolumn{2}{c}{Sharklet AF, 1--4 h}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Species & model $L$ & observed & predicted & model & observed \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# species table
fields = [("body_radius", "$R$", r"\si{\micro\metre}"), ("body_length", "$L_c$", r"\si{\micro\metre}"), ("compliance", "$c$", "--"),
          ("body_adhesion", r"$\gamma_b$", "eff."), ("adhesion_range", r"$\lambda_a$", r"\si{\micro\metre}"), ("compliance_penalty", "$k_c$", "eff."),
          ("steric_penalty", "$k_s$", "eff."), ("slope_penalty", r"$\lambda_\nabla$", "eff."), ("appendage_count", "$N_p$", "--"),
          ("appendage_length", "$L_p$", r"\si{\micro\metre}"), ("appendage_binding", r"$k_{\mathrm{bind}}$", "eff."), ("appendage_strength", r"$\gamma_p$", "eff."),
          ("appendage_slope_penalty", r"$\mu_p$", "eff."), ("motility_um_s", "$U$", r"\si{\micro\metre\per\second}"), ("transport_sigma", r"$\sigma_t$", r"\si{\micro\metre}"),
          ("transport_slope_penalty", r"$\alpha_s$", "eff."), ("transport_relief_penalty", r"$\alpha_r$", "eff."), ("detach_rate", "$k_{TF}$", "$t_m^{-1}$"),
          ("anchor_rate", "$k_{TA}$", "$t_m^{-1}$"), ("reorientation_rate", "$k_{AT}$", "$t_m^{-1}$"), ("eps_activation_rate", "$k_{AE}$", "$t_m^{-1}$"),
          ("maturation_rate", "$k_{EM}$", "$t_m^{-1}$"), ("eps_production", r"$s_{\mathrm{EPS}}$", "$t_m^{-1}$")]
lines = []
for fld, sym, unit in fields:
    vals = " & ".join(f"{getattr(SPECIES[s], fld):g}" for s in SP)
    label = "swimming speed" if fld == "motility_um_s" else fld.replace("_", " ")
    lines.append(f"{label} & {sym} & {vals} & {unit} \\\\")
(M / "tab_species.tex").write_text(r"""\begingroup
\small
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{0.86}
\setlength{\LTpre}{0pt}
\setlength{\LTpost}{0pt}
\begin{longtable}{llcccl}
\caption{Species coefficients. Cell dimensions and swimming speeds are literature-informed; the remaining coefficients are phenomenological (``eff.'') and were not fitted. $t_m$ denotes model time.}\label{tab:species}\\
\toprule
Quantity & Symbol & \textit{E.~coli} & \textit{S.~aureus} & \textit{P.~aeruginosa} & Unit \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{longtable}
\endgroup
""")

c = ModelConfig()
shared = [("Production / screening grid spacing", r"$\Delta x$", "0.10 / 0.20", r"\si{\micro\metre}"),
          ("Sensing scale", r"$\ell_s$", f"{c.sense_scale}", r"\si{\micro\metre}"),
          ("Curvature smoothing", r"$\ell_c$", f"{c.curv_scale}", r"\si{\micro\metre}"),
          ("State time step / horizon", "$h$, $t_{\\mathrm{end}}$", f"{c.h_state} / {c.horizon}", "$t_m$"),
          ("Attachment scale", r"$\beta$", f"{c.beta}", "eff."),
          ("EPS nonlinear coefficient / exponent", r"$D_0$, $\nu$", f"{c.eps_diffusivity} / {c.eps_diffusion_exponent}", r"\si{\micro\metre\squared}\,$t_m^{-1}$ / --"),
          ("EPS linear diffusivity", r"$D_{\mathrm{lin}}$", f"{c.eps_linear_diffusivity}", r"\si{\micro\metre\squared}\,$t_m^{-1}$"),
          ("EPS decay / yield", r"$k_d$, $\theta_y$", f"{c.eps_decay} / {c.eps_yield}", r"$t_m^{-1}$ / $\si{\per\micro\metre\squared}$"),
          ("EPS gate thresholds", r"$e_{\mathrm{att}}$, $e_{\mathrm{mat}}$", f"{c.eps_attach_threshold} / {c.eps_mature_threshold}", "eff."),
          ("Arrival / capture / replenishment", r"$\lambda_{\mathrm{arr}}$, $k_{\mathrm{cap}}$, $\rho$", f"{c.arrival_rate} / {c.capture_rate} / {c.replenish_rate}", "-- / $t_m^{-1}$ / $t_m^{-1}$"),
          ("Roughness correlation length", r"$\ell_\eta$", "0.6", r"\si{\micro\metre}")]
(M / "tab_shared.tex").write_text(r"""\begin{table}[h]\centering\small
\caption{Shared model settings.}\label{tab:shared}
\begin{tabular}{llll}\toprule Quantity & Symbol & Value & Unit\\\midrule
""" + "\n".join(f"{a} & {b} & {v} & {u}\\\\" for a, b, v, u in shared) + r"""
\bottomrule\end{tabular}\end{table}
""")

pr = []
names = {"pitch_f": r"Following-tip spacing $p_f$ (\si{\micro\metre})", "pitch_a": r"Adjacent-tip spacing $p_a$ (\si{\micro\metre})",
         "length": r"Projected blade length $\ell$ (\si{\micro\metre})", "base_width": r"Blade base width $w_0$ (\si{\micro\metre})",
         "taper": r"Taper exponent $q$", "tip_height": r"Tip height $h_t$ (\si{\micro\metre})",
         "pit_a": r"Pit semi-axis along blade $a_p$ (\si{\micro\metre})", "pit_b": r"Pit semi-axis across blade $b_p$ (\si{\micro\metre})",
         "pit_depth": r"Pit depth $d_p$ (\si{\micro\metre})", "jitter": r"Lattice jitter $\sigma_j$ (\si{\micro\metre})",
         "rough_amp": r"Roughness SD $a_\eta$ (\si{\micro\metre})"}
SEMEST = {"base_width", "pit_a", "pit_b", "taper"}
for k, spec in GEOM_PRIOR.items():
    if spec[0] == "normal":
        d = f"$\\mathcal N({spec[1]},{spec[2]}^2)$ on $[{spec[3]},{spec[4]}]$"
        src = "measured \\cite{Peroutka2026}"
    else:
        d = f"$\\mathcal U({spec[1]},{spec[2]})$"
        src = "SEM estimate, Fig.~1b of \\cite{Peroutka2026}" if k in SEMEST else "unmeasured"
    pr.append(f"{names[k]} & {d} & {src}\\\\")
(M / "tab_prior.tex").write_text(r"""\begin{table}[h]\centering\small
\caption{Geometric prior for the blade-array ensemble.}\label{tab:prior}
\begin{tabular}{lll}\toprule Quantity & Prior & Source\\\midrule
""" + "\n".join(pr) + r"""
\bottomrule\end{tabular}\end{table}
""")

nu = pd.read_csv("results_v2/data/numerics.csv")
nu["L"] = -np.log10(nu.Y)
gp = nu[nu.test == "grid"].pivot_table(index=["surface", "species"], columns="dx", values="L")
lines = []
for (surf, s), r in gp.iterrows():
    lines.append(f"{surf} & {it(s)} & " + " & ".join(f"{r[d]:.2f}" for d in (0.3, 0.2, 0.15, 0.1, 0.07)) + r"\\")
sh = pd.read_csv("results_v2/data/numerics_sharklet.csv")
sh["L"] = -np.log10(sh.Y)
for s in SP:
    r = sh[sh.species == s].set_index("dx").L
    lines.append(f"Sharklet & {it(s)} & " + " & ".join(f"{r[d]:.2f}" for d in (0.3, 0.2, 0.15, 0.1, 0.07)) + r"\\")
(M / "tab_numerics.tex").write_text(r"""\begin{table}[!ht]
\centering\small
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.05}
\caption{Grid refinement. Entries are $L=-\log_{10}Y$ for the nominal blade array (0), three draws from the prior (1--3) and the Sharklet AF-type cell. For the four blade arrays, the maximum $|\Delta L|$ between $\Delta x=0.10$ and \SI{0.07}{\micro\metre} is \NumMax{} (mean \NumMean{}); the Sharklet cell changes by at most \NumShark{}. Halving or doubling the state time step changes $L$ by at most \NumTime{}.}\label{tab:numerics}
\begin{tabular}{llccccc}
\toprule Geometry & Species & \multicolumn{5}{c}{$\Delta x$ (\si{\micro\metre})}\\
 & & 0.30 & 0.20 & 0.15 & 0.10 & 0.07\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

sgt = pd.read_csv(T / "sobol_geometry.csv")
lines = []
for p in list(sgt[sgt.species == "E. coli"].sort_values("ST", ascending=False).parameter):
    cells = []
    for s in SP:
        r = sgt[(sgt.species == s) & (sgt.parameter == p)].iloc[0]
        cells.append(f"{r.ST:.3f} [{max(r.ST_lo, 0):.3f}, {r.ST_hi:.3f}]")
    lines.append(f"{p.replace('_', ' ')} & " + " & ".join(cells) + r"\\")
(M / "tab_sobol_geom.tex").write_text(r"""\begin{table}[h]\centering\small
\caption{Total-order Sobol indices of $L$ with respect to blade-array geometry (Saltelli design, $n=64$ base samples, 832 geometries, $\Delta x=\SI{0.2}{\micro\metre}$, common lattice and roughness seeds; 95\% bootstrap intervals). Each input was sampled uniformly over the bounded support of its prior (table~\ref{tab:prior}); the indices describe variance over this design space, not variance under the truncated-normal production priors.}\label{tab:sobolgeom}
\begin{tabular}{lccc}\toprule Parameter & \textit{E.~coli} & \textit{S.~aureus} & \textit{P.~aeruginosa}\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}\end{table}
""")

smt = pd.read_csv(T / "sobol_mechanism.csv")
smt = smt[smt.surface == "snake"]
lines = []
order = list(smt[smt.species == "E. coli"].sort_values("ST", ascending=False).parameter)
for p in order:
    cells = []
    for s in SP + ["contrast"]:
        r = smt[(smt.species == s) & (smt.parameter == p)].iloc[0]
        cells.append(f"{r.ST:.3f}")
    lines.append(f"{p.replace('_', ' ')} & " + " & ".join(cells) + r"\\")
(M / "tab_sobol_mech.tex").write_text(r"""\begin{table}[h]\centering\small
\caption{Total-order Sobol indices with respect to mechanistic coefficients on the nominal blade array. Each multiplier is log-uniform on $[1/2,2]$ and scales the coefficient for all three species ($n=64$ base samples, 896 coefficient sets, $\Delta x=\SI{0.2}{\micro\metre}$). First-order estimates were unstable at this sample size and are not reported. ``Contrast'' is $L_{Sa}/L_{Ec}$.}\label{tab:sobolmech}
\begin{tabular}{lcccc}\toprule Multiplier & $L$ \textit{E.~coli} & $L$ \textit{S.~aureus} & $L$ \textit{P.~aeruginosa} & contrast\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}\end{table}
""")

ch = pd.read_csv(T / "chemistry_L.csv")
lines = []
for s in SP:
    g = ch[ch.species == s].set_index("flat_adhesion_factor").L
    lines.append(f"{it(s)} & " + " & ".join(f"{g[x]:.2f}" for x in (0.8, 0.9, 1.0, 1.1, 1.25)) + r"\\")
(M / "tab_chem.tex").write_text(r"""\begin{table}[h]\centering\small
\caption{Sensitivity of $L$ to a material difference between patterned surface and control. The control is simulated with body adhesion $\gamma_b$ scaled by $f$ while propensities on both surfaces are referenced to the unscaled flat energy (8 ensemble geometries, $\Delta x=\SI{0.15}{\micro\metre}$).}\label{tab:chem}
\begin{tabular}{lccccc}\toprule Species & $f=0.8$ & 0.9 & 1.0 & 1.1 & 1.25\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}\end{table}
""")
print("tables written")
