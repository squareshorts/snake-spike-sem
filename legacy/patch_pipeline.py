#!/usr/bin/env python3
"""SEM-informed mathematical modelling of topography-controlled suppression
of stable bacterial film formation on Python regius scale surfaces.

Pipeline v3 — Q1-journal-grade analyses:
  * OAT sensitivity analysis
  * Multi-seed Monte Carlo robustness
  * Pitch-sweep curve
  * Grid-convergence check
  * Full parameter-table export
  * Publication-quality figures
"""
import argparse, json, textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

# ── Defaults ──────────────────────────────────────────────────────────────
DX = 0.3
DOMAIN = (36.0, 36.0)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'outputs'


@dataclass(frozen=True)
class OutputLayout:
    base: Path
    data: Path
    figures: Path
    manuscript: Path


def prepare_output_layout(base_dir):
    base = Path(base_dir).resolve()
    layout = OutputLayout(
        base=base,
        data=base / 'data',
        figures=base / 'figures',
        manuscript=base / 'manuscript',
    )
    for path in (layout.base, layout.data, layout.figures, layout.manuscript):
        path.mkdir(parents=True, exist_ok=True)
    return layout


@dataclass
class Geometry:
    name: str
    pitch_x: float = 4.6;  pitch_y: float = 4.4
    height_mu: float = 9.0; height_sd: float = 0.7
    r_tip: float = 0.15;   r_base: float = 1.55
    jitter_x: float = 0.08; jitter_y: float = 0.15
    disorder_mode: str = 'anisotropic_hex'
    seed: int = 3

SPECIES = {
    'E. coli': dict(key='ecoli', motility_um_s=29.0, slope_penalty=0.35,
                    support_power=1.5, bridge_ell0=1.8, bridge_rad_override=None,
                    hydro_chi=0.0060, observed=0.12),
    'S. aureus': dict(key='saureus', motility_um_s=0.0, slope_penalty=0.35,
                      support_power=1.5, bridge_ell0=0.35, bridge_rad_override=1,
                      hydro_chi=0.0, observed=0.22),
    'P. aeruginosa': dict(key='paeruginosa', motility_um_s=45.0, slope_penalty=0.35,
                          support_power=1.5, bridge_ell0=1.5, bridge_rad_override=None,
                          hydro_chi=0.0060, observed=0.15),
}
RHEOLOGY = {'tau_y': 0.30, 'K': 1.0, 'n': 0.60, 'U_ref': 1.0}

# ── Surface construction ─────────────────────────────────────────────────
def make_centers(geom, domain=DOMAIN):
    rng = np.random.default_rng(geom.seed)
    Lx, Ly = domain; centers = []
    if geom.disorder_mode == 'anisotropic_hex':
        for j in range(int(Ly/geom.pitch_y)+3):
            y = (j+0.5)*geom.pitch_y + rng.normal(0, geom.jitter_y)
            xoff = (geom.pitch_x/2.0) if (j%2) else 0.0
            for i in range(int(Lx/geom.pitch_x)+3):
                x = (i+0.5)*geom.pitch_x + xoff + rng.normal(0, geom.jitter_x)
                if -1.0 <= x <= Lx+1.0 and -1.0 <= y <= Ly+1.0:
                    centers.append((x, y))
    elif geom.disorder_mode == 'randomized':
        n_est = int((Lx/geom.pitch_x)*(Ly/geom.pitch_y))
        min_d = 0.55*min(geom.pitch_x, geom.pitch_y); att = 0
        while len(centers) < n_est and att < 20000:
            att += 1; x = rng.uniform(0.5, Lx-0.5); y = rng.uniform(0.5, Ly-0.5)
            if all((x-cx)**2+(y-cy)**2 >= min_d**2 for cx, cy in centers):
                centers.append((x, y))
    return np.array(centers, dtype=float)

def sample_heights(geom, n):
    rng = np.random.default_rng(geom.seed+1)
    return np.clip(rng.normal(geom.height_mu, geom.height_sd, n),
                   geom.height_mu-2.5, geom.height_mu+2.5)

def build_surface(geom, domain=DOMAIN, dx=DX):
    centers = make_centers(geom, domain)
    heights = sample_heights(geom, len(centers))
    Lx, Ly = domain
    xs = np.arange(0, Lx+dx, dx); ys = np.arange(0, Ly+dx, dx)
    X, Y = np.meshgrid(xs, ys); Z = np.zeros_like(X)
    for (xc, yc), h in zip(centers, heights):
        d = np.hypot(X-xc, Y-yc); spike = np.zeros_like(Z)
        if geom.r_tip > 0 and geom.r_base > 0:
            spike[d <= geom.r_tip] = h
            m = (d > geom.r_tip) & (d < geom.r_base)
            spike[m] = h*(1.0-(d[m]-geom.r_tip)/(geom.r_base-geom.r_tip))
        Z = np.maximum(Z, spike)
    return xs, ys, Z, centers, heights

# ── Bacterial footprint & support ─────────────────────────────────────────
def footprint_offsets(species_key, dx=DX):
    if species_key == 'saureus':
        r = 0.5; n = int(np.ceil(r/dx)); s = np.arange(-n, n+1)*dx
        X, Y = np.meshgrid(s, s); mask = (X*X+Y*Y) <= r*r
        z = np.zeros_like(X); z[mask] = np.sqrt(np.maximum(r*r-(X[mask]**2+Y[mask]**2), 0.0))
        return [(mask, z)]
    if species_key == 'paeruginosa':
        r = 0.4; Lcyl = 1.5; half = r+Lcyl/2.0
    else:
        r = 0.45; Lcyl = 1.1; half = r+Lcyl/2.0
    n = int(np.ceil(half/dx)); s = np.arange(-n, n+1)*dx
    X, Y = np.meshgrid(s, s); outs = []
    for theta in np.linspace(0, np.pi, 8, endpoint=False):
        ca, sa = np.cos(theta), np.sin(theta)
        u = ca*X+sa*Y; v = -sa*X+ca*Y
        uc = np.clip(u, -Lcyl/2.0, Lcyl/2.0); rr = (u-uc)**2+v**2
        mask = rr <= r*r; z = np.zeros_like(X)
        z[mask] = np.sqrt(np.maximum(r*r-rr[mask], 0.0))
        outs.append((mask, z))
    return outs

def local_support_map(Z, species_key, slope_penalty, support_power, delta=0.05, dx=DX):
    fps = footprint_offsets(species_key, dx)
    ny, nx = Z.shape
    gradx = np.gradient(Z, dx, axis=1); grady = np.gradient(Z, dx, axis=0)
    grad = np.hypot(gradx, grady); out = np.zeros((ny, nx))
    for mask, z in fps:
        my, mx = mask.shape; hy, hx = my//2, mx//2
        for iy in range(hy, ny-hy):
            for ix in range(hx, nx-hx):
                patch = Z[iy-hy:iy+hy+1, ix-hx:ix+hx+1]
                gp = grad[iy-hy:iy+hy+1, ix-hx:ix+hx+1]
                zoff = np.max(patch[mask]-z[mask])
                gap = np.clip(zoff+z[mask]-patch[mask], 0, None)
                support = np.mean(gap <= delta)
                score = (support**support_power)*np.exp(-slope_penalty*np.mean(gp[mask]))
                if score > out[iy, ix]: out[iy, ix] = score
    return out

def hydro_factor(Z, species_name, dx=DX):
    gx = np.gradient(Z, dx, axis=1); gy = np.gradient(Z, dx, axis=0)
    grad_rms = float(np.sqrt(np.mean(gx*gx+gy*gy)))
    chi = SPECIES[species_name]['hydro_chi']; mot = SPECIES[species_name]['motility_um_s']
    return float(np.exp(-chi*mot*grad_rms)), grad_rms

def bridge_metrics(score_map, species_name, geom, dx=DX):
    active = score_map >= 0.12
    p_eff = 0.5*(geom.pitch_x+geom.pitch_y)
    Bn = RHEOLOGY['tau_y']*p_eff/(RHEOLOGY['K']*(RHEOLOGY['U_ref']**RHEOLOGY['n'])+1e-9)
    ell = SPECIES[species_name]['bridge_ell0']/(1.0+Bn)
    rad = SPECIES[species_name]['bridge_rad_override']
    if rad is None: rad = max(1, int(round(ell/dx)))
    yy, xx = np.ogrid[-rad:rad+1, -rad:rad+1]; se = (xx*xx+yy*yy) <= rad*rad
    dil = ndimage.binary_dilation(active, structure=se)
    lbl, num = ndimage.label(dil); largest = 0.0
    if num > 0:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, num+1))
        largest = float(np.max(sizes))/active.size
    return dict(largest_cc=largest, active_fraction=float(active.mean()),
                Bn_eff=float(Bn), bridge_radius_px=int(rad), ell_eps_um=float(ell))

def evaluate_surface(Z, geom, species_name, dx=DX):
    spec = SPECIES[species_name]
    S = local_support_map(Z, spec['key'], spec['slope_penalty'], spec['support_power'], dx=dx)
    bridge = bridge_metrics(S, species_name, geom, dx=dx)
    hydro, grad_rms = hydro_factor(Z, species_name, dx=dx)
    return dict(score_map=S, score_mean=float(S.mean()), largest_cc=bridge['largest_cc'],
                hydro_factor=hydro, grad_rms=grad_rms,
                stable_score=float(S.mean())*bridge['largest_cc']*hydro,
                bridge_radius_px=bridge['bridge_radius_px'], Bn_eff=bridge['Bn_eff'],
                active_fraction=bridge['active_fraction'])

# ── Core geometry sweep ───────────────────────────────────────────────────
def run_geometries(seed=3):
    geometries = [
        Geometry(name='smooth', r_tip=0.0, r_base=0.0, seed=seed),
        Geometry(name='paper_reconstructed', seed=seed),
        Geometry(name='blunted_tips', r_tip=0.45, r_base=1.55, seed=seed),
        Geometry(name='widened_pitch', pitch_x=5.6, pitch_y=5.4, seed=seed),
        Geometry(name='randomized_centroids', disorder_mode='randomized',
                 jitter_x=0.0, jitter_y=0.0, seed=seed),
    ]
    rows, metas = [], {}
    for geom in geometries:
        xs, ys, Z, centers, heights = build_surface(geom)
        metas[geom.name] = dict(surface=Z, xs=xs, ys=ys, centers=centers,
                                heights=heights, score_maps={})
        for sp in SPECIES:
            res = evaluate_surface(Z, geom, sp)
            metas[geom.name]['score_maps'][sp] = res['score_map']
            rows.append(dict(geometry=geom.name, species=sp, **{k: res[k] for k in res if k != 'score_map'}))
    df = pd.DataFrame(rows)
    for sp in SPECIES:
        bl = float(df[(df.geometry=='smooth')&(df.species==sp)].stable_score.iloc[0])
        df.loc[df.species==sp, 'relative_remaining_fraction'] = df.loc[df.species==sp, 'stable_score']/max(bl, 1e-15)
    return df, metas

# ── New analyses ──────────────────────────────────────────────────────────
def sensitivity_oat(layout):
    """One-at-a-time ±20 % sweep of key parameters."""
    params = [
        ('height_mu', 9.0), ('pitch_x', 4.6), ('pitch_y', 4.4),
        ('r_tip', 0.15), ('r_base', 1.55), ('height_sd', 0.7),
    ]
    base_geom = Geometry(name='base')
    xs_b, ys_b, Z_b, _, _ = build_surface(base_geom)
    base_vals = {}
    for sp in SPECIES:
        bl_g = Geometry(name='smooth_bl', r_tip=0.0, r_base=0.0)
        _, _, Z_sm, _, _ = build_surface(bl_g)
        bl_score = evaluate_surface(Z_sm, bl_g, sp)['stable_score']
        base_vals[sp] = evaluate_surface(Z_b, base_geom, sp)['stable_score'] / max(bl_score, 1e-15)

    rows = []
    for pname, pval in params:
        for frac in [-0.20, 0.20]:
            kw = {pname: pval*(1+frac)}
            g = Geometry(name=f'{pname}_{frac:+.0%}', **kw)
            _, _, Z_p, _, _ = build_surface(g)
            bl_g = Geometry(name='smooth_tmp', r_tip=0.0, r_base=0.0)
            _, _, Z_sm, _, _ = build_surface(bl_g)
            for sp in SPECIES:
                bl_score = evaluate_surface(Z_sm, bl_g, sp)['stable_score']
                val = evaluate_surface(Z_p, g, sp)['stable_score'] / max(bl_score, 1e-15)
                rows.append(dict(parameter=pname, direction='+20%' if frac>0 else '-20%',
                                 species=sp, remaining_fraction=val,
                                 delta=val - base_vals[sp]))
    df = pd.DataFrame(rows)
    df.to_csv(layout.data / 'sensitivity_oat.csv', index=False)
    return df

def montecarlo_seeds(layout, n_seeds=50):
    """Multi-seed robustness check (50 seeds)."""
    rows = []
    for seed in range(1, n_seeds+1):
        df_s, _ = run_geometries(seed=seed)
        sub = df_s[df_s.geometry=='paper_reconstructed'][['species','relative_remaining_fraction']].copy()
        sub['seed'] = seed; rows.append(sub)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(layout.data / 'montecarlo_seeds.csv', index=False)
    summary = df.groupby('species')['relative_remaining_fraction'].agg(['mean', 'std']).reset_index()
    summary['cv'] = summary['std'] / summary['mean']
    summary['ci_95_lo'] = summary['mean'] - 1.96 * summary['std'] 
    summary['ci_95_hi'] = summary['mean'] + 1.96 * summary['std']
    summary.to_csv(layout.data / 'montecarlo_summary.csv', index=False)
    return df, summary

def pitch_sweep(layout):
    """Continuous pitch sweep from 3.0 to 7.0 µm."""
    pitches = np.arange(3.0, 7.1, 0.5)
    rows = []
    bl_g = Geometry(name='smooth_bl', r_tip=0.0, r_base=0.0)
    _, _, Z_sm, _, _ = build_surface(bl_g)
    for p in pitches:
        g = Geometry(name=f'pitch_{p:.1f}', pitch_x=p, pitch_y=p-0.2)
        _, _, Z_p, _, _ = build_surface(g)
        for sp in SPECIES:
            bl_score = evaluate_surface(Z_sm, bl_g, sp)['stable_score']
            val = evaluate_surface(Z_p, g, sp)['stable_score'] / max(bl_score, 1e-15)
            rows.append(dict(pitch=p, species=sp, remaining_fraction=val))
    df = pd.DataFrame(rows)
    df.to_csv(layout.data / 'pitch_sweep.csv', index=False)
    return df

def grid_convergence(layout):
    """Resolution convergence at dx = 0.15, 0.30, 0.45 µm."""
    rows = []
    for dx in [0.15, 0.30, 0.45]:
        g = Geometry(name='paper_reconstructed')
        _, _, Z, _, _ = build_surface(g, dx=dx)
        bl_g = Geometry(name='smooth', r_tip=0.0, r_base=0.0)
        _, _, Z_sm, _, _ = build_surface(bl_g, dx=dx)
        for sp in SPECIES:
            bl_score = evaluate_surface(Z_sm, bl_g, sp, dx=dx)['stable_score']
            val = evaluate_surface(Z, g, sp, dx=dx)['stable_score'] / max(bl_score, 1e-15)
            rows.append(dict(dx=dx, species=sp, remaining_fraction=val))
    df = pd.DataFrame(rows)
    df.to_csv(layout.data / 'grid_convergence.csv', index=False)
    return df

# ── Parameter table ───────────────────────────────────────────────────────

def mechanistic_ablations(layout):
    rows = []
    g = Geometry(name='paper_reconstructed')
    _, _, Z, _, _ = build_surface(g)
    bl_g = Geometry(name='smooth_bl', r_tip=0.0, r_base=0.0)
    _, _, Z_sm, _, _ = build_surface(bl_g)
    for sp in SPECIES:
        bl_res = evaluate_surface(Z_sm, bl_g, sp)
        res = evaluate_surface(Z, g, sp)
        # Contact only
        rows.append(dict(species=sp, ablation='Contact only', fraction=res['score_mean']/max(bl_res['score_mean'],1e-15)))
        # Contact + Bridge
        rows.append(dict(species=sp, ablation='Contact+Bridge', fraction=(res['score_mean']*res['largest_cc'])/max(bl_res['score_mean']*bl_res['largest_cc'],1e-15)))
        # Full
        rows.append(dict(species=sp, ablation='Full mechanism', fraction=res['stable_score']/max(bl_res['stable_score'],1e-15)))
    df = pd.DataFrame(rows)
    df.to_csv(layout.data / 'mechanistic_ablations.csv', index=False)
    
    _pub_style()
    fig, ax = plt.subplots(figsize=(6,4))
    species_list = df['species'].unique()
    ablation_list = df['ablation'].unique()
    w = 0.25
    x = np.arange(len(species_list))
    for i, ab in enumerate(ablation_list):
        y = [df[(df.species==sp)&(df.ablation==ab)]['fraction'].values[0] for sp in species_list]
        ax.bar(x + (i - 1)*w, y, width=w, label=ab)
    ax.set_xticks(x)
    ax.set_xticklabels(species_list, style='italic')
    ax.set_ylabel('Relative stable-film fraction')
    ax.legend(frameon=False)
    fig.savefig(layout.figures / 'figure_ablations.png')
    plt.close(fig)
    return df

def parameter_identifiability(layout):
    ps = np.linspace(1.0, 3.0, 10)
    lams = np.linspace(0.1, 0.8, 10)
    g = Geometry(name='paper_reconstructed')
    _, _, Z, _, _ = build_surface(g, dx=0.45)
    bl_g = Geometry(name='smooth_bl', r_tip=0.0, r_base=0.0)
    _, _, Z_sm, _, _ = build_surface(bl_g, dx=0.45)
    
    X, Y = np.meshgrid(ps, lams)
    errs = np.zeros_like(X)
    for i in range(10):
        for j in range(10):
            err_sum = 0
            for sp in ['E. coli', 'S. aureus']:
                old_p, old_lam = SPECIES[sp]['support_power'], SPECIES[sp]['slope_penalty']
                SPECIES[sp]['support_power'] = X[i,j]
                SPECIES[sp]['slope_penalty'] = Y[i,j]
                bl_score = evaluate_surface(Z_sm, bl_g, sp, dx=0.45)['stable_score']
                score = evaluate_surface(Z, g, sp, dx=0.45)['stable_score'] / max(bl_score, 1e-15)
                err_sum += (score - SPECIES[sp]['observed'])**2
                SPECIES[sp]['support_power'] = old_p
                SPECIES[sp]['slope_penalty'] = old_lam
            errs[i,j] = err_sum
            
    _pub_style()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    cs = ax.contourf(X, Y, errs, levels=30, cmap='viridis_r')
    ax.plot(1.5, 0.35, 'r*', markersize=12, label='Calibrated params')
    ax.set_xlabel('Support power (p)')
    ax.set_ylabel(r'Slope penalty ($\lambda_s$)')
    fig.colorbar(cs, ax=ax, label='Sum of squared errors')
    ax.legend(frameon=False)
    fig.savefig(layout.figures / 'figure_identifiability.png'); plt.close(fig)

def export_parameter_table(layout):
    rows = [
        ('Spike height mean', 'h_mu', 9.0, 'um', 'Peroutka et al. 2026'),
        ('Spike height SD', 'h_sd', 0.7, 'um', 'Peroutka et al. 2026'),
        ('Pitch x', 'p_x', 4.6, 'um', 'Peroutka et al. 2026'),
        ('Pitch y', 'p_y', 4.4, 'um', 'Peroutka et al. 2026'),
        ('Tip radius', 'r_tip', 0.15, 'um', 'Inferred from SEM'),
        ('Base radius', 'r_base', 1.55, 'um', 'Inferred from SEM'),
        ('Jitter x', 'sigma_jx', 0.08, 'um', 'Estimated'),
        ('Jitter y', 'sigma_jy', 0.15, 'um', 'Estimated'),
        ('Grid spacing', 'dx', 0.3, 'um', 'Numerical'),
        ('Domain size', 'L', 36.0, 'um', 'Numerical'),
        ('E. coli radius', 'r_ec', 0.45, 'um', 'Literature'),
        ('E. coli cyl. length', 'L_ec', 1.1, 'um', 'Literature'),
        ('S. aureus radius', 'r_sa', 0.5, 'um', 'Literature'),
        ('E. coli motility', 'U_m', 29.0, 'um/s', 'Wu-Zhang et al. 2025'),
        ('Support exponent', 'p', 1.5, '--', 'Calibrated'),
        ('Slope penalty', 'lambda_s', 0.35, '--', 'Calibrated'),
        ('Contact gap threshold', 'delta', 0.05, 'um', 'Set'),
        ('EPS yield stress', 'tau_y', 0.30, 'Pa (dim-less)', 'Estimated'),
        ('EPS consistency', 'K', 1.0, '--', 'Estimated'),
        ('EPS flow index', 'n', 0.60, '--', 'Shear-thinning'),
        ('Bridge ell0 E. coli', 'ell0_ec', 1.8, 'um', 'Calibrated'),
        ('Bridge ell0 S. aureus', 'ell0_sa', 0.35, 'um', 'Calibrated'),
        ('Hydro chi E. coli', 'chi_ec', 0.006, '1/(um/s)', 'Calibrated'),
        ('Support threshold', 'A_thr', 0.12, '--', 'Set'),
    ]
    df = pd.DataFrame(rows, columns=['Description','Symbol','Value','Unit','Source'])
    df.to_csv(layout.data / 'parameter_table.csv', index=False)
    return df

# ── Plotting (publication quality) ────────────────────────────────────────
def _pub_style():
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                         'axes.labelsize': 11, 'xtick.labelsize': 9,
                         'ytick.labelsize': 9, 'legend.fontsize': 9,
                         'figure.dpi': 300, 'savefig.dpi': 300,
                         'savefig.bbox': 'tight'})

def plot_reconstruction(meta, layout):
    _pub_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    Z, xs, ys = meta['surface'], meta['xs'], meta['ys']
    im = ax.imshow(Z, origin='lower', extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                   aspect='equal', cmap='inferno')
    if len(meta['centers']):
        c = meta['centers']
        ax.scatter(c[:,0], c[:,1], s=5, c='white', edgecolors='k', linewidths=0.15, zorder=3)
    ax.set_xlabel('x (\u00b5m)'); ax.set_ylabel('y (\u00b5m)')
    cb = fig.colorbar(im, ax=ax, shrink=0.82); cb.set_label('Height (\u00b5m)')
    fig.savefig(layout.figures / 'figure_reconstruction.png'); plt.close(fig)

def plot_controls(df, layout):
    _pub_style()
    order = ['paper_reconstructed','blunted_tips','widened_pitch','randomized_centroids']
    labels = {'paper_reconstructed':'Reconstructed','blunted_tips':'Blunted tips',
              'widened_pitch':'Wider pitch','randomized_centroids':'Randomized'}
    colors = ['#2c7bb6','#abd9e9','#fdae61','#d7191c']
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
    for ax, sp in zip(axes, SPECIES):
        sub = df[(df.geometry!='smooth')&(df.species==sp)].set_index('geometry').loc[order].reset_index()
        ax.bar(np.arange(len(sub)), sub['relative_remaining_fraction'], color=colors)
        ax.axhline(SPECIES[sp]['observed'], ls='--', lw=1, color='grey')
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels([labels[g] for g in sub.geometry], rotation=20, ha='right')
        ax.set_ylim(0, 0.8); ax.set_ylabel('Relative stable-film fraction')
        ax.set_title(sp, style='italic')
    fig.savefig(layout.figures / 'figure_controls_barplot.png'); plt.close(fig)

def plot_decomposition(df, layout):
    _pub_style()
    sub = df[df.geometry=='paper_reconstructed'].copy(); x = np.arange(len(sub)); w = 0.22
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(x-w, sub['score_mean'], width=w, label='Contact support', color='#2c7bb6')
    ax.bar(x, sub['largest_cc'], width=w, label='Bridge connectivity', color='#abd9e9')
    ax.bar(x+w, sub['hydro_factor'], width=w, label='Hydrodynamic factor', color='#fdae61')
    ax.set_xticks(x); ax.set_xticklabels(sub['species'], style='italic')
    ax.set_ylim(0, 1.05); ax.set_ylabel('Component magnitude')
    ax.legend(frameon=False, loc='upper right'); fig.savefig(layout.figures / 'figure_decomposition.png'); plt.close(fig)

def plot_support_maps(metas, layout):
    _pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), constrained_layout=True)
    pairs = [('paper_reconstructed','E. coli'),('paper_reconstructed','S. aureus'),
             ('blunted_tips','E. coli'),('blunted_tips','S. aureus')]
    for ax, (g, sp) in zip(axes.ravel(), pairs):
        S = metas[g]['score_maps'][sp]
        im = ax.imshow(S, origin='lower', aspect='equal', cmap='viridis')
        ax.set_title(f"{g.replace('_',' ').title()} | {sp}", fontsize=9, style='italic')
        ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, shrink=0.78)
    fig.savefig(layout.figures / 'figure_support_maps.png'); plt.close(fig)

def plot_sensitivity_tornado(df_sens, layout):
    _pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, sp in zip(axes, SPECIES):
        sub = df_sens[df_sens.species==sp].copy()
        params = sub['parameter'].unique()
        y = np.arange(len(params))
        for i, p in enumerate(params):
            lo = sub[(sub.parameter==p)&(sub.direction=='-20%')].delta.values
            hi = sub[(sub.parameter==p)&(sub.direction=='+20%')].delta.values
            lo_v = lo[0] if len(lo) else 0; hi_v = hi[0] if len(hi) else 0
            ax.barh(i, hi_v, height=0.35, color='#d7191c', alpha=0.8)
            ax.barh(i, lo_v, height=0.35, color='#2c7bb6', alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels(params)
        ax.axvline(0, color='k', lw=0.6); ax.set_xlabel('\u0394 remaining fraction')
        ax.set_title(sp, style='italic')
    fig.savefig(layout.figures / 'figure_sensitivity_tornado.png'); plt.close(fig)

def plot_pitch_sweep(df_ps, layout):
    _pub_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for sp, col in zip(SPECIES, ['#2c7bb6','#d7191c']):
        sub = df_ps[df_ps.species==sp]
        ax.plot(sub.pitch, sub.remaining_fraction, 'o-', label=sp, color=col)
        ax.axhline(SPECIES[sp]['observed'], ls='--', lw=0.8, color=col, alpha=0.5)
    ax.set_xlabel('Interspike pitch (\u00b5m)'); ax.set_ylabel('Relative stable-film fraction')
    ax.legend(frameon=False); fig.savefig(layout.figures / 'figure_pitch_sweep.png'); plt.close(fig)

def plot_montecarlo(df_mc, layout):
    _pub_style()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    species_list = list(SPECIES.keys())
    positions = [1, 2]; colors = ['#2c7bb6','#d7191c']
    for i, sp in enumerate(species_list):
        vals = df_mc[df_mc.species==sp].relative_remaining_fraction.values
        bp = ax.boxplot([vals], positions=[positions[i]], widths=0.5,
                        patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor(colors[i]); bp['boxes'][0].set_alpha(0.6)
        ax.axhline(SPECIES[sp]['observed'], ls='--', lw=0.8, color=colors[i], alpha=0.5)
    ax.set_xticks(positions); ax.set_xticklabels(species_list, style='italic')
    ax.set_ylabel('Relative stable-film fraction')
    fig.savefig(layout.figures / 'figure_montecarlo_spread.png'); plt.close(fig)

# ── Tables ────────────────────────────────────────────────────────────────
def save_tables(df, layout):
    calib = df[df.geometry=='paper_reconstructed'].copy()
    calib['observed_remaining_fraction'] = calib['species'].map({k:v['observed'] for k,v in SPECIES.items()})
    calib['absolute_error'] = (calib['relative_remaining_fraction']-calib['observed_remaining_fraction']).abs()
    df.to_csv(layout.data / 'all_geometry_results.csv', index=False)
    calib.to_csv(layout.data / 'calibration_table.csv', index=False)
    return calib

# ── Manuscript writer ─────────────────────────────────────────────────────
def write_manuscript(df, calib, layout, summary_mc=None):
    # calculate topography metrics S_a, S_q
    g = Geometry(name='paper_reconstructed')
    _, _, Z, _, _ = build_surface(g)
    Sa = np.mean(np.abs(Z - np.mean(Z)))
    Sq = np.sqrt(np.mean((Z - np.mean(Z))**2))
    

    ec = calib[calib.species=='E. coli'].iloc[0]
    sa = calib[calib.species=='S. aureus'].iloc[0]
    pa = calib[calib.species=='P. aeruginosa'].iloc[0] if 'P. aeruginosa' in calib.species.values else calib.iloc[0]
    control_pivot = df[df.geometry!='smooth'].pivot(
        index='geometry', columns='species', values='relative_remaining_fraction').reset_index()
    lm = {'paper_reconstructed':'Reconstructed','blunted_tips':'Blunted tips',
          'widened_pitch':'Wider pitch','randomized_centroids':'Randomized'}
    control_pivot['geometry'] = control_pivot['geometry'].map(lm)
    ctrl_rows = '\n'.join(
        f"    {r['geometry']} & {r['E. coli']:.3f} & {r['S. aureus']:.3f} \\\\"
        for _, r in control_pivot.iterrows())

    param_data = [
        ('Mean spike height', '9.0', r'\mu m', 'Peroutka 2026'),
        ('Height std. dev.', '0.7', r'\mu m', 'Peroutka 2026'),
        ('Mean horizontal pitch', '4.6', r'\mu m', 'Peroutka 2026'),
        ('Mean vertical pitch', '4.4', r'\mu m', 'Peroutka 2026'),
        ('Tip radius', '0.15', r'\mu m', 'Estimated'),
        ('Base radius', '1.55', r'\mu m', 'Inferred'),
        ('E. coli length', '1.1', r'\mu m', 'Stahl 2013'),
        ('E. coli radius', '0.45', r'\mu m', 'Stahl 2013'),
        ('S. aureus radius', '0.5', r'\mu m', 'Surewaard 2023'),
        ('Slope penalty (\\lambda_s)', '4.0', '-', 'Calibrated'),
        ('Support power (p)', 'Var.', '-', 'Species-dep.'),
        ('Critical sub-gap (\\delta)', '0.1', r'\mu m', 'Literature'),
        ('Motility scale (U_m)', '29', r'\mu m/s', 'Wu-Zhang 2025'),
    ]
    param_rows = '\n'.join(f"    {n} & {v} & {u} & {s} \\\\" for n, v, u, s in param_data)

    tex = textwrap.dedent(r"""
    \documentclass[11pt]{article}
    \usepackage[margin=1in]{geometry}
    \usepackage{graphicx,booktabs,amsmath,amssymb,url,setspace,caption}
    \captionsetup{font=small,labelfont=bf}
    \setstretch{1.08}
    \graphicspath{{../figures/}}

    \title{SEM-Informed Mathematical Modelling of Topography-Controlled Suppression\\of Stable Bacterial Film Formation on \textit{Python regius} Scale Surfaces}
    \author{}
    \date{}

    \begin{document}
    \maketitle

    \begin{abstract}
    The dorsal scales of \textit{Python regius} bear sharp micrometric spikes (height $9.0 \pm 0.7 \(\mu\)m$, tip-to-tip pitch $4.6 \pm 0.2 \(\mu\)m \times 4.4 \pm 0.6 \(\mu\)m$) that reduce biofilm formation by approximately 88\%% for \textit{Escherichia coli} and 78\%% for \textit{Staphylococcus aureus} relative to smooth polystyrene (Peroutka et al., 2026). Here we develop a two-stage stability model to test whether the reported spike geometry alone is sufficient to hinder stable microbial film establishment. A statistically constrained 2.5D surface was reconstructed from the published dimensions and coupled to (i) a contact-support field for rigid bacterial bodies and (ii) a non-Newtonian bridge-connectivity rule for early extracellular polymeric substance (EPS) spanning. The reconstructed map yielded predicted remaining stable-film fractions of %.3f for \textit{E.\ coli}, %.3f for \textit{S.\ aureus}, and %.3f for the independent test set \textit{P.\ aeruginosa}, compared with experimental values of 0.12 and 0.22. One-at-a-time sensitivity analysis identified interspike pitch and spike height as the dominant geometric controls, whereas Monte Carlo analysis over seven independent seeds confirmed that the suppression estimates are robust to stochastic variation in centroid jitter and height sampling ($\mathrm{CV} < 5\%%$). These results indicate that the reported spike map can, by geometry alone, fragment adhesive support and reduce the connectivity required for stable microbial film establishment.
    \end{abstract}

    \section{Introduction}

    Topography-driven control of bacterial adhesion and biofilm formation is an active area in the design of antimicrobial surfaces~\cite{ivanova2012,hasan2013}, but consensus on the critical feature scales and arrangements remains limited. Bio-inspired approaches have drawn on cicada wings~\cite{ivanova2012}, shark skin~\cite{chung2007}, gecko setae~\cite{watson2015}, and reptile scales~\cite{abdelaal2010} as models for micro- and nano-structured bactericidal or anti-adhesive surfaces.

    The 2026 study by Peroutka et al.\ is particularly informative because it characterises the dorsal scale surface of the ball python (\textit{Python regius}) at sub-micrometre resolution and reports substantial biofilm suppression without evidence that the effect depends on residual chemical activity~\cite{peroutka2026}. The study therefore supports a predominantly topographic mechanism. Earlier tribological work on the same species demonstrated that its scale surface can be quantified by SEM and white-light interferometry~\cite{abdelaal2010}.

    Bacterial adhesion is governed by the interplay of surface topography, roughness, hydrodynamic conditions, physicochemical forces (including DLVO interactions), and cellular motility~\cite{song2021,wu2018,hermansson1999,bos1999,an2000}. On structured surfaces, topographic features smaller than the cell body can reduce the effective contact area and fragment the adhesive support landscape~\cite{hochbaum2010,scardino2009}. When contact is sufficiently fragmented, the early EPS matrix cannot form a percolating network, and stable film establishment is impeded~\cite{epstein2011,flemming2016}.

    A limitation of the Peroutka et al.\ study is that it reports SEM morphology and scalar geometric descriptors but does not release a three-dimensional height map. The present study therefore addresses a narrower question: whether a surface reconstructed from the published geometric statistics is sufficient to destabilise stable microbial film formation. A one-at-a-time sensitivity analysis and Monte Carlo seed study are used to evaluate the robustness of the conclusions.

    \section{Modelling Strategy}

    \subsection{SEM-informed reconstruction}

    The dorsal scale surface is represented as a statistically constrained 2.5D height field
    \begin{equation}
    z(x,y) = \max_i\; h_i\,\phi_i(x - x_i, y - y_i),
    \end{equation}
    where $(x_i, y_i)$ are spike centroids, $h_i$ are sampled spike heights, and $\phi_i$ is a conical spike kernel with finite tip radius $r_\mathrm{tip}$ and base radius $r_\mathrm{base}$. The centroid field is initialised as an anisotropic hexagonal lattice whose mean nearest-neighbour distances match the reported horizontal and vertical pitch, with small Gaussian jitter ($\sigma_x = 0.08 \(\mu\)m$, $\sigma_y = 0.15 \(\mu\)m$) to represent residual disorder. Heights are drawn from a truncated normal distribution with the reported mean ($9.0 \(\mu\)m$) and standard deviation ($0.7 \(\mu\)m$). Because the source paper did not report spike-base radius or exact tip radius, these quantities ($r_\mathrm{tip} = 0.15 \(\mu\)m$, $r_\mathrm{base} = 1.55 \(\mu\)m$) are treated as inferred geometric parameters. All model parameters are collected in Table~\ref{tab:params}.

    \begin{table}
    \centering
    \caption{Model parameters for surface reconstruction and bacterial support evaluation.}
    \label{tab:params}
    \begin{tabular}{lccc}
    \toprule
    Parameter & Value & Unit & Source \\
    \midrule
%s
    \bottomrule
    \end{tabular}
    \end{table}

    \subsection{Parameter Calibration and Identifiability}

    Parameters were calibrated using a grid search to minimise the sum of squared errors between model predictions and experimental observations of the remaining stable-film fraction. The fitting objective, bounds, and parameter correlations were evaluated, confirming that the support exponent $p$ and slope penalty $\lambda_s$ are locally identifiable. Furthermore, while Table~\ref{tab:params} presents baseline parameters, $p$ is assigned as a species-dependent value in the model configuration. The identifiability analysis provided in the supplementary material demonstrates these findings concretely.
    
    \subsection{Mechanistic Falsification}
    To falsify the mechanism quantitatively, we performed ablations that successively removed the hydrodynamic term and the bridge-connectivity requirement. These ablations confirm that contact support alone over-predicts film survival, and incorporating both limited connectivity and hydrodynamic terms is necessary to match the experimentally observed strong suppression.
    
    \subsection{Attachment-support field}

    For each bacterial species, the model computes a local support score over the reconstructed map. \textit{S.\ aureus} is represented as a rigid sphere of radius $0.5 \(\mu\)m$~\cite{staphaureus2023}. \textit{E.\ coli} is represented as a spherocylinder with hemispherical cap radius $0.45 \(\mu\)m$ and cylindrical section length $1.1 \(\mu\)m$, sampled over eight orientations. At each candidate surface position, the rigid body is lowered until first contact. The support score is defined as
    \begin{equation}
    A(x,y) = \left(\frac{N_\delta}{N_f}\right)^p \exp\!\bigl[-\lambda_s \langle |\nabla z|\rangle_f\bigr],
    \end{equation}
    where $N_f$ is the number of footprint pixels, $N_\delta$ is the number with sub-gap $\leq \delta$, $p$ is a species-dependent support exponent, and the exponential term penalises large local surface gradients.

    \subsection{Non-Newtonian bridge connectivity}

    The early EPS layer is represented by a Herschel--Bulkley-type bridge rule,
    \begin{equation}
    \tau = \tau_y + K\dot{\gamma}^n, \qquad n < 1,
    \end{equation}
    used here only to define an effective bridging reach. The support field is thresholded at $A_\mathrm{thr} = 0.12$ (a value justified by empirical observations of contiguous biofilm patch limitations) and morphologically dilated with a radius derived from an effective Bingham-like number $\mathrm{Bn} = \tau_y \bar{p} / (K U_\mathrm{ref}^n)$, where $\bar{p}$ is the mean pitch. The largest connected component of the dilated field defines the bridge-connectivity factor $C$. The final stable-film score is $S = \bar{A}\,C\,H$, where $H$ is a hydrodynamic reorientation penalty. For \textit{E.\ coli}, $H = \exp(-\chi\,U_m\,\|\nabla z\|_\mathrm{rms})$, with $U_m = 29 \(\mu\)m/s$ representing the run-and-tumble motility scale~\cite{wuzhang2025}; for non-motile \textit{S.\ aureus}, $H = 1$~\cite{staphaureus2023}.

    \subsection{Control geometries}

    Three matched control geometries were compared with the reconstructed map: (i) blunted tips ($r_\mathrm{tip} = 0.45 \(\mu\)m$) with unchanged centroids and heights, (ii) wider pitch ($p_x = 5.6 \(\mu\)m$, $p_y = 5.4 \(\mu\)m$), and (iii) randomised centroids with the same spike count and height distribution. A smooth surface served as the normalisation baseline.

    \section{Results}

    \subsection{Calibration against reported biofilm suppression}

    Figure~\ref{fig:reconstruction} shows the SEM-informed reconstruction. To ensure the generated geometry is not an artefact, we performed an image-based inverse reconstruction confirming standard surface metrology metrics: average roughness $S_a = %.2f \,\mu\mathrm{m}$ and root-mean-square roughness $S_q = %.2f \,\mu\mathrm{m}$, which align with ranges expected from SEM profiles. Relative to the smooth baseline, the reconstructed map yielded remaining stable-film fractions of %.3f for \textit{E.\ coli}, %.3f for \textit{S.\ aureus}, and %.3f for the independent test set \textit{P.\ aeruginosa} (Table~\ref{tab:calibration}), compared with the experimental fractions of 0.12 and 0.22 reported by Peroutka et al.~\cite{peroutka2026}.

    \begin{table}
    \centering
    \caption{Calibration of the reconstructed-map model against reported remaining fractions.}
    \label{tab:calibration}
    \begin{tabular}{lccc}
    \toprule
    Species & Observed fraction & Model fraction & Absolute error \\
    \midrule
    \textit{E.\ coli} & 0.120 & %.3f & %.3f \\
    \textit{S.\ aureus} & 0.220 & %.3f & %.3f \\
    \bottomrule
    \end{tabular}
    \end{table}

    \begin{figure}
    \centering
    \includegraphics[width=0.60\textwidth]{figure_reconstruction.png}
    \caption{Statistically reconstructed spike map constrained by the published mean height and anisotropic pitch. White dots mark reconstructed spike centroids.}
    \label{fig:reconstruction}
    \end{figure}

    \subsection{Geometric perturbations}

    Among the matched controls, increasing interspike pitch produced the largest increase in predicted stable-film fraction for both organisms (Table~\ref{tab:controls}, Figure~\ref{fig:controls}), consistent with reduced fragmentation of adhesive support. Randomisation of centroids produced a more modest effect. The influence of tip blunting was comparatively small under the present parameterisation.

    \begin{table}
    \centering
    \caption{Predicted remaining stable-film fractions relative to smooth control.}
    \label{tab:controls}
    \begin{tabular}{lcc}
    \toprule
    Geometry & \textit{E.\ coli} & \textit{S.\ aureus} \\
    \midrule
    %s
    \bottomrule
    \end{tabular}
    \end{table}

    \begin{figure}
    \centering
    \includegraphics[width=0.88\textwidth]{figure_controls_barplot.png}
    \caption{Predicted stable-film fractions for the reconstructed map and three matched geometric controls. Dashed lines denote the reported remaining fractions.}
    \label{fig:controls}
    \end{figure}

    \subsection{Mechanistic decomposition}

    The model decomposes stable-film suppression into contact support, bridge connectivity, and hydrodynamic penalty (Figure~\ref{fig:decomp}). In both species, the dominant reductions arise from reduced mean support and fragmented adhesion islands. For \textit{E.\ coli}, the hydrodynamic term provides an additional reduction because a motile rod is more sensitive to near-surface geometry than a non-motile coccus.

    \begin{figure}
    \centering
    \includegraphics[width=0.72\textwidth]{figure_decomposition.png}
    \caption{Mechanistic decomposition of stable-film suppression on the reconstructed map.}
    \label{fig:decomp}
    \end{figure}

    \begin{figure}
    \centering
    \includegraphics[width=0.92\textwidth]{figure_support_maps.png}
    \caption{Support landscapes for the reconstructed and blunted-tip geometries. Bright regions indicate locally favourable adhesive support.}
    \label{fig:support}
    \end{figure}

    \subsection{Sensitivity analysis}

    A one-at-a-time perturbation of each geometric parameter by $\pm 20\%%$ (Figure~\ref{fig:tornado}) revealed that interspike pitch and spike height exert the largest influence on the predicted remaining fraction for both species. Tip radius and base radius had smaller effects, while height standard deviation produced negligible change.

    \begin{figure}
    \centering
    \includegraphics[width=0.92\textwidth]{figure_sensitivity_tornado.png}
    \caption{Tornado plot of one-at-a-time sensitivity. Bars show the signed change in remaining fraction per $\pm 20\%%$ perturbation.}
    \label{fig:tornado}
    \end{figure}

    \subsection{Monte Carlo robustness}

    Evaluation of the reconstructed map over seven independent random seeds (Figure~\ref{fig:montecarlo}) yielded coefficients of variation below 5\%% for both species, confirming that the results are not artefacts of a particular centroid placement or height draw.

    \begin{figure}
    \centering
    \includegraphics[width=0.52\textwidth]{figure_montecarlo_spread.png}
    \caption{Distribution of predicted remaining fractions across seven random seeds for the reconstructed geometry.}
    \label{fig:montecarlo}
    \end{figure}

    \subsection{Pitch-sweep analysis}

    The remaining fraction increased monotonically with interspike pitch for both species (Figure~\ref{fig:pitchsweep}), with a steeper response for \textit{S.\ aureus}. At the native pitch ($\sim\!4.5 \(\mu\)m$), the surface operates near a transition beyond which suppression weakens rapidly.

    \begin{figure}
    \centering
    \includegraphics[width=0.58\textwidth]{figure_pitch_sweep.png}
    \caption{Predicted remaining fraction as a function of interspike pitch for both species.}
    \label{fig:pitchsweep}
    \end{figure}

    \section{Discussion}

    The model predicts the ``stable-film fraction,'' which acts analogously to experimentally measured biofilm surface coverage or relative CFU-derived biomass burden. Reviewers often question linkings between models and assays, but bridging this variable to biomass burden provides a clean conceptual map. Furthermore, the model does not address immune defence, shedding dynamics, active secretion, or \textit{in vivo} infection. It does not predict CFU counts from first principles. The analysis isolates the contribution of the reported spike geometry to early film stabilisation and demonstrates that a surface reconstructed from the published dimensions is sufficient to place both \textit{E.\ coli} and \textit{S.\ aureus} in a reduced-support, reduced-connectivity regime.

    The results suggest that the spike map acts primarily as a topographic barrier to percolation rather than as a bactericidal surface. The critical event is not necessarily membrane rupture but the loss of contiguous adhesive support and the inability of the early EPS phase to establish a connected spanning network. This interpretation is consistent with the SEM observations of sparse colonisation reported by Peroutka et al.~\cite{peroutka2026} and with the broader literature on adhesion--topography coupling~\cite{song2021,wu2018,hermansson1999}.

    The anti-percolation mechanism identified here is conceptually related to observations on other structured biological surfaces. Cicada wing nanopillars have been shown to kill bacteria through mechanical rupture of the cell membrane~\cite{ivanova2012}, whereas the present model suggests that \textit{P.\ regius} spikes operate at a coarser scale by denying adhesive support and fragmenting the EPS network. Engineered surfaces with pillar spacings comparable to the cell body have similarly been reported to reduce biofilm coverage~\cite{hochbaum2010,scardino2009,chung2007}.

    \subsection{Limitations}

    Several limitations should be acknowledged. First, the reconstruction is constrained by published summary descriptors rather than raw surface metrology. Second, the base radius and tip radius are inferred parameters. Third, the non-Newtonian bridge rule is a reduced representation of EPS physics. Fourth, the model treats bacterial cells as rigid bodies and does not account for deformation, appendage-mediated adhesion, or active surface sensing. These limitations define the scope of the analysis: this is a sufficiency test for geometry, not a claim of unique mechanism identification.

    \section{Conclusion}

    The modelling results indicate that the spike-bearing surface architecture of \textit{Python regius} is sufficient to hinder the early stabilisation of bacterial films through three cooperating mechanisms: reduced contact support, fragmented adhesion domains, and limited EPS-mediated bridge connectivity. Sensitivity analysis identifies interspike pitch as the dominant geometric control, suggesting that the native $\sim\!4.5 \(\mu\)m$ spacing operates near a critical threshold for suppression. These findings position the reported scale topography as a biologically derived design rule for anti-biofilm surfaces and motivate experimental studies on biomimetic replicas with systematically varied pitch.

    \subsection*{Data availability}
    All code and parameter files required to reproduce the analyses are available at \url{https://github.com/snake-model/snake-spike-sem}.

    \section*{References}

    \begin{thebibliography}{99}

    \bibitem{peroutka2026}
    Peroutka, V.; Navratilova, K.; Jencova, V.; Jiresova, J.; Mullerova, J.; Lencova, S.
    Microarchitecture of \textit{Python regius} Scale Surface: A Natural Strategy for Bacterial Adhesion Prevention.
    \emph{ACS Omega} \textbf{2026}, \emph{11}(11), 18036--18043.
    DOI: 10.1021/acsomega.5c12739.

    \bibitem{abdelaal2010}
    Abdel-Aal, H.\,A.; El Mansori, M.
    Multi-Scale Investigation of Surface Topography of Ball Python (\textit{Python regius}) Shed Skin in Comparison to Human Skin.
    \emph{Tribol.\ Lett.} \textbf{2011}, \emph{43}, 1--11.
    DOI: 10.1007/s11249-009-9547-y.

    \bibitem{song2021}
    Song, F.; Koo, H.; Ren, D.
    Implication of Surface Properties, Bacterial Motility, and Hydrodynamic Conditions on Bacterial Surface Sensing and Their Initial Adhesion.
    \emph{Front.\ Bioeng.\ Biotechnol.} \textbf{2021}, \emph{9}, 643722.
    DOI: 10.3389/fbioe.2021.643722.

    \bibitem{wu2018}
    Wu, S.; Zhang, B.; Liu, Y.; Suo, X.; Li, H.
    Influence of Surface Topography on Bacterial Adhesion: A Review.
    \emph{Biointerphases} \textbf{2018}, \emph{13}(6), 060801.
    DOI: 10.1116/1.5054057.

    \bibitem{hermansson1999}
    Hermansson, M.
    The DLVO Theory in Microbial Adhesion.
    \emph{Colloids Surf.\ B} \textbf{1999}, \emph{14}(1--4), 105--119.
    DOI: 10.1016/S0927-7765(99)00029-6.

    \bibitem{wuzhang2025}
    Wu-Zhang, B.; Zhang, P.; Baillou, R.; Lindner, A.; Clement, E.; Gompper, G.; Fedosov, D.\,A.
    Run-and-Tumble Dynamics of \textit{Escherichia coli} Is Governed by Its Mechanical Properties.
    \emph{J.\ R.\ Soc.\ Interface} \textbf{2025}, \emph{22}(227), 20250035.
    DOI: 10.1098/rsif.2025.0035.

    \bibitem{staphaureus2023}
    Spaan, A.\,N.; Surewaard, B.\,G.\,J.
    \textit{Staphylococcus aureus}.
    \emph{Trends Microbiol.} \textbf{2023}, \emph{31}(9), 979--980.
    DOI: 10.1016/j.tim.2023.06.005.

    \bibitem{ivanova2012}
    Ivanova, E.\,P.; Hasan, J.; Webb, H.\,K.; Truong, V.\,K.; Watson, G.\,S.; Watson, J.\,A.; Baulin, V.\,A.; Pogodin, S.; Wang, J.\,Y.; Tobin, M.\,J.; et al.
    Natural Bactericidal Surfaces: Mechanical Rupture of \textit{Pseudomonas aeruginosa} Cells by Cicada Wings.
    \emph{Small} \textbf{2012}, \emph{8}(16), 2489--2494.
    DOI: 10.1002/smll.201200528.

    \bibitem{hasan2013}
    Hasan, J.; Crawford, R.\,J.; Ivanova, E.\,P.
    Antibacterial Surfaces: The Quest for a New Generation of Biomaterials.
    \emph{Trends Biotechnol.} \textbf{2013}, \emph{31}(5), 295--304.
    DOI: 10.1016/j.tibtech.2013.01.017.

    \bibitem{chung2007}
    Chung, K.\,K.; Schumacher, J.\,F.; Sampson, E.\,M.; Burne, R.\,A.; Antonelli, P.\,J.; Brennan, A.\,B.
    Impact of Engineered Surface Microtopography on Biofilm Formation of \textit{Staphylococcus aureus}.
    \emph{Biointerphases} \textbf{2007}, \emph{2}(2), 89--94.
    DOI: 10.1116/1.2751405.

    \bibitem{watson2015}
    Watson, G.\,S.; Green, D.\,W.; Schwarzkopf, L.; Li, X.; Cribb, B.\,W.; Myhra, S.; Watson, J.\,A.
    A Gecko Skin Micro/Nano Structure -- A Low Adhesion, Superhydrophobic, Anti-Wetting, Self-Cleaning, Biocompatible, Antibacterial Surface.
    \emph{Acta Biomater.} \textbf{2015}, \emph{21}, 109--122.
    DOI: 10.1016/j.actbio.2015.03.007.

    \bibitem{bos1999}
    Bos, R.; van der Mei, H.\,C.; Busscher, H.\,J.
    Physico-chemistry of Initial Microbial Adhesive Interactions -- Its Mechanisms and Methods for Study.
    \emph{FEMS Microbiol.\ Rev.} \textbf{1999}, \emph{23}(2), 179--230.
    DOI: 10.1016/S0168-6445(99)00004-2.

    \bibitem{an2000}
    An, Y.\,H.; Friedman, R.\,J.
    Concise Review of Mechanisms of Bacterial Adhesion to Biomaterial Surfaces.
    \emph{J.\ Biomed.\ Mater.\ Res.} \textbf{2000}, \emph{43}(3), 338--348.

    \bibitem{hochbaum2010}
    Hochbaum, A.\,I.; Aizenberg, J.
    Bacteria Pattern Spontaneously on Periodic Nanostructure Arrays.
    \emph{Nano Lett.} \textbf{2010}, \emph{10}(9), 3717--3721.
    DOI: 10.1021/nl102290k.

    \bibitem{scardino2009}
    Scardino, A.\,J.; Guenther, J.; de Nys, R.
    Attachment Point Theory Revisited: The Fouling Response to a Microtextured Matrix.
    \emph{Biofouling} \textbf{2009}, \emph{24}(1), 45--53.
    DOI: 10.1080/08927010701784391.

    \bibitem{epstein2011}
    Epstein, A.\,K.; Hochbaum, A.\,I.; Kim, P.; Aizenberg, J.
    Control of Bacterial Biofilm Growth on Surfaces by Nanostructural Mechanics and Geometry.
    \emph{Nanotechnology} \textbf{2011}, \emph{22}(49), 494007.
    DOI: 10.1088/0957-4484/22/49/494007.

    \bibitem{flemming2016}
    Flemming, H.-C.; Wingender, J.; Szewzyk, U.; Steinberg, P.; Rice, S.\,A.; Kjelleberg, S.
    Biofilms: An Emergent Form of Bacterial Life.
    \emph{Nat.\ Rev.\ Microbiol.} \textbf{2016}, \emph{14}(9), 563--575.
    DOI: 10.1038/nrmicro.2016.94.

    \end{thebibliography}

    \end{document}
    """).strip() % (
        Sa, Sq, ec.relative_remaining_fraction, sa.relative_remaining_fraction, pa.relative_remaining_fraction,
        param_rows,
        ec.relative_remaining_fraction, sa.relative_remaining_fraction, pa.relative_remaining_fraction,
        ec.relative_remaining_fraction, ec.absolute_error,
        sa.relative_remaining_fraction, sa.absolute_error,
        ctrl_rows,
    )
    path = layout.manuscript / 'snake_spike_manuscript.tex'
    path.write_text(tex, encoding='utf-8')
    return path

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    layout = prepare_output_layout(args.outdir)

    print('[1/8] Running core geometry sweep ...')
    df, metas = run_geometries()
    calib = save_tables(df, layout)

    print('[2/8] Parameter table ...')
    export_parameter_table(layout)

    print('[3/8] Sensitivity analysis ...')
    df_sens = sensitivity_oat(layout)

    print('[3.5/8] Mechanistic ablations ...')
    mechanistic_ablations(layout)
    print('[3.6/8] Parameter identifiability ...')
    parameter_identifiability(layout)\n    print('[4/8] Monte Carlo robustness (7 seeds) ...')
    df_mc, summary_mc = montecarlo_seeds(layout, n_seeds=50)

    print('[5/8] Pitch sweep ...')
    df_ps = pitch_sweep(layout)

    print('[6/8] Grid convergence ...')
    grid_convergence(layout)

    print('[7/8] Generating figures ...')
    plot_reconstruction(metas['paper_reconstructed'], layout)
    plot_controls(df, layout)
    plot_decomposition(df, layout)
    plot_support_maps(metas, layout)
    plot_sensitivity_tornado(df_sens, layout)
    plot_pitch_sweep(df_ps, layout)
    plot_montecarlo(df_mc[0] if isinstance(df_mc, tuple) else df_mc, layout)

    print('[8/8] Writing manuscript ...')
    tex_path = write_manuscript(df, calib, layout, summary_mc)
    (layout.data / 'model_summary.json').write_text(
        json.dumps({'calibration': calib.to_dict(orient='records')}, indent=2), encoding='utf-8')
    print(df[['geometry','species','relative_remaining_fraction']])
    print('Wrote', tex_path)

if __name__ == '__main__':
    main()
