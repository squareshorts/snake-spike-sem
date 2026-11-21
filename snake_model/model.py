"""Colonisation model for imbricated blade arrays and reference topographies.

Each patterned surface and a flat reference are simulated with identical physics, and the
model reports the ratio of retained surface-associated activity, Y = B_pattern / B_flat,
the quantity measured by CFU assays against a smooth control.  Attachment propensity and
arrival are referenced to the analytically computed flat-surface values of each species.
Surface-associated cells follow a capacity-limited compartment system, so summed
occupancy stays below one.  All smoothing operators and the background roughness field are
defined in physical units, which makes results independent of grid spacing and time step,
and all domains are doubly periodic and commensurate with the surface lattice.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage

from .species import SpeciesProfile, body_masks, default_species, shell_mask, sigmoid


# ----------------------------------------------------------------------------- config
@dataclass
class ModelConfig:
    dx: float = 0.2
    domain_x: float = 36.0
    domain_y: float = 36.0
    margin: float = 0.0              # periodic domains: no boundary margin is required
    horizon: float = 4.5
    h_state: float = 0.05            # state-model sub-step (model-time units)
    beta: float = 4.5
    eps_diffusivity: float = 0.14
    eps_diffusion_exponent: float = 1.3
    eps_linear_diffusivity: float = 0.2592  # um^2 per model-time unit
    eps_decay: float = 0.05
    eps_yield: float = 0.02
    eps_attach_threshold: float = 0.0015
    eps_mature_threshold: float = 0.003
    arrival_rate: float = 0.24
    capture_rate: float = 0.90
    replenish_rate: float = 0.5
    use_compliance: bool = True
    use_appendages: bool = True
    use_dynamic_eps: bool = True
    use_transport: bool = True
    use_slope: bool = True
    arrival_mode: str = "dual"   # dual | supply | capture
    use_steric: bool = True
    curv_scale: float = 0.5
    sense_scale: float = 0.25
    record_times: tuple = (1.0, 2.0, 3.0, 4.5, 6.0, 9.0)

    def grid(self, lx=None, ly=None):
        lx = self.domain_x if lx is None else lx
        ly = self.domain_y if ly is None else ly
        nx, ny = int(round(lx / self.dx)), int(round(ly / self.dx))
        return np.meshgrid(np.arange(nx) * self.dx, np.arange(ny) * self.dx)

    def interior(self, shape):
        m = int(round(self.margin / self.dx))
        mask = np.zeros(shape, bool)
        mask[m:shape[0] - m, m:shape[1] - m] = True
        return mask


def _wrap(d, period):
    return (d + 0.5 * period) % period - 0.5 * period


def periodic_gradient(z, dx):
    gx = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) / (2 * dx)
    gy = (np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)) / (2 * dx)
    return gy, gx


# ---------------------------------------------------------------------------- surfaces
def fourier_roughness(xx, yy, amp, corr, seed, n_modes=256, lx=None, ly=None):
    """Finite-mode Gaussian random field evaluated in physical coordinates.

    The unsnapped wave-vector distribution has the target Gaussian covariance and unit
    ensemble variance.  Wave vectors are snapped to the reciprocal lattice when a
    periodic cell is specified, so the realised covariance is an approximation to that
    target while periodicity is exact.  The amplitude is independent of grid spacing.
    """
    if amp <= 0:
        return np.zeros_like(xx)
    rng = np.random.default_rng(seed)
    k = rng.normal(0.0, 1.0 / (np.sqrt(2.0) * corr), size=(n_modes, 2))
    if lx is not None:
        k[:, 0] = 2 * np.pi / lx * np.round(k[:, 0] * lx / (2 * np.pi))
        k[:, 1] = 2 * np.pi / ly * np.round(k[:, 1] * ly / (2 * np.pi))
        zero = (k[:, 0] == 0) & (k[:, 1] == 0)
        k[zero, 0] = 2 * np.pi / lx
    phi = rng.uniform(0, 2 * np.pi, n_modes)
    out = np.zeros_like(xx)
    for (kx, ky), p in zip(k, phi):
        out += np.cos(kx * xx + ky * yy + p)
    return amp * np.sqrt(2.0 / n_modes) * out


def _blade_rng(seed, i, j):
    """Return the deterministic random generator for blade lattice index (i, j)."""
    return np.random.default_rng(np.random.SeedSequence([seed, i + 4096, j + 4096]))


def build_flat_surface(cfg: ModelConfig):
    """Return a flat periodic reference surface."""
    xx, _ = cfg.grid(12.0, 12.0)
    return np.zeros_like(xx)


def build_sharklet_af(cfg: ModelConfig, height=3.0, width=2.0, gap=2.0):
    """Build the seven-rib Sharklet AF-type comparator in its 40 x 32 um cell."""
    lx, ly = 40.0, 32.0
    xx, yy = cfg.grid(lx, ly)
    z = np.zeros_like(xx)
    lengths = [4, 8, 12, 16, 12, 8, 4]
    pitch = width + gap
    for a in range(2):
        for b in range(2):
            cx, cy = 20.0 * a + 10.0 * b + 10.0, 16.0 * b + 8.0
            for k, length in enumerate(lengths):
                y0 = cy + (k - 3) * pitch
                mask = (np.abs(_wrap(yy - y0, ly)) <= width / 2) & (np.abs(_wrap(xx - cx, lx)) <= length / 2)
                z[mask] = height
    return z


# ------------------------------------------------------------------------- attachment
def _attachment_energy(z, sp: SpeciesProfile, cfg: ModelConfig):
    # Cells and appendages sense topography at a finite length scale.  All geometric
    # descriptors are computed on the surface smoothed with a Gaussian of physical width
    # sense_scale, which gives the discrete operators a well-defined limit as dx -> 0.
    if cfg.sense_scale > 0:
        z = ndimage.gaussian_filter(z, sigma=cfg.sense_scale / cfg.dx, mode="wrap")
    gy, gx = periodic_gradient(z, cfg.dx)
    slope = np.hypot(gx, gy)
    # Curvature is sensed at the cell scale: evaluate it on the surface smoothed with a
    # Gaussian of physical width curv_scale, so that kinks do not produce 1/dx spikes.
    zs = ndimage.gaussian_filter(z, sigma=cfg.curv_scale / cfg.dx, mode="wrap")
    curv = np.abs(ndimage.laplace(zs, mode="wrap")) / cfg.dx ** 2
    energies, contacts, reliefs = [], [], []
    for mask in body_masks(sp, cfg.dx):
        k = mask.astype(float)
        k /= k.sum()
        mean = ndimage.convolve(z, k, mode="wrap")
        zmax = ndimage.maximum_filter(z, footprint=mask, mode="wrap")
        zmin = ndimage.minimum_filter(z, footprint=mask, mode="wrap")
        relief = np.clip(zmax - zmin, 0, None)
        gap = np.clip(zmax - mean, 0, None)
        reliefs.append(relief)
        eff = gap / (1.0 + sp.compliance) if cfg.use_compliance else gap
        e = -sp.body_adhesion * np.exp(-eff / sp.adhesion_range) + 0.5 * sp.compliance_penalty * eff ** 2
        if cfg.use_steric:
            e = e + sp.steric_penalty * np.clip(relief - sp.body_radius * (1.2 + 0.8 * sp.compliance), 0, None) ** 2
        if cfg.use_slope:
            e = e + sp.slope_penalty * ndimage.convolve(slope, k, mode="wrap")
        energies.append(e)
        contacts.append(np.exp(-gap / sp.adhesion_range))
    energy = np.min(np.stack(energies), axis=0)
    contact = np.max(np.stack(contacts), axis=0)
    # Relief used for arrival: the orientation that minimises relief (most accessible approach).
    relief_ref = np.min(np.stack(reliefs), axis=0)
    if cfg.use_appendages and sp.appendage_count > 0:
        ring = shell_mask(sp, cfg.dx)
        ss = ndimage.maximum_filter(slope, footprint=ring, mode="wrap")
        sc = ndimage.maximum_filter(contact, footprint=ring, mode="wrap")
        app = 1.0 - np.exp(-sp.appendage_binding * sp.appendage_count * np.exp(-sp.appendage_slope_penalty * ss) * sc)
        energy = energy - sp.appendage_strength * app
    else:
        app = np.zeros_like(z)
    return energy, app, slope, curv, relief_ref


def attachment_layers(z, sp: SpeciesProfile, cfg: ModelConfig, flat_energy: float):
    energy, app, slope, curv, relief = _attachment_energy(z, sp, cfg)
    attach = np.clip(np.exp(-cfg.beta * (energy - flat_energy)), 0.0, 2.0)
    return {"attach": attach, "appendage": app, "slope": slope, "curv": curv, "relief": relief}


def flat_reference_energy(sp: SpeciesProfile, cfg: ModelConfig):
    small = replace(cfg, domain_x=6.0, domain_y=6.0, margin=0.0)
    z = build_flat_surface(small)
    e, *_ = _attachment_energy(z, sp, small)
    return float(np.median(e))


def arrival(layers, sp: SpeciesProfile, cfg: ModelConfig):
    base = np.exp(-sp.transport_slope_penalty * layers["slope"] - sp.transport_relief_penalty * layers["relief"])
    sig = sp.transport_sigma / cfg.dx
    if cfg.use_transport and sp.motility_um_s > 0:
        drive = 1.0 + 0.018 * sp.motility_um_s * np.exp(-0.15 * layers["curv"])
        flux = ndimage.gaussian_filter(base * drive, sigma=sig, mode="wrap")
        flat = 1.0 + 0.018 * sp.motility_um_s
    elif cfg.use_transport:
        flux = ndimage.gaussian_filter(base, sigma=0.7 * sig, mode="wrap")
        flat = 1.0
    else:
        flux = np.ones_like(base)
        flat = 1.0
    drive = 0.80 + 0.004 * sp.motility_um_s + 0.012 * sp.appendage_count + 0.035 * sp.appendage_length
    return np.clip(drive * flux / flat, 0.0, 2.5)


# --------------------------------------------------------------------------- dynamics
def _eps_step(eps, src, cfg: ModelConfig, h_total):
    """Advance eps_t = D0 m(L) L + D_lin lap(eps) + src - k eps by h_total.

    Strang splitting: the linear diffusion half-steps are exact Gaussian propagators with
    width sqrt(2 D_lin h/2) in micrometres; the nonlinear yield-stress term, source and
    decay are integrated explicitly with CFL-limited sub-steps.
    """
    sig = np.sqrt(cfg.eps_linear_diffusivity * h_total) / cfg.dx  # sqrt(2 D (h/2)) / dx
    if sig > 0:
        eps = ndimage.gaussian_filter(eps, sigma=sig, mode="wrap")
    t = 0.0
    while t < h_total - 1e-12:
        emax = float(eps.max())
        d_nl = cfg.eps_diffusivity * cfg.eps_diffusion_exponent * max(emax, 1e-8) ** (cfg.eps_diffusion_exponent - 1)
        h = min(h_total - t, 0.2 * cfg.dx ** 2 / max(d_nl, 1e-9))
        nl = np.power(np.maximum(eps, 1e-8), cfg.eps_diffusion_exponent)
        lap_nl = ndimage.laplace(nl, mode="wrap") / cfg.dx ** 2
        mob = sigmoid(np.abs(lap_nl), center=cfg.eps_yield, scale=0.03)
        eps = np.maximum(eps + h * (cfg.eps_diffusivity * mob * lap_nl + src - cfg.eps_decay * eps), 0.0)
        t += h
    if sig > 0:
        eps = ndimage.gaussian_filter(eps, sigma=sig, mode="wrap")
    return eps


def evolve(layers, flux, sp: SpeciesProfile, cfg: ModelConfig, interior):
    shape = flux.shape
    app = layers["appendage"] if cfg.use_appendages else np.zeros(shape)
    att = layers["attach"]
    slope = layers["slope"] if cfg.use_slope else np.zeros(shape)
    anchor_support = np.clip(att * (1.0 + 0.90 * app), 0.0, 1.8)
    retention = np.clip(np.exp(-0.22 * slope) * (1.0 + 0.60 * app + 0.35 * att), 0.30, 2.2)
    j_cap = flux if cfg.arrival_mode in ("dual", "capture") else np.ones_like(flux)
    j_sup = flux if cfg.arrival_mode in ("dual", "supply") else np.ones_like(flux)
    l_ft = cfg.capture_rate * j_cap * (0.55 + 0.45 * np.minimum(att, 1.0))
    l_tf = sp.detach_rate * np.exp(0.25 * slope) / retention
    l_ta = sp.anchor_rate * anchor_support
    l_at = sp.reorientation_rate * np.exp(0.15 * slope) / np.clip(0.85 + 1.10 * app, 0.35, None)
    act_base = sp.eps_activation_rate * (0.40 + 0.60 * np.minimum(att, 1.0) + 0.15 * app)
    mat_base = sp.maturation_rate * (0.50 + 0.50 * np.minimum(att, 1.0))
    src_mod = np.clip(0.65 + 0.45 * np.minimum(att, 1.0) + 0.10 * app, 0.0, None)
    f_target = np.clip(cfg.arrival_rate * j_sup, 0.0, 1.0)
    f = f_target.copy()
    t_, a, e, m = (np.zeros(shape) for _ in range(4))
    eps = np.zeros(shape)
    h = cfg.h_state
    n = int(round(max(cfg.horizon, max(cfg.record_times)) / h))
    rec = {}
    rec_steps = {int(round(rt / h)): rt for rt in cfg.record_times}
    for step in range(1, n + 1):
        cap = np.clip(1.0 - (t_ + a + e + m), 0.0, 1.0)
        if cfg.use_dynamic_eps:
            g_a = sigmoid(eps, center=cfg.eps_attach_threshold, scale=max(0.35 * cfg.eps_attach_threshold, 5e-4))
            g_m = sigmoid(eps, center=cfg.eps_mature_threshold, scale=max(0.35 * cfg.eps_mature_threshold, 8e-4))
        else:
            g_a = g_m = 1.0
        cap_flux = l_ft * f * cap
        j_tf = l_tf * t_
        j_ta = l_ta * t_
        j_at = l_at * a
        j_ae = act_base * g_a * a
        j_em = mat_base * g_m * e
        f = f + h * (cfg.replenish_rate * (f_target - f) - cap_flux + j_tf)
        t_ = t_ + h * (cap_flux - j_tf - j_ta + j_at)
        a = a + h * (j_ta - j_at - j_ae)
        e = e + h * (j_ae - j_em)
        m = m + h * j_em
        f, t_, a, e, m = (np.maximum(x, 0.0) for x in (f, t_, a, e, m))
        if cfg.use_dynamic_eps:
            src = sp.eps_production * (0.35 * a + 0.85 * e + 1.25 * m) * src_mod
            eps = _eps_step(eps, src, cfg, h)
        if step in rec_steps:
            rec[rec_steps[step]] = {
                "B": float(np.mean((a + e + m)[interior])),
                "T": float(np.mean(t_[interior])),
                "A": float(np.mean(a[interior])),
                "E": float(np.mean(e[interior])),
                "M": float(np.mean(m[interior])),
                "eps": float(np.mean(eps[interior])),
            }
        if step == int(round(cfg.horizon / h)):
            final = {"eps_field": eps.copy(), "M_field": m.copy(), "B_field": (a + e + m).copy()}
    occ_max = float(np.max(t_ + a + e + m))
    return rec, final, occ_max


def simulate(z, sp: SpeciesProfile, cfg: ModelConfig, flat_energy=None, keep_fields=False):
    if flat_energy is None:
        flat_energy = flat_reference_energy(sp, cfg)
    interior = cfg.interior(z.shape)
    layers = attachment_layers(z, sp, cfg, flat_energy)
    flux = arrival(layers, sp, cfg)
    rec, final, occ_max = evolve(layers, flux, sp, cfg, interior)
    out = {"B": rec[cfg.horizon]["B"], "series": rec, "occ_max": occ_max,
           "attach_mean": float(np.mean(layers["attach"][interior])),
           "arrival_mean": float(np.mean(flux[interior]))}
    eps = final["eps_field"]
    mask = (eps >= 0.006) & interior
    lab, nlab = ndimage.label(mask)
    out["eps_cover"] = float(mask[interior].mean())
    out["eps_lcc"] = float(np.bincount(lab.ravel())[1:].max() / interior.sum()) if nlab else 0.0
    if keep_fields:
        out.update(layers=layers, flux=flux, **final)
    return out


class FlatCache:
    """Caches flat-reference simulations, keyed by species and configuration."""

    def __init__(self):
        self._c = {}

    def get(self, sp: SpeciesProfile, cfg: ModelConfig):
        key = (repr(sp), repr(cfg))
        if key not in self._c:
            z = build_flat_surface(cfg)
            fe = flat_reference_energy(sp, cfg)
            self._c[key] = (simulate(z, sp, cfg, fe), fe)
        return self._c[key]


def ratio(z, sp, cfg, cache: FlatCache, times=None):
    flat, fe = cache.get(sp, cfg)
    pat = simulate(z, sp, cfg, fe)
    out = {"Y": pat["B"] / flat["B"], "B_pattern": pat["B"], "B_flat": flat["B"],
           "occ_max": pat["occ_max"], "attach_mean": pat["attach_mean"], "arrival_mean": pat["arrival_mean"],
           "eps_cover": pat["eps_cover"], "eps_lcc": pat["eps_lcc"]}
    for tt in (times or cfg.record_times):
        out[f"Y_t{tt:g}"] = pat["series"][tt]["B"] / flat["series"][tt]["B"]
    return out


SPECIES = default_species()


# ------------------------------------------------------------- imbricated blade arrays
@dataclass
class BladeGeometry:
    """Imbricated array of tapered blades, as imaged on P. regius dorsal scales.

    Blades point along +x.  Each blade rises from the scale sheet at its base to a free
    tip of height tip_height at projected length `length`; tips stand over elliptical pits.
    Following tips are pitch_f apart along the blade axis; adjacent columns are pitch_a
    apart and staggered by half a pitch."""
    pitch_f: float = 4.6          # following-tip spacing, Peroutka et al.: 4.6 +/- 0.2 um
    pitch_a: float = 4.4          # adjacent-tip spacing, Peroutka et al.: 4.4 +/- 0.6 um
    length: float = 9.0           # projected blade length, Peroutka et al.: 9.0 +/- 0.7 um
    base_width: float = 2.2       # SEM estimate (Fig. 1b of Peroutka et al.)
    taper: float = 1.0            # width ~ (1 - u/length)^taper
    tip_height: float = 1.5       # unmeasured
    pit_a: float = 1.2            # pit semi-axis along the blade (SEM estimate)
    pit_b: float = 0.6            # pit semi-axis across the blade (SEM estimate)
    pit_depth: float = 1.5        # unmeasured
    jitter: float = 0.12
    rough_amp: float = 0.05
    rough_corr: float = 0.6
    lattice_seed: int = 1
    row_phase: float = 0.5
    rough_seed: int = 2


def blade_cell(g: BladeGeometry, cfg: ModelConfig):
    """Columns are staggered by half a following pitch, so the adjacent-tip distance pitch_a
    is the diagonal neighbour distance and columns are sqrt(pitch_a^2 - (pitch_f/2)^2) apart."""
    perp = np.sqrt(max(g.pitch_a ** 2 - (0.5 * g.pitch_f) ** 2, 0.25))
    nx = max(4, int(round(cfg.domain_x / g.pitch_f)))
    ny = max(4, 2 * int(round(cfg.domain_y / perp / 2)))
    lx = cfg.dx * round(nx * g.pitch_f / cfg.dx)
    ly = cfg.dx * round(ny * perp / cfg.dx)
    return nx, ny, lx, ly


def build_blade_surface(g: BladeGeometry, cfg: ModelConfig):
    nx, ny, lx, ly = blade_cell(g, cfg)
    pf, pc = lx / nx, ly / ny
    xx, yy = cfg.grid(lx, ly)
    specs = []
    for j in range(ny):
        for i in range(nx):
            r = _blade_rng(g.lattice_seed, i, j)
            jx, jy = r.normal(0.0, g.jitter, 2)
            z4 = r.standard_normal(4)
            specs.append(((i + g.row_phase * (j % 2)) * pf + jx, (j + 0.5) * pc + jy,
                          max(g.length * (1 + 0.05 * z4[0]), 1.0), max(g.base_width * (1 + 0.08 * z4[1]), 0.3),
                          max(g.tip_height * (1 + 0.1 * z4[2]), 0.05)))
    # pits: elliptical openings in the scale sheet centred under each tip
    pit = np.zeros_like(xx)
    owner = np.full(xx.shape, -1)
    for k, (xb, yb, L, w0, ht) in enumerate(specs):
        du = _wrap(xx - (xb + L), lx)
        v = _wrap(yy - yb, ly)
        rr = (du / g.pit_a) ** 2 + (v / g.pit_b) ** 2
        d = np.where(rr < 1, g.pit_depth * np.sqrt(np.clip(1 - rr, 0, 1)), 0.0)
        upd = d > pit
        pit[upd] = d[upd]
        owner[upd] = k
    in_pit = pit > 0
    z = -pit
    blade_mask = np.zeros(xx.shape, bool)
    for k, (xb, yb, L, w0, ht) in enumerate(specs):
        u = (xx - xb) % lx
        v = _wrap(yy - yb, ly)
        half = 0.5 * w0 * np.clip(1.0 - u / L, 0.0, None) ** g.taper
        inside = (u <= L) & (np.abs(v) <= half) & (~in_pit | (owner == k))
        zb = np.where(inside, ht * u / L, -np.inf)
        blade_mask |= inside
        np.maximum(z, zb, out=z)
    z = z + fourier_roughness(xx, yy, g.rough_amp, g.rough_corr, g.rough_seed, lx=lx, ly=ly)
    z = z - z.min()
    info = {"n_blades": nx * ny, "lx": lx, "ly": ly, "pit_fraction": float(in_pit.mean()),
            "blade_fraction": float(blade_mask.mean()), "relief": float(np.percentile(z, 99.5) - np.percentile(z, 0.5)),
            "occupied": float(blade_mask.mean())}
    return z, info
