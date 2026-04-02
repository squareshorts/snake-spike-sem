#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

DEFAULT_DX = 0.6
DOMAIN = (36.0, 36.0)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"


@dataclass(frozen=True)
class OutputLayout:
    base: Path
    data: Path
    figures: Path
    manuscript: Path
    supporting: Path


def prepare_output_layout(base_dir: Path) -> OutputLayout:
    base = Path(base_dir).resolve()
    layout = OutputLayout(base, base / "data", base / "figures", base / "manuscript", base / "supporting_information")
    for path in (layout.base, layout.data, layout.figures, layout.manuscript, layout.supporting):
        path.mkdir(parents=True, exist_ok=True)
    return layout


@dataclass
class GeometryParams:
    name: str
    pitch_x: float = 4.6
    pitch_y: float = 4.4
    height_mu: float = 9.0
    height_sd: float = 0.7
    tip_radius_mu: float = 0.15
    base_radius_mu: float = 1.55
    sharpness_mu: float = 2.4
    jitter_x: float = 0.08
    jitter_y: float = 0.15
    latent_amplitude: float = 0.18
    disorder_mode: str = "anisotropic_hex"
    seed: int = 3


@dataclass
class SpeciesProfile:
    name: str
    observed: float
    body_radius: float
    body_length: float
    compliance: float
    body_adhesion: float
    adhesion_range: float
    compliance_penalty: float
    steric_penalty: float
    slope_penalty: float
    appendage_count: int
    appendage_length: float
    appendage_binding: float
    appendage_strength: float
    appendage_slope_penalty: float
    motility_um_s: float
    transport_sigma: float
    transport_slope_penalty: float
    transport_relief_penalty: float
    detach_rate: float
    anchor_rate: float
    reorientation_rate: float
    eps_activation_rate: float
    maturation_rate: float
    eps_production: float
    bridge_ell0: float
    hydro_chi: float


@dataclass
class Config:
    dx: float = DEFAULT_DX
    domain_x: float = DOMAIN[0]
    domain_y: float = DOMAIN[1]
    ensemble_size: int = 8
    calibration_surfaces: int = 8
    n_steps: int = 18
    dt: float = 0.25
    beta: float = 4.5
    eps_diffusivity: float = 0.14
    eps_diffusion_exponent: float = 1.3
    eps_decay: float = 0.05
    eps_yield: float = 0.02
    eps_bridge_threshold: float = 0.006
    eps_attach_threshold: float = 0.0015
    eps_mature_threshold: float = 0.003
    arrival_rate: float = 0.24
    capture_rate: float = 0.90
    use_compliance: bool = True
    use_appendages: bool = True
    use_dynamic_eps: bool = True
    use_state_dynamics: bool = True
    use_transport: bool = True
    obs_gain: float = 9.0
    obs_bias: float = 0.48
    uq_samples: int = 18
    phase_pitch_values: tuple[float, ...] = (3.8, 4.4, 5.0, 5.6, 6.2)
    phase_appendage_scales: tuple[float, ...] = (0.6, 0.9, 1.2, 1.5, 1.8)


@dataclass
class Surface:
    params: GeometryParams
    surface: np.ndarray
    centers: np.ndarray
    descriptors: dict
    loss: float


def default_species() -> dict[str, SpeciesProfile]:
    return {
        "E. coli": SpeciesProfile("E. coli", 0.12, 0.45, 1.10, 0.45, 1.05, 0.20, 1.00, 1.40, 0.28, 6, 1.15, 0.18, 0.12, 0.12, 29.0, 1.1, 1.2, 0.9, 0.22, 0.58, 0.18, 0.62, 0.55, 0.34, 1.80, 0.006),
        "S. aureus": SpeciesProfile("S. aureus", 0.22, 0.50, 0.00, 0.28, 1.15, 0.18, 1.15, 1.10, 0.22, 2, 0.45, 0.06, 0.03, 0.08, 0.0, 0.8, 0.7, 0.8, 0.16, 0.64, 0.10, 0.88, 0.92, 0.48, 0.35, 0.0),
        "P. aeruginosa": SpeciesProfile("P. aeruginosa", 0.15, 0.40, 1.50, 0.50, 0.98, 0.22, 0.95, 1.35, 0.30, 8, 1.55, 0.22, 0.16, 0.15, 45.0, 1.3, 1.35, 0.95, 0.24, 0.62, 0.20, 0.60, 0.58, 0.33, 1.50, 0.006),
    }


def sigmoid(x, center=0.0, scale=0.08):
    arr = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-(arr - center) / max(scale, 1e-6)))


def make_centers(params: GeometryParams, domain):
    rng = np.random.default_rng(params.seed)
    lx, ly = domain
    centers = []
    if params.disorder_mode == "anisotropic_hex":
        for j in range(int(ly / params.pitch_y) + 3):
            y = (j + 0.5) * params.pitch_y + rng.normal(0.0, params.jitter_y)
            xoff = 0.5 * params.pitch_x if (j % 2) else 0.0
            for i in range(int(lx / params.pitch_x) + 3):
                x = (i + 0.5) * params.pitch_x + xoff + rng.normal(0.0, params.jitter_x)
                if -1.0 <= x <= lx + 1.0 and -1.0 <= y <= ly + 1.0:
                    centers.append((x, y))
    else:
        n_est = max(1, int((lx / params.pitch_x) * (ly / params.pitch_y)))
        min_d = 0.55 * min(params.pitch_x, params.pitch_y)
        attempts = 0
        while len(centers) < n_est and attempts < 30000:
            attempts += 1
            x = rng.uniform(0.5, lx - 0.5)
            y = rng.uniform(0.5, ly - 0.5)
            if all((x - cx) ** 2 + (y - cy) ** 2 >= min_d ** 2 for cx, cy in centers):
                centers.append((x, y))
    return np.asarray(centers, dtype=float)


def paircorr(centers, domain, n_bins=10, max_r=10.0):
    if len(centers) < 2:
        return np.zeros(n_bins)
    d = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=-1))
    d = d[np.triu_indices(len(centers), 1)]
    hist, edges = np.histogram(d, bins=n_bins, range=(0.0, max_r))
    shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    density = len(centers) / (domain[0] * domain[1])
    expected = np.maximum(0.5 * len(centers) * density * shell, 1e-6)
    return hist / expected


def build_surface(params: GeometryParams, dx: float, domain):
    rng = np.random.default_rng(params.seed + 11)
    centers = make_centers(params, domain)
    heights = np.clip(rng.normal(params.height_mu, params.height_sd, len(centers)), params.height_mu - 2.8, params.height_mu + 2.8)
    tip = np.clip(rng.normal(params.tip_radius_mu, 0.30 * params.tip_radius_mu + 0.02, len(centers)), 0.05, None)
    base = np.clip(rng.normal(params.base_radius_mu, 0.16 * params.base_radius_mu + 0.08, len(centers)), tip + 0.15, None)
    sharp = np.clip(rng.normal(params.sharpness_mu, 0.35, len(centers)), 1.2, 4.5)
    xs = np.arange(0.0, domain[0] + dx, dx)
    ys = np.arange(0.0, domain[1] + dx, dx)
    xx, yy = np.meshgrid(xs, ys)
    surface = np.zeros_like(xx)
    for (xc, yc), h, rt, rb, m in zip(centers, heights, tip, base, sharp):
        dist = np.hypot(xx - xc, yy - yc)
        spike = np.zeros_like(surface)
        spike[dist <= rt] = h
        mask = (dist > rt) & (dist < rb)
        if np.any(mask):
            u = (dist[mask] - rt) / max(rb - rt, 1e-6)
            spike[mask] = h * np.maximum(0.0, 1.0 - u ** m)
        surface = np.maximum(surface, spike)
    if params.latent_amplitude > 0.0:
        noise = ndimage.gaussian_filter(rng.normal(0.0, 1.0, surface.shape), sigma=max(1.0, 0.6 / dx))
        surface = np.maximum(surface + params.latent_amplitude * noise, 0.0)
    tree = cKDTree(centers) if len(centers) else None
    nn = tree.query(centers, k=2)[0][:, 1] if tree is not None else np.array([0.0])
    cov = np.cov(centers.T) if len(centers) > 1 else np.eye(2)
    eig = np.linalg.eigvalsh(cov)
    anisotropy = float(np.sqrt(max(eig.max(), 1e-6) / max(eig.min(), 1e-6)))
    descriptors = {
        "density": float(len(centers) / (domain[0] * domain[1])),
        "nn_mean": float(np.mean(nn)),
        "nn_std": float(np.std(nn)),
        "anisotropy": anisotropy,
        "paircorr": paircorr(centers, domain),
        "height_mean": float(np.mean(heights)) if len(heights) else 0.0,
        "height_std": float(np.std(heights)) if len(heights) else 0.0,
        "tip_mean": float(np.mean(tip)) if len(tip) else 0.0,
        "base_mean": float(np.mean(base)) if len(base) else 0.0,
        "sharp_mean": float(np.mean(sharp)) if len(sharp) else 0.0,
        "occupied": float(np.mean(surface > 0.15 * max(np.mean(heights) if len(heights) else 1.0, 1e-6))),
    }
    return surface, centers, descriptors


def reference_target(config: Config):
    params = GeometryParams("reference", seed=3)
    _, _, target = build_surface(params, config.dx, (config.domain_x, config.domain_y))
    tol = {
        "density": 0.12 * target["density"] + 1e-6,
        "nn_mean": 0.18 * target["nn_mean"] + 1e-6,
        "nn_std": 0.35 * max(target["nn_std"], 0.08),
        "anisotropy": 0.18 * target["anisotropy"] + 1e-6,
        "height_mean": 0.10 * target["height_mean"] + 1e-6,
        "height_std": 0.22 * max(target["height_std"], 0.08),
        "tip_mean": 0.28 * target["tip_mean"] + 1e-6,
        "base_mean": 0.22 * target["base_mean"] + 1e-6,
        "sharp_mean": 0.22 * target["sharp_mean"] + 1e-6,
        "occupied": 0.18 * target["occupied"] + 1e-6,
        "paircorr": 0.22,
    }
    return target, tol


def descriptor_loss(desc, target, tol):
    keys = ["density", "nn_mean", "nn_std", "anisotropy", "height_mean", "height_std", "tip_mean", "base_mean", "sharp_mean", "occupied"]
    loss = sum(float(((desc[k] - target[k]) / tol[k]) ** 2) for k in keys)
    loss += float(np.mean(((desc["paircorr"] - target["paircorr"]) / tol["paircorr"]) ** 2))
    return loss

def sample_params(rng, idx, variant="native"):
    params = GeometryParams(f"{variant}_{idx}", seed=int(rng.integers(1, 1_000_000)))
    params = replace(
        params,
        pitch_x=float(np.clip(rng.normal(params.pitch_x, 0.30), 3.6, 6.4)),
        pitch_y=float(np.clip(rng.normal(params.pitch_y, 0.30), 3.4, 6.2)),
        height_mu=float(np.clip(rng.normal(params.height_mu, 0.35), 7.8, 10.2)),
        height_sd=float(np.clip(rng.normal(params.height_sd, 0.12), 0.35, 1.10)),
        tip_radius_mu=float(np.clip(rng.normal(params.tip_radius_mu, 0.04), 0.06, 0.34)),
        base_radius_mu=float(np.clip(rng.normal(params.base_radius_mu, 0.18), 1.0, 2.2)),
        sharpness_mu=float(np.clip(rng.normal(params.sharpness_mu, 0.30), 1.4, 3.6)),
        jitter_x=float(np.clip(rng.normal(params.jitter_x, 0.02), 0.03, 0.18)),
        jitter_y=float(np.clip(rng.normal(params.jitter_y, 0.03), 0.05, 0.30)),
        latent_amplitude=float(np.clip(rng.normal(params.latent_amplitude, 0.05), 0.05, 0.35)),
    )
    if variant == "wider_pitch":
        params.pitch_x += 0.8
        params.pitch_y += 0.8
    elif variant == "blunt":
        params.tip_radius_mu = min(params.tip_radius_mu + 0.18, 0.50)
        params.sharpness_mu = max(params.sharpness_mu - 0.7, 1.2)
    elif variant == "randomized":
        params.disorder_mode = "randomized"
        params.jitter_x = 0.0
        params.jitter_y = 0.0
    return params


def generate_ensemble(config: Config, variant="native", seed=13):
    rng = np.random.default_rng(seed)
    target, tol = reference_target(config)
    accepted = []
    for idx in range(max(80, 18 * config.ensemble_size)):
        params = sample_params(rng, idx, variant)
        surface, centers, desc = build_surface(params, config.dx, (config.domain_x, config.domain_y))
        accepted.append(Surface(params, surface, centers, desc, descriptor_loss(desc, target, tol)))
        if len([x for x in accepted if x.loss <= 12.0]) >= config.ensemble_size:
            break
    accepted.sort(key=lambda item: item.loss)
    accepted = accepted[: config.ensemble_size]
    rows = []
    for item in accepted:
        row = {"surface_id": item.params.name, "loss": item.loss}
        for key, value in item.descriptors.items():
            if key == "paircorr":
                for idx, v in enumerate(value):
                    row[f"paircorr_{idx}"] = float(v)
            else:
                row[key] = float(value)
        rows.append(row)
    return accepted, pd.DataFrame(rows)


def body_masks(species: SpeciesProfile, dx: float):
    if species.body_length <= 1e-6:
        n = int(np.ceil(species.body_radius / dx))
        s = np.arange(-n, n + 1) * dx
        xx, yy = np.meshgrid(s, s)
        return [(xx * xx + yy * yy) <= species.body_radius ** 2]
    masks = []
    for theta in np.linspace(0.0, np.pi, 4, endpoint=False):
        half = species.body_radius + species.body_length / 2.0
        n = int(np.ceil(half / dx))
        s = np.arange(-n, n + 1) * dx
        xx, yy = np.meshgrid(s, s)
        ca, sa = np.cos(theta), np.sin(theta)
        u = ca * xx + sa * yy
        v = -sa * xx + ca * yy
        uc = np.clip(u, -species.body_length / 2.0, species.body_length / 2.0)
        rr = (u - uc) ** 2 + v ** 2
        masks.append(rr <= species.body_radius ** 2)
    return masks


def shell_mask(species: SpeciesProfile, dx: float):
    outer = species.body_radius + species.appendage_length + 0.5 * max(species.body_length, 0.0)
    n = int(np.ceil(outer / dx))
    s = np.arange(-n, n + 1) * dx
    xx, yy = np.meshgrid(s, s)
    shell = (xx * xx + yy * yy) <= outer ** 2
    inner = (xx * xx + yy * yy) <= (species.body_radius + 0.15 * max(species.body_length, 0.0)) ** 2
    return shell & (~inner)


def attachment_layers(surface, species: SpeciesProfile, config: Config):
    dy, dxg = np.gradient(surface, config.dx, config.dx)
    slope = np.hypot(dxg, dy)
    curv = np.abs(ndimage.laplace(surface, mode="nearest")) / max(config.dx * config.dx, 1e-6)
    energies, contacts = [], []
    relief_ref = np.zeros_like(surface)
    for mask in body_masks(species, config.dx):
        kernel = mask.astype(float)
        kernel /= max(kernel.sum(), 1.0)
        local_mean = ndimage.convolve(surface, kernel, mode="nearest")
        local_max = ndimage.maximum_filter(surface, footprint=mask, mode="nearest")
        local_min = ndimage.minimum_filter(surface, footprint=mask, mode="nearest")
        relief = np.clip(local_max - local_min, 0.0, None)
        gap = np.clip(local_max - local_mean, 0.0, None)
        relief_ref = relief
        eff_gap = gap / (1.0 + species.compliance if config.use_compliance else 1.0)
        body = -species.body_adhesion * np.exp(-eff_gap / max(species.adhesion_range, 1e-6))
        deform = 0.5 * species.compliance_penalty * eff_gap ** 2
        steric = species.steric_penalty * np.clip(relief - species.body_radius * (1.2 + 0.8 * species.compliance), 0.0, None) ** 2
        slope_cost = species.slope_penalty * ndimage.convolve(slope, kernel, mode="nearest")
        energies.append(body + deform + steric + slope_cost)
        contacts.append(np.exp(-gap / max(species.adhesion_range, 1e-6)))
    energy = np.min(np.stack(energies), axis=0)
    body_contact = np.max(np.stack(contacts), axis=0)
    if config.use_appendages and species.appendage_count > 0:
        ring = shell_mask(species, config.dx)
        shell_slope = ndimage.maximum_filter(slope, footprint=ring, mode="nearest")
        shell_contact = ndimage.maximum_filter(body_contact, footprint=ring, mode="nearest")
        appendage = np.exp(-species.appendage_slope_penalty * shell_slope) * shell_contact
        appendage_prob = 1.0 - np.exp(-species.appendage_binding * species.appendage_count * appendage)
        energy = energy - species.appendage_strength * appendage_prob
    else:
        appendage_prob = np.zeros_like(surface)
    attach = np.exp(-config.beta * (energy - np.min(energy)))
    attach /= max(float(np.max(attach)), 1e-8)
    return {"attach": attach, "appendage": appendage_prob, "slope": slope, "curv": curv, "relief": relief_ref}


def reduced_score(layers, species: SpeciesProfile, surface: Surface, config: Config):
    active = layers["attach"] >= 0.52
    mean_pitch = 0.5 * (surface.params.pitch_x + surface.params.pitch_y)
    bn = 0.30 * mean_pitch / (1.0 * (1.0 ** 0.60) + 1e-9)
    ell = species.bridge_ell0 / (1.0 + bn)
    rad = max(1, int(round(ell / config.dx)))
    yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
    se = (xx * xx + yy * yy) <= rad * rad
    dil = ndimage.binary_dilation(active, structure=se)
    labels, num = ndimage.label(dil)
    lcc = float(np.max(ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, num + 1))) / active.size) if num else 0.0
    hydro = float(np.exp(-species.hydro_chi * species.motility_um_s * float(np.sqrt(np.mean(layers["slope"] ** 2)))))
    raw = float(np.mean(layers["attach"])) * lcc * hydro
    return {"stable_fraction": float(sigmoid(raw, center=config.obs_bias, scale=1.0 / max(config.obs_gain, 1e-6))), "raw_fraction": raw, "largest_cc": lcc}


def arrival_flux(layers, species: SpeciesProfile, config: Config):
    base = np.exp(-species.transport_slope_penalty * layers["slope"] - species.transport_relief_penalty * layers["relief"])
    sigma_px = max(species.transport_sigma / max(config.dx, 1e-6), 0.8)
    if config.use_transport and species.motility_um_s > 0.0:
        drive = 1.0 + 0.018 * species.motility_um_s * np.exp(-0.15 * layers["curv"])
        flux = ndimage.gaussian_filter(base * drive, sigma=sigma_px, mode="nearest")
    elif config.use_transport:
        flux = ndimage.gaussian_filter(base, sigma=max(0.7 * sigma_px, 0.8), mode="nearest")
    else:
        flux = np.ones_like(base)
    flux = np.clip(flux, 0.0, None)
    ref = max(float(np.percentile(flux, 95.0)), 1e-8)
    shape = np.clip(flux / ref, 0.0, None)
    arrival_drive = 0.80 + 0.004 * species.motility_um_s + 0.012 * species.appendage_count + 0.035 * species.appendage_length
    return np.clip(arrival_drive * shape, 0.0, 2.5)


def evolve(layers, flux, species: SpeciesProfile, config: Config):
    shape = layers["attach"].shape
    appendage = layers["appendage"] if config.use_appendages else np.zeros(shape)
    anchor_support = np.clip(layers["attach"] * (1.0 + 0.90 * appendage), 0.0, 1.8)
    retention = np.clip(np.exp(-0.22 * layers["slope"]) * (1.0 + 0.60 * appendage + 0.35 * layers["attach"]), 0.30, 2.2)
    f = np.clip(config.arrival_rate * flux.copy(), 0.0, 1.0)
    t = np.zeros(shape)
    a = np.zeros(shape)
    e = np.zeros(shape)
    m = np.zeros(shape)
    eps = np.zeros(shape)
    series = []
    for step in range(config.n_steps):
        if config.use_state_dynamics:
            l_ft = config.capture_rate * flux * (0.55 + 0.45 * layers["attach"])
            l_tf = species.detach_rate * np.exp(0.25 * layers["slope"]) / retention
            l_ta = species.anchor_rate * anchor_support
            l_at = species.reorientation_rate * np.exp(0.15 * layers["slope"]) / np.clip(0.85 + 1.10 * appendage, 0.35, None)
            eps_attach_gate = sigmoid(eps, center=config.eps_attach_threshold, scale=max(0.35 * config.eps_attach_threshold, 5e-4))
            eps_mature_gate = sigmoid(eps, center=config.eps_mature_threshold, scale=max(0.35 * config.eps_mature_threshold, 8e-4))
            l_ae = species.eps_activation_rate * (0.40 + 0.60 * layers["attach"] + 0.15 * appendage) * eps_attach_gate
            l_em = species.maturation_rate * (0.50 + 0.50 * layers["attach"]) * eps_mature_gate
            replenishment = config.arrival_rate * flux * (0.12 + 0.10 * appendage)
            f = np.clip(f + config.dt * (-l_ft * f + l_tf * t + l_at * a + replenishment), 0.0, 1.0)
            t = np.clip(t + config.dt * (l_ft * f - (l_tf + l_ta) * t), 0.0, 1.0)
            a = np.clip(a + config.dt * (l_ta * t - (l_at + l_ae) * a), 0.0, 1.0)
            e = np.clip(e + config.dt * (l_ae * a - l_em * e), 0.0, 1.0)
            m = np.clip(m + config.dt * (l_em * e), 0.0, 1.0)
        else:
            t = np.clip(layers["attach"] * flux, 0.0, 1.0)
            a = np.clip(anchor_support, 0.0, 1.0)
            e = a.copy()
            m = np.clip(a * sigmoid(flux, center=0.45, scale=0.18), 0.0, 1.0)
        if config.use_dynamic_eps:
            source = species.eps_production * (0.35 * a + 0.85 * e + 1.25 * m) * np.clip(0.65 + 0.45 * layers["attach"] + 0.10 * appendage, 0.0, None)
            nonlinear = np.power(np.maximum(eps, 1e-8), config.eps_diffusion_exponent)
            diff = ndimage.laplace(nonlinear, mode="nearest") / max(config.dx * config.dx, 1e-6)
            mob = sigmoid(np.abs(diff), center=config.eps_yield, scale=0.03)
            eps = np.clip(eps + config.dt * (config.eps_diffusivity * mob * diff + source - config.eps_decay * eps), 0.0, None)
            eps = ndimage.gaussian_filter(eps, sigma=0.6, mode="nearest")
        else:
            eps = np.clip(0.65 * a + 0.35 * m, 0.0, None)
        series.append({"step": step, "free_mean": float(np.mean(f)), "transient_mean": float(np.mean(t)), "anchored_mean": float(np.mean(a)), "eps_attached_mean": float(np.mean(e)), "microcolony_mean": float(np.mean(m)), "eps_mean": float(np.mean(eps))})
    mask = eps >= config.eps_bridge_threshold
    labels, num = ndimage.label(mask)
    lcc = float(np.max(ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, num + 1))) / mask.size) if num else 0.0
    eps_cover = float(np.mean(mask))
    raw_core = float(np.mean(0.62 * m + 0.25 * e + 0.13 * a))
    support_gate = 0.10 + 0.50 * lcc + 0.35 * float(np.mean(layers["attach"])) + 0.25 * float(np.mean(anchor_support))
    environment_gate = 0.40 + 0.40 * eps_cover + 0.20 * min(float(np.mean(flux)), 1.5)
    raw = raw_core * support_gate * environment_gate
    stable = float(sigmoid(raw, center=config.obs_bias, scale=1.0 / max(config.obs_gain, 1e-6)))
    return {"stable_fraction": stable, "raw_fraction": raw, "largest_cc": lcc, "eps": eps, "eps_cover": eps_cover, "anchor_support_mean": float(np.mean(anchor_support)), "timeseries": pd.DataFrame(series)}


def evaluate_surface(surface: Surface, species: SpeciesProfile, config: Config):
    layers = attachment_layers(surface.surface, species, config)
    flux = arrival_flux(layers, species, config)
    full = evolve(layers, flux, species, config)
    full.update({"attachment_mean": float(np.mean(layers["attach"])), "appendage_mean": float(np.mean(layers["appendage"])), "arrival_mean": float(np.mean(flux)), "eps_mean": float(np.mean(full["eps"])), "layers": layers})
    reduced = reduced_score(layers, species, surface, config)
    return full, reduced

def calibrate_obs(surfaces, species, config: Config):
    target_names = ("E. coli", "S. aureus")
    subset = surfaces[: min(config.calibration_surfaces, len(surfaces))]
    raw_samples = {name: [] for name in target_names}
    for surface in subset:
        for name in target_names:
            full, _ = evaluate_surface(surface, species[name], config)
            raw_samples[name].append(float(full["raw_fraction"]))
    raw_means = {name: float(np.mean(values)) for name, values in raw_samples.items()}
    raw_spread = {name: float(np.std(values)) for name, values in raw_samples.items()}
    ordered = sorted(((raw_means[name], species[name].observed, name) for name in target_names), key=lambda item: item[0])
    x1, y1, _ = ordered[0]
    x2, y2, _ = ordered[1]
    gap = max(abs(x2 - x1), 1e-6)
    spread = max(float(np.mean(list(raw_spread.values()))), 1e-4)
    logit = lambda y: np.log(np.clip(y, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - y, 1e-6, 1.0))
    analytic_gain = max((logit(y2) - logit(y1)) / gap, 1.0)
    analytic_bias = x1 - logit(y1) / analytic_gain
    gain_floor = max(10.0, 0.18 * analytic_gain)
    gain_cap = min(140.0, max(50.0, 2.6 / spread))
    gains = np.linspace(gain_floor, gain_cap, 61)
    bias_half_width = 3.0 * max(gap, spread)
    biases = np.linspace(analytic_bias - bias_half_width, analytic_bias + bias_half_width, 81)
    best = None
    for gain in gains:
        for bias in biases:
            sse = 0.0
            preds = {}
            pred_spread = {}
            for name, values in raw_samples.items():
                pred_vec = sigmoid(np.asarray(values), center=bias, scale=1.0 / gain)
                preds[name] = float(np.mean(pred_vec))
                pred_spread[name] = float(np.std(pred_vec))
                sse += (preds[name] - species[name].observed) ** 2
                sse += 0.05 * pred_spread[name] ** 2
            sse += 1e-4 * (gain / gain_cap) ** 2
            if best is None or sse < best["sse"]:
                best = {"obs_gain": float(gain), "obs_bias": float(bias), "sse": float(sse), "predictions": preds, "prediction_spread": pred_spread, "raw_means": raw_means, "raw_spread": raw_spread, "fit_species": list(target_names), "calibration_surfaces": len(subset)}
    return best


def summarise(surfaces, species, config: Config):
    rows, rep = [], {}
    for idx, surface in enumerate(surfaces):
        for sp in species.values():
            full, reduced = evaluate_surface(surface, sp, config)
            rows.append({"surface_id": surface.params.name, "surface_index": idx, "species": sp.name, "mechanism": "full_hierarchical", "loss": surface.loss, "stable_fraction": full["stable_fraction"], "raw_fraction": full["raw_fraction"], "largest_cc": full["largest_cc"], "attachment_mean": full["attachment_mean"], "appendage_mean": full["appendage_mean"], "arrival_mean": full["arrival_mean"], "eps_mean": full["eps_mean"], "eps_cover": full["eps_cover"], "anchor_support_mean": full["anchor_support_mean"]})
            rows.append({"surface_id": surface.params.name, "surface_index": idx, "species": sp.name, "mechanism": "reduced", "loss": surface.loss, "stable_fraction": reduced["stable_fraction"], "raw_fraction": reduced["raw_fraction"], "largest_cc": reduced["largest_cc"], "attachment_mean": full["attachment_mean"], "appendage_mean": full["appendage_mean"], "arrival_mean": full["arrival_mean"], "eps_mean": full["eps_mean"], "eps_cover": full["eps_cover"], "anchor_support_mean": full["anchor_support_mean"]})
            if idx == 0:
                rep[sp.name] = {"surface": surface, "full": full, "reduced": reduced}
    return pd.DataFrame(rows), rep


def lhs(n, bounds, seed=101):
    rng = np.random.default_rng(seed)
    cols = {}
    for key, (low, high) in bounds.items():
        edges = np.linspace(0.0, 1.0, n + 1)
        vals = edges[:-1] + (edges[1:] - edges[:-1]) * rng.random(n)
        rng.shuffle(vals)
        cols[key] = low + (high - low) * vals
    return pd.DataFrame(cols)


def run_uq(surfaces, species, config: Config):
    bounds = uq_bounds()
    samples = lhs(config.uq_samples, bounds, seed=205)
    rows = []
    base = surfaces[0]
    for idx, sample in samples.iterrows():
        params = replace(base.params, name=f"uq_{idx:03d}", tip_radius_mu=float(sample.tip_radius_mu), base_radius_mu=float(sample.base_radius_mu), sharpness_mu=float(sample.sharpness_mu), seed=base.params.seed + idx + 1)
        srf, centers, desc = build_surface(params, config.dx, (config.domain_x, config.domain_y))
        surface = Surface(params, srf, centers, desc, 0.0)
        cfg = replace(config, eps_diffusivity=float(sample.eps_diffusivity), capture_rate=float(sample.capture_rate))
        for sp in species.values():
            tuned = replace(sp, compliance=sp.compliance * float(sample.compliance_scale), appendage_length=sp.appendage_length * float(sample.appendage_length_scale))
            full, _ = evaluate_surface(surface, tuned, cfg)
            rows.append({"sample_id": idx, "species": sp.name, **sample.to_dict(), "stable_fraction": full["stable_fraction"]})
    sample_df = pd.DataFrame(rows)
    out = []
    for name in species:
        subset = sample_df[sample_df.species == name]
        for key in bounds:
            rho = subset[[key, "stable_fraction"]].corr(method="spearman").iloc[0, 1]
            out.append({"species": name, "parameter": key, "spearman_rho": float(rho)})
    return sample_df, pd.DataFrame(out)


def phase_diagram(species, config: Config):
    rows = []
    for pitch in config.phase_pitch_values:
        for scale in config.phase_appendage_scales:
            params = GeometryParams(f"phase_{pitch:.2f}_{scale:.2f}", pitch_x=pitch, pitch_y=max(2.8, pitch - 0.2), seed=int(1000 * pitch + 100 * scale))
            srf, centers, desc = build_surface(params, config.dx, (config.domain_x, config.domain_y))
            surface = Surface(params, srf, centers, desc, 0.0)
            for sp in species.values():
                tuned = replace(sp, appendage_length=sp.appendage_length * scale)
                full, _ = evaluate_surface(surface, tuned, config)
                rows.append({"pitch": pitch, "appendage_scale": scale, "species": sp.name, "stable_fraction": full["stable_fraction"]})
    return pd.DataFrame(rows)


def ablations(surface: Surface, species, config: Config):
    rows = []
    variants = {
        "reduced_limit": replace(config, use_compliance=False, use_appendages=False, use_dynamic_eps=False, use_state_dynamics=False, use_transport=False),
        "mechanics_only": replace(config, use_appendages=False, use_dynamic_eps=False, use_state_dynamics=False, use_transport=False),
        "mechanics_plus_appendages": replace(config, use_appendages=True, use_dynamic_eps=False, use_state_dynamics=False, use_transport=False),
        "mechanics_plus_eps": replace(config, use_appendages=True, use_dynamic_eps=True, use_state_dynamics=False, use_transport=False),
        "full_hierarchical": config,
    }
    for sp in species.values():
        for name, cfg in variants.items():
            full, _ = evaluate_surface(surface, sp, cfg)
            rows.append({"species": sp.name, "mechanism": name, "stable_fraction": full["stable_fraction"], "raw_fraction": full["raw_fraction"], "largest_cc": full["largest_cc"]})
    return pd.DataFrame(rows)


def surface_prior_table():
    return [
        {"parameter": "pitch_x", "distribution": "truncated_normal", "center": 4.6, "scale": 0.30, "lower": 3.6, "upper": 6.4, "units": "um", "notes": "anisotropic horizontal pitch prior"},
        {"parameter": "pitch_y", "distribution": "truncated_normal", "center": 4.4, "scale": 0.30, "lower": 3.4, "upper": 6.2, "units": "um", "notes": "anisotropic vertical pitch prior"},
        {"parameter": "height_mu", "distribution": "truncated_normal", "center": 9.0, "scale": 0.35, "lower": 7.8, "upper": 10.2, "units": "um", "notes": "mean spike height prior"},
        {"parameter": "height_sd", "distribution": "truncated_normal", "center": 0.7, "scale": 0.12, "lower": 0.35, "upper": 1.10, "units": "um", "notes": "height heterogeneity prior"},
        {"parameter": "tip_radius_mu", "distribution": "truncated_normal", "center": 0.15, "scale": 0.04, "lower": 0.06, "upper": 0.34, "units": "um", "notes": "tip rounding prior"},
        {"parameter": "base_radius_mu", "distribution": "truncated_normal", "center": 1.55, "scale": 0.18, "lower": 1.0, "upper": 2.2, "units": "um", "notes": "base radius prior"},
        {"parameter": "sharpness_mu", "distribution": "truncated_normal", "center": 2.4, "scale": 0.30, "lower": 1.4, "upper": 3.6, "units": "1", "notes": "sidewall sharpness prior"},
        {"parameter": "jitter_x", "distribution": "truncated_normal", "center": 0.08, "scale": 0.02, "lower": 0.03, "upper": 0.18, "units": "um", "notes": "centroid jitter along x"},
        {"parameter": "jitter_y", "distribution": "truncated_normal", "center": 0.15, "scale": 0.03, "lower": 0.05, "upper": 0.30, "units": "um", "notes": "centroid jitter along y"},
        {"parameter": "latent_amplitude", "distribution": "truncated_normal", "center": 0.18, "scale": 0.05, "lower": 0.05, "upper": 0.35, "units": "um", "notes": "latent roughness background"},
    ]


def uq_bounds():
    return {
        "tip_radius_mu": (0.08, 0.30),
        "base_radius_mu": (1.10, 1.95),
        "sharpness_mu": (1.4, 3.3),
        "compliance_scale": (0.7, 1.4),
        "appendage_length_scale": (0.6, 1.8),
        "eps_diffusivity": (0.08, 0.22),
        "capture_rate": (0.35, 0.75),
    }


def grid_coordinates(dx: float, domain):
    xs = np.arange(0.0, domain[0] + dx, dx)
    ys = np.arange(0.0, domain[1] + dx, dx)
    xx, yy = np.meshgrid(xs, ys)
    return xs, ys, xx, yy


def build_smooth_control_surface(config: Config):
    _, _, xx, _ = grid_coordinates(config.dx, (config.domain_x, config.domain_y))
    params = GeometryParams(
        "smooth_control",
        pitch_x=6.0,
        pitch_y=6.0,
        height_mu=0.0,
        height_sd=0.0,
        tip_radius_mu=0.0,
        base_radius_mu=0.0,
        sharpness_mu=1.0,
        latent_amplitude=0.0,
        seed=0,
    )
    return Surface(params, np.zeros_like(xx), np.zeros((0, 2)), {"family": "smooth_control"}, float("nan"))


def rectangular_bar_mask(xx, yy, center_x, center_y, length, width, angle_deg=0.0):
    theta = np.deg2rad(angle_deg)
    x_local = np.cos(theta) * (xx - center_x) + np.sin(theta) * (yy - center_y)
    y_local = -np.sin(theta) * (xx - center_x) + np.cos(theta) * (yy - center_y)
    return (np.abs(x_local) <= 0.5 * length) & (np.abs(y_local) <= 0.5 * width)


def build_sharklet_like_surface(config: Config):
    _, _, xx, yy = grid_coordinates(config.dx, (config.domain_x, config.domain_y))
    surface = np.zeros_like(xx)
    feature_height = 3.0
    feature_width = 2.0
    feature_spacing = 2.0
    feature_lengths = [4.0, 8.0, 12.0, 16.0, 12.0, 8.0, 4.0]
    centers = []
    for origin_x in np.arange(7.0, config.domain_x + 1e-9, 16.0):
        for origin_y in np.arange(6.0, config.domain_y + 1e-9, 17.0):
            for idx, length in enumerate(feature_lengths):
                center_x = origin_x + 0.75 * idx
                center_y = origin_y + 0.78 * idx * (feature_width + feature_spacing)
                if center_y + 0.5 * feature_width > config.domain_y - 0.5:
                    continue
                mask = rectangular_bar_mask(xx, yy, center_x, center_y, length, feature_width, angle_deg=32.0)
                surface[mask] = feature_height
                centers.append((center_x, center_y))
    params = GeometryParams(
        "sharklet_like_positive_control",
        pitch_x=4.0,
        pitch_y=4.0,
        height_mu=feature_height,
        height_sd=0.0,
        tip_radius_mu=1.0,
        base_radius_mu=1.0,
        sharpness_mu=1.0,
        latent_amplitude=0.0,
        seed=0,
    )
    return Surface(
        params,
        surface,
        np.asarray(centers, dtype=float) if centers else np.zeros((0, 2)),
        {
            "family": "sharklet_like_positive_control",
            "feature_height": feature_height,
            "feature_width": feature_width,
            "feature_spacing": feature_spacing,
            "feature_lengths": ",".join(f"{length:.1f}" for length in feature_lengths),
        },
        float("nan"),
    )


def build_positive_control_surfaces(config: Config):
    return {
        "smooth_control": build_smooth_control_surface(config),
        "sharklet_like_positive_control": build_sharklet_like_surface(config),
    }


def evaluate_benchmark_validation(native_surfaces, species, config: Config):
    rows = []
    for surface in native_surfaces:
        for sp in species.values():
            full, _ = evaluate_surface(surface, sp, config)
            rows.append(
                {
                    "surface_family": "native_ensemble",
                    "surface_id": surface.params.name,
                    "species": sp.name,
                    "stable_fraction": full["stable_fraction"],
                    "raw_fraction": full["raw_fraction"],
                    "largest_cc": full["largest_cc"],
                }
            )
    for family, surface in build_positive_control_surfaces(config).items():
        for sp in species.values():
            full, _ = evaluate_surface(surface, sp, config)
            rows.append(
                {
                    "surface_family": family,
                    "surface_id": surface.params.name,
                    "species": sp.name,
                    "stable_fraction": full["stable_fraction"],
                    "raw_fraction": full["raw_fraction"],
                    "largest_cc": full["largest_cc"],
                }
            )
    benchmark_df = pd.DataFrame(rows)
    summary = (
        benchmark_df.groupby(["surface_family", "species"])
        .agg(
            stable_fraction_mean=("stable_fraction", "mean"),
            stable_fraction_std=("stable_fraction", "std"),
            raw_fraction_mean=("raw_fraction", "mean"),
            largest_cc_mean=("largest_cc", "mean"),
            n_surfaces=("surface_id", "nunique"),
        )
        .reset_index()
    )
    summary["stable_fraction_std"] = summary["stable_fraction_std"].fillna(0.0)
    return benchmark_df, summary


def apply_species_scalars(profile: SpeciesProfile, scalars: dict[str, float]):
    if not scalars:
        return profile
    updates = {field: getattr(profile, field) * factor for field, factor in scalars.items()}
    return replace(profile, **updates)


def rebuild_surface(surface: Surface, config: Config, tag: str, param_updates: dict[str, float]):
    if not param_updates:
        return surface
    new_params = replace(surface.params, name=f"{surface.params.name}_{tag}", **param_updates)
    rebuilt, centers, descriptors = build_surface(new_params, config.dx, (config.domain_x, config.domain_y))
    return Surface(new_params, rebuilt, centers, descriptors, surface.loss)


def device_condition_scenarios(surface: Surface, config: Config):
    rounded_surface = {
        "tip_radius_mu": min(surface.params.tip_radius_mu + 0.08, 0.45),
        "base_radius_mu": min(surface.params.base_radius_mu + 0.12, 2.40),
        "sharpness_mu": max(surface.params.sharpness_mu - 0.60, 1.2),
        "height_mu": max(surface.params.height_mu - 0.50, 6.5),
    }
    return [
        {
            "name": "reference_lab",
            "description": "Current buffered in vitro reference used for the base analysis.",
            "config_updates": {},
            "species_scales": {},
            "surface_updates": {},
        },
        {
            "name": "protein_conditioned",
            "description": "Protein-conditioning film raises short-range adhesion and appendage engagement.",
            "config_updates": {"capture_rate": min(config.capture_rate * 1.05, 0.99), "eps_decay": config.eps_decay * 0.92},
            "species_scales": {"body_adhesion": 1.18, "appendage_binding": 1.25, "adhesion_range": 1.10},
            "surface_updates": {},
        },
        {
            "name": "low_shear_loading",
            "description": "Low-shear aqueous loading reduces replenishment while increasing residence time.",
            "config_updates": {"arrival_rate": config.arrival_rate * 0.82, "capture_rate": min(config.capture_rate * 1.10, 0.99)},
            "species_scales": {"detach_rate": 0.88, "transport_slope_penalty": 0.92},
            "surface_updates": {},
        },
        {
            "name": "soft_polymer_interface",
            "description": "Compliant polymer contact increases effective mechanical conformity of early attachment.",
            "config_updates": {"capture_rate": min(config.capture_rate * 1.02, 0.99)},
            "species_scales": {"compliance": 1.22, "compliance_penalty": 0.92, "steric_penalty": 0.95},
            "surface_updates": {},
        },
        {
            "name": "fabrication_rounding",
            "description": "Replication-induced tip rounding and partial feature-height loss.",
            "config_updates": {},
            "species_scales": {},
            "surface_updates": rounded_surface,
        },
        {
            "name": "combined_device_envelope",
            "description": "Protein-conditioned, low-shear, compliant silicone-device envelope with rounded features.",
            "config_updates": {"arrival_rate": config.arrival_rate * 0.82, "capture_rate": min(config.capture_rate * 1.12, 0.99), "eps_decay": config.eps_decay * 0.92},
            "species_scales": {
                "body_adhesion": 1.18,
                "appendage_binding": 1.25,
                "adhesion_range": 1.10,
                "detach_rate": 0.88,
                "transport_slope_penalty": 0.92,
                "compliance": 1.22,
                "compliance_penalty": 0.92,
                "steric_penalty": 0.95,
            },
            "surface_updates": rounded_surface,
        },
    ]


def run_boundary_condition_robustness(native_surface: Surface, species, config: Config):
    smooth_control = build_smooth_control_surface(config)
    rows = []
    for scenario in device_condition_scenarios(native_surface, config):
        cfg = replace(config, **scenario["config_updates"])
        scenario_native = rebuild_surface(native_surface, config, scenario["name"], scenario["surface_updates"])
        for family, eval_surface in {"native_reference": scenario_native, "smooth_control": smooth_control}.items():
            for sp in species.values():
                tuned = apply_species_scalars(sp, scenario["species_scales"])
                full, _ = evaluate_surface(eval_surface, tuned, cfg)
                rows.append(
                    {
                        "scenario": scenario["name"],
                        "scenario_description": scenario["description"],
                        "surface_family": family,
                        "surface_id": eval_surface.params.name,
                        "species": sp.name,
                        "stable_fraction": full["stable_fraction"],
                        "raw_fraction": full["raw_fraction"],
                    }
                )
    boundary_df = pd.DataFrame(rows)
    ratio_rows = []
    for scenario_name in boundary_df["scenario"].unique():
        for species_name in boundary_df["species"].unique():
            subset = boundary_df[(boundary_df["scenario"] == scenario_name) & (boundary_df["species"] == species_name)]
            native_value = float(subset[subset["surface_family"] == "native_reference"]["stable_fraction"].iloc[0])
            smooth_value = float(subset[subset["surface_family"] == "smooth_control"]["stable_fraction"].iloc[0])
            ratio_rows.append(
                {
                    "scenario": scenario_name,
                    "species": species_name,
                    "native_stable_fraction": native_value,
                    "smooth_stable_fraction": smooth_value,
                    "native_to_smooth_ratio": native_value / max(smooth_value, 1e-8),
                    "suppression_vs_smooth": 1.0 - native_value / max(smooth_value, 1e-8),
                }
            )
    return boundary_df, pd.DataFrame(ratio_rows)


def add_scale_bar(ax, length_um: float, config: Config, color: str = "white"):
    x0 = config.domain_x - length_um - 2.0
    y0 = 2.0
    ax.plot([x0, x0 + length_um], [y0, y0], color=color, linewidth=2.4, solid_capstyle="butt")
    ax.text(x0 + 0.5 * length_um, y0 + 1.2, f"{length_um:.0f} um", color=color, fontsize=8, ha="center", va="bottom")


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def plot_surfaces(surfaces, descriptor_df, layout: OutputLayout, config: Config):
    plt.rcParams.update({"font.family": "serif", "figure.dpi": 180, "savefig.dpi": 180, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), constrained_layout=True)
    extent = (0.0, config.domain_x, 0.0, config.domain_y)
    vmax = max(max(float(np.max(surface.surface)), 1.0) for surface in surfaces[:4])
    image = None
    for ax, surface in zip(axes.ravel(), surfaces[:4]):
        image = ax.imshow(surface.surface, origin="lower", cmap="inferno", extent=extent, vmin=0.0, vmax=vmax)
        ax.set_title(f"{surface.params.name}\nloss={surface.loss:.2f}", fontsize=9)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_xticks([0, 10, 20, 30])
        ax.set_yticks([0, 10, 20, 30])
        add_scale_bar(ax, 10.0, config)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.84, label="height (um)")
    fig.savefig(layout.figures / "figure_surface_ensemble.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.hist(descriptor_df["loss"], bins=min(10, len(descriptor_df)), color="#4C78A8", edgecolor="white", linewidth=1.2, rwidth=0.95)
    ax.set_xlabel("Descriptor loss")
    ax.set_ylabel("Accepted surfaces")
    fig.savefig(layout.figures / "figure_surface_losses.png")
    plt.close(fig)


def plot_ablations(ablation_df, layout: OutputLayout):
    plt.rcParams.update({"font.family": "serif", "figure.dpi": 180, "savefig.dpi": 180, "savefig.bbox": "tight"})
    mechanisms = list(dict.fromkeys(ablation_df["mechanism"]))
    species = list(dict.fromkeys(ablation_df["species"]))
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    x = np.arange(len(species))
    width = 0.16
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    handles = []
    labels = []
    label_map = {"mechanics_plus_eps": "mechanics plus EPS"}
    for idx, mechanism in enumerate(mechanisms):
        vals = [ablation_df[(ablation_df.species == sp) & (ablation_df.mechanism == mechanism)]["raw_fraction"].iloc[0] for sp in species]
        display_label = label_map.get(mechanism, mechanism.replace("_", " "))
        bar = ax.bar(x + (idx - 2) * width, vals, width=width, label=display_label, color=colors[idx % len(colors)])
        handles.append(bar[0])
        labels.append(display_label)
    ax.set_xticks(x); ax.set_xticklabels(species, style="italic")
    ax.set_ylabel("Raw colonisation signal")
    fig.subplots_adjust(bottom=0.26)
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.02))
    fig.savefig(layout.figures / "figure_mechanism_ablation.png")
    plt.close(fig)


def plot_phase(phase_df, layout: OutputLayout):
    plt.rcParams.update({"font.family": "serif", "figure.dpi": 180, "savefig.dpi": 180, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.0), constrained_layout=True)
    for ax, name in zip(axes, ["E. coli", "S. aureus", "P. aeruginosa"]):
        subset = phase_df[phase_df.species == name]
        pivot = subset.pivot(index="appendage_scale", columns="pitch", values="stable_fraction")
        im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns))); ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index))); ax.set_yticklabels([f"{y:.1f}" for y in pivot.index])
        ax.set_xlabel("Pitch (um)"); ax.set_ylabel("Appendage scale")
        ax.set_title(name, style="italic")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(layout.figures / "figure_phase_diagram.png")
    plt.close(fig)


def plot_benchmarks(benchmark_summary: pd.DataFrame, layout: OutputLayout):
    plt.rcParams.update({"font.family": "serif", "figure.dpi": 180, "savefig.dpi": 180, "savefig.bbox": "tight"})
    order = ["smooth_control", "sharklet_like_positive_control", "native_ensemble"]
    labels = {
        "smooth_control": "smooth control",
        "sharklet_like_positive_control": "Sharklet-like benchmark",
        "native_ensemble": "snake ensemble",
    }
    colors = {"smooth_control": "#B3B3B3", "sharklet_like_positive_control": "#4C78A8", "native_ensemble": "#E45756"}
    species_order = ["E. coli", "S. aureus", "P. aeruginosa"]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(species_order))
    width = 0.24
    for idx, family in enumerate(order):
        subset = benchmark_summary[benchmark_summary.surface_family == family].set_index("species").reindex(species_order)
        ax.bar(
            x + (idx - 1) * width,
            subset["stable_fraction_mean"],
            width=width,
            yerr=subset["stable_fraction_std"],
            capsize=3,
            color=colors[family],
            label=labels[family],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(species_order, style="italic")
    ax.set_ylabel("Stable fraction")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    fig.savefig(layout.figures / "figure_benchmark_validation.png")
    plt.close(fig)


def plot_boundary_conditions(boundary_summary: pd.DataFrame, layout: OutputLayout):
    plt.rcParams.update({"font.family": "serif", "figure.dpi": 180, "savefig.dpi": 180, "savefig.bbox": "tight"})
    scenario_order = [
        "reference_lab",
        "protein_conditioned",
        "low_shear_loading",
        "soft_polymer_interface",
        "fabrication_rounding",
        "combined_device_envelope",
    ]
    scenario_labels = [
        "reference",
        "protein\nconditioned",
        "low-shear\nloading",
        "soft-polymer\ninterface",
        "fabrication\nrounding",
        "combined\nenvelope",
    ]
    species_order = ["E. coli", "S. aureus", "P. aeruginosa"]
    native_pivot = boundary_summary.pivot(index="scenario", columns="species", values="native_stable_fraction").reindex(index=scenario_order, columns=species_order)
    suppression_pivot = boundary_summary.pivot(index="scenario", columns="species", values="suppression_vs_smooth").reindex(index=scenario_order, columns=species_order)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), constrained_layout=True)
    panels = [
        (axes[0], native_pivot, "Native-surface stable fraction", "magma", 0.0, float(np.nanmax(native_pivot.values))),
        (axes[1], suppression_pivot, "Suppression relative to smooth", "viridis", float(np.nanmin(suppression_pivot.values)), float(np.nanmax(suppression_pivot.values))),
    ]
    for ax, pivot, title, cmap, vmin, vmax in panels:
        image = ax.imshow(pivot.values, origin="upper", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(species_order)))
        ax.set_xticklabels(species_order, rotation=15, ha="right", style="italic")
        ax.set_yticks(np.arange(len(scenario_labels)))
        ax.set_yticklabels(scenario_labels)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.82)
    fig.savefig(layout.figures / "figure_device_boundary_conditions.png")
    plt.close(fig)


def write_supporting_information(layout: OutputLayout, config: Config, cal: dict, benchmark_summary: pd.DataFrame, boundary_summary: pd.DataFrame):
    target, tol = reference_target(config)
    descriptor_rows = []
    for key in ["density", "nn_mean", "nn_std", "anisotropy", "height_mean", "height_std", "tip_mean", "base_mean", "sharp_mean", "occupied"]:
        descriptor_rows.append(
            {
                "descriptor": key,
                "target_value": target[key],
                "tolerance": tol[key],
                "inverse_tolerance_sq": 1.0 / max(tol[key] ** 2, 1e-12),
                "acceptance_cutoff": 12.0,
            }
        )
    descriptor_rows.append(
        {
            "descriptor": "paircorr_bins_0_9",
            "target_value": float(np.mean(target["paircorr"])),
            "tolerance": tol["paircorr"],
            "inverse_tolerance_sq": 1.0 / max(tol["paircorr"] ** 2, 1e-12),
            "acceptance_cutoff": 12.0,
        }
    )
    pd.DataFrame(surface_prior_table()).to_csv(layout.supporting / "si_surface_priors.csv", index=False)
    pd.DataFrame(descriptor_rows).to_csv(layout.supporting / "si_descriptor_acceptance.csv", index=False)
    pd.DataFrame([{"parameter": key, "lower": bounds[0], "upper": bounds[1]} for key, bounds in uq_bounds().items()]).to_csv(layout.supporting / "si_uq_bounds.csv", index=False)
    pd.DataFrame(
        [
            {
                "figure": "figure_surface_ensemble.png",
                "generator": "plot_surfaces",
                "data_dependencies": "surface_ensemble_descriptors.csv",
                "scale_information": f"Each panel spans {config.domain_x:.0f} x {config.domain_y:.0f} um; white scale bar = 10 um.",
            },
            {
                "figure": "figure_surface_losses.png",
                "generator": "plot_surfaces",
                "data_dependencies": "surface_ensemble_descriptors.csv",
                "scale_information": "Histogram; not image-scale dependent.",
            },
            {
                "figure": "figure_mechanism_ablation.png",
                "generator": "plot_ablations",
                "data_dependencies": "hierarchical_mechanism_ablation.csv",
                "scale_information": "Bar chart; not image-scale dependent.",
            },
            {
                "figure": "figure_phase_diagram.png",
                "generator": "plot_phase",
                "data_dependencies": "hierarchical_phase_diagram.csv",
                "scale_information": "Axes report pitch in um and appendage scaling factor.",
            },
            {
                "figure": "figure_benchmark_validation.png",
                "generator": "plot_benchmarks",
                "data_dependencies": "orthogonal_validation_benchmark.csv",
                "scale_information": "Grouped benchmark comparison; not image-scale dependent.",
            },
            {
                "figure": "figure_device_boundary_conditions.png",
                "generator": "plot_boundary_conditions",
                "data_dependencies": "deployment_boundary_conditions.csv",
                "scale_information": "Heatmaps of device-envelope robustness metrics; not image-scale dependent.",
            },
        ]
    ).to_csv(layout.supporting / "si_figure_manifest.csv", index=False)
    calibration_payload = {
        "fit_species": cal["fit_species"],
        "observed_targets": {name: default_species()[name].observed for name in cal["fit_species"]},
        "raw_means": cal["raw_means"],
        "raw_spread": cal["raw_spread"],
        "selected_obs_gain": cal["obs_gain"],
        "selected_obs_bias": cal["obs_bias"],
        "objective": "Minimise squared error between observed and predicted stable fractions for E. coli and S. aureus, with a weak penalty on prediction spread and large observation-gain values.",
        "calibration_surfaces": cal["calibration_surfaces"],
    }
    (layout.supporting / "si_calibration_objective.json").write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")
    benchmark_rows = []
    for species_name in ["E. coli", "S. aureus", "P. aeruginosa"]:
        subset = benchmark_summary[benchmark_summary.species == species_name].set_index("surface_family")
        benchmark_rows.append([species_name, f"{subset.loc['smooth_control', 'stable_fraction_mean']:.3f}", f"{subset.loc['sharklet_like_positive_control', 'stable_fraction_mean']:.3f}", f"{subset.loc['native_ensemble', 'stable_fraction_mean']:.3f} +/- {subset.loc['native_ensemble', 'stable_fraction_std']:.3f}"])
    boundary_rows = []
    for species_name in ["E. coli", "S. aureus", "P. aeruginosa"]:
        subset = boundary_summary[boundary_summary.species == species_name]
        boundary_rows.append([species_name, f"{100.0 * subset['suppression_vs_smooth'].min():.1f}%", f"{100.0 * subset['suppression_vs_smooth'].max():.1f}%", f"{float(subset[subset.scenario == 'combined_device_envelope']['native_stable_fraction'].iloc[0]):.3f}"])
    supporting_text = textwrap.dedent(
        f"""
        # Supporting Information Package

        This directory provides the reproducibility package requested for the submission bundle. All files are generated from `python -m snake_model.pipeline` and are intended to be submitted alongside the manuscript.

        ## Included files

        - `si_surface_priors.csv`: truncated-normal priors and hard bounds used to sample candidate surfaces.
        - `si_descriptor_acceptance.csv`: descriptor targets, tolerances, and the ensemble acceptance cutoff (`loss <= 12`).
        - `si_uq_bounds.csv`: Latin-hypercube uncertainty bounds used for the global sensitivity analysis.
        - `si_calibration_objective.json`: calibration targets, raw means/spread, and the objective used to choose `obs_gain` and `obs_bias`.
        - `si_figure_manifest.csv`: figure-generation provenance, data dependencies, and scale information.

        ## Surface-generation algorithm

        1. Generate an anisotropic hexagonal centroid lattice across a {config.domain_x:.0f} x {config.domain_y:.0f} um domain with Gaussian centroid jitter.
        2. Sample spike height, tip radius, base radius, and sharpness from the priors in `si_surface_priors.csv`.
        3. Reconstruct a 2.5D height field by taking the pointwise maximum across all spikes and adding low-amplitude correlated latent roughness.
        4. Compute ensemble descriptors (density, nearest-neighbour statistics, anisotropy, pair-correlation trace, height statistics, tip/base radii, sharpness, occupied area fraction).
        5. Accept surfaces whose descriptor loss against the Peroutka-derived reference target remains below 12.0.

        ## Calibration objective

        The observation map is calibrated only on the reported E. coli and S. aureus remaining fractions. The optimisation minimises squared error between observed and predicted stable fractions, adds a weak penalty for large prediction spread across accepted surfaces, and regularises excessively steep observation gains.

        ## Sensitivity ranges

        - Phase-diagram pitch values: {', '.join(f'{value:.1f}' for value in config.phase_pitch_values)} um.
        - Phase-diagram appendage scales: {', '.join(f'{value:.1f}' for value in config.phase_appendage_scales)}.
        - Global-sensitivity parameters: {', '.join(uq_bounds().keys())}.

        ## Benchmark summary

        {markdown_table(["Species", "Smooth", "Sharklet-like", "Snake ensemble"], benchmark_rows)}

        ## Device-envelope robustness summary

        {markdown_table(["Species", "Min suppression vs smooth", "Max suppression vs smooth", "Combined-envelope stable fraction"], boundary_rows)}
        """
    ).strip() + "\n"
    supporting_text = "\n".join(line[8:] if line.startswith("        ") else line for line in supporting_text.splitlines()) + "\n"
    (layout.supporting / "README.md").write_text(supporting_text, encoding="utf-8")


def write_manuscript_addendum(layout: OutputLayout, payload: dict, benchmark_summary: pd.DataFrame, boundary_summary: pd.DataFrame, config: Config):
    benchmark_rows = []
    for species_name in ["E. coli", "S. aureus", "P. aeruginosa"]:
        subset = benchmark_summary[benchmark_summary.species == species_name].set_index("surface_family")
        benchmark_rows.append([species_name, f"{subset.loc['smooth_control', 'stable_fraction_mean']:.3f}", f"{subset.loc['sharklet_like_positive_control', 'stable_fraction_mean']:.3f}", f"{subset.loc['native_ensemble', 'stable_fraction_mean']:.3f} +/- {subset.loc['native_ensemble', 'stable_fraction_std']:.3f}"])
    boundary_rows = []
    for species_name in ["E. coli", "S. aureus", "P. aeruginosa"]:
        subset = boundary_summary[boundary_summary.species == species_name]
        boundary_rows.append([species_name, f"{100.0 * subset['suppression_vs_smooth'].min():.1f}% to {100.0 * subset['suppression_vs_smooth'].max():.1f}%", f"{float(subset[subset.scenario == 'combined_device_envelope']['native_stable_fraction'].iloc[0]):.3f}"])
    report_lines = [
        "# Hierarchical Snake-Scale Colonisation Report",
        "",
        f"Observation gain: {payload['calibration']['obs_gain']:.3f}",
        f"Observation bias: {payload['calibration']['obs_bias']:.3f}",
        f"Calibration SSE: {payload['calibration']['sse']:.5f}",
        "",
        "## Full hierarchical means",
    ]
    for row in payload["full_species_summary"]:
        report_lines.append(f"- {row['species']}: {row['mean']:.3f} +/- {row['std']:.3f}")
    report_lines.extend([
        "",
        "## Orthogonal validation layer",
        "An external Sharklet-like micropattern benchmark (2 um feature width, 2 um spacing, 3 um relief, 4-16 um motif lengths) is now evaluated without recalibrating the observation map.",
        "",
        markdown_table(["Species", "Smooth", "Sharklet-like", "Snake ensemble"], benchmark_rows),
        "",
        "## Device-envelope robustness",
        "A protein-conditioned, low-shear, compliant silicone-device envelope is now evaluated as an application-relevant computational robustness layer.",
        "",
        markdown_table(["Species", "Suppression vs smooth across scenarios", "Combined-envelope stable fraction"], boundary_rows),
        "",
        "## Supporting information package",
        "- `supporting_information/README.md`: SI manifest and algorithm summary.",
        "- `supporting_information/si_surface_priors.csv`: sampling priors and hard bounds.",
        "- `supporting_information/si_descriptor_acceptance.csv`: descriptor weights/tolerances and acceptance threshold.",
        "- `supporting_information/si_uq_bounds.csv`: uncertainty-analysis ranges.",
        "- `supporting_information/si_calibration_objective.json`: calibration targets and optimisation details.",
        "- `supporting_information/si_figure_manifest.csv`: figure provenance and scale notes.",
        "",
        "## Figure-scale note",
        f"Representative surface panels now report x/y axes in um and include a 10 um scale bar; each panel spans {config.domain_x:.0f} x {config.domain_y:.0f} um laterally.",
    ])
    (layout.manuscript / "hierarchical_model_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    addendum_text = textwrap.dedent(
        f"""
        # Submission Strengthening Addendum

        This addendum implements the reviewer-facing strengthening items directly in the submission bundle.

        ## 1. Orthogonal validation beyond the Peroutka-derived reconstruction

        We added an external positive-control benchmark based on the Sharklet-style anti-biofilm micropattern reported for silicone medical-device surfaces (2 um feature width, 2 um spacing, 3 um relief, 4-16 um repeating motif lengths). The observation map was not recalibrated for this analysis.

        {markdown_table(["Species", "Smooth", "Sharklet-like", "Snake ensemble"], benchmark_rows)}

        This creates an orthogonal validation layer that is no longer closed within the Peroutka-derived topography family.

        ## 2. Application-relevant boundary conditions

        We added a device-envelope robustness analysis motivated by protein-conditioned, low-shear silicone-device deployment. The scenarios include protein conditioning, low-shear loading, compliant polymer contact, fabrication rounding, and their combined envelope.

        {markdown_table(["Species", "Suppression vs smooth across scenarios", "Combined-envelope stable fraction"], boundary_rows)}

        These outputs are written to `data/deployment_boundary_conditions.csv`, `data/deployment_boundary_condition_summary.csv`, and `figures/figure_device_boundary_conditions.png`.

        ## 3. Supporting information package

        The submission bundle now includes a real SI directory under `supporting_information/` with priors, descriptor tolerances, calibration details, sensitivity bounds, and figure-generation metadata.

        ## 4. Explicit scale information for image panels

        `figure_surface_ensemble.png` now includes x/y axes in um plus a 10 um scale bar. Caption language can now state explicitly that each admissible reconstructed surface panel spans {config.domain_x:.0f} x {config.domain_y:.0f} um.
        """
    ).strip() + "\n"
    (layout.manuscript / "submission_strengthening_addendum.md").write_text(addendum_text, encoding="utf-8")


def run_pipeline(config: Config, layout: OutputLayout):
    species = default_species()
    print("[1/8] Surface ensemble ...")
    surfaces, descriptor_df = generate_ensemble(config)
    descriptor_df.to_csv(layout.data / "surface_ensemble_descriptors.csv", index=False)
    print("[2/8] Calibrating observation map ...")
    cal = calibrate_obs(surfaces, species, config)
    config = replace(config, obs_gain=cal["obs_gain"], obs_bias=cal["obs_bias"])
    print("[3/8] Evaluating reduced and full models ...")
    summary_df, rep = summarise(surfaces, species, config)
    summary_df.to_csv(layout.data / "hierarchical_ensemble_summary.csv", index=False)
    print("[4/8] Mechanism ablations ...")
    ablation_df = ablations(surfaces[0], species, config)
    ablation_df.to_csv(layout.data / "hierarchical_mechanism_ablation.csv", index=False)
    print("[5/8] UQ, phase sweep, and orthogonal validation ...")
    uq_samples, uq_summary = run_uq(surfaces, species, config)
    uq_samples.to_csv(layout.data / "hierarchical_uq_samples.csv", index=False)
    uq_summary.to_csv(layout.data / "hierarchical_global_sensitivity.csv", index=False)
    phase_df = phase_diagram(species, config)
    phase_df.to_csv(layout.data / "hierarchical_phase_diagram.csv", index=False)
    benchmark_df, benchmark_summary = evaluate_benchmark_validation(surfaces, species, config)
    benchmark_df.to_csv(layout.data / "orthogonal_validation_benchmark.csv", index=False)
    benchmark_summary.to_csv(layout.data / "orthogonal_validation_summary.csv", index=False)
    print("[6/8] Device-boundary robustness ...")
    boundary_df, boundary_summary = run_boundary_condition_robustness(surfaces[0], species, config)
    boundary_df.to_csv(layout.data / "deployment_boundary_conditions.csv", index=False)
    boundary_summary.to_csv(layout.data / "deployment_boundary_condition_summary.csv", index=False)
    print("[7/8] Figures ...")
    plot_surfaces(surfaces, descriptor_df, layout, config)
    plot_ablations(ablation_df, layout)
    plot_phase(phase_df, layout)
    plot_benchmarks(benchmark_summary, layout)
    plot_boundary_conditions(boundary_summary, layout)
    print("[8/8] Summary and supporting information ...")
    payload = {
        "config": asdict(config),
        "calibration": cal,
        "full_species_summary": summary_df[summary_df.mechanism == "full_hierarchical"].groupby("species")["stable_fraction"].agg(["mean", "std"]).reset_index().to_dict(orient="records"),
        "reduced_species_summary": summary_df[summary_df.mechanism == "reduced"].groupby("species")["stable_fraction"].agg(["mean", "std"]).reset_index().to_dict(orient="records"),
        "benchmark_summary": benchmark_summary.to_dict(orient="records"),
        "boundary_summary": boundary_summary.to_dict(orient="records"),
    }
    (layout.data / "hierarchical_model_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_supporting_information(layout, config, cal, benchmark_summary, boundary_summary)
    write_manuscript_addendum(layout, payload, benchmark_summary, boundary_summary, config)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--uq-samples", type=int, default=18)
    parser.add_argument("--dx", type=float, default=DEFAULT_DX)
    args = parser.parse_args()
    config = Config(ensemble_size=args.ensemble_size, calibration_surfaces=args.ensemble_size, uq_samples=args.uq_samples, dx=args.dx)
    layout = prepare_output_layout(args.outdir)
    summary = run_pipeline(config, layout)
    print(json.dumps(summary["full_species_summary"], indent=2))
    print("Wrote outputs to", layout.base)


if __name__ == "__main__":
    main()









