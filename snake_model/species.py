"""Species-specific cell and interaction parameters for the blade-array model."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpeciesProfile:
    name: str
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


def default_species() -> dict[str, SpeciesProfile]:
    """Return the fixed, non-fitted species coefficients used in the manuscript."""
    return {
        "E. coli": SpeciesProfile(
            name="E. coli", body_radius=0.45, body_length=1.10, compliance=0.45,
            body_adhesion=1.05, adhesion_range=0.20, compliance_penalty=1.00,
            steric_penalty=1.40, slope_penalty=0.28, appendage_count=6,
            appendage_length=1.15, appendage_binding=0.18, appendage_strength=0.12,
            appendage_slope_penalty=0.12, motility_um_s=29.0, transport_sigma=1.1,
            transport_slope_penalty=1.2, transport_relief_penalty=0.9,
            detach_rate=0.22, anchor_rate=0.58, reorientation_rate=0.18,
            eps_activation_rate=0.62, maturation_rate=0.55, eps_production=0.34,
        ),
        "S. aureus": SpeciesProfile(
            name="S. aureus", body_radius=0.50, body_length=0.0, compliance=0.28,
            body_adhesion=1.15, adhesion_range=0.18, compliance_penalty=1.15,
            steric_penalty=1.10, slope_penalty=0.22, appendage_count=2,
            appendage_length=0.45, appendage_binding=0.06, appendage_strength=0.03,
            appendage_slope_penalty=0.08, motility_um_s=0.0, transport_sigma=0.8,
            transport_slope_penalty=0.7, transport_relief_penalty=0.8,
            detach_rate=0.16, anchor_rate=0.64, reorientation_rate=0.10,
            eps_activation_rate=0.88, maturation_rate=0.92, eps_production=0.48,
        ),
        "P. aeruginosa": SpeciesProfile(
            name="P. aeruginosa", body_radius=0.40, body_length=1.50, compliance=0.50,
            body_adhesion=0.98, adhesion_range=0.22, compliance_penalty=0.95,
            steric_penalty=1.35, slope_penalty=0.30, appendage_count=8,
            appendage_length=1.55, appendage_binding=0.22, appendage_strength=0.16,
            appendage_slope_penalty=0.15, motility_um_s=45.0, transport_sigma=1.3,
            transport_slope_penalty=1.35, transport_relief_penalty=0.95,
            detach_rate=0.24, anchor_rate=0.62, reorientation_rate=0.20,
            eps_activation_rate=0.60, maturation_rate=0.58, eps_production=0.33,
        ),
    }


def sigmoid(x, center: float = 0.0, scale: float = 0.08):
    arr = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-(arr - center) / max(scale, 1e-6)))


def body_masks(species: SpeciesProfile, dx: float):
    """Discrete disc or four-orientation spherocylinder footprints."""
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
    """Annular footprint reached by appendages beyond the cell body."""
    outer = species.body_radius + species.appendage_length + 0.5 * max(species.body_length, 0.0)
    n = int(np.ceil(outer / dx))
    s = np.arange(-n, n + 1) * dx
    xx, yy = np.meshgrid(s, s)
    shell = (xx * xx + yy * yy) <= outer ** 2
    inner = (xx * xx + yy * yy) <= (species.body_radius + 0.15 * max(species.body_length, 0.0)) ** 2
    return shell & (~inner)
