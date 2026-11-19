import numpy as np

from snake_model.model import (
    BladeGeometry,
    FlatCache,
    ModelConfig,
    SPECIES,
    build_blade_surface,
    build_flat_surface,
    build_sharklet_af,
    ratio,
)


def _small_config():
    return ModelConfig(
        dx=0.30,
        domain_x=12.0,
        domain_y=12.0,
        horizon=0.5,
        h_state=0.05,
        record_times=(0.5,),
    )


def test_species_profiles_are_complete_and_fixed():
    assert set(SPECIES) == {"E. coli", "S. aureus", "P. aeruginosa"}
    for species in SPECIES.values():
        assert species.body_radius > 0
        assert species.appendage_count >= 0
        assert species.detach_rate > 0
        assert species.anchor_rate > 0


def test_blade_and_sharklet_surfaces_are_finite():
    cfg = _small_config()
    blade, info = build_blade_surface(BladeGeometry(), cfg)
    sharklet = build_sharklet_af(cfg)
    assert blade.ndim == 2 and blade.size > 0
    assert sharklet.ndim == 2 and sharklet.size > 0
    assert np.isfinite(blade).all()
    assert np.isfinite(sharklet).all()
    assert info["n_blades"] > 0
    assert info["lx"] > 0 and info["ly"] > 0


def test_flat_reference_ratio_is_one_for_each_species():
    cfg = _small_config()
    flat = build_flat_surface(cfg)
    for species in SPECIES.values():
        result = ratio(flat, species, cfg, FlatCache(), times=(0.5,))
        assert np.isfinite(result["Y"])
        assert abs(result["Y"] - 1.0) < 1e-10
        assert abs(result["Y_t0.5"] - 1.0) < 1e-10


def test_blade_array_ratio_is_finite_and_capacity_limited():
    cfg = _small_config()
    blade, _ = build_blade_surface(BladeGeometry(), cfg)
    cache = FlatCache()
    for species in SPECIES.values():
        result = ratio(blade, species, cfg, cache, times=(0.5,))
        assert np.isfinite(result["Y"])
        assert result["Y"] > 0
        assert np.isfinite(result["occ_max"])
        assert 0 <= result["occ_max"] < 1
