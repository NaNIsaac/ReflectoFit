"""The conventions of DESIGN.md section 2, pinned before the code that follows them.

Absorption sign: Im SLD > 0 is absorption. Layer order: ambient first, substrate last; a layer's
roughness is that of the interface above it. Units: Å, 1e-6 Å^-2, Å^-1, degrees.
"""

import math

import numpy as np
import pytest

from reflectofit import Layer, Stack, forward, q_to_tth, tth_to_q

SI = 20.07  # 1e-6 Å^-2, real part of Si at Cu Kα (DECISIONS #8)


def q_c(sld: float) -> float:
    """Critical wavevector transfer for an SLD step from vacuum: q_c = 4 sqrt(pi rho)."""
    return 4 * math.sqrt(math.pi * sld * 1e-6)


# --- units -------------------------------------------------------------------------------------


def test_tth_to_q_uses_degrees_and_angstrom():
    assert tth_to_q(1.0, 1.5406) == pytest.approx(
        4 * math.pi / 1.5406 * math.sin(math.radians(0.5))
    )
    assert q_to_tth(tth_to_q(2.5, 1.5406), 1.5406) == pytest.approx(2.5)


def test_default_wavelength_is_cu_kalpha1():
    assert tth_to_q(1.0) == pytest.approx(tth_to_q(1.0, 1.5406))


def test_sld_unit_sets_the_critical_edge():
    """SLD in 1e-6 Å^-2: total reflection below q_c = 4 sqrt(pi rho), a sharp drop above it."""
    qc = q_c(SI)
    q = np.array([0.9 * qc, 0.98 * qc, 1.05 * qc])
    r = forward(q, Stack(layers=[], substrate=Layer(SI)))
    assert r[0] == pytest.approx(1.0, abs=1e-12)
    assert r[1] == pytest.approx(1.0, abs=1e-12)
    assert r[2] < 0.5


def test_fresnel_tail_scales_as_q_minus_4():
    """Far above the edge R -> a^2 (1 + a) / 16 with a = (q_c / q)^2, for a sharp substrate."""
    q = np.array([0.4, 0.8, 1.2])
    a = (q_c(SI) / q) ** 2
    r = forward(q, Stack(layers=[], substrate=Layer(SI)))
    np.testing.assert_allclose(r, a**2 * (1 + a) / 16, rtol=3 * a.max() ** 2)


def test_thickness_unit_sets_the_kiessig_period():
    """Thickness in Å: above both edges the fringes repeat every 2 pi / d in the film's own
    wavevector transfer q_film = sqrt(q^2 - q_c,film^2) (real Fresnel coefficients, no roughness)."""
    d = 300.0
    q = np.linspace(0.12, 0.40, 20000)
    r = forward(q, Stack(layers=[Layer(45.0, d)], substrate=Layer(SI)))
    q_film = np.sqrt(q**2 - q_c(45.0) ** 2)
    minima = q_film[1:-1][(r[1:-1] < r[:-2]) & (r[1:-1] < r[2:])]
    assert np.mean(np.diff(minima)) == pytest.approx(2 * math.pi / d, rel=2e-3)


# --- absorption sign ---------------------------------------------------------------------------


def test_absorbing_substrate_reflects_less_than_one_below_the_edge():
    qc = q_c(SI)
    q = np.linspace(0.2 * qc, 0.95 * qc, 50)
    r = forward(q, Stack(layers=[], substrate=Layer(SI + 0.46j)))
    assert np.all(r < 1.0)
    assert np.all(r > 0.5)


def test_absorption_never_creates_reflectivity_above_one():
    q = np.linspace(0.005, 0.6, 2000)
    stack = Stack(
        layers=[Layer(45 + 1.5j, 240, 6), Layer(18.86 + 0.24j, 10, 3)],
        substrate=Layer(SI + 0.46j, roughness=3),
    )
    r = forward(q, stack)
    assert np.all(r <= 1.0 + 1e-12)


def test_absorbing_film_damps_its_fringes():
    """A thick absorbing film reflects less than the same film without absorption."""
    q = np.linspace(0.05, 0.3, 500)
    clear = forward(q, Stack(layers=[Layer(45.0, 2000)], substrate=Layer(SI)))
    lossy = forward(q, Stack(layers=[Layer(45 + 5j, 2000)], substrate=Layer(SI)))
    clear_swing = np.ptp(np.log10(clear[200:300]))
    lossy_swing = np.ptp(np.log10(lossy[200:300]))
    assert lossy_swing < clear_swing


def test_gain_is_rejected():
    with pytest.raises(ValueError, match="imaginary"):
        forward(np.array([0.1]), Stack(layers=[Layer(45 - 1j, 100)], substrate=Layer(SI)))


# --- layer order -------------------------------------------------------------------------------


def test_first_layer_is_on_top():
    """A layer matching the ambient is invisible on top and not invisible at the bottom."""
    q = np.linspace(0.02, 0.4, 400)
    film = Layer(45.0, 200, 0)
    vacuum = Layer(0.0, 80, 0)
    bare = forward(q, Stack(layers=[film], substrate=Layer(SI)))
    on_top = forward(q, Stack(layers=[vacuum, film], substrate=Layer(SI)))
    at_bottom = forward(q, Stack(layers=[film, vacuum], substrate=Layer(SI)))
    np.testing.assert_allclose(on_top, bare, rtol=1e-12)
    assert np.max(np.abs(at_bottom / bare - 1)) > 0.1


def test_last_layer_is_on_the_substrate():
    """A layer matching the substrate, with a sharp interface below it, merges into the substrate."""
    q = np.linspace(0.02, 0.4, 400)
    film = Layer(45.0, 200, 4)
    bare = forward(q, Stack(layers=[film], substrate=Layer(SI, roughness=0)))
    merged = forward(q, Stack(layers=[film, Layer(SI, 50, 0)], substrate=Layer(SI, roughness=0)))
    np.testing.assert_allclose(merged, bare, rtol=1e-12)


def test_reversing_two_layers_changes_the_curve():
    q = np.linspace(0.02, 0.4, 400)
    a, b = Layer(60.0, 50, 3), Layer(20.0, 200, 3)
    r_ab = forward(q, Stack(layers=[a, b], substrate=Layer(SI)))
    r_ba = forward(q, Stack(layers=[b, a], substrate=Layer(SI)))
    assert np.max(np.abs(np.log10(r_ab / r_ba))) > 0.1


def test_roughness_belongs_to_the_interface_above():
    """Substrate roughness alone gives the Névot-Croce factor |exp(-2 k0 k1 sigma^2)|^2."""
    sigma = 5.0
    q = np.linspace(0.05, 0.4, 100)
    sharp = forward(q, Stack(layers=[], substrate=Layer(SI)))
    rough = forward(q, Stack(layers=[], substrate=Layer(SI, roughness=sigma)))
    k0 = q / 2
    k1 = np.sqrt(k0**2 - 4 * math.pi * SI * 1e-6 + 0j)
    np.testing.assert_allclose(
        rough / sharp, np.abs(np.exp(-2 * k0 * k1 * sigma**2)) ** 2, rtol=1e-10
    )


def test_film_roughness_is_its_top_surface():
    """Film roughness damps the film/ambient interface, which carries most of the contrast here."""
    q = np.linspace(0.1, 0.4, 200)
    smooth = forward(q, Stack(layers=[Layer(45.0, 200, 0)], substrate=Layer(SI, roughness=0)))
    rough_top = forward(q, Stack(layers=[Layer(45.0, 200, 8)], substrate=Layer(SI, roughness=0)))
    rough_bottom = forward(q, Stack(layers=[Layer(45.0, 200, 0)], substrate=Layer(SI, roughness=8)))
    # ambient/film contrast 45 is about 2x the film/substrate contrast 25, so damping it lowers R more
    assert (
        np.mean(np.log10(rough_top)) < np.mean(np.log10(rough_bottom)) < np.mean(np.log10(smooth))
    )


# --- shapes ------------------------------------------------------------------------------------


def test_batch_rows_equal_single_calls():
    q = np.linspace(0.01, 0.5, 300)
    stacks = [
        Stack(layers=[Layer(45.0, d, 5)], substrate=Layer(SI + 0.46j, roughness=3))
        for d in (100, 250)
    ]
    batch = forward(q, [s.packed() for s in stacks])
    for row, s in zip(batch, stacks, strict=True):
        np.testing.assert_allclose(row, forward(q, s), rtol=1e-14)


def test_packed_order_is_ambient_first():
    sld, thickness, roughness = Stack(
        layers=[Layer(45 + 1j, 200, 5)], substrate=Layer(SI, roughness=3)
    ).packed()
    np.testing.assert_array_equal(sld, [0, 45 + 1j, SI])
    np.testing.assert_array_equal(thickness, [0, 200, 0])
    np.testing.assert_array_equal(roughness, [0, 5, 3])
