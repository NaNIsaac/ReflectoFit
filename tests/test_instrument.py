"""Instrument effects, each against its closed form or a brute-force integral (DESIGN.md section 2)."""

import math

import numpy as np
import pytest
from scipy.special import erf

from reflectofit import Instrument, Layer, Stack, forward, q_to_tth, tth_to_q

STACK = Stack(
    [Layer(45 + 1.5j, 240, 6), Layer(18.86 + 0.24j, 10, 3)], Layer(20.07 + 0.46j, roughness=3)
)
Q = tth_to_q(np.linspace(0.1, 6.0, 300))


def test_no_instrument_effects_is_the_bare_curve():
    np.testing.assert_allclose(forward(Q, STACK, Instrument()), forward(Q, STACK), rtol=1e-10)


def test_i0_and_background_are_linear():
    bare = forward(Q, STACK)
    np.testing.assert_allclose(
        forward(Q, STACK, Instrument(i0=2.5, background=1e-7)), 2.5 * bare + 1e-7, rtol=1e-10
    )


def test_tth_offset_is_added_to_the_angle():
    shifted = tth_to_q(q_to_tth(Q) + 0.02)
    np.testing.assert_allclose(
        forward(Q, STACK, Instrument(tth_offset=0.02)), forward(shifted, STACK), rtol=1e-10
    )


def test_kalpha2_is_the_same_angle_at_the_second_wavelength():
    inst = Instrument(kalpha2_ratio=0.5)
    q2 = Q * inst.wavelength / inst.kalpha2_wavelength
    expected = (forward(Q, STACK) + 0.5 * forward(q2, STACK)) / 1.5
    np.testing.assert_allclose(forward(Q, STACK, inst), expected, rtol=1e-10)


def test_footprint_is_the_centred_gaussian_beam_factor():
    inst = Instrument(sample_length=10.0, beam_sigma=0.05)
    sin_t = Q * inst.wavelength / (4 * math.pi)
    expected = forward(Q, STACK) * erf(10.0 * sin_t / (2 * math.sqrt(2) * 0.05))
    np.testing.assert_allclose(forward(Q, STACK, inst), expected, rtol=1e-10)


@pytest.mark.parametrize(("sigma_q", "dq_over_q"), [(1e-3, 0.0), (0.0, 0.02), (5e-4, 0.01)])
def test_resolution_matches_a_brute_force_convolution(sigma_q, dq_over_q):
    q = tth_to_q(np.linspace(0.6, 4.0, 40))
    got = forward(q, STACK, Instrument(sigma_q=sigma_q, dq_over_q=dq_over_q))
    sigma = np.sqrt(sigma_q**2 + (dq_over_q * q) ** 2)
    u = np.linspace(-8, 8, 4001)
    nodes = q[:, None] + sigma[:, None] * u[None, :]
    weights = np.exp(-0.5 * u**2)
    brute = (forward(nodes.ravel(), STACK).reshape(nodes.shape) * weights).sum(1) / weights.sum()
    np.testing.assert_allclose(got, brute, rtol=1e-6)


def test_per_curve_instrument_values():
    stacks = [STACK, STACK]
    batch = forward(
        Q, stacks, Instrument(i0=np.array([1.0, 2.0]), tth_offset=np.array([0.0, 0.01]))
    )
    np.testing.assert_allclose(batch[0], forward(Q, STACK), rtol=1e-10)
    np.testing.assert_allclose(
        batch[1], forward(Q, STACK, Instrument(i0=2.0, tth_offset=0.01)), rtol=1e-10
    )
