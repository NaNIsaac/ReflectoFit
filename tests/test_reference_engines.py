"""The kernel against independent engines at 1e-10 relative (BACKLOG X1b, DESIGN.md section 5.1).

References: the numpy Parratt specification, refnx (its own C kernel), ReflectoLearn's Abeles kernel
(b6fb0c4, copied verbatim) run in complex128, and a JAX Parratt. A difference above 1e-10 is not
loosened here: it goes to the physics lane for a verdict.
"""

import math

import numpy as np
import pytest

from reflectofit import Layer, Stack, forward, tth_to_q
from tests.reference.parratt import parratt

SI = 20.07 + 0.46j
SIO2 = 18.9 + 0.24j
TOL = 1e-10


def _structures() -> dict[str, Stack]:
    sl = []
    for _ in range(25):
        sl += [Layer(30 + 0.5j, 25, 3), Layer(55 + 2j, 35, 3.5)]
    return {
        "single": Stack([Layer(45 + 1.5j, 240, 6)], Layer(SI, roughness=3)),
        "oxide_under": Stack(
            [Layer(45 + 1.5j, 240, 6), Layer(SIO2, 10, 3)], Layer(SI, roughness=3)
        ),
        "bilayer": Stack(
            [Layer(30 + 0.5j, 120, 4), Layer(55 + 2j, 200, 5), Layer(SIO2, 10, 3)],
            Layer(SI, roughness=3),
        ),
        "five_layer": Stack(
            [
                Layer(40 + 1j, 30, 4),
                Layer(55 + 2j, 80, 5),
                Layer(30 + 0.5j, 120, 4),
                Layer(45 + 1.5j, 60, 3),
                Layer(SIO2, 15, 3),
            ],
            Layer(SI, roughness=3),
        ),
        "superlattice50": Stack(sl, Layer(SI, roughness=3)),
    }


STRUCTURES = _structures()
Q = tth_to_q(np.linspace(0.05, 8.0, 2000))


def _rel(a, b):
    return float(np.max(np.abs(a - b) / np.abs(b)))


@pytest.mark.parametrize("name", STRUCTURES)
def test_matches_numpy_specification(name):
    stack = STRUCTURES[name]
    assert _rel(forward(Q, stack), parratt(Q, *stack.packed())) < TOL


@pytest.mark.parametrize("name", STRUCTURES)
def test_matches_refnx(name):
    reflect = pytest.importorskip("refnx.reflect")
    sld, thickness, roughness = STRUCTURES[name].packed()
    w = np.column_stack([thickness, sld.real, sld.imag, roughness])
    assert _rel(forward(Q, STRUCTURES[name]), reflect.reflectivity(Q, w, dq=0.0)) < TOL


@pytest.mark.parametrize("name", STRUCTURES)
def test_matches_reflectolearn_kernel_in_complex128(name):
    torch = pytest.importorskip("torch")
    from tests.reference.reflectolearn_abeles import abeles_core

    sld, thickness, roughness = STRUCTURES[name].packed()
    # ReflectoLearn's Abeles form takes Im SLD > 0 as is (exp(+i k d) with Im k <= 0).
    rho = torch.tensor((sld - sld[0]) * 1e-6, dtype=torch.complex128)
    ref = abeles_core(
        torch.tensor(Q, dtype=torch.complex128)[None, :, None],
        torch.tensor(thickness[:-1])[None, None],
        torch.tensor(roughness[1:])[None, None],
        rho[None, None],
    )[0].numpy()
    assert _rel(forward(Q, STRUCTURES[name]), ref) < TOL


@pytest.mark.parametrize("name", ["single", "bilayer"])
def test_matches_jax(name):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    sld, thickness, roughness = STRUCTURES[name].packed()
    rho = jnp.asarray(np.conj(sld - sld[0]) * 1e-6)
    k = jnp.sqrt((jnp.asarray(Q)[:, None] / 2) ** 2 - 4 * math.pi * rho[None, :] + 0j)
    x = jnp.zeros(len(Q), complex)
    n = len(sld)
    for j in range(n - 2, -1, -1):
        k1, k2 = k[:, j], k[:, j + 1]
        r = (k1 - k2) / (k1 + k2) * jnp.exp(-2 * k1 * k2 * roughness[j + 1] ** 2)
        p = jnp.exp(2j * k2 * (thickness[j + 1] if j + 1 < n - 1 else 0.0))
        x = (r + x * p) / (1 + r * x * p)
    assert _rel(forward(Q, STRUCTURES[name]), np.asarray(jnp.abs(x) ** 2)) < TOL


@pytest.mark.parametrize("im", [1e-12, 1e-6, 0.46])
def test_square_root_branches(im):
    """float64 accuracy near Re k² = 0 at small absorption, on the plateau and in the tail. The
    closed form uses the same conjugate and principal root, and |r| of a bare substrate does not
    depend on the sign convention, so this does not test the branch or the absorption sign; those
    are pinned by the refnx and ReflectoLearn comparisons with absorbing layers above."""
    sub = 20.07 + im * 1j
    q = np.concatenate([np.linspace(0.005, 0.03, 50), np.linspace(0.04, 1.0, 50)])
    k0 = q / 2
    k1 = np.sqrt(k0**2 - 4 * math.pi * np.conj(sub) * 1e-6 + 0j)
    closed = np.abs((k0 - k1) / (k0 + k1)) ** 2
    assert _rel(forward(q, Stack([], Layer(sub))), closed) < TOL
