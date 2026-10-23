"""The physics reviews of ReflectoFit#2: resolution with the default rule (thick films, wide
resolution, weakly absorbing and real-SLD edges, waveguides), absorbing ambient, gradient at zero
resolution."""

import numpy as np
import pytest

from reflectofit import Instrument, Layer, Stack, forward, resolution, tth_to_q

SI = 20.07 + 0.46j
NI = 64 + 1.35j
Q = np.concatenate([np.linspace(0.015, 0.07, 200), tth_to_q(np.linspace(1.0, 6.0, 150))])


def _brute(q, stack, sigma, n=24001):
    """Gaussian convolution in q on n points over ±9 σ (σ may vary with q)."""
    u = np.linspace(-9, 9, n)
    w = np.exp(-0.5 * u**2)
    nodes = np.clip(q[:, None] + np.asarray(sigma)[..., None] * u[None, :], 1e-12, None)
    return (forward(nodes.ravel(), stack).reshape(nodes.shape) * w).sum(1) / w.sum()


CASES = {
    # the first review's table: GH17 was off by 1.4e-3 up to 0.97 for thick absorbing films
    "45+1.5i 2000 A, sigma 1e-3": (Stack([Layer(45 + 1.5j, 2000, 4)], Layer(SI)), 1e-3, 0.0),
    "45+1.5i 4000 A, sigma 2e-3": (Stack([Layer(45 + 1.5j, 4000, 4)], Layer(SI)), 2e-3, 0.0),
    "45+1.5i 1000 A, dq/q 0.02": (Stack([Layer(45 + 1.5j, 1000, 4)], Layer(SI)), 0.0, 0.02),
    # the recheck: weakly absorbing films and a real SLD were off by 1e-2 at their edges
    "PS-like 9.6+0.012i 1000 A": (
        Stack([Layer(9.6 + 0.012j, 1000, 4)], Layer(SI, roughness=3)),
        5e-4,
        0.0,
    ),
    "45+0.05i 1000 A": (Stack([Layer(45 + 0.05j, 1000, 4)], Layer(SI, roughness=3)), 5e-4, 0.0),
    "bare Si, real SLD": (Stack([], Layer(20.07)), 5e-4, 0.0),
    "PS-like, sigma 2e-3": (
        Stack([Layer(9.6 + 0.012j, 1000, 4)], Layer(SI, roughness=3)),
        2e-3,
        0.0,
    ),
    "25-period superlattice": (
        Stack([Layer(30 + 0.5j, 25, 3), Layer(55 + 2j, 35, 3.5)] * 25, Layer(SI, roughness=3)),
        5e-4,
        0.0,
    ),
    # the final review (ReflectoFit#4): resonances between the edges are narrower than one edge
    # width, and a real-SLD substrate sets the narrowest width to EDGE_FLOOR
    "Ni/C/Ni waveguide, sigma 1e-4": (
        Stack([Layer(NI, 100, 2), Layer(18 + 0.02j, 800, 2)], Layer(NI, roughness=2)),
        1e-4,
        0.0,
    ),
    "PS-like on real Si, sigma 1e-4": (
        Stack([Layer(9.6 + 0.012j, 1000, 3)], Layer(20.07, roughness=3)),
        1e-4,
        0.0,
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_default_resolution_rule_converges(name):
    stack, sigma_q, dq_over_q = CASES[name]
    got = forward(Q, stack, Instrument(sigma_q=sigma_q, dq_over_q=dq_over_q))
    sigma = np.sqrt(sigma_q**2 + (dq_over_q * Q) ** 2)
    np.testing.assert_allclose(got, _brute(Q, stack, sigma), rtol=1e-5)


@pytest.mark.parametrize(
    ("name", "constant", "loose"),
    [
        ("Ni/C/Ni waveguide, sigma 1e-4", "RESONANCE", 2.0),
        ("PS-like on real Si, sigma 1e-4", "EDGE_FLOOR", 1e-5),
    ],
)
def test_the_waveguide_cases_fail_with_a_looser_constant(monkeypatch, name, constant, loose):
    stack, sigma_q, _ = CASES[name]
    monkeypatch.setattr(resolution, constant, loose)
    got = forward(Q, stack, Instrument(sigma_q=sigma_q))
    assert np.abs(got / _brute(Q, stack, sigma_q) - 1).max() > 1e-5


def test_absorbing_ambient_is_rejected():
    with pytest.raises(ValueError, match="ambient must not absorb"):
        forward(Q, Stack([Layer(45.0, 300, 3)], Layer(20.07), ambient=0.5j))


def test_gradient_is_finite_for_a_curve_without_resolution_in_a_mixed_batch():
    torch = pytest.importorskip("torch")
    stack = Stack([Layer(45 + 1.5j, 240, 6)], Layer(SI, roughness=3))
    packed = tuple(torch.tensor(np.stack([a, a])) for a in stack.packed())
    sigma = torch.tensor([0.0, 6e-4], dtype=torch.float64, requires_grad=True)
    r = forward(torch.tensor(Q), packed, Instrument(sigma_q=sigma, nodes=27), backend="torch")
    (g,) = torch.autograd.grad(r.sum(), sigma)
    assert torch.isfinite(g).all()
    assert g[0] == 0


def test_a_curve_does_not_depend_on_the_rest_of_its_batch():
    stacks = [CASES["PS-like 9.6+0.012i 1000 A"][0], Stack([Layer(45 + 1.5j, 3000, 4)], Layer(SI))]
    batch = forward(Q, stacks, Instrument(sigma_q=np.array([5e-4, 2e-3])))
    alone = forward(Q, stacks[0], Instrument(sigma_q=5e-4))
    np.testing.assert_array_equal(batch[0], alone)
