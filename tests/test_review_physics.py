"""The physics review of ReflectoFit#2: resolution convergence with the default nodes, absorbing
ambient, and the gradient at zero resolution."""

import numpy as np
import pytest

from reflectofit import Instrument, Layer, Stack, forward, tth_to_q
from reflectofit.forward import resolution_nodes

Q = tth_to_q(np.linspace(0.3, 6.0, 240))


def _brute(q, stack, sigma):
    """Gaussian convolution in q on 6001 points over ±9 σ (σ may vary with q)."""
    u = np.linspace(-9, 9, 6001)
    w = np.exp(-0.5 * u**2)
    nodes = np.clip(q[:, None] + np.asarray(sigma)[..., None] * u[None, :], 1e-12, None)
    return (forward(nodes.ravel(), stack).reshape(nodes.shape) * w).sum(1) / w.sum()


@pytest.mark.parametrize(
    ("thickness", "sigma_q", "dq_over_q"),
    [(2000, 1e-3, 0.0), (4000, 2e-3, 0.0), (500, 2e-3, 0.0), (1000, 0.0, 0.02)],
)
def test_default_nodes_converge_for_thick_films_and_wide_resolution(thickness, sigma_q, dq_over_q):
    """The review's table: GH17 was off by 1.4e-3 (2000 Å, 1e-3) up to 0.97 (4000 Å, 2e-3); the
    largest errors sat at the film's critical edge, which this grid includes (q 0.048-0.057)."""
    stack = Stack([Layer(45 + 1.5j, thickness, 4)], Layer(20.07 + 0.46j, roughness=0))
    got = forward(Q, stack, Instrument(sigma_q=sigma_q, dq_over_q=dq_over_q))
    sigma = np.sqrt(sigma_q**2 + (dq_over_q * Q) ** 2)
    np.testing.assert_allclose(got, _brute(Q, stack, sigma), rtol=1e-6)


def test_node_count_follows_resolution_and_thickness():
    assert resolution_nodes(0.0, 1000) == 1
    assert resolution_nodes(5e-4, 240) == 27
    assert resolution_nodes(2e-3, 500) == 89  # the kink spacing NODE_DQ / σ = 0.15 decides
    assert resolution_nodes(1e-3, 500) == 45  # kink: h = 0.3
    assert resolution_nodes(1e-3, 8000) == 51  # fringes: h = 2π / 24
    with pytest.warns(UserWarning, match="capped"):
        assert resolution_nodes(0.05, 5000) == 401


def test_absorbing_ambient_is_rejected():
    with pytest.raises(ValueError, match="ambient must not absorb"):
        forward(Q, Stack([Layer(45.0, 300, 3)], Layer(20.07), ambient=0.5j))


def test_gradient_is_finite_for_a_curve_without_resolution_in_a_mixed_batch():
    torch = pytest.importorskip("torch")
    stack = Stack([Layer(45 + 1.5j, 240, 6)], Layer(20.07 + 0.46j, roughness=3))
    packed = tuple(torch.tensor(np.stack([a, a])) for a in stack.packed())
    sigma = torch.tensor([0.0, 6e-4], dtype=torch.float64, requires_grad=True)
    r = forward(torch.tensor(Q), packed, Instrument(sigma_q=sigma, nodes=27), backend="torch")
    (g,) = torch.autograd.grad(r.sum(), sigma)
    assert torch.isfinite(g).all()
    assert g[0] == 0
