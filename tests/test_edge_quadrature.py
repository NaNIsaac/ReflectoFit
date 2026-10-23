"""The resolution integral at critical edges with real SLDs (ReflectoFit#6): default rule against a
brute-force convolution within 1e-4, a bounded node count, and the torch backend on the same nodes.
The full confirmation grid is docs/design-bench/x1e/confirm.py."""

import numpy as np
import pytest

from reflectofit import Instrument, Layer, Stack, forward, resolution

Q = np.linspace(0.01, 0.12, 300)


def _brute(q, stack, sigma, n=48001):
    u = np.linspace(-9, 9, n)
    w = np.exp(-0.5 * u**2)
    w /= w.sum()
    out = np.empty(q.size)
    for i in range(0, q.size, 25):
        nodes = np.clip(q[i : i + 25, None] + sigma * u, 1e-9, None)
        out[i : i + 25] = (forward(nodes.ravel(), stack).reshape(nodes.shape) * w).sum(1)
    return out


CASES = {
    "bare Si, real": Stack([], Layer(20.07)),
    "dense film on real Si": Stack([Layer(60.0, 800, 5)], Layer(20.0, roughness=3)),
    "real waveguide": Stack([Layer(25.0, 30, 3), Layer(5.0, 300, 3)], Layer(20.0, roughness=3)),
}


@pytest.mark.parametrize("sigma_q", [1e-4, 2e-3])
@pytest.mark.parametrize("name", CASES)
def test_real_sld_edges_within_1e_4(name, sigma_q):
    stack = CASES[name]
    got = forward(Q, stack, Instrument(sigma_q=sigma_q))
    np.testing.assert_allclose(got, _brute(Q, stack, sigma_q), rtol=1e-4)


def test_node_count_is_bounded_on_real_slds():
    stack = CASES["dense film on real Si"]
    e = resolution.edges(*stack.packed()[:2], kalpha2_scale=1.0025)
    counts, _, _ = resolution.layout(Q, np.full(Q.size, 1e-4), [e])
    assert counts.max() <= 16 * resolution.PANEL  # was up to 8001 (ReflectoFit#6)


def test_torch_backend_on_the_same_nodes():
    torch = pytest.importorskip("torch")
    stack = CASES["dense film on real Si"]
    inst = Instrument(sigma_q=5e-4, kalpha2_ratio=0.4, sample_length=10.0, beam_sigma=0.05)
    ref = forward(Q, stack, inst)
    got = forward(torch.as_tensor(Q), stack, inst, backend="torch").numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-10)
