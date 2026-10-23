"""The torch backend equals the numba path and is differentiable."""

import numpy as np
import pytest

from reflectofit import Instrument, Layer, Stack, forward, tth_to_q

torch = pytest.importorskip("torch")

STACK = Stack(
    [Layer(45 + 1.5j, 240, 6), Layer(18.86 + 0.24j, 10, 3)], Layer(20.07 + 0.46j, roughness=3)
)
Q = tth_to_q(np.linspace(0.05, 6.0, 300))
FULL = Instrument(
    i0=1.3,
    background=2e-8,
    tth_offset=0.01,
    sigma_q=4e-4,
    dq_over_q=0.01,
    kalpha2_ratio=0.5,
    sample_length=12.0,
    beam_sigma=0.04,
)


@pytest.mark.parametrize("inst", [None, FULL], ids=["bare", "full"])
def test_torch_equals_numba(inst):
    got = forward(Q, STACK, inst, backend="torch")
    assert got.dtype == torch.float64
    np.testing.assert_allclose(got.numpy(), forward(Q, STACK, inst), rtol=1e-10)


def test_chunked_batch_equals_one_call():
    from reflectofit.torch_backend import forward as torch_forward

    stacks = [
        Stack([Layer(45 + 1j, d, 5)], Layer(20.07 + 0.46j, roughness=3))
        for d in (100, 150, 200, 250, 300)
    ]
    inst = Instrument(i0=np.array([1.0, 1.1, 1.2, 1.3, 1.4]), sigma_q=5e-4)
    np.testing.assert_allclose(
        torch_forward(Q, stacks, inst, chunk=2).numpy(),
        torch_forward(Q, stacks, inst).numpy(),
        rtol=1e-14,
    )


def test_gradients_match_finite_differences():
    sld, thickness, roughness = (torch.tensor(a) for a in STACK.packed())
    thickness.requires_grad_(True)
    offset = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(Q)

    def f(t, off):
        inst = FULL.replace(tth_offset=off)
        return forward(q, (sld, t, roughness), inst, backend="torch")

    assert torch.autograd.gradcheck(f, (thickness, offset), eps=1e-6, atol=1e-7, rtol=1e-4)
