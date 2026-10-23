"""The six points of the xrr-policy review on ReflectoFit#2, each pinned by a test."""

import math

import numpy as np
import pytest

from reflectofit import Instrument, Layer, Stack, forward, q_to_tth, tth_to_q
from reflectofit.data import Curve
from reflectofit.estimate import critical_q, kiessig_thickness
from reflectofit.fit import lsq
from reflectofit.metrics import residuals
from reflectofit.model import Model

STACK = Stack([Layer(45 + 1.5j, 240, 6, "film")], Layer(20.07 + 0.46j, roughness=3))
Q = tth_to_q(np.linspace(0.1, 6.0, 300))


def test_1_kiessig_ignores_zero_counts():
    """At 1e6 counts the tail has zeros; they must not enter the transform as log10(tiny)."""
    r = forward(Q, STACK, Instrument(background=1e-9))
    y = np.random.default_rng(0).poisson(1e6 * r).astype(float)
    assert (y == 0).sum() > 10
    assert kiessig_thickness(Q, y, sld=45.0) == pytest.approx(240, abs=3)
    keep = Q < 0.35
    assert kiessig_thickness(Q, y, sld=45.0, mask=keep) == pytest.approx(240, abs=3)


def test_2_jacfwd_through_the_torch_forward():
    """torch.func.jacfwd over thickness, SLD, sigma_q and offset, against central differences."""
    torch = pytest.importorskip("torch")
    sld0, d0, s0 = (torch.tensor(a) for a in STACK.packed())
    q = torch.tensor(Q)
    base = Instrument(sample_length=10.0, beam_sigma=0.04, kalpha2_ratio=0.5, dq_over_q=0.005)

    def f(theta):
        d = torch.cat([d0[:1], theta[:1], d0[2:]])
        sld = torch.complex(torch.cat([sld0.real[:1], theta[1:2], sld0.real[2:]]), sld0.imag)
        inst = base.replace(sigma_q=theta[2], tth_offset=theta[3])
        return forward(q, (sld, d, s0), inst, backend="torch")

    theta = torch.tensor([240.0, 45.0, 6e-4, 0.01], dtype=torch.float64)
    jac = torch.func.jacfwd(f)(theta).numpy()
    h = np.array([1e-4, 1e-5, 1e-8, 1e-7])
    for j in range(4):
        e = torch.zeros(4, dtype=torch.float64)
        e[j] = h[j]
        fd = (f(theta + e) - f(theta - e)).numpy() / (2 * h[j])
        np.testing.assert_allclose(jac[:, j], fd, rtol=1e-4, atol=1e-6 * np.abs(fd).max())


def test_3_offset_comes_before_the_resolution():
    """The resolution is a Gaussian in q around q(2θ + offset), as DESIGN.md section 2 now says."""
    offset, sigma = 0.03, 2e-3
    inst = Instrument(tth_offset=offset, sigma_q=sigma, nodes=41)
    q_true = tth_to_q(q_to_tth(Q) + offset)
    u = np.linspace(-8, 8, 4001)
    w = np.exp(-0.5 * u**2)
    nodes = np.clip(q_true[:, None] + sigma * u[None, :], 1e-12, None)
    brute = (forward(nodes.ravel(), STACK).reshape(nodes.shape) * w).sum(1) / w.sum()
    np.testing.assert_allclose(forward(Q, STACK, inst), brute, rtol=1e-6)


def test_4_poisson_residual_gradient_is_finite_at_lam_equal_y():
    torch = pytest.importorskip("torch")
    y = torch.tensor([0.0, 7.0, 100.0], dtype=torch.float64)
    lam = torch.tensor([3.0, 7.0, 100.0], dtype=torch.float64, requires_grad=True)
    r = residuals("poisson", lam, y)
    (g,) = torch.autograd.grad(r.sum(), lam)
    assert torch.isfinite(g).all()
    np.testing.assert_allclose(g[1:].numpy(), 1 / np.sqrt([7.0, 100.0]), rtol=1e-6)
    # away from lam = y it is the plain form
    lam_np, y_np = np.array([3.0, 9.0]), np.array([0.0, 7.0])
    d = 2 * (
        lam_np - y_np + np.where(y_np > 0, y_np * np.log(np.where(y_np > 0, y_np, 1) / lam_np), 0)
    )
    np.testing.assert_allclose(
        residuals("poisson", lam_np, y_np), np.sign(lam_np - y_np) * np.sqrt(d)
    )


def test_5_half_plateau_rule():
    """The half-plateau rule lands where the smoothed curve crosses half the plateau."""
    q = np.linspace(0.005, 0.1, 400)
    y = 1e6 * forward(q, Stack([], Layer(20.07 + 0.46j, roughness=3)))
    qc = critical_q(q, y, rule="half_plateau")
    assert np.interp(qc, q, y) == pytest.approx(0.5 * y[:40].max(), rel=0.02)
    assert qc > 4 * math.sqrt(math.pi * 20.07e-6)  # above the edge, as the rule's known bias
    assert critical_q(q, y) == pytest.approx(4 * math.sqrt(math.pi * 20.07e-6), rel=0.02)
    with pytest.raises(ValueError, match="unknown rule"):
        critical_q(q, y, rule="half")


def test_6_lsq_reports_njev_and_status_and_fits_a_subset():
    truth = Model(STACK, Instrument(), {"film.thickness": (150, 350), "film.sld": (20, 80)})
    curve = Curve(Q, forward(Q, STACK))
    start = truth.with_values(np.array([250.0, 40.0]))
    capped = lsq(start, curve, max_nfev=2)
    assert capped.status == 0
    full = lsq(start, curve)
    assert full.status > 0 and full.njev > 0
    only_d = lsq(start.select(["film.thickness"]), curve)
    assert list(only_d.values) == ["film.thickness"]
    assert start.select([False, True]).names == ["film.sld"]


def test_2b_kalpha2_and_resolution_can_be_traced_by_jacfwd_and_vmap():
    """Before the review fix, a traced kalpha2_ratio raised ("Cannot access data pointer")."""
    torch = pytest.importorskip("torch")
    packed = tuple(torch.tensor(a) for a in STACK.packed())
    q = torch.tensor(Q)

    def f(a):
        return forward(q, packed, Instrument(kalpha2_ratio=a, sigma_q=5e-4), backend="torch")

    jac = torch.func.jacfwd(f)(torch.tensor(0.5, dtype=torch.float64)).numpy()
    a, h = torch.tensor(0.5, dtype=torch.float64), 1e-6
    fd = (f(a + h) - f(a - h)).numpy() / (2 * h)
    np.testing.assert_allclose(jac, fd, rtol=1e-5, atol=1e-9)
    batch = torch.func.vmap(f)(torch.tensor([0.3, 0.5], dtype=torch.float64)).numpy()
    np.testing.assert_allclose(
        batch[1], f(torch.tensor(0.5, dtype=torch.float64)).numpy(), rtol=1e-12
    )
