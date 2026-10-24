"""metrics, data, estimate and materials."""

import math

import numpy as np
import pytest

from reflectofit import Layer, Stack, forward, materials, tth_to_q
from reflectofit.data import block_split, read, stitch, write
from reflectofit.estimate import critical_q, kiessig_thickness
from reflectofit.metrics import deviance, information_criteria, loss, residuals


def test_poisson_residuals_square_to_the_deviance():
    rng = np.random.default_rng(1)
    lam = rng.uniform(0.01, 200, 500)
    y = rng.poisson(lam).astype(float)
    y[:20] = 0.0
    assert y.min() == 0
    np.testing.assert_allclose(loss("poisson", lam, y), deviance(lam, y), rtol=1e-12)


def test_residual_kinds():
    lam, y = np.array([10.0, 100.0]), np.array([0.0, 120.0])
    np.testing.assert_allclose(residuals("log", lam, y), np.log10(lam) - np.log10([0.5, 120.0]))
    np.testing.assert_allclose(residuals("raw", lam, y, scale=1e6), (lam - y) / 1e6)
    np.testing.assert_allclose(residuals("chi2", lam, y, sigma=np.array([1.0, 10.0])), [10.0, -2.0])
    assert residuals("poisson", lam, y)[0] == pytest.approx(math.sqrt(20.0))
    with pytest.raises(ValueError, match="unknown kind"):
        residuals("l1", lam, y)


def test_deviance_is_zero_at_the_data_and_positive_elsewhere():
    y = np.array([0.0, 3.0, 50.0])
    assert deviance(np.where(y > 0, y, 1e-300), y) == pytest.approx(0.0, abs=1e-250)
    assert deviance(y + 1.0, y) > 0


def test_information_criteria():
    c = information_criteria(100.0, k=5, n=200)
    assert c.aic == 110.0
    assert c.bic == pytest.approx(100.0 + 5 * math.log(200))


def test_metrics_accept_torch_tensors():
    torch = pytest.importorskip("torch")
    lam = torch.tensor([10.0, 100.0], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.0, 120.0], dtype=torch.float64)
    d = loss("poisson", lam, y)
    d.backward()
    np.testing.assert_allclose(d.item(), deviance(lam.detach().numpy(), y.numpy()), rtol=1e-12)
    assert torch.isfinite(lam.grad).all()


def test_read_two_and_three_columns(tmp_path):
    f = tmp_path / "scan.dat"
    f.write_text("# 2theta counts error\n1.0 100 10\n0.5, 400, 20\nnot data\n")
    c = read(f, x="tth")
    np.testing.assert_allclose(c.q, tth_to_q([0.5, 1.0]))
    np.testing.assert_allclose(c.y, [400, 100])
    np.testing.assert_allclose(c.sigma, [20, 10])
    out = tmp_path / "out.dat"
    write(out, c)
    np.testing.assert_allclose(read(out).q, c.q, rtol=1e-12)


def test_stitch_boundary_takes_the_last_scan_as_reference():
    a = np.array([[0.1, 1000.0, 10.0], [0.2, 500.0, 5.0]])
    b = np.array([[0.2, 50.0, 1.0], [0.3, 20.0, 1.0]])
    out = stitch([a, b])
    np.testing.assert_allclose(out[:, 0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(out[:, 1], [100.0, 50.0, 20.0])
    np.testing.assert_allclose(out[0, 2], 1.0)


def test_stitch_median_takes_the_first_scan_as_reference():
    x = np.linspace(0.1, 1.0, 10)
    true = 1e6 * np.exp(-5 * x)
    a = np.column_stack([x[:6], true[:6]])
    b = np.column_stack([x[4:], true[4:] / 30.0])
    out = stitch([b, a], method="median")
    late = out[out[:, 0] > x[5] + 1e-9]
    np.testing.assert_allclose(late[:, 1], true[6:], rtol=1e-12)
    with pytest.raises(ValueError, match="fewer than two"):
        stitch([a[:3], b[3:]], method="median")


def test_block_split_keeps_low_q_and_holds_whole_blocks():
    q = np.linspace(0.0, 1.0, 101)
    train, held = block_split(q, n_blocks=10, held=(2, 6), q_min=0.3)
    assert not held[q < 0.3].any()
    assert held.sum() == 2 * 7
    assert np.all(train ^ held)
    idx = np.flatnonzero(held)
    assert np.count_nonzero(np.diff(idx) > 1) == 1


def test_critical_q_finds_the_edge():
    q = np.linspace(0.005, 0.2, 2000)
    r = forward(q, Stack([], Layer(20.07 + 0.46j, roughness=3)))
    assert critical_q(q, r) == pytest.approx(4 * math.sqrt(math.pi * 20.07e-6), rel=0.02)


def test_kiessig_thickness_recovers_the_film():
    q = tth_to_q(np.linspace(0.1, 6.0, 600))
    r = forward(q, Stack([Layer(45 + 1j, 240, 4)], Layer(20.07 + 0.46j, roughness=3)))
    assert kiessig_thickness(q, r, sld=45.0) == pytest.approx(240, abs=3)


def test_materials_from_density():
    assert materials.sld("Si", 2.33) == pytest.approx(20.071 + 0.457j, abs=1e-3)
    assert materials.sld("SiO2", 2.20) == pytest.approx(18.865 + 0.244j, abs=1e-3)
