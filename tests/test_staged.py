"""Staged gradient fitting (the ReflectoLearn refine strategy, ported for X1c).

Pre-registered (written before reflectofit.staged existed):
  1. The losses equal ReflectoLearn's `refine/losses.py` formulas, written out here, within 1e-12 in
     float64 (log10 MSE, linear MSE, 1 - Pearson correlation, value + 10 x slope MSE, 20 x linear +
     log, Poisson deviance with zero counts), each a per-curve masked mean then a batch mean.
  2. Bounds: the state starts at the given values (within 1e-9 relative after the logit round
     trip, inside the clamp of 1e-4 of the range) and never leaves [lo, hi]; a stage of 0 steps
     changes nothing; keys outside a stage are passed to the model unchanged.
  3. `active` is the union of the keys of the stages run so far, and the model sees it.
  4. The Occam penalty pulls a parameter the data do not constrain towards its neutral value.
  5. On a noise-free single-film curve from a start 5 % off in thickness and SLD, two log-MSE
     stages (lr 0.01, then 1e-3) bring thickness within 0.2 % and SLD within 1 % of the truth.
  6. `fit.staged` on a Model with Poisson counts ends at a deviance no higher than its start, with
     every value inside its bounds, and its result says method "staged" with the stages run.
A failure is reported as a failure; the criteria are not changed after the run.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reflectofit import Instrument, Layer, Stack, forward, tth_to_q  # noqa: E402
from reflectofit.data import Curve  # noqa: E402
from reflectofit.fit import staged as fit_staged  # noqa: E402
from reflectofit.model import Model  # noqa: E402
from reflectofit.staged import LOSSES, Param, Stage, StagedFit, loss_fn  # noqa: E402

Q = tth_to_q(np.linspace(0.2, 6.0, 300))
SI = 20.07 + 0.46j


def _film_simulate(q):
    """thickness, sld (B,) -> R (B, Q) for one film on Si, roughness 4 / 3 Å."""
    qt = torch.as_tensor(q)

    def simulate(vals, active):
        b = vals["thickness"].shape[0]
        z = torch.zeros(b, dtype=torch.float64)
        sld = torch.stack([z.to(torch.complex128), vals["sld"] + 0.5j, z + SI], 1)
        thk = torch.stack([z, vals["thickness"], z], 1)
        rgh = torch.stack([z, z + 4.0, z + 3.0], 1)
        return forward(qt, (sld, thk, rgh), backend="torch")

    return simulate


def _ref_losses(r, log_y, m):
    """ReflectoLearn refine/losses.py, written out."""
    m = m.to(r.dtype)

    def mean(err, mm=m):
        return ((err * mm).sum(-1) / mm.sum(-1).clamp_min(1.0)).mean()

    lr = torch.log10(r.clamp_min(1e-12))
    out = {
        "log_mse": mean((lr - log_y) ** 2),
        "linear_mse": mean((r - 10.0**log_y) ** 2),
    }
    n = m.sum(-1).clamp_min(1.0)
    vx = (lr - ((lr * m).sum(-1) / n).unsqueeze(-1)) * m
    vy = (log_y - ((log_y * m).sum(-1) / n).unsqueeze(-1)) * m
    den = torch.sqrt((vx**2).sum(-1)) * torch.sqrt((vy**2).sum(-1)) + 1e-9
    out["correlation"] = (1.0 - (vx * vy).sum(-1) / den).mean()
    md = m[:, 1:] * m[:, :-1]
    grad = mean(((lr[:, 1:] - lr[:, :-1]) - (log_y[:, 1:] - log_y[:, :-1])) ** 2, md)
    out["log_gradient"] = out["log_mse"] + 10.0 * grad
    out["hybrid"] = 20.0 * out["linear_mse"] + out["log_mse"]
    return out


def test_losses_are_the_reflectolearn_formulas():
    g = torch.Generator().manual_seed(1)
    r = 10.0 ** (-6 * torch.rand(4, 50, generator=g, dtype=torch.float64))
    log_y = torch.log10(r) + 0.1 * torch.randn(4, 50, generator=g, dtype=torch.float64)
    mask = (torch.rand(4, 50, generator=g) > 0.2).double()
    ref = _ref_losses(r, log_y, mask)
    for kind, expected in ref.items():
        got = LOSSES[kind](r, log_y, mask)
        np.testing.assert_allclose(got.item(), expected.item(), rtol=1e-12, err_msg=kind)
    s = 1e6
    y = torch.poisson(s * r, generator=g) / s
    y[0, :5] = 0.0
    lam, yc = s * r, s * y
    dev = 2.0 * (lam - yc + torch.special.xlogy(yc, yc / lam))
    expected = ((dev * mask).sum(-1) / mask.sum(-1)).mean()
    log_y0 = torch.log10(y)  # -inf where y = 0: 10**log_y gives the zero count back
    got = loss_fn("poisson", counts_scale=s)(r, log_y0, mask)
    np.testing.assert_allclose(got.item(), expected.item(), rtol=1e-12)


def _fit(truth=(240.0, 45.0), start=(240.0, 45.0), **kw):
    sim = _film_simulate(Q)
    t = {"thickness": torch.tensor([truth[0]]), "sld": torch.tensor([truth[1]])}
    log_y = torch.log10(sim(t, set()))
    params = {
        "thickness": Param(start[0], 150.0, 350.0),
        "sld": Param(start[1], 20.0, 80.0),
    }
    return StagedFit(sim, params, log_y, **kw)


def test_state_starts_at_the_values_and_stays_in_bounds():
    f = _fit(start=(252.0, 42.75))
    assert f.state["thickness"].item() == pytest.approx(252.0, rel=1e-9)
    f.run(Stage("none", ("thickness", "sld"), steps=0))
    assert f.state["thickness"].item() == pytest.approx(252.0, rel=1e-9)
    assert f.state["sld"].item() == pytest.approx(42.75, rel=1e-9)
    f.run(Stage("big steps", ("thickness",), lr=5.0, steps=50))
    lo, hi = f.bounds["thickness"]
    assert lo <= f.state["thickness"].item() <= hi
    assert f.state["sld"].item() == pytest.approx(42.75, rel=1e-9)


def test_active_is_the_union_of_the_stages_run():
    seen = []
    base = _film_simulate(Q)

    def sim(vals, active):
        seen.append(set(active))
        return base(vals, active)

    t = {"thickness": torch.tensor([240.0]), "sld": torch.tensor([45.0])}
    f = StagedFit(
        sim, {k: Param(v.item(), 1.0, 500.0) for k, v in t.items()}, torch.log10(base(t, set()))
    )
    f.run(Stage("a", ("thickness",), steps=1))
    assert seen[-1] == {"thickness"}
    f.run(Stage("b", ("sld",), steps=1))
    assert seen[-1] == {"thickness", "sld"}
    assert f.active == {"thickness", "sld"}


def test_penalty_pulls_an_unconstrained_parameter_to_neutral():
    base = _film_simulate(Q)
    t = {"thickness": torch.tensor([240.0]), "sld": torch.tensor([45.0])}
    params = {
        "thickness": Param(240.0, 150.0, 350.0),
        "sld": Param(45.0, 20.0, 80.0),
        "unused": Param(5.0, 0.0, 60.0),
    }
    f = StagedFit(
        base, params, torch.log10(base(t, set())), penalty={"unused": 0.0}, penalty_weight=0.2
    )
    f.run(Stage("pull", ("unused",), steps=300, lr=0.05))
    assert f.state["unused"].item() < 1.0


def test_two_log_stages_recover_a_film():
    f = _fit(start=(252.0, 42.75))
    f.run(Stage("coarse", ("thickness", "sld"), steps=600, lr=0.01))
    f.run(Stage("fine", ("thickness", "sld"), steps=600, lr=1e-3))
    assert f.state["thickness"].item() == pytest.approx(240.0, rel=2e-3)
    assert f.state["sld"].item() == pytest.approx(45.0, rel=1e-2)


def test_fit_staged_on_a_model():
    s = 1e7
    truth = Model(
        Stack([Layer(45 + 1.5j, 240, 6, "film")], Layer(SI, roughness=3, name="substrate")),
        Instrument(background=1e-7),
        {"film.thickness": (150, 350), "film.sld": (20, 80), "i0": (0.5, 2.0)},
    )
    r = truth.curves(Q, truth.start())[0]
    curve = Curve(Q, np.random.default_rng(4).poisson(s * r).astype(float))
    start = truth.with_values(np.array([250.0, 43.0, 1.1]))
    stages = [
        Stage("scale", ("i0",), loss="linear_mse", steps=100, lr=0.05),
        Stage("all", ("film.thickness", "film.sld", "i0"), loss="poisson", steps=400, lr=0.01),
    ]
    res = fit_staged(start, curve, stages, counts_scale=s)
    lo, hi = start.bounds
    theta = np.array([res.values[k] for k in start.names])
    assert np.all((lo <= theta) & (theta <= hi))
    dev0 = fit_staged(start, curve, [], counts_scale=s).objective
    assert res.objective <= dev0
    assert res.method == "staged"
    assert [st["name"] for st in res.settings["stages"]] == ["scale", "all"]
    assert math.isfinite(res.objective)
