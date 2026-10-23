"""Fits recover a known structure from synthetic Poisson data; models round-trip through TOML."""

import numpy as np
import pytest

from reflectofit import Instrument, Layer, Stack, tth_to_q
from reflectofit.data import Curve
from reflectofit.fit import de, lsq, score, trust_bounds
from reflectofit.model import Model, dumps, load

S = 1e8
TRUTH = Model(
    Stack([Layer(45 + 1.5j, 240, 6, "film")], Layer(20.07 + 0.46j, roughness=3, name="substrate")),
    Instrument(i0=1.0, background=1e-7, sigma_q=5e-4),
    {
        "film.thickness": (150, 350),
        "film.roughness": (1, 15),
        "film.sld": (20, 80),
        "i0": (0.5, 2.0),
        "log10_background": (-10, -6),
    },
)
Q = tth_to_q(np.linspace(0.1, 9.0, 400))


def _data(seed=3) -> Curve:
    r = TRUTH.curves(Q, TRUTH.start())[0]
    counts = np.random.default_rng(seed).poisson(S * r).astype(float)
    return Curve(Q, counts)


def _start(**changes) -> Model:
    values = dict(zip(TRUTH.names, TRUTH.start(), strict=True)) | changes
    return TRUTH.with_values(np.array([values[k] for k in TRUTH.names]))


def test_lsq_poisson_recovers_the_truth_within_its_errors():
    res = lsq(
        _start(**{"film.thickness": 252, "film.sld": 42, "film.roughness": 4}),
        _data(),
        loss="poisson",
        counts_scale=S,
    )
    assert res.success
    truth = dict(zip(TRUTH.names, TRUTH.start(), strict=True))
    for k, v in res.values.items():
        assert abs(v - truth[k]) < 4 * res.errors[k], k
    assert res.at_bound == []
    assert res.objective < 1.3 * Q.size
    assert np.isfinite(res.bic)


def test_lsq_log_loss_on_reflectivity_data():
    data = _data()
    curve = Curve(data.q, data.y / S)
    res = lsq(_start(**{"film.thickness": 246}), curve, loss="log")
    assert res.values["film.thickness"] == pytest.approx(240, abs=0.5)
    assert np.isnan(res.bic)


def test_de_finds_the_basin_from_the_bounds_alone():
    res = de(TRUTH, _data(), loss="poisson", counts_scale=S, popsize=15, maxiter=60, seed=1)
    assert res.values["film.thickness"] == pytest.approx(240, abs=1.0)
    assert res.method == "de+lsq"


def test_trust_window_narrows_the_bounds():
    lo, hi = trust_bounds(_start(**{"film.thickness": 200}), 0.1)
    i = TRUTH.names.index("film.thickness")
    assert (lo[i], hi[i]) == pytest.approx((180, 220))


def test_poisson_needs_counts():
    with pytest.raises(ValueError, match="counts_scale"):
        lsq(TRUTH, _data(), loss="poisson")


def test_score_reports_every_available_kind():
    s = score(TRUTH, _data(), counts_scale=S)
    assert set(s) >= {"log", "raw", "poisson", "poisson_aic", "poisson_bic"}
    assert s["poisson"] < 1.3 * Q.size


def test_model_toml_round_trip(tmp_path):
    f = tmp_path / "m.toml"
    f.write_text(dumps(TRUTH), encoding="utf-8")
    back = load(f)
    assert back.names == TRUTH.names
    np.testing.assert_allclose(back.start(), TRUTH.start())
    np.testing.assert_allclose(
        back.curves(Q, back.start()), TRUTH.curves(Q, TRUTH.start()), rtol=1e-12
    )


def test_material_layers_in_toml(tmp_path):
    f = tmp_path / "m.toml"
    f.write_text(
        '[[layer]]\nname = "oxide"\nmaterial = "SiO2"\ndensity = 2.2\nthickness = 10\nroughness = 3\n'
        '[substrate]\nmaterial = "Si"\ndensity = 2.33\nroughness = 2\n',
        encoding="utf-8",
    )
    m = load(f)
    assert m.stack.substrate.sld == pytest.approx(20.071 + 0.457j, abs=1e-3)
    assert m.stack.names == ["oxide", "substrate"]


def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="unknown parameter"):
        Model(TRUTH.stack, free={"flim.thickness": (1, 2)})


def test_a_parameter_pinned_on_its_bound_is_reported():
    pinned = Model(TRUTH.stack, TRUTH.instrument, TRUTH.free | {"film.thickness": (150, 230)})
    res = lsq(pinned, _data(), loss="poisson", counts_scale=S)
    assert res.at_bound == ["film.thickness"]
