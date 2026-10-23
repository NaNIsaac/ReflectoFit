"""Posterior sampling (`fit.mcmc`, X1f): emcee in coordinates whitened by the start fit's covariance,
differential-evolution moves, a convergence check on the integrated autocorrelation time τ.

Pre-registered (written before fit.mcmc existed; the design follows ReflectoLearn's R2c amendment 3,
where whitening and DE moves converged chains that the stretch move did not, ReflectoLearn#29):
  1. Convergence: on a Poisson curve of one film (240 Å, 6 Å, 45 + 1.5i on Si; 10⁶ counts at R = 1;
     thickness, roughness, SLD, i0 and log10 background free), started from the least-squares fit,
     64 walkers, the kept half reaches 50 τ (largest over parameters) within 20000 steps.
  2. Width: on that curve the 68 % half-width (p84 − p16) / 2 of the thickness and of the SLD lies
     within 0.5 to 2 times the profile-likelihood half-width (mean distance to the Δ deviance = 1
     crossings, the other parameters refitted by least squares at each fixed value).
  3. Every sample lies inside the model's bounds; the same seed gives the same samples.
  4. A chain stopped short of 50 τ says so: converged False, τ reported, a RuntimeWarning.
  5. The log likelihood (Gaussian in log10 R with one fitted spread, as ReflectoLearn's refine.mcmc)
     runs and reports the spread; no width criterion (R2c found it up to 2.5 times too wide).
A failure is reported as a failure; the criteria are not changed after the run.
First run (ce04cc6): all held, with half-width ratios 0.955 and 0.958.

Second registration (ReflectoFit#10, written before the sampler is changed): emcee 3.1.6's
DESnookerMove does not leave the target invariant, and the 0.5-2 band of criterion 2 let its bias
through (physics review of ReflectoFit#9: on N(0, I) the 0.8 DE + 0.2 snooker mix gives variance
0.934 at d = 5 and 0.861 at d = 13). The sampler becomes DEMove alone. Criteria 1-5 stay as they
are (2 is not re-graded); added:
  6. The ensemble on a known target, N(0, I_d) through the sampler's own moves (64 walkers started
     in the target, 6000 steps, first half discarded, seed 0): the variance averaged over the
     coordinates within 1 ± 0.03, and so the 68 % half-width, for d = 5 and d = 13 (about five
     standard errors at the τ the review measured; it separates 0.934 and 0.861 from 1).
  7. On the curve of criterion 2, the half-width ratios of thickness and SLD within 1 ± 0.05 (the
     least-squares errors equal the profile half-widths to 1e-3 there, so the expected ratio is 1).
"""

import math

import numpy as np
import pytest

from reflectofit import Instrument, Layer, Stack, tth_to_q
from reflectofit.data import Curve
from reflectofit.fit import _ensemble, lsq, mcmc
from reflectofit.model import Model

emcee = pytest.importorskip("emcee")

S = 1e6
TRUTH = Model(
    Stack([Layer(45 + 1.5j, 240, 6, "film")], Layer(20.07 + 0.46j, roughness=3, name="substrate")),
    Instrument(i0=1.0, background=1e-6),
    {
        "film.thickness": (150, 350),
        "film.roughness": (1, 15),
        "film.sld": (20, 80),
        "i0": (0.5, 2.0),
        "log10_background": (-8, -4),
    },
)
Q = tth_to_q(np.linspace(0.2, 6.0, 300))


@pytest.fixture(scope="module")
def data():
    r = TRUTH.curves(Q, TRUTH.start())[0]
    curve = Curve(Q, np.random.default_rng(5).poisson(S * r).astype(float))
    best = lsq(TRUTH, curve, loss="poisson", counts_scale=S, max_nfev=500)
    return curve, best


def _profile_half_width(curve, best, key):
    """Mean distance from the ML value to the Δ deviance = 1 crossings of the profile of key."""
    model = best.model
    others = [k for k in model.names if k != key]
    x_ml, d_ml = best.values[key], best.objective
    lo, hi = model.free[key]

    def deviance_at(x):
        fixed = model.with_values(
            np.array([x if k == key else best.values[k] for k in model.names])
        )
        sub = fixed.select(others)
        return lsq(sub, curve, loss="poisson", counts_scale=S, max_nfev=300).objective

    step = best.errors[key] if math.isfinite(best.errors[key]) else 0.01 * (hi - lo)
    widths = []
    for sign in (+1, -1):
        a, b = 0.0, step
        while deviance_at(x_ml + sign * b) - d_ml < 1.0:
            a, b = b, 2 * b
        for _ in range(30):
            mid = 0.5 * (a + b)
            if deviance_at(x_ml + sign * mid) - d_ml < 1.0:
                a = mid
            else:
                b = mid
        widths.append(0.5 * (a + b))
    return float(np.mean(widths))


@pytest.fixture(scope="module")
def chain(data):
    curve, best = data
    return mcmc(
        TRUTH, curve, best, loss="poisson", counts_scale=S, steps=5000, max_steps=20000, seed=0
    )


def test_chain_converges_within_20000_steps(chain):
    assert chain.converged
    assert chain.tau_margin >= 50


@pytest.mark.parametrize("key", ["film.thickness", "film.sld"])
def test_width_matches_the_profile_likelihood(data, chain, key):
    curve, best = data
    p16, p84 = chain.intervals[key][0], chain.intervals[key][2]
    ratio = 0.5 * (p84 - p16) / _profile_half_width(curve, best, key)
    assert 0.5 <= ratio <= 2.0, ratio


def test_samples_stay_in_bounds_and_repeat(data, chain):
    curve, best = data
    lo, hi = TRUTH.bounds
    assert np.all((chain.samples >= lo) & (chain.samples <= hi))
    again = mcmc(
        TRUTH, curve, best, loss="poisson", counts_scale=S, steps=300, max_steps=300, seed=0
    )
    once = mcmc(
        TRUTH, curve, best, loss="poisson", counts_scale=S, steps=300, max_steps=300, seed=0
    )
    np.testing.assert_array_equal(again.samples, once.samples)


def test_a_short_chain_says_it_is_not_converged(data):
    curve, best = data
    with pytest.warns(RuntimeWarning, match="autocorrelation"):
        res = mcmc(
            TRUTH, curve, best, loss="poisson", counts_scale=S, steps=200, max_steps=200, seed=1
        )
    assert not res.converged
    assert set(res.tau) == set(TRUTH.names)


def test_log_likelihood_runs_and_reports_its_spread(data):
    curve, best = data
    res = mcmc(TRUTH, curve, best, loss="log", counts_scale=S, steps=400, max_steps=400, seed=2)
    assert "log_sigma" in res.intervals
    assert np.isfinite(res.intervals["log_sigma"]).all()


@pytest.mark.parametrize("d", [5, 13])
def test_the_ensemble_samples_a_known_gaussian(d):
    def log_prob(z):
        return -0.5 * np.sum(np.atleast_2d(z) ** 2, axis=1)

    z0 = np.random.default_rng(0).standard_normal((64, d))
    chain, _, _, _ = _ensemble(log_prob, z0, steps=6000, max_steps=6000, seed=0)
    flat = chain.reshape(-1, d)
    variance = flat.var(axis=0).mean()
    p16, p84 = np.percentile(flat, [16, 84], axis=0)
    half_width = (0.5 * (p84 - p16)).mean()
    assert abs(variance - 1) <= 0.03, variance
    assert abs(half_width - 1) <= 0.03, half_width


@pytest.mark.parametrize("key", ["film.thickness", "film.sld"])
def test_width_ratio_is_near_one_on_the_gaussian_curve(data, chain, key):
    curve, best = data
    p16, p84 = chain.intervals[key][0], chain.intervals[key][2]
    ratio = 0.5 * (p84 - p16) / _profile_half_width(curve, best, key)
    assert abs(ratio - 1) <= 0.05, ratio
