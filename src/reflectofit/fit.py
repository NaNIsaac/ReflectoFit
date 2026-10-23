"""Fitting: least squares in a trust window, differential evolution, and scoring.

`counts_scale` S says the data are counts with S counts at R = 1; then lam = S·R and the Poisson
kind is the likelihood. Without it the data are reflectivity and lam = R.
"""

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution, least_squares

from reflectofit import metrics
from reflectofit.data import Curve
from reflectofit.model import Model


@dataclass
class FitResult:
    model: Model
    values: dict[str, float]
    errors: dict[str, float]
    loss: str
    objective: float
    n_points: int
    aic: float
    bic: float
    success: bool
    message: str
    nfev: int
    method: str
    njev: int = 0
    status: int | None = None
    """scipy's status: for lsq 0 means the evaluation budget ran out."""
    settings: dict = field(default_factory=dict)
    covariance: NDArray | None = None
    at_bound: list[str] = field(default_factory=list)
    """Parameters that ended on a bound; their errors from the Jacobian are not meaningful."""

    def to_json(self) -> str:
        d = asdict(self) | {
            "covariance": None if self.covariance is None else self.covariance.tolist()
        }
        d.pop("model")
        return json.dumps(d, indent=1)


def _setup(model: Model, curve: Curve, loss: str, counts_scale: float | None, mask):
    if loss == "poisson" and counts_scale is None:
        raise ValueError("the poisson loss needs counts and counts_scale")
    if loss == "chi2" and curve.sigma is None:
        raise ValueError("the chi2 loss needs an error column")
    scale = 1.0 if counts_scale is None else float(counts_scale)
    keep = np.ones(curve.q.size, bool) if mask is None else np.asarray(mask, bool)
    y = curve.y[keep]
    floor = 0.5 if counts_scale is not None else 0.5 * float(np.min(y[y > 0]))
    sigma = None if curve.sigma is None else curve.sigma[keep]
    q = curve.q[keep]

    def resid(theta: NDArray) -> NDArray:
        lam = scale * model.curves(q, theta)
        return metrics.residuals(loss, lam, y[None], scale=scale, sigma=sigma, floor=floor)

    return q, resid


def _result(model, theta, r, jac, loss, n, success, message, nfev, method, settings, bounds):
    lo, hi = bounds
    tol = 1e-6 * (hi - lo)
    on_bound = (theta - lo <= tol) | (hi - theta <= tol)
    at_bound = [name for name, b in zip(model.names, on_bound, strict=True) if b]
    k = theta.size
    objective = float(np.sum(r**2))
    # AIC and BIC need a likelihood: the Poisson deviance or chi2; nan for the log and raw losses
    likelihood = loss in ("poisson", "chi2")
    criteria = metrics.information_criteria(objective, k, n) if likelihood else (np.nan, np.nan)
    cov = None
    errors = dict.fromkeys(model.names, float("nan"))
    if jac is not None and n > k:
        jtj = jac.T @ jac
        try:
            cov = np.linalg.inv(jtj)
            if not likelihood:
                cov = cov * objective / (n - k)
            errors = dict(
                zip(model.names, np.sqrt(np.clip(np.diag(cov), 0, None)).tolist(), strict=True)
            )
        except np.linalg.LinAlgError:
            cov = None
    return FitResult(
        model=model.with_values(theta),
        values=dict(zip(model.names, theta.tolist(), strict=True)),
        errors=errors,
        loss=loss,
        objective=objective,
        n_points=n,
        aic=float(criteria[0]),
        bic=float(criteria[1]),
        success=bool(success),
        message=str(message),
        nfev=int(nfev),
        method=method,
        settings=settings,
        covariance=cov,
        at_bound=at_bound,
    )


def trust_bounds(model: Model, margin: float) -> tuple[NDArray, NDArray]:
    """The model's bounds narrowed to start · (1 ± margin) for every parameter with a non-zero start
    (the trust window of ReflectoLearn refine around a network guess)."""
    lo, hi = model.bounds
    x0 = model.start()
    half = np.abs(x0) * margin
    use = x0 != 0
    return np.where(use, np.maximum(lo, x0 - half), lo), np.where(
        use, np.minimum(hi, x0 + half), hi
    )


def lsq(
    model: Model,
    curve: Curve,
    loss: str = "log",
    counts_scale: float | None = None,
    trust: float | None = None,
    max_nfev: int = 200,
    mask=None,
) -> FitResult:
    """Trust-region reflective least squares (scipy) on the residuals of `loss`, with a central
    finite-difference Jacobian computed as one batched forward call."""
    if not model.free:
        raise ValueError("the model has no free parameters")
    q, resid = _setup(model, curve, loss, counts_scale, mask)
    lo, hi = trust_bounds(model, trust) if trust is not None else model.bounds
    x0 = np.clip(model.start(), lo, hi)
    step = 1e-6 * (hi - lo)

    def fun(theta):
        return resid(theta)[0]

    def jac(theta):
        p = theta.size
        rows = np.repeat(theta[None], 2 * p, axis=0)
        h = np.minimum(step, np.minimum(theta - lo, hi - theta).clip(step * 1e-3))
        idx = np.arange(p)
        rows[idx, idx] += h
        rows[p + idx, idx] -= h
        r = resid(rows)
        return ((r[:p] - r[p:]) / (2 * h[:, None])).T

    res = least_squares(
        fun, x0, jac=jac, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=max_nfev
    )
    settings = {"trust": trust, "max_nfev": max_nfev, "counts_scale": counts_scale}
    out = _result(
        model,
        res.x,
        res.fun,
        res.jac,
        loss,
        q.size,
        res.success,
        res.message,
        res.nfev,
        "lsq",
        settings,
        (lo, hi),
    )
    out.njev, out.status = int(res.njev or 0), int(res.status)
    return out


def de(
    model: Model,
    curve: Curve,
    loss: str = "log",
    counts_scale: float | None = None,
    popsize: int = 20,
    maxiter: int = 300,
    seed: int | None = 0,
    polish: bool = True,
    mask=None,
) -> FitResult:
    """Differential evolution (scipy) over the model's bounds; each generation is one batched
    forward call. With `polish`, the best member is refined by `lsq`."""
    if not model.free:
        raise ValueError("the model has no free parameters")
    q, resid = _setup(model, curve, loss, counts_scale, mask)

    def cost(pop):  # pop (P, S) in vectorized mode
        return np.sum(resid(np.asarray(pop).T) ** 2, axis=1)

    lo, hi = model.bounds
    res = differential_evolution(
        cost,
        list(zip(lo, hi, strict=True)),
        popsize=popsize,
        maxiter=maxiter,
        seed=seed,
        vectorized=True,
        updating="deferred",
        polish=False,
        init="latinhypercube",
    )
    settings = {"popsize": popsize, "maxiter": maxiter, "seed": seed, "counts_scale": counts_scale}
    out = _result(
        model,
        res.x,
        resid(res.x)[0],
        None,
        loss,
        q.size,
        res.success,
        res.message,
        res.nfev,
        "de",
        settings,
        (lo, hi),
    )
    if not polish:
        return out
    polished = lsq(out.model, curve, loss=loss, counts_scale=counts_scale, mask=mask)
    polished.method = "de+lsq"
    polished.nfev += out.nfev
    polished.settings = settings | {"polish": "lsq"}
    return polished


def staged(
    model: Model,
    curve: Curve,
    stages,
    counts_scale: float | None = None,
    mask=None,
    penalty: dict[str, float] | None = None,
    penalty_weight: float = 0.0,
) -> FitResult:
    """Staged gradient fit (`reflectofit.staged`, needs torch): each stage frees some of the
    model's parameters and runs Adam on one loss within the model's bounds. The result's objective
    is the Poisson deviance with counts_scale, else the log residual sum; it has no covariance."""
    import torch

    from reflectofit import staged as st

    if not model.free:
        raise ValueError("the model has no free parameters")
    kind = "poisson" if counts_scale is not None else "log"
    q, resid = _setup(model, curve, kind, counts_scale, mask)
    keep = np.ones(curve.q.size, bool) if mask is None else np.asarray(mask, bool)
    with np.errstate(divide="ignore"):
        log_y = np.log10(curve.y[keep] / (1.0 if counts_scale is None else counts_scale))
    lo, hi = model.bounds
    params = {
        k: st.Param(float(x), float(a), float(b))
        for k, x, a, b in zip(model.names, model.start(), lo, hi, strict=True)
    }
    fit = st.StagedFit(
        st.model_simulate(model, q),
        params,
        torch.as_tensor(log_y[None], dtype=torch.float64),
        q=torch.as_tensor(q, dtype=torch.float64),
        counts_scale=counts_scale,
        penalty=penalty,
        penalty_weight=penalty_weight,
    )
    for stage in stages:
        fit.run(stage)
    theta = np.array([fit.state[k].item() for k in model.names])
    settings = {
        "stages": [asdict(s) for s in stages],
        "counts_scale": counts_scale,
        "penalty": penalty,
        "penalty_weight": penalty_weight,
    }
    steps = sum(s.steps for s in stages)
    return _result(
        model,
        theta,
        resid(theta)[0],
        None,
        kind,
        q.size,
        True,
        "stages done",
        steps,
        "staged",
        settings,
        (lo, hi),
    )


LOG_SIGMA_BOUNDS = (-8.0, 1.0)
"""Bounds of ln σ, the spread of the log10 residuals in the log likelihood (as ReflectoLearn)."""
TAU_FACTOR = 50
"""Kept steps per integrated autocorrelation time for a usable estimate."""


@dataclass
class MCMCResult:
    names: list[str]
    samples: NDArray
    """Kept samples (N, P), the model's parameters in order (ln σ not included)."""
    intervals: dict[str, tuple[float, float, float]]
    """16th, 50th and 84th percentile of every parameter (and of `log_sigma` for the log loss)."""
    tau: dict[str, float]
    """Integrated autocorrelation time of every parameter over the kept half, in steps."""
    tau_margin: float
    """Kept steps / largest τ."""
    converged: bool
    steps: int
    walkers: int
    acceptance: float
    log_prob_max: float
    best: dict[str, float]
    settings: dict = field(default_factory=dict)


def _ensemble(log_prob, z0, steps, max_steps, seed, to_params=None):
    """Run emcee's DEMove ensemble from z0 (walkers, ndim): `steps`, then doubled while the kept half
    (the first half discarded) is shorter than TAU_FACTOR times the largest integrated
    autocorrelation time, up to `max_steps`. Returns the kept chain (steps, walkers, ndim) mapped by
    `to_params`, its τ per coordinate, the total steps and the sampler."""
    import emcee

    walkers, ndim = z0.shape
    sampler = emcee.EnsembleSampler(
        walkers, ndim, log_prob, vectorize=True, moves=[(emcee.moves.DEMove(), 1.0)]
    )
    sampler.random_state = np.random.RandomState(seed).get_state()
    state = sampler.run_mcmc(z0, steps, progress=False)
    total = steps
    while True:
        chain = sampler.get_chain(discard=total // 2)
        if to_params is not None:
            chain = to_params(chain)
        tau = emcee.autocorr.integrated_time(chain, quiet=True)
        if (total - total // 2) / np.max(tau) >= TAU_FACTOR or total >= max_steps:
            return chain, tau, total, sampler
        more = min(total, max_steps - total)
        state = sampler.run_mcmc(state, more, progress=False)
        total += more


def mcmc(
    model: Model,
    curve: Curve,
    start: FitResult | None = None,
    loss: str = "poisson",
    counts_scale: float | None = None,
    steps: int = 5000,
    max_steps: int = 20000,
    walkers: int = 64,
    seed: int = 0,
    mask=None,
    n_independent: int | None = None,
) -> MCMCResult:
    """Posterior sampling with emcee (extra `mcmc`), flat priors within the model's bounds.

    The walkers move by differential evolution (emcee's DEMove), which mixed in correlated
    posteriors where the stretch move did not (ReflectoLearn#29, R2c). Coordinates are whitened by
    the start fit's covariance (θ = θ_start + L z, L its Cholesky factor); DE is affine-invariant, so
    this sets the shape of the initial ensemble, a 0.1 σ ball around the start. emcee's
    DESnookerMove is not used: it does not leave the target invariant (ReflectoFit#10). The first
    half of the chain is discarded. While the kept half is shorter than TAU_FACTOR times the largest
    integrated autocorrelation time, the chain is extended, up to `max_steps`; if it is still short,
    `converged` is False and a RuntimeWarning says so. `converged` is that τ criterion only: a basin
    the walkers never reach from the start does not show in τ.

    `loss` "poisson" is the likelihood of counts (needs counts_scale); "log" is a Gaussian in log10
    R with one fitted spread ln σ, scaled to `n_independent` points if given (as ReflectoLearn's
    refine.mcmc).
    """
    import warnings

    if not model.free:
        raise ValueError("the model has no free parameters")
    if loss not in ("poisson", "log"):
        raise ValueError(f"loss must be 'poisson' or 'log', got {loss!r}")
    start_model = start.model if start is not None else model
    q, resid = _setup(start_model, curve, loss, counts_scale, mask)
    n = q.size
    lo, hi = start_model.bounds
    theta0 = start_model.start()
    p = theta0.size
    cov = start.covariance if start is not None else None
    try:
        chol = np.linalg.cholesky(cov) if cov is not None else None
    except np.linalg.LinAlgError:
        chol = None
    if chol is None:
        chol = np.diag(1e-2 * (hi - lo))
    log_like = loss == "log"
    weight = n_independent / n if n_independent else 1.0

    def to_theta(z):
        return theta0 + z[:, :p] @ chol.T

    def log_prob(z):
        z = np.atleast_2d(z)
        theta = to_theta(z)
        out = np.full(z.shape[0], -np.inf)
        ok = np.all((theta >= lo) & (theta <= hi), axis=1)
        if log_like:
            ok &= (z[:, p] > LOG_SIGMA_BOUNDS[0]) & (z[:, p] < LOG_SIGMA_BOUNDS[1])
        if ok.any():
            sq = np.sum(resid(theta[ok]) ** 2, axis=1)
            if log_like:
                s = np.exp(z[ok, p])
                out[ok] = weight * (-0.5 * sq / s**2 - n * np.log(s))
            else:
                out[ok] = -0.5 * sq
        return out

    ndim = p + log_like
    walkers = max(walkers, 2 * ndim + 2)
    rng = np.random.default_rng(seed)
    z0 = np.zeros((walkers, ndim))
    z0[:, :p] = 0.1 * rng.standard_normal((walkers, p))
    theta = np.clip(to_theta(z0), lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
    z0[:, :p] = np.linalg.solve(chol, (theta - theta0).T).T
    if log_like:
        rms = np.sqrt(np.sum(resid(theta0[None])[0] ** 2) / max(n, 1))
        z0[:, p] = np.clip(np.log(max(rms, 1e-4)), *LOG_SIGMA_BOUNDS) + 0.01 * rng.standard_normal(
            walkers
        )

    def to_params(chain):
        th = theta0 + chain[..., :p] @ chol.T
        return np.concatenate([th, chain[..., p:]], axis=-1)

    chain, tau, total, sampler = _ensemble(log_prob, z0, steps, max_steps, seed, to_params)
    kept = total - total // 2
    margin = float(kept / np.max(tau))
    converged = margin >= TAU_FACTOR
    if not converged:
        warnings.warn(
            f"mcmc: {kept} kept steps < {TAU_FACTOR} x the largest integrated autocorrelation time "
            f"({np.max(tau):.0f}); the intervals are not converged, run a longer chain",
            RuntimeWarning,
            stacklevel=2,
        )
    flat = chain.reshape(-1, ndim)
    names = list(start_model.names)
    labels = [*names, "log_sigma"] if log_like else names
    lp = sampler.get_log_prob(discard=total // 2, flat=True)
    best = flat[int(np.argmax(lp))]
    return MCMCResult(
        names=names,
        samples=flat[:, :p],
        intervals={
            k: tuple(float(v) for v in np.percentile(flat[:, i], [16, 50, 84]))
            for i, k in enumerate(labels)
        },
        tau=dict(zip(labels, map(float, tau), strict=True)),
        tau_margin=margin,
        converged=converged,
        steps=total,
        walkers=walkers,
        acceptance=float(np.mean(sampler.acceptance_fraction)),
        log_prob_max=float(lp.max()),
        best=dict(zip(names, map(float, best[:p]), strict=True)),
        settings={
            "loss": loss,
            "counts_scale": counts_scale,
            "seed": seed,
            "max_steps": max_steps,
            "n_independent": n_independent,
            "whitened": cov is not None,
        },
    )


def score(model: Model, curve: Curve, counts_scale: float | None = None, mask=None) -> dict:
    """Every loss kind that the data allow, and AIC/BIC with the model's free-parameter count."""
    out = {}
    for kind in metrics.KINDS:
        try:
            _, resid = _setup(model, curve, kind, counts_scale, mask)
        except ValueError:
            continue
        theta = model.start() if model.free else np.zeros(0)
        r = resid(theta)[0]
        n = r.size
        out[kind] = float(np.sum(r**2))
        if kind in ("poisson", "chi2"):
            c = metrics.information_criteria(out[kind], len(model.free), n)
            out[f"{kind}_aic"], out[f"{kind}_bic"] = c.aic, c.bic
    return out
