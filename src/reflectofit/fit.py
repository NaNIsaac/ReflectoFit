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
