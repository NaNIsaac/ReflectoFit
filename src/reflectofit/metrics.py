"""Residuals, losses and information criteria with one signature (DESIGN.md section 3.2).

`lam` is the model's expected counts and `y` the measured counts, both per point; `scale` is the
counts at R = 1. Data given as reflectivity are counts with scale 1 for the log, raw and chi2 kinds;
the Poisson kind needs real counts. Works on numpy arrays and torch tensors alike.
"""

from typing import NamedTuple

import numpy as np

KINDS = ("log", "raw", "poisson", "chi2")


def _xp(x):
    if hasattr(x, "detach"):
        import torch

        return torch
    return np


def _clamp_min(x, v):
    return x.clamp_min(v) if hasattr(x, "clamp_min") else np.maximum(x, v)


def residuals(kind: str, lam, y, scale: float = 1.0, sigma=None, floor: float = 0.5):
    """Per-point residuals whose sum of squares is the loss of that kind.

    log: log10 lam - log10 max(y, floor); raw: (lam - y) / scale;
    poisson: sign(lam - y) sqrt(2 [lam - y + y ln(y / lam)]), zero counts included;
    chi2: (lam - y) / sigma.
    """
    xp = _xp(lam)
    match kind:
        case "log":
            return xp.log10(lam) - xp.log10(_clamp_min(y, floor))
        case "raw":
            return (lam - y) / scale
        case "poisson":
            d = _deviance_terms(lam, y)
            return xp.sign(lam - y) * xp.sqrt(_clamp_min(d, 0.0))
        case "chi2":
            if sigma is None:
                raise ValueError("chi2 residuals need sigma")
            return (lam - y) / sigma
    raise ValueError(f"unknown kind {kind!r}; use one of {KINDS}")


def loss(kind: str, lam, y, scale: float = 1.0, sigma=None, mask=None, floor: float = 0.5):
    """Sum of squared residuals over the mask (last axis); the Poisson kind sums to the deviance."""
    r2 = residuals(kind, lam, y, scale=scale, sigma=sigma, floor=floor) ** 2
    if mask is not None:
        r2 = r2 * mask
    return r2.sum(-1)


def _deviance_terms(lam, y):
    xp = _xp(lam)
    if xp is np:
        from scipy.special import xlogy

        return 2.0 * (lam - y + xlogy(y, y) - xlogy(y, lam))
    return 2.0 * (lam - y + xp.special.xlogy(y, y) - xp.special.xlogy(y, lam))


def deviance(lam, y, mask=None):
    """Poisson deviance 2 sum [lam - y + y ln(y / lam)] over the last axis, zero counts included."""
    d = _deviance_terms(lam, y)
    if mask is not None:
        d = d * mask
    return d.sum(-1)


class Criteria(NamedTuple):
    aic: float
    bic: float


def information_criteria(deviance: float, k: int, n: int) -> Criteria:
    """AIC = D + 2k and BIC = D + k ln n, with the deviance standing in for -2 ln L (the saturated
    model's constant cancels in any comparison on the same data)."""
    return Criteria(deviance + 2.0 * k, deviance + k * np.log(n))
