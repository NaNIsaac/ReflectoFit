"""Quick estimators read directly off a curve, without a fit."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def critical_q(
    q: NDArray, y: NDArray, rule: Literal["steepest", "half_plateau"] = "steepest"
) -> float:
    """q of the critical edge.

    steepest: the steepest descent of log R above the plateau (the minimum of d log10 R / dq,
    refined by a parabola through it and its neighbours).
    half_plateau: where the 3-point mean of y first falls to half the plateau, the plateau being
    the largest 3-point mean in the first 10 % of points; linear interpolation between the two
    points around the crossing (the rule of ReflectoLearn#35 protocol A1, step 2). NaN if it never
    falls that far.

    Neither is exactly the film's q_c: the half-plateau point is about 9 % high for 250 Å films and
    can be several times low for 20-40 Å dense films whose edge tunnels (physics review on
    ReflectoLearn#35).
    """
    q = np.asarray(q, float)
    y = np.asarray(y, float)
    if rule == "half_plateau":
        ys = np.convolve(y, np.ones(3) / 3, mode="same")
        ys[0], ys[-1] = y[0], y[-1]
        top = int(np.argmax(ys[: max(len(ys) // 10, 3)]))
        below = np.flatnonzero(ys[top:] < 0.5 * ys[top])
        if below.size == 0:
            return float("nan")
        k = top + int(below[0])
        if k == 0:
            return float(q[0])
        f = (ys[k - 1] - 0.5 * ys[top]) / (ys[k - 1] - ys[k])
        return float(q[k - 1] + f * (q[k] - q[k - 1]))
    if rule != "steepest":
        raise ValueError(f"unknown rule {rule!r}; use 'steepest' or 'half_plateau'")
    logy = np.log10(np.clip(y, np.finfo(float).tiny, None))
    start = int(np.argmax(logy))
    slope = np.gradient(logy[start:], q[start:])
    i = int(np.argmin(slope))
    if 0 < i < slope.size - 1:
        a, b, c = slope[i - 1 : i + 2]
        denom = a - 2 * b + c
        if denom > 0:
            dq = q[start + i + 1] - q[start + i]
            return float(q[start + i] + 0.5 * (a - c) / denom * dq)
    return float(q[start + i])


def kiessig_thickness(
    q: NDArray,
    y: NDArray,
    d_range: tuple[float, float] = (20.0, 2000.0),
    sld: float | None = None,
    d_min: float = 60.0,
    mask: NDArray | None = None,
) -> float:
    """Film thickness [Å] from the Kiessig period: the strongest peak of the real-space transform
    of the detrended log R (port of ReflectoLearn `refine.fft_thickness`, b6fb0c4).

    Only points with y > 0 (and inside `mask`, if given) are used: a zero-count point has no
    logarithm. Fringes are periodic in the film's own wavevector sqrt(q² − q_c²); with `sld`
    [1e-6 Å⁻²] the transform runs over it, otherwise over q (which overestimates d by a few percent
    near the edge). NaN with fewer than 40 usable points or no peak in [max(d_min, lo), hi].
    """
    q = np.asarray(q, float)
    y = np.asarray(y, float)
    ok = y > 0 if mask is None else (y > 0) & np.asarray(mask, bool)
    usable = np.flatnonzero(ok)
    if usable.size < 40:
        return float("nan")
    logy = np.log10(y[usable])
    qu = q[usable]
    qc2 = 16e-6 * np.pi * max(sld, 0.0) if sld is not None else 0.0
    z = np.arange(max(d_min, d_range[0]) - 1.0, d_range[1] + 2.0, 1.0)
    start = int(np.argmax(logy))  # skip the plateau and the critical edge
    idx = np.flatnonzero((np.arange(qu.size) > start + 5) & (qu**2 > 1.2 * qc2))
    if idx.size < 40:
        return float("nan")
    qq, yy = np.sqrt(qu[idx] ** 2 - qc2), logy[idx]
    yy = yy - np.polyval(np.polyfit(qq, yy, 5), qq)
    yy = yy * np.hanning(yy.size)
    amp = np.abs(np.exp(-1j * np.outer(z, qq)) @ yy)
    peak = (amp[1:-1] > amp[:-2]) & (amp[1:-1] >= amp[2:])
    return float(z[1:-1][peak][np.argmax(amp[1:-1][peak])]) if peak.any() else float("nan")
