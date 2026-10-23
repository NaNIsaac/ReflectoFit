"""Quick estimators read directly off a curve, without a fit."""

import numpy as np
from numpy.typing import NDArray


def critical_q(q: NDArray, y: NDArray) -> float:
    """q of the critical edge: the steepest descent of log R above the plateau (the minimum of
    d log10 R / dq, refined by a parabola through it and its neighbours)."""
    q = np.asarray(q, float)
    logy = np.log10(np.clip(np.asarray(y, float), np.finfo(float).tiny, None))
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
) -> float:
    """Film thickness [Å] from the Kiessig period: the strongest peak of the real-space transform
    of the detrended log R (port of ReflectoLearn `refine.fft_thickness`, b6fb0c4).

    Fringes are periodic in the film's own wavevector sqrt(q² − q_c²); with `sld` [1e-6 Å⁻²] the
    transform runs over it, otherwise over q (which overestimates d by a few percent near the edge).
    NaN with fewer than 40 usable points or no peak in [max(d_min, lo), hi].
    """
    q = np.asarray(q, float)
    logy = np.log10(np.clip(np.asarray(y, float), np.finfo(float).tiny, None))
    qc2 = 16e-6 * np.pi * max(sld, 0.0) if sld is not None else 0.0
    z = np.arange(max(d_min, d_range[0]) - 1.0, d_range[1] + 2.0, 1.0)
    start = int(np.argmax(logy))  # skip the plateau and the critical edge
    idx = np.flatnonzero((np.arange(q.size) > start + 5) & (q**2 > 1.2 * qc2))
    if idx.size < 40:
        return float("nan")
    qq, yy = np.sqrt(q[idx] ** 2 - qc2), logy[idx]
    yy = yy - np.polyval(np.polyfit(qq, yy, 5), qq)
    yy = yy * np.hanning(yy.size)
    amp = np.abs(np.exp(-1j * np.outer(z, qq)) @ yy)
    peak = (amp[1:-1] > amp[:-2]) & (amp[1:-1] >= amp[2:])
    return float(z[1:-1][peak][np.argmax(amp[1:-1][peak])]) if peak.any() else float("nan")
