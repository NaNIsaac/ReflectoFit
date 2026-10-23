"""Float64 Parratt recursion compiled with numba.

k_z² = (q/2)² − 4π(ρ − ρ_ambient) with ρ in Å⁻², Névot-Croce roughness. The SLD is conjugated so
that the principal square root gives Im k_z ≥ 0 (decay into absorbers) with the exp(+2i k_z d)
recursion; callers pass Im SLD > 0 for absorption.
"""

import math

import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit(cache=True)
def _curve_points(q, sld, thickness, roughness, start, stop, i, out):
    n = sld.shape[1]
    kz = np.empty(n, np.complex128)
    rho = np.empty(n, np.complex128)
    for j in range(n):
        rho[j] = 4.0e-6 * np.pi * np.conj(sld[i, j] - sld[i, 0])
    for iq in range(start, stop):
        h = 0.5 * q[i, iq]
        for j in range(n):
            kz[j] = np.sqrt(h * h - rho[j])
        x = 0j
        for j in range(n - 2, -1, -1):
            k1 = kz[j]
            k2 = kz[j + 1]
            s = roughness[i, j + 1]
            r = (k1 - k2) / (k1 + k2) * np.exp(-2.0 * k1 * k2 * s * s)
            if j + 1 < n - 1:
                p = np.exp(2j * k2 * thickness[i, j + 1])
                xp = x * p
            else:
                xp = 0j
            x = (r + xp) / (1.0 + r * xp)
        out[i, iq] = x.real * x.real + x.imag * x.imag


@numba.njit(parallel=True, cache=True)
def _parratt(q, sld, thickness, roughness, out):
    b, nq = q.shape
    chunk = 64
    per_curve = (nq + chunk - 1) // chunk
    for t in numba.prange(b * per_curve):
        i = t // per_curve
        start = (t % per_curve) * chunk
        _curve_points(q, sld, thickness, roughness, start, min(start + chunk, nq), i, out)


def parratt(
    q: NDArray[np.float64],
    sld: NDArray[np.complex128],
    thickness: NDArray[np.float64],
    roughness: NDArray[np.float64],
) -> NDArray[np.float64]:
    """|r|² for q (B, Q) and media arrays (B, N), ambient first. Returns (B, Q)."""
    q = np.ascontiguousarray(q, dtype=np.float64)
    out = np.empty(q.shape)
    _parratt(
        q,
        np.ascontiguousarray(sld, dtype=np.complex128),
        np.ascontiguousarray(thickness, dtype=np.float64),
        np.ascontiguousarray(roughness, dtype=np.float64),
        out,
    )
    return out


@numba.njit(cache=True)
def _reflectivity(qv, rho, thickness, roughness, i):
    """|r|² at one q for curve i (rho = 4π · conj(ρ − ρ_ambient) [Å⁻²], ambient first)."""
    n = rho.shape[1]
    h = 0.5 * qv
    k2 = np.sqrt(h * h - rho[i, n - 1])
    x = 0j
    for j in range(n - 2, -1, -1):
        k1 = np.sqrt(h * h - rho[i, j])
        s = roughness[i, j + 1]
        r = (k1 - k2) / (k1 + k2) * np.exp(-2.0 * k1 * k2 * s * s)
        xp = x * np.exp(2j * k2 * thickness[i, j + 1]) if j + 1 < n - 1 else 0j
        x = (r + xp) / (1.0 + r * xp)
        k2 = k1
    return x.real * x.real + x.imag * x.imag


@numba.njit(parallel=True, cache=True)
def _smear(
    q_true, sigma, starts, u, w, rho, thickness, roughness, wl1, wl2, ratio, length, beam, out
):
    b, nq = q_true.shape
    for t in numba.prange(b * nq):
        i, iq = t // nq, t % nq
        acc = 0.0
        for m in range(starts[t], starts[t + 1]):
            sin_t = min(
                max((q_true[i, iq] + sigma[i, iq] * u[m]) * wl1 / (4.0 * np.pi), 1e-12), 1.0
            )
            r = _reflectivity(4.0 * np.pi / wl1 * sin_t, rho, thickness, roughness, i)
            if wl2 > 0:
                r2 = _reflectivity(4.0 * np.pi / wl2 * sin_t, rho, thickness, roughness, i)
                r = (r + ratio[i] * r2) / (1.0 + ratio[i])
            if length[i] == length[i]:
                r *= math.erf(length[i] * sin_t / (2.0 * np.sqrt(2.0) * beam[i]))
            acc += w[m] * r
        out[i, iq] = acc


def smear(
    q_true: NDArray[np.float64],
    sigma: NDArray[np.float64],
    counts: NDArray[np.int64],
    u: NDArray[np.float64],
    w: NDArray[np.float64],
    sld: NDArray[np.complex128],
    thickness: NDArray[np.float64],
    roughness: NDArray[np.float64],
    wl1: float,
    wl2: float,
    ratio: NDArray[np.float64],
    length: NDArray[np.float64],
    beam: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Σ w · R over each point's nodes q_true + sigma · u, for (B, Q) points whose nodes are listed
    point after point (counts (B·Q,) nodes each): Kα₁ (and Kα₂ at the same angle when wl2 > 0, mixed
    by ratio), times the footprint factor where length is not NaN. Returns (B, Q)."""
    sld = np.ascontiguousarray(sld, dtype=np.complex128)
    rho = 4.0e-6 * np.pi * np.conj(sld - sld[:, :1])
    starts = np.concatenate([[0], np.cumsum(counts)])
    out = np.empty(q_true.shape)
    _smear(
        np.ascontiguousarray(q_true, dtype=np.float64),
        np.ascontiguousarray(sigma, dtype=np.float64),
        starts,
        np.ascontiguousarray(u, dtype=np.float64),
        np.ascontiguousarray(w, dtype=np.float64),
        rho,
        np.ascontiguousarray(thickness, dtype=np.float64),
        np.ascontiguousarray(roughness, dtype=np.float64),
        float(wl1),
        float(wl2),
        np.ascontiguousarray(ratio, dtype=np.float64),
        np.ascontiguousarray(length, dtype=np.float64),
        np.ascontiguousarray(beam, dtype=np.float64),
        out,
    )
    return out
