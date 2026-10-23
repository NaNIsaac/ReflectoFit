"""Float64 Parratt recursion compiled with numba.

k_z² = (q/2)² − 4π(ρ − ρ_ambient) with ρ in Å⁻², Névot-Croce roughness. The SLD is conjugated so
that the principal square root gives Im k_z ≥ 0 (decay into absorbers) with the exp(+2i k_z d)
recursion; callers pass Im SLD > 0 for absorption.
"""

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
