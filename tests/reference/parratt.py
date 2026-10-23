"""The specification: the Parratt recursion in plain numpy, float64, one structure at a time.

Same recursion as ReflectoLearn `benchmark.parratt` (b6fb0c4). Kept readable rather than fast; the
package kernel must match it to 1e-10 relative (DESIGN.md section 5.1).
"""

import numpy as np


def parratt(q, sld, thickness, roughness):
    """|r|² for media ambient first; sld in 1e-6 Å^-2 with Im > 0 for absorption."""
    rho = np.conj(np.asarray(sld, complex) - sld[0]) * 1e-6
    kz = np.sqrt((np.asarray(q)[:, None] / 2) ** 2 - 4 * np.pi * rho[None, :] + 0j)
    n = len(rho)
    x = np.zeros(len(q), complex)
    for j in range(n - 2, -1, -1):
        k1, k2 = kz[:, j], kz[:, j + 1]
        r = (k1 - k2) / (k1 + k2) * np.exp(-2 * k1 * k2 * roughness[j + 1] ** 2)
        d = thickness[j + 1] if j + 1 < n - 1 else 0.0
        p = np.exp(2j * k2 * d)
        x = (r + x * p) / (1 + r * x * p)
    return np.abs(x) ** 2
