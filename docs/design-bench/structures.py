"""Reference structures and the float64 numpy Parratt used as the common baseline.

Layer tuple: (sld [1e-6 A^-2, complex, +imag = absorption], thickness [A], roughness of the
interface above this medium [A]). Ambient first, substrate last.
"""

import numpy as np

WAVELENGTH = 1.5406
SI = 20.07 + 0.46j
SIO2 = 18.9 + 0.24j


def q_grid(n: int = 2000) -> np.ndarray:
    tth = np.linspace(0.05, 8.0, n)
    return 4 * np.pi / WAVELENGTH * np.sin(np.radians(tth / 2))


def structures() -> dict[str, list[tuple[complex, float, float]]]:
    amb = (0j, 0.0, 0.0)
    film = (45 + 1.5j, 240.0, 6.0)
    sl = []
    for _ in range(10):
        sl += [(30 + 0.5j, 25.0, 3.0), (55 + 2.0j, 35.0, 3.5)]
    return {
        "single": [amb, film, (SI, 0.0, 3.0)],
        "oxide_under": [amb, film, (SIO2, 10.0, 3.0), (SI, 0.0, 3.0)],
        "bilayer": [
            amb,
            (30 + 0.5j, 120.0, 4.0),
            (55 + 2.0j, 200.0, 5.0),
            (SIO2, 10.0, 3.0),
            (SI, 0.0, 3.0),
        ],
        "superlattice": [amb, (40 + 1.0j, 20.0, 4.0), *sl, (SIO2, 10.0, 3.0), (SI, 0.0, 3.0)],
    }


def arrays(layers):
    sld = np.array([lay[0] for lay in layers], complex)
    d = np.array([lay[1] for lay in layers], float)
    sig = np.array([lay[2] for lay in layers], float)
    return sld, d, sig


def parratt(q: np.ndarray, sld: np.ndarray, d: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Batched Parratt. q (Q,), sld/d/sig (B, N) or (N,). Same recursion as ReflectoLearn benchmark.parratt."""
    sld = np.atleast_2d(sld)
    d = np.atleast_2d(d)
    sig = np.atleast_2d(sig)
    rho = np.conj(sld - sld[:, :1]) * 1e-6
    kz = np.sqrt((q[None, :, None] / 2) ** 2 - 4 * np.pi * rho[:, None, :] + 0j)  # (B, Q, N)
    n = sld.shape[1]
    x = np.zeros(kz.shape[:2], complex)
    for j in range(n - 2, -1, -1):
        k1, k2 = kz[..., j], kz[..., j + 1]
        r = (k1 - k2) / (k1 + k2) * np.exp(-2 * k1 * k2 * sig[:, None, j + 1] ** 2)
        dj = d[:, None, j + 1] if j + 1 < n - 1 else 0.0
        p = np.exp(2j * k2 * dj)
        x = (r + x * p) / (1 + r * x * p)
    return np.abs(x) ** 2


def abeles_np(q, sld, d, sig):
    """Batched Abeles matrix product in complex128. Takes Im SLD > 0 directly: this kernel uses exp(+i k d)
    with Im k < 0, the mirror of the Parratt convention, so the SLD is not conjugated."""
    sld = np.atleast_2d(sld)
    d = np.atleast_2d(d)
    sig = np.atleast_2d(sig)
    rho = (sld - sld[:, :1]) * 1e-6
    k = np.sqrt((q[None, :, None] / 2) ** 2 - 4 * np.pi * rho[:, None, :] + 0j)
    n = sld.shape[1]
    m00 = np.ones(k.shape[:2], complex)
    m01 = np.zeros_like(m00)
    m10 = np.zeros_like(m00)
    m11 = np.ones_like(m00)
    for j in range(n - 1):
        k1, k2 = k[..., j], k[..., j + 1]
        r = (k1 - k2) / (k1 + k2) * np.exp(-2 * k1 * k2 * sig[:, None, j + 1] ** 2)
        b = np.exp(1j * k1 * d[:, None, j]) if j > 0 else 1.0
        bi = 1 / b
        a00, a01, a10, a11 = b, r * b, r * bi, bi
        m00, m01, m10, m11 = (
            m00 * a00 + m01 * a10,
            m00 * a01 + m01 * a11,
            m10 * a00 + m11 * a10,
            m10 * a01 + m11 * a11,
        )
    return np.abs(m10 / m00) ** 2


def parratt_exact(q, sld, d, sig, wavelength=WAVELENGTH):
    """Parratt with k_z from the exact n^2 (as GenX ReflQ) instead of the linear 1 - 2 delta."""
    sld = np.atleast_2d(sld)
    d = np.atleast_2d(d)
    sig = np.atleast_2d(sig)
    n = 1 - wavelength**2 * np.conj(sld) * 1e-6 / (2 * np.pi)
    q0 = 4 * np.pi / wavelength
    kz = 0.5 * np.sqrt(
        (n[:, None, :] ** 2 - n[:, None, :1] ** 2) * q0**2
        + n[:, None, :1] ** 2 * q[None, :, None] ** 2
        + 0j
    )
    nn = sld.shape[1]
    x = np.zeros(kz.shape[:2], complex)
    for j in range(nn - 2, -1, -1):
        k1, k2 = kz[..., j], kz[..., j + 1]
        r = (k1 - k2) / (k1 + k2) * np.exp(-2 * k1 * k2 * sig[:, None, j + 1] ** 2)
        dj = d[:, None, j + 1] if j + 1 < nn - 1 else 0.0
        x = (r + x * np.exp(2j * k2 * dj)) / (1 + r * x * np.exp(2j * k2 * dj))
    return np.abs(x) ** 2
