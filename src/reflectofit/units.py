"""Angles, wavelengths and resolution widths. Lengths in Å, angles in degrees."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

CU_KALPHA1 = 1.5406
"""Cu Kα₁ wavelength [Å], the default everywhere."""

CU_KALPHA2 = 1.54439
"""Cu Kα₂ wavelength [Å]."""

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def tth_to_q(tth: ArrayLike, wavelength: float = CU_KALPHA1) -> NDArray[np.float64] | float:
    """Scattering vector q = 4π/λ sin(2θ/2) [Å⁻¹] from 2θ [deg]."""
    q = 4.0 * np.pi / wavelength * np.sin(np.radians(np.asarray(tth, dtype=float) / 2.0))
    return float(q) if q.ndim == 0 else q


def q_to_tth(q: ArrayLike, wavelength: float = CU_KALPHA1) -> NDArray[np.float64] | float:
    """2θ [deg] from q [Å⁻¹]."""
    s = np.asarray(q, dtype=float) * wavelength / (4.0 * np.pi)
    tth = 2.0 * np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))
    return float(tth) if tth.ndim == 0 else tth


def fwhm_to_sigma(fwhm: ArrayLike) -> NDArray[np.float64] | float:
    """Gaussian standard deviation from a full width at half maximum."""
    s = np.asarray(fwhm, dtype=float) / FWHM_PER_SIGMA
    return float(s) if s.ndim == 0 else s
