"""Observed reflectivity: the layer kernel plus the instrument.

For each quadrature node of the Gaussian resolution in q: the nominal 2θ plus the alignment offset
gives the true angle; R at that angle for Kα₁ (and Kα₂, mixed by intensity ratio) times the footprint
factor at that angle. The weighted sum over nodes is scaled by I0 and the background is added
(DESIGN.md section 2). Every instrument field may be a scalar or a (B,) array, one value per curve.
"""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf

from reflectofit.kernel import parratt
from reflectofit.structure import as_batch
from reflectofit.units import CU_KALPHA1, CU_KALPHA2

MAX_TTH = 179.0


@dataclass(frozen=True)
class Instrument:
    """Instrument model. Angles in degrees, lengths in mm, resolution as Gaussian σ in q."""

    wavelength: float = CU_KALPHA1
    i0: Any = 1.0
    background: Any = 0.0
    tth_offset: Any = 0.0
    sigma_q: Any = 0.0
    """Constant resolution term, Gaussian σ [Å⁻¹]."""
    dq_over_q: Any = 0.0
    """Relative resolution term: σ = dq_over_q · q. Combined in quadrature with sigma_q."""
    kalpha2_ratio: Any = 0.0
    """Kα₂/Kα₁ intensity ratio at the detector; 0 for a monochromatic beam."""
    kalpha2_wavelength: float = CU_KALPHA2
    sample_length: Any = None
    """Sample length along the beam [mm]; with beam_sigma enables the footprint factor."""
    beam_sigma: Any = None
    """Gaussian σ of the beam profile at the sample [mm] (FWHM / 2.3548), beam centred."""
    nodes: int = 17
    """Gauss-Hermite nodes for the resolution integral."""

    def replace(self, **changes) -> "Instrument":
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        return Instrument(**(values | changes))


class _NumpyOps:
    asarray = staticmethod(np.asarray)
    sqrt = staticmethod(np.sqrt)
    sin = staticmethod(np.sin)
    arcsin = staticmethod(np.arcsin)
    clip = staticmethod(np.clip)
    erf = staticmethod(erf)

    @staticmethod
    def column(x, b):
        return np.broadcast_to(np.asarray(x, dtype=float), (b,)).reshape(b, 1, 1)


def _observed(ops, kernel, q, b, packed, inst: Instrument):
    """Shared pipeline. q (Q,) nominal; returns (B, Q)."""
    q = q.reshape(1, -1, 1)
    sigma2 = ops.column(inst.sigma_q, b) ** 2 + (ops.column(inst.dq_over_q, b) * q) ** 2
    if inst.nodes > 1 and bool((sigma2 > 0).any()):
        x, w = np.polynomial.hermite.hermgauss(inst.nodes)
        nodes = q + np.sqrt(2.0) * ops.sqrt(sigma2) * ops.asarray(x).reshape(1, 1, -1)
        weights = ops.asarray(w / np.sqrt(np.pi)).reshape(1, 1, -1)
    else:
        nodes = q + 0.0 * sigma2
        weights = None
    wl1 = inst.wavelength
    s1 = ops.clip(nodes * (wl1 / (4.0 * np.pi)), 0.0, 1.0)
    theta_deg = ops.arcsin(s1) * (180.0 / np.pi) + 0.5 * ops.column(inst.tth_offset, b)
    theta = ops.clip(theta_deg, 1e-9, 0.5 * MAX_TTH) * (np.pi / 180.0)
    sin_t = ops.sin(theta)
    shape = sin_t.shape
    r = kernel((4.0 * np.pi / wl1) * sin_t.reshape(b, -1), *packed).reshape(shape)
    if inst.kalpha2_ratio is not None and bool(np.any(np.asarray(_detach(inst.kalpha2_ratio)) > 0)):
        a = ops.column(inst.kalpha2_ratio, b)
        r2 = kernel((4.0 * np.pi / inst.kalpha2_wavelength) * sin_t.reshape(b, -1), *packed)
        r = (r + a * r2.reshape(shape)) / (1.0 + a)
    if inst.sample_length is not None and inst.beam_sigma is not None:
        arg = (
            ops.column(inst.sample_length, b)
            * sin_t
            / (2.0 * np.sqrt(2.0) * ops.column(inst.beam_sigma, b))
        )
        r = r * ops.erf(arg)
    r = (r * weights).sum(-1) if weights is not None else r[..., 0]
    return ops.column(inst.i0, b)[..., 0] * r + ops.column(inst.background, b)[..., 0]


def _detach(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else x


def forward(
    q: ArrayLike, model, instrument: Instrument | None = None, backend: str = "numba"
) -> NDArray[np.float64]:
    """Observed reflectivity for one model (returns (Q,)) or a batch (returns (B, Q)).

    `model` is a Stack, a packed (sld, thickness, roughness) tuple of (N,) or (B, N) arrays, or a
    sequence of either. Without an instrument this is the bare |r|² of the stack.
    """
    if backend == "torch":
        from reflectofit.backends.torch import forward as torch_forward

        return torch_forward(q, model, instrument)
    if backend != "numba":
        raise ValueError(f"unknown backend {backend!r}; use 'numba' or 'torch'")
    packed, single = as_batch(model)
    q = np.asarray(q, dtype=float)
    if q.ndim != 1:
        raise ValueError("q must be one-dimensional; all curves of a batch share the q grid")
    b = packed[0].shape[0]
    if instrument is None:
        r = parratt(np.broadcast_to(q, (b, q.size)), *packed)
    else:
        r = _observed(_NumpyOps, parratt, q, b, packed, instrument)
    return r[0] if single else r
