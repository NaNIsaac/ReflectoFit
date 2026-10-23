"""Observed reflectivity: the layer kernel plus the instrument.

The nominal 2θ plus the alignment offset is the true angle; the resolution is a Gaussian in q around
the q of that angle. At each quadrature node: R for Kα₁ (and Kα₂ at the same angle, mixed by intensity
ratio) times the footprint factor at that angle. The weighted sum over nodes is scaled by I0 and the
background is added (DESIGN.md section 2). Every instrument field may be a scalar or a (B,) array, one value per curve.
"""

import warnings
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
    nodes: int | None = None
    """Nodes of the resolution integral; None chooses them from the resolution width and the stack's
    total thickness (`resolution_nodes`)."""

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


def _active(x) -> bool:
    """Whether an instrument term is switched on. A tensor counts as on: its value may be traced
    (torch.func.jacfwd, vmap), so no branch may depend on it."""
    if x is None:
        return False
    if hasattr(x, "detach"):
        return True
    return bool(np.any(np.asarray(x) != 0))


NODE_SPAN = 6.5
"""The resolution integral runs over ±NODE_SPAN standard deviations."""
NODE_DQ = 3e-4
"""Largest node spacing in q [Å⁻¹]; resolves the square-root kink of R at a critical edge."""
MAX_NODES = 401


def resolution_nodes(sigma_max: float, thickness: float) -> int:
    """Node count of the uniform (Gaussian-weighted trapezoid) resolution rule.

    Spacing h, in units of σ, is the smallest of 0.5, NODE_DQ / σ and 2π / (2 σ T + 8): the last
    keeps the rule exact for fringes of a film of total thickness T up to well past the point where
    the resolution damps them. Against a brute-force convolution this stays within 1e-8 relative for
    σ up to 2e-3 Å⁻¹ and T up to 4000 Å (physics review on ReflectoFit#2). Above MAX_NODES the count
    is capped with a warning.
    """
    if sigma_max <= 0:
        return 1
    h = min(0.5, NODE_DQ / sigma_max, 2 * np.pi / (2 * sigma_max * thickness + 8))
    n = 2 * int(np.ceil(NODE_SPAN / h)) + 1
    if n > MAX_NODES:
        warnings.warn(
            f"resolution needs {n} nodes (σ_q up to {sigma_max:.2g} Å⁻¹, thickness {thickness:.0f} Å);"
            f" capped at {MAX_NODES}, the smeared curve may be inaccurate",
            stacklevel=3,
        )
        n = MAX_NODES
    return n


def _value(x) -> float:
    if hasattr(x, "detach"):
        x = x.detach()
    return float(x.max()) if hasattr(x, "max") else float(x)


def _observed(ops, kernel, q, b, packed, inst: Instrument):
    """Shared pipeline. q (Q,) nominal; returns (B, Q).

    The offset comes first: the true angle is the nominal 2θ plus the offset, and the resolution
    is a Gaussian in q around the q of that true angle. Each quadrature node is an angle at which
    Kα₁, Kα₂ and the footprint are evaluated.
    """
    q = q.reshape(1, -1, 1)
    wl1 = inst.wavelength
    to_deg = 180.0 / np.pi
    theta = ops.arcsin(ops.clip(q * (wl1 / (4.0 * np.pi)), 0.0, 1.0)) * to_deg
    theta = ops.clip(theta + 0.5 * ops.column(inst.tth_offset, b), 1e-9, 0.5 * MAX_TTH)
    q_true = (4.0 * np.pi / wl1) * ops.sin(theta / to_deg)
    if inst.nodes != 1 and (_active(inst.sigma_q) or _active(inst.dq_over_q)):
        sigma2 = ops.column(inst.sigma_q, b) ** 2 + (ops.column(inst.dq_over_q, b) * q_true) ** 2
        n = inst.nodes
        if n is None:
            # node count from values: fine under jacfwd; under vmap over the resolution pass nodes
            total = _value(packed[1].sum(-1))
            n = resolution_nodes(_value(sigma2) ** 0.5, total)
        u = np.linspace(-NODE_SPAN, NODE_SPAN, n)
        w = np.exp(-0.5 * u**2)
        # the floor keeps d sqrt / d sigma finite (zero) for a curve without resolution
        sigma = ops.sqrt(ops.clip(sigma2, 1e-300, None))
        nodes = q_true + sigma * ops.asarray(u).reshape(1, 1, -1)
        weights = ops.asarray(w / w.sum()).reshape(1, 1, -1)
    else:
        nodes = q_true
        weights = None
    # a node below q = 0 is the mirror image of a grazing angle; keep it at grazing incidence
    sin_t = ops.clip(nodes * (wl1 / (4.0 * np.pi)), 1e-12, 1.0)
    shape = sin_t.shape
    r = kernel((4.0 * np.pi / wl1) * sin_t.reshape(b, -1), *packed).reshape(shape)
    if _active(inst.kalpha2_ratio):
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


def forward(
    q: ArrayLike, model, instrument: Instrument | None = None, backend: str = "numba"
) -> NDArray[np.float64]:
    """Observed reflectivity for one model (returns (Q,)) or a batch (returns (B, Q)).

    `model` is a Stack, a packed (sld, thickness, roughness) tuple of (N,) or (B, N) arrays, or a
    sequence of either. Without an instrument this is the bare |r|² of the stack.
    """
    if backend == "torch":
        from reflectofit.torch_backend import forward as torch_forward

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
