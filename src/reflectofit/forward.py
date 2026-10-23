"""Observed reflectivity: the layer kernel plus the instrument.

The nominal 2θ plus the alignment offset is the true angle; the resolution is a Gaussian in q around
the q of that angle. At each quadrature node: R for Kα₁ (and Kα₂ at the same angle, mixed by intensity
ratio) times the footprint factor at that angle. The weighted sum over nodes is scaled by I0 and the
background is added (DESIGN.md section 2). Every instrument field may be a scalar or a (B,) array, one value per curve.
"""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf

from reflectofit import resolution
from reflectofit.kernel import parratt, smear
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
    """None: each point gets the node count its resolution width, the fringes and any nearby
    critical edge need (`reflectofit.resolution`). An int: that many nodes for every point, faster
    and of the accuracy the count allows; also the way to vmap over resolution parameters."""

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
    exp = staticmethod(np.exp)
    cat = staticmethod(np.concatenate)

    @staticmethod
    def column(x, b):
        return np.broadcast_to(np.asarray(x, dtype=float), (b,)).reshape(b, 1, 1)

    @staticmethod
    def take_rows(values, idx):
        """values (B, Q) gathered at idx (B, M)."""
        return np.take_along_axis(values, idx, axis=1)

    @staticmethod
    def sum_rows(values, idx, width):
        """(B, width) sums of values (B, M) by idx (B, M)."""
        b = values.shape[0]
        flat = (idx + width * np.arange(b)[:, None]).ravel()
        return np.bincount(flat, values.ravel(), minlength=b * width).reshape(b, width)


def _numpy(x) -> np.ndarray:
    """Values of an array or tensor as numpy, without its graph (also under torch.func.jacfwd)."""
    if not hasattr(x, "detach"):
        return np.asarray(x)
    x = x.detach()
    try:
        return x.cpu().numpy()
    except RuntimeError:  # a functorch-wrapped tensor has no storage; its values are still readable
        return np.array(x.tolist())


def _active(x) -> bool:
    """Whether an instrument term is switched on. A tensor counts as on: its value may be traced
    (torch.func.jacfwd, vmap), so no branch may depend on it."""
    if x is None:
        return False
    if hasattr(x, "detach"):
        return True
    return bool(np.any(np.asarray(x) != 0))


ROWS_PER_BLOCK = 256
"""Curves per block of the resolution integral (bounds the flat node arrays)."""
NODES_PER_CALL = 2_000_000
"""Padded resolution nodes per kernel call (bounds the memory of a block)."""


def _at_angles(ops, kernel, x, packed, inst: Instrument):
    """R at nominal-true q values x (B, M): Kα₁ (and Kα₂ at the same angle) times the footprint."""
    b = x.shape[0]
    wl1 = inst.wavelength
    # a point below q = 0 is the mirror image of a grazing angle; keep it at grazing incidence
    sin_t = ops.clip(x * (wl1 / (4.0 * np.pi)), 1e-12, 1.0)
    r = kernel((4.0 * np.pi / wl1) * sin_t, *packed)
    if _active(inst.kalpha2_ratio):
        a = ops.column(inst.kalpha2_ratio, b)[..., 0]
        r2 = kernel((4.0 * np.pi / inst.kalpha2_wavelength) * sin_t, *packed)
        r = (r + a * r2) / (1.0 + a)
    if inst.sample_length is not None and inst.beam_sigma is not None:
        length = ops.column(inst.sample_length, b)[..., 0]
        beam = ops.column(inst.beam_sigma, b)[..., 0]
        r = r * ops.erf(length * sin_t / (2.0 * np.sqrt(2.0) * beam))
    return r


def _point_smear(ops, kernel, q_true, sigma, packed, inst: Instrument):
    """Resolution with a node count per point (`reflectofit.resolution`) for one block of curves:
    all nodes of a curve in one flat list, one kernel call, summed back per point."""
    b, nq = q_true.shape[0], q_true.shape[1]
    qt, sg = _numpy(q_true)[..., 0], _numpy(sigma)[..., 0]
    sld, thickness = _numpy(packed[0]), _numpy(packed[1])
    ka2 = inst.kalpha2_wavelength / inst.wavelength if _active(inst.kalpha2_ratio) else None
    curves = [resolution.edges(sld[i], thickness[i], ka2) for i in range(b)]
    counts, u_all, w_all = resolution.layout(qt, sg, curves)
    if ops is _NumpyOps:
        # one fused numba loop over the points and their nodes (no padding, no temporaries)
        footprint = inst.sample_length is not None and inst.beam_sigma is not None
        nan = np.full(b, np.nan)
        return smear(
            qt,
            sg,
            counts,
            u_all,
            w_all,
            *packed,
            inst.wavelength,
            inst.kalpha2_wavelength if ka2 is not None else 0.0,
            ops.column(inst.kalpha2_ratio, b).ravel(),
            ops.column(inst.sample_length, b).ravel() if footprint else nan,
            ops.column(inst.beam_sigma, b).ravel() if footprint else nan,
        )
    # per curve: owner point of each node, its u and weight
    ends = np.cumsum(counts.reshape(b, nq).sum(axis=1))
    layouts = [
        (np.repeat(np.arange(nq), c), x, y)
        for c, x, y in zip(
            counts.reshape(b, nq),
            np.split(u_all, ends[:-1]),
            np.split(w_all, ends[:-1]),
            strict=True,
        )
    ]
    sizes = [o.size for o, _, _ in layouts]
    out = []
    start = 0
    while start < b:
        # curves in one kernel call while the padded node array stays within NODES_PER_CALL
        stop, m = start + 1, sizes[start]
        while stop < b and max(m, sizes[stop]) * (stop + 1 - start) <= NODES_PER_CALL:
            m = max(m, sizes[stop])
            stop += 1
        rows = slice(start, stop)
        part = layouts[rows]
        # padding nodes sit on point 0 with weight 0, so a curve's result does not depend on its batch
        owner = np.stack([np.pad(o, (0, m - o.size)) for o, _, _ in part])
        u = np.stack([np.pad(x, (0, m - x.size)) for _, x, _ in part])
        w = np.stack([np.pad(x, (0, m - x.size)) for _, _, x in part])
        nodes = ops.take_rows(q_true[rows, :, 0], owner) + ops.take_rows(
            sigma[rows, :, 0], owner
        ) * ops.asarray(u)
        r = _at_angles(ops, kernel, nodes, tuple(a[rows] for a in packed), _rows(inst, rows, b))
        out.append(ops.sum_rows(r * ops.asarray(w), owner, nq))
        start = stop
    return ops.cat(out) if len(out) > 1 else out[0]


def _observed(ops, kernel, q, b, packed, inst: Instrument):
    """Shared pipeline. q (Q,) nominal; returns (B, Q).

    The offset comes first: the true angle is the nominal 2θ plus the offset, and the resolution
    is a Gaussian in q around the q of that true angle. Kα₁, Kα₂ and the footprint are evaluated at
    the angle of each point of the resolution integral.
    """
    q = q.reshape(1, -1, 1)
    wl1 = inst.wavelength
    to_deg = 180.0 / np.pi
    theta = ops.arcsin(ops.clip(q * (wl1 / (4.0 * np.pi)), 0.0, 1.0)) * to_deg
    theta = ops.clip(theta + 0.5 * ops.column(inst.tth_offset, b), 1e-9, 0.5 * MAX_TTH)
    q_true = (4.0 * np.pi / wl1) * ops.sin(theta / to_deg)  # (B, Q, 1)
    resolved = _active(inst.sigma_q) or _active(inst.dq_over_q)
    if not resolved or inst.nodes == 1:
        r = _at_angles(ops, kernel, q_true[..., 0], packed, inst)
    else:
        sigma2 = ops.column(inst.sigma_q, b) ** 2 + (ops.column(inst.dq_over_q, b) * q_true) ** 2
        # the floor keeps d sqrt / d sigma finite (zero) for a curve without resolution
        sigma = ops.sqrt(ops.clip(sigma2, 1e-300, None))
        if inst.nodes is None:
            blocks = []
            for start in range(0, b, ROWS_PER_BLOCK):
                rows = slice(start, min(start + ROWS_PER_BLOCK, b))
                part = tuple(a[rows] for a in packed)
                blocks.append(
                    _point_smear(ops, kernel, q_true[rows], sigma[rows], part, _rows(inst, rows, b))
                )
            r = ops.cat(blocks) if len(blocks) > 1 else blocks[0]
        else:
            u = np.linspace(-resolution.SPAN, resolution.SPAN, inst.nodes)
            w = np.exp(-0.5 * u**2)
            nodes = q_true + sigma * ops.asarray(u).reshape(1, 1, -1)
            shape = nodes.shape
            r = _at_angles(ops, kernel, nodes.reshape(b, -1), packed, inst).reshape(shape)
            r = (r * ops.asarray(w / w.sum()).reshape(1, 1, -1)).sum(-1)
    return ops.column(inst.i0, b)[..., 0] * r + ops.column(inst.background, b)[..., 0]


def _rows(inst: Instrument, rows: slice, total: int) -> Instrument:
    """The instrument for a block of curves (per-curve fields sliced)."""
    changes = {}
    for f in fields(inst):
        v = getattr(inst, f.name)
        per_curve = v is not None and np.ndim(v) == 1 and len(v) == total
        if f.name not in ("wavelength", "kalpha2_wavelength", "nodes") and per_curve:
            changes[f.name] = v[rows]
    return inst.replace(**changes) if changes else inst


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
