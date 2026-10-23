"""Nodes and weights for the resolution integral, chosen per measured point.

Each point integrates R against its Gaussian over ±SPAN σ. R has a square-root kink at the critical
edge q_c of every medium (kz = √(q² − q_c²)/2 changes from imaginary to real there; absorption
rounds it over a width 8π Im ρ / q_c). Uniform nodes converge slowly on such a kink, so the window
is split at every edge inside it (Kα₂ edges included), and each piece [l, r] is mapped from s in
[0, 1] so that the distance to an edge end goes as s²:

- edge at both ends: u = l + (r − l) sin²(π s / 2);
- edge at the left end: u = l + (r − l) s²; at the right end: u = r − (r − l) (1 − s)²;
- no edge: u = l + (r − l) s.

R is then smooth in s, and each piece gets Gauss–Legendre panels of PANEL nodes, as many as the
larger of: its fringe phase change T (√(q_r² − q_e²) − √(q_l² − q_e²)) over PHASE_PER_PANEL, with
T the total film thickness and q_e the highest edge at or below the piece; and its length over
GAUSS_PER_PANEL σ. Where a medium absorbs, waveguide resonances between the lowest and highest
critical q are about as narrow as the narrowest absorbing edge width w, and the pieces there get
nodes RESONANCE · w apart (in q). So only points near edges pay for them, and the count depends only
on the curve's own stack and instrument, never on the rest of a batch.

The counts and node positions are read from parameter values and are not traced; gradients come
through R, the node positions q + σ u and the weights.
"""

from functools import cache
from typing import NamedTuple

import numba
import numpy as np

SPAN = 6.5
"""Nodes run over ±SPAN σ."""
PANEL = 16
"""Gauss–Legendre nodes per panel."""
PHASE_PER_PANEL = 6.0
"""Fringe phase [rad] one panel resolves."""
GAUSS_PER_PANEL = 3.0
"""Length of a panel in units of σ, at most."""
RESONANCE = 0.5
"""Node spacing between the lowest and highest critical q, in units of the narrowest absorbing
edge width.

Review (physics, ReflectoFit#2): at 2.0 a Ni/C/Ni waveguide with sigma_q = 1e-4 was off by 1.6e-3;
the resonances there are narrower than one edge width, so the spacing is a quarter of it."""
MAX_POINT_NODES = 8001
SPLIT_AT_EDGES = True
"""False integrates across the edges with plain panels (for tests that show the split matters)."""


class Edges(NamedTuple):
    """The critical edges of one curve: q_c [Å⁻¹] of every medium denser than the ambient, its edge
    width 8π Im ρ / q_c (0 for a real SLD), its thickness (0 for the substrate) and its image (0 for
    Kα₁, 1 for the Kα₂ copy at q_c λ₂/λ₁ in nominal q); `light` is the total thickness of the layers
    not denser than the ambient, which have no edge."""

    qc: np.ndarray
    width: np.ndarray
    depth: np.ndarray
    image: np.ndarray
    light: float


def edges(sld: np.ndarray, thickness: np.ndarray, kalpha2_scale: float | None = None) -> Edges:
    """The edges of one curve (sld, thickness (N,), ambient first; 1e-6 Å⁻², Å). With
    `kalpha2_scale` = λ₂/λ₁ the Kα₂ edges are added."""
    rho = (sld[1:] - sld[0]) * 1e-6
    dense = rho.real > 0
    qc = 4.0 * np.sqrt(np.pi * rho.real[dense])
    width = 8.0 * np.pi * rho.imag[dense] / qc
    depth = np.asarray(thickness[1:], float)[dense]
    light = float(np.asarray(thickness[1:], float)[~dense].sum())
    image = np.zeros(qc.size, int)
    if kalpha2_scale is not None and qc.size:
        qc = np.concatenate([qc, qc * kalpha2_scale])
        width = np.concatenate([width, width * kalpha2_scale])
        depth = np.concatenate([depth, depth])
        image = np.concatenate([image, image + 1])
    return Edges(qc, width, depth, image, light)


@cache
def _gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


@numba.njit(parallel=True, cache=True)
def _pieces(q, sigma, qc, n_edges, depth, image, light, n_images, spacing, split, consts):
    """Pieces of every point's window, points of all curves flattened (B·Q, max pieces): ends in u,
    edge flags and panel counts. Edge arrays are (B, max edges), the first n_edges[b] valid."""
    span, phase_per, gauss_per, panel, cap = consts
    n_b, n_q = q.shape
    m = qc.shape[1] + 1
    lo = np.zeros((n_b * n_q, m))
    hi = np.zeros((n_b * n_q, m))
    le = np.zeros((n_b * n_q, m), np.bool_)
    re = np.zeros((n_b * n_q, m), np.bool_)
    panels = np.zeros((n_b * n_q, m), np.int64)
    for t in numba.prange(n_b * n_q):
        c, i = t // n_q, t % n_q
        n_e = n_edges[c]
        bp = np.empty(n_e + 2)
        edge = np.empty(n_e + 2, np.bool_)
        change = np.empty(n_images[c])
        qmin, qmax = np.inf, -np.inf
        for j in range(n_e):
            qmin, qmax = min(qmin, qc[c, j]), max(qmax, qc[c, j])
        # breakpoints: the window ends and the edges strictly inside, in order (insertion sort)
        bp[0], edge[0] = -span, False
        nb = 1
        for j in range(n_e):
            u = (qc[c, j] - q[c, i]) / sigma[c, i]
            if split and -span < u < span:
                k = nb
                while k > 1 and bp[k - 1] > u:
                    bp[k], edge[k] = bp[k - 1], edge[k - 1]
                    k -= 1
                bp[k], edge[k] = u, True
                nb += 1
        bp[nb], edge[nb] = span, False
        nb += 1
        total = 0
        for p in range(nb - 1):
            a, b = bp[p], bp[p + 1]
            if b <= a:
                continue
            ql = max(q[c, i] + sigma[c, i] * a, 0.0)
            qr = max(q[c, i] + sigma[c, i] * b, 0.0)
            # every layer's phase or evanescent decay change across the piece; images apart
            change[:] = 0.0
            for j in range(n_e):
                x, y = ql * ql - qc[c, j] ** 2, qr * qr - qc[c, j] ** 2
                dr = (np.sqrt(y) if y > 0 else 0.0) - (np.sqrt(x) if x > 0 else 0.0)
                di = (np.sqrt(-y) if y < 0 else 0.0) - (np.sqrt(-x) if x < 0 else 0.0)
                change[image[c, j]] += depth[c, j] * np.hypot(dr, di)
            phase = light[c] * (qr - ql) + (change.max() if n_e else 0.0)
            stretch = 1.0
            if edge[p] and edge[p + 1]:
                stretch = 0.5 * np.pi
            elif edge[p] or edge[p + 1]:
                stretch = 2.0
            n = max(
                np.ceil(stretch * phase / phase_per), np.ceil(stretch * (b - a) / gauss_per), 1.0
            )
            if spacing[c] > 0 and ql <= qmax and qr >= qmin:
                n = max(n, np.ceil((qr - ql) / (spacing[c] * panel)))
            lo[t, p], hi[t, p], le[t, p], re[t, p] = a, b, edge[p], edge[p + 1]
            panels[t, p] = int(n)
            total += int(n)
        if total > cap:
            for p in range(nb - 1):
                if panels[t, p] > 0:
                    panels[t, p] = max(int(np.floor(panels[t, p] * cap / total)), 1)
    return lo, hi, le, re, panels


@numba.njit(parallel=True, cache=True)
def _fill(lo, hi, le, re, panels, starts, sg, wg):
    n_t, m = panels.shape
    n_panel = sg.size
    u = np.empty(starts[-1])
    w = np.empty(starts[-1])
    for t in numba.prange(n_t):
        pos = starts[t]
        for p in range(m):
            k = panels[t, p]
            a, b = lo[t, p], hi[t, p]
            length = b - a
            for c in range(k):
                for g in range(n_panel):
                    s = (c + sg[g]) / k
                    if le[t, p] and re[t, p]:
                        x = a + length * np.sin(0.5 * np.pi * s) ** 2
                        d = length * 0.5 * np.pi * np.sin(np.pi * s)
                    elif le[t, p]:
                        x = a + length * s * s
                        d = 2.0 * length * s
                    elif re[t, p]:
                        x = b - length * (1.0 - s) ** 2
                        d = 2.0 * length * (1.0 - s)
                    else:
                        x = a + length * s
                        d = length
                    u[pos] = x
                    w[pos] = wg[g] / k * d * np.exp(-0.5 * x * x)
                    pos += 1
        norm = w[starts[t] : pos].sum()
        w[starts[t] : pos] /= norm
    return u, w


def layout(
    q: np.ndarray, sigma: np.ndarray, curves: list[Edges]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodes of a batch of curves (q, sigma (B, Q)), by the rules of the module docstring: node
    counts per point (B·Q,), and for every node, point after point, its position u in units of σ
    and its weight (Gaussian, normalised per point)."""
    q = np.ascontiguousarray(np.atleast_2d(q), float)
    sigma = np.ascontiguousarray(np.atleast_2d(sigma), float)
    n_b = q.shape[0]
    width = max((e.qc.size for e in curves), default=0)

    def table(name, dtype, fill):
        out = np.full((n_b, max(width, 1)), fill, dtype)
        for c, e in enumerate(curves):
            out[c, : e.qc.size] = getattr(e, name)
        return out

    spacing = np.array(
        [RESONANCE * e.width[e.width > 0].min() if (e.width > 0).any() else 0.0 for e in curves]
    )
    consts = (SPAN, PHASE_PER_PANEL, GAUSS_PER_PANEL, float(PANEL), MAX_POINT_NODES // PANEL)
    lo, hi, le, re, panels = _pieces(
        q,
        sigma,
        table("qc", float, 0.0)[:, :width],
        np.array([e.qc.size for e in curves], np.int64),
        table("depth", float, 0.0)[:, :width],
        table("image", np.int64, 0)[:, :width],
        np.array([e.light for e in curves], float),
        np.array([int(e.image.max()) + 1 if e.qc.size else 1 for e in curves], np.int64),
        spacing,
        SPLIT_AT_EDGES,
        consts,
    )
    counts = panels.sum(axis=1) * PANEL
    starts = np.concatenate([[0], np.cumsum(counts)])
    u, w = _fill(lo, hi, le, re, panels, starts, *_gauss_legendre(PANEL))
    return counts, u, w


def point_layout(
    q: np.ndarray, sigma: np.ndarray, e: Edges
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat node list for one curve (q, sigma (Q,)): owner point of each node, its position u in
    units of σ, and its weight (Gaussian, normalised per point)."""
    counts, u, w = layout(q, sigma, [e])
    return np.repeat(np.arange(np.size(q)), counts), u, w
