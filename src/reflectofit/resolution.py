"""Nodes and weights for the resolution integral, chosen per measured point.

Each point integrates R against its Gaussian over ±SPAN σ. R has a square-root kink at the critical
edge q_c of the substrate (kz = √(q² − q_c²)/2 of a semi-infinite medium changes from imaginary to
real there; absorption rounds it over a width 8π Im ρ / q_c). A finite layer enters R only through
even functions of its own kz, Névot–Croce factors included, so its edge is smooth, though R can
change fast there. Uniform nodes converge slowly on a kink, so the window is split at every edge
inside it (Kα₂ edges included; splitting at a smooth edge costs nodes, not accuracy), and each
piece [l, r] is mapped from s in [0, 1] so that the distance to an edge end goes as s²:

- edge at both ends: u = l + (r − l) sin²(π s / 2);
- edge at the left end: u = l + (r − l) s²; at the right end: u = r − (r − l) (1 − s)²;
- no edge: u = l + (r − l) s.

R is then smooth in s, and each piece gets Gauss–Legendre panels of PANEL nodes, as many as the
larger of: the change of every layer's phase or evanescent decay d_j |√(q_r² − q_cj²) − √(q_l² −
q_cj²)| across it, summed over the layers (Kα₁ and Kα₂ images apart, the larger), over
PHASE_PER_PANEL; and its length over GAUSS_PER_PANEL σ; both times the largest stretch of the map
(2 with an edge at one end, π/2 at both). Where a medium absorbs, waveguide resonances between the
lowest and highest critical q are about as narrow as the narrowest absorbing edge width w, and the
pieces there get nodes RESONANCE · w apart (in q).

Between the lowest and highest critical q, multilayers without absorption also have narrow
features whose width no edge sets: in a superlattice, the transmission resonances of its first
Kronig–Penney miniband, which narrow with the number of periods (physics review of ReflectoFit#8).
There each panel is compared with its two halves and split while they differ by more than
REFINE_TOL of the point's integral over that range, up to MAX_DEPTH halvings. The tolerance holds
per panel; the sum of those differences is the point's estimated relative error, which `layout`
returns, and `forward` warns when it exceeds REFINE_TOL or a point stops at MAX_DEPTH.

So only points near edges pay for them, and the nodes depend only on the curve's own stack and
instrument, never on the rest of a batch.

The counts and node positions are read from parameter values and are not traced; gradients come
through R, the node positions q + σ u and the weights.
"""

from functools import cache
from typing import NamedTuple

import numba
import numpy as np

from reflectofit.kernel import _reflectivity

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
REFINE_TOL = 1e-5
"""Relative tolerance of the refinement between the critical q; inf switches the refinement off."""
MAX_DEPTH = 12
"""Halvings of a panel at most."""
LEAF_ROOM = 4
"""Leaves kept per refined panel (and one more) between the two passes; only speed depends on it."""


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
    edge flags, panel counts, and whether the piece reaches the range between the lowest and
    highest critical q. Edge arrays are (B, max edges), the first n_edges[b] valid."""
    span, phase_per, gauss_per, panel, cap = consts
    n_b, n_q = q.shape
    m = qc.shape[1] + 1
    lo = np.zeros((n_b * n_q, m))
    hi = np.zeros((n_b * n_q, m))
    le = np.zeros((n_b * n_q, m), np.bool_)
    re = np.zeros((n_b * n_q, m), np.bool_)
    bt = np.zeros((n_b * n_q, m), np.bool_)
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
            between = n_e > 0 and ql <= qmax and qr >= qmin
            if spacing[c] > 0 and between:
                n = max(n, np.ceil((qr - ql) / (spacing[c] * panel)))
            lo[t, p], hi[t, p], le[t, p], re[t, p] = a, b, edge[p], edge[p + 1]
            bt[t, p] = between
            panels[t, p] = int(n)
            total += int(n)
        if total > cap:
            for p in range(nb - 1):
                if panels[t, p] > 0:
                    panels[t, p] = max(int(np.floor(panels[t, p] * cap / total)), 1)
    return lo, hi, le, re, bt, panels


@numba.njit(cache=True)
def _map(s, a, b, left, right):
    """u(s) and du/ds of a piece [a, b] with edge flags (module docstring)."""
    length = b - a
    if left and right:
        return a + length * np.sin(0.5 * np.pi * s) ** 2, length * 0.5 * np.pi * np.sin(np.pi * s)
    if left:
        return a + length * s * s, 2.0 * length * s
    if right:
        return b - length * (1.0 - s) ** 2, 2.0 * length * (1.0 - s)
    return a + length * s, length


@numba.njit(cache=True)
def _panel(s0, s1, a, b, left, right, qv, sv, c, rho, thickness, roughness, wl1, wl2, sg, wg):
    """Gauss–Legendre values of R · Gaussian · du/ds over [s0, s1] of a piece, for Kα₁ and (when
    wl2 > 0) Kα₂ apart, without footprint (a smooth factor that does not change where R needs
    nodes). Refining each line on its own bounds any mix of the two, so the nodes do not depend on
    the intensity ratio (which may be a traced value)."""
    acc1 = 0.0
    acc2 = 0.0
    for g in range(sg.size):
        x, d = _map(s0 + (s1 - s0) * sg[g], a, b, left, right)
        sin_t = min(max((qv + sv * x) * wl1 / (4.0 * np.pi), 1e-12), 1.0)
        f = wg[g] * (s1 - s0) * d * np.exp(-0.5 * x * x)
        acc1 += f * _reflectivity(4.0 * np.pi / wl1 * sin_t, rho, thickness, roughness, c)
        if wl2 > 0:
            acc2 += f * _reflectivity(4.0 * np.pi / wl2 * sin_t, rho, thickness, roughness, c)
    return acc1, acc2


@numba.njit(cache=True)
def _emit(s0, s1, halves, a, b, left, right, sg, wg, u, w, pos):
    """Write the nodes of panel [s0, s1] of a piece, as `halves` equal parts, at pos; returns the
    next position."""
    for h in range(halves):
        h0 = s0 + (s1 - s0) * h / halves
        h1 = s0 + (s1 - s0) * (h + 1) / halves
        for g in range(sg.size):
            x, d = _map(h0 + (h1 - h0) * sg[g], a, b, left, right)
            u[pos] = x
            w[pos] = wg[g] * (h1 - h0) * d * np.exp(-0.5 * x * x)
            pos += 1
    return pos


@numba.njit(parallel=True, cache=True)
def _refine(
    q, sigma, lo, hi, le, re, bt, panels, rho, thickness, roughness, wl1, wl2, tol, max_depth,
    sg, wg, lstart, leaf0, leaf1, leafp, nleaf, counts, err, capped,
):  # fmt: skip
    """First pass: the nodes per point, and for the pieces between the critical q, panels split
    while a panel and its two halves differ by more than tol of the point's integral over those
    pieces (each wavelength on its own). Every tested panel's value is reused: an initial panel's
    whole is its share of the reference integral, a half becomes its child's whole. The leaves are
    kept (lstart gives each point's room; nleaf is −1 where it did not suffice) so that `_fill`
    writes the nodes without evaluating R again; err and capped as in the module docstring."""
    n_t, m = panels.shape
    n_q = q.shape[1]
    k_nodes = sg.size
    for t in numba.prange(n_t):
        c, i = t // n_q, t % n_q
        qv, sv = q[c, i], sigma[c, i]
        n_init = 0
        if tol < np.inf:
            for p in range(m):
                if bt[t, p]:
                    n_init += panels[t, p]
        init1 = np.empty(n_init)
        init2 = np.empty(n_init)
        ref1 = 0.0
        ref2 = 0.0
        j = 0
        for p in range(m):
            if bt[t, p] and tol < np.inf:
                for k in range(panels[t, p]):
                    v1, v2 = _panel(k / panels[t, p], (k + 1) / panels[t, p], lo[t, p], hi[t, p],
                                    le[t, p], re[t, p], qv, sv, c, rho, thickness, roughness, wl1,
                                    wl2, sg, wg)  # fmt: skip
                    init1[j], init2[j] = v1, v2
                    ref1 += v1
                    ref2 += v2
                    j += 1
        n_out = 0
        est1 = 0.0
        est2 = 0.0
        hit = False
        room = lstart[t + 1] - lstart[t]
        kept = 0
        overflow = False
        size = 2 * max_depth + 4
        stack0 = np.empty(size)
        stack1 = np.empty(size)
        stackd = np.empty(size, np.int64)
        stackw1 = np.empty(size)
        stackw2 = np.empty(size)
        j = 0
        for p in range(m):
            a, b, left, right = lo[t, p], hi[t, p], le[t, p], re[t, p]
            tested = bt[t, p] and ref1 > 0 and tol < np.inf
            for k in range(panels[t, p]):
                if not tested:
                    n_out += k_nodes
                    if bt[t, p] and tol < np.inf:
                        j += 1
                    continue
                stack0[0], stack1[0], stackd[0] = k / panels[t, p], (k + 1) / panels[t, p], 0
                stackw1[0], stackw2[0] = init1[j], init2[j]
                j += 1
                top = 1
                while top > 0:
                    top -= 1
                    s0, s1, dep = stack0[top], stack1[top], stackd[top]
                    w1, w2 = stackw1[top], stackw2[top]
                    sm = 0.5 * (s0 + s1)
                    l1, l2 = _panel(s0, sm, a, b, left, right, qv, sv, c, rho, thickness,
                                    roughness, wl1, wl2, sg, wg)  # fmt: skip
                    r1, r2 = _panel(sm, s1, a, b, left, right, qv, sv, c, rho, thickness,
                                    roughness, wl1, wl2, sg, wg)  # fmt: skip
                    e1, e2 = abs(w1 - l1 - r1), abs(w2 - l2 - r2)
                    over = e1 > tol * ref1 or (ref2 > 0 and e2 > tol * ref2)
                    if over and dep < max_depth:
                        stack0[top], stack1[top], stackd[top] = s0, sm, dep + 1
                        stackw1[top], stackw2[top] = l1, l2
                        stack0[top + 1], stack1[top + 1], stackd[top + 1] = sm, s1, dep + 1
                        stackw1[top + 1], stackw2[top + 1] = r1, r2
                        top += 2
                        continue
                    est1 += e1
                    est2 += e2
                    hit = hit or over
                    n_out += 2 * k_nodes
                    if kept < room:
                        leaf0[lstart[t] + kept] = s0
                        leaf1[lstart[t] + kept] = s1
                        leafp[lstart[t] + kept] = p
                    else:
                        overflow = True
                    kept += 1
        counts[t] = n_out
        nleaf[t] = -1 if overflow else kept
        e = est1 / ref1 if ref1 > 0 else 0.0
        if ref2 > 0:
            e = max(e, est2 / ref2)
        err[t] = e
        capped[t] = hit


@numba.njit(parallel=True, cache=True)
def _fill(
    q, sigma, lo, hi, le, re, bt, panels, rho, thickness, roughness, wl1, wl2, tol, max_depth,
    sg, wg, lstart, leaf0, leaf1, leafp, nleaf, starts, u, w,
):  # fmt: skip
    """Second pass: write the nodes of every point from starts, untested panels whole and every
    kept leaf as its two halves. A point whose leaves did not fit is refined again (`_replay`)."""
    n_t, m = panels.shape
    for t in numba.prange(n_t):
        pos = starts[t]
        if nleaf[t] < 0:
            pos = _replay(t, q, sigma, lo, hi, le, re, bt, panels, rho, thickness, roughness, wl1,
                          wl2, tol, max_depth, sg, wg, u, w, pos)  # fmt: skip
        else:
            kept = lstart[t]
            for p in range(m):
                a, b, left, right = lo[t, p], hi[t, p], le[t, p], re[t, p]
                tested = kept < lstart[t] + nleaf[t] and leafp[kept] == p
                if not tested:
                    for k in range(panels[t, p]):
                        pos = _emit(k / panels[t, p], (k + 1) / panels[t, p], 1, a, b, left,
                                    right, sg, wg, u, w, pos)  # fmt: skip
                    continue
                while kept < lstart[t] + nleaf[t] and leafp[kept] == p:
                    pos = _emit(leaf0[kept], leaf1[kept], 2, a, b, left, right, sg, wg, u, w, pos)
                    kept += 1
        norm = w[starts[t] : pos].sum()
        w[starts[t] : pos] /= norm


@numba.njit(cache=True)
def _replay(t, q, sigma, lo, hi, le, re, bt, panels, rho, thickness, roughness, wl1, wl2, tol,
            max_depth, sg, wg, u, w, pos):  # fmt: skip
    """The refinement of `_refine` for one point, writing the nodes as it goes."""
    m = panels.shape[1]
    n_q = q.shape[1]
    c, i = t // n_q, t % n_q
    qv, sv = q[c, i], sigma[c, i]
    n_init = 0
    for p in range(m):
        if bt[t, p] and tol < np.inf:
            n_init += panels[t, p]
    init1 = np.empty(n_init)
    init2 = np.empty(n_init)
    ref1 = 0.0
    ref2 = 0.0
    j = 0
    for p in range(m):
        if bt[t, p] and tol < np.inf:
            for k in range(panels[t, p]):
                v1, v2 = _panel(k / panels[t, p], (k + 1) / panels[t, p], lo[t, p], hi[t, p],
                                le[t, p], re[t, p], qv, sv, c, rho, thickness, roughness, wl1,
                                wl2, sg, wg)  # fmt: skip
                init1[j], init2[j] = v1, v2
                ref1 += v1
                ref2 += v2
                j += 1
    size = 2 * max_depth + 4
    stack0 = np.empty(size)
    stack1 = np.empty(size)
    stackd = np.empty(size, np.int64)
    stackw1 = np.empty(size)
    stackw2 = np.empty(size)
    j = 0
    for p in range(m):
        a, b, left, right = lo[t, p], hi[t, p], le[t, p], re[t, p]
        tested = bt[t, p] and ref1 > 0 and tol < np.inf
        for k in range(panels[t, p]):
            if not tested:
                pos = _emit(k / panels[t, p], (k + 1) / panels[t, p], 1, a, b, left, right, sg,
                            wg, u, w, pos)  # fmt: skip
                if bt[t, p]:
                    j += 1
                continue
            stack0[0], stack1[0], stackd[0] = k / panels[t, p], (k + 1) / panels[t, p], 0
            stackw1[0], stackw2[0] = init1[j], init2[j]
            j += 1
            top = 1
            while top > 0:
                top -= 1
                s0, s1, dep = stack0[top], stack1[top], stackd[top]
                w1, w2 = stackw1[top], stackw2[top]
                sm = 0.5 * (s0 + s1)
                l1, l2 = _panel(s0, sm, a, b, left, right, qv, sv, c, rho, thickness, roughness,
                                wl1, wl2, sg, wg)  # fmt: skip
                r1, r2 = _panel(sm, s1, a, b, left, right, qv, sv, c, rho, thickness, roughness,
                                wl1, wl2, sg, wg)  # fmt: skip
                over = abs(w1 - l1 - r1) > tol * ref1 or (
                    ref2 > 0 and abs(w2 - l2 - r2) > tol * ref2
                )
                if over and dep < max_depth:
                    stack0[top], stack1[top], stackd[top] = s0, sm, dep + 1
                    stackw1[top], stackw2[top] = l1, l2
                    stack0[top + 1], stack1[top + 1], stackd[top + 1] = sm, s1, dep + 1
                    stackw1[top + 1], stackw2[top + 1] = r1, r2
                    top += 2
                    continue
                pos = _emit(s0, s1, 2, a, b, left, right, sg, wg, u, w, pos)
    return pos


def layout(
    q: np.ndarray,
    sigma: np.ndarray,
    packed: tuple[np.ndarray, np.ndarray, np.ndarray],
    wavelength: float,
    kalpha2_wavelength: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nodes of a batch of curves (q, sigma (B, Q) true q and σ; packed (B, N) media), by the rules
    of the module docstring: node counts per point (B·Q,); for every node, point after point, its
    position u in units of σ and its weight (Gaussian, normalised per point); and per point the
    estimated relative error of the refined range and whether refinement stopped at MAX_DEPTH."""
    q = np.ascontiguousarray(np.atleast_2d(q), float)
    sigma = np.ascontiguousarray(np.atleast_2d(sigma), float)
    sld, thickness, roughness = (np.ascontiguousarray(np.atleast_2d(a)) for a in packed)
    sld = sld.astype(np.complex128)
    n_b = q.shape[0]
    ka2 = kalpha2_wavelength / wavelength if kalpha2_wavelength else None
    curves = [edges(sld[c], thickness[c], ka2) for c in range(n_b)]
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
    lo, hi, le, re, bt, panels = _pieces(
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
    rho = 4.0e-6 * np.pi * np.conj(sld - sld[:, :1])
    sg, wg = _gauss_legendre(PANEL)
    n_t = q.size
    counts = np.zeros(n_t, np.int64)
    err = np.zeros(n_t)
    capped = np.zeros(n_t, np.bool_)
    # room for the leaves of the refined panels; a point that needs more is refined again in _fill
    room = LEAF_ROOM * ((panels * bt).sum(axis=1) + 1)
    lstart = np.concatenate([[0], np.cumsum(room)])
    leaf0, leaf1 = np.empty(lstart[-1]), np.empty(lstart[-1])
    leafp = np.empty(lstart[-1], np.int64)
    nleaf = np.zeros(n_t, np.int64)
    args = (q, sigma, lo, hi, le, re, bt, panels, rho, np.ascontiguousarray(thickness, float),
            np.ascontiguousarray(roughness, float), float(wavelength),
            float(kalpha2_wavelength) if ka2 is not None else 0.0, REFINE_TOL, MAX_DEPTH,
            sg, wg, lstart, leaf0, leaf1, leafp, nleaf)  # fmt: skip
    _refine(*args, counts, err, capped)
    starts = np.concatenate([[0], np.cumsum(counts)])
    u, w = np.empty(starts[-1]), np.empty(starts[-1])
    _fill(*args, starts, u, w)
    return counts, u, w, err, capped
