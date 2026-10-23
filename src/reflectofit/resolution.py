"""Node counts for the resolution integral, chosen per measured point.

Each point integrates R against its Gaussian with uniform nodes over ±SPAN σ and Gaussian weights
(spectrally accurate for a smooth integrand). The node spacing h, in units of σ, is the smallest of
- 0.5;
- 2π / (2 σ T_x + 8), so the fringes stay resolved until the resolution has damped them; T_x is
  the total thickness T stretched by x / sqrt(x² − q_c²) at the lowest x of the window above an
  edge (fringes run in the film wavevector; capped where absorption smooths the edge);
- where the window [q − SPAN σ, q + SPAN σ] contains a critical edge q_c,j, KINK · w_j / σ, with
  w_j = 8π Im ρ_j / q_c,j the width over which absorption smooths the square-root kink of R there
  (floor EDGE_FLOOR for a real SLD);
- where the window reaches the range between the lowest and highest critical q, RESONANCE · w / σ
  with w the narrowest edge width: waveguide resonances there are about that narrow.
So only the points near edges pay for them. The count depends only on the curve's own stack and
instrument, never on the rest of a batch.

The counts are read from parameter values and are not traced; gradients come through R, the node
positions and the Gaussian weights.
"""

import numpy as np

SPAN = 6.5
"""Nodes run over ±SPAN σ."""
EDGE_FLOOR = 1e-6
"""Smallest edge width [Å⁻¹] used for the spacing, reached by a medium with a real SLD."""
MAX_POINT_NODES = 8001
KINK = 0.75
"""Spacing near a critical edge, in units of that edge's width."""
RESONANCE = 0.5
"""Spacing between the lowest and highest critical q, in units of the narrowest edge width.

Review (physics, ReflectoFit#2): at 2.0 a Ni/C/Ni waveguide with sigma_q = 1e-4 was off by 1.6e-3;
the resonances there are narrower than one edge width, so the spacing is a quarter of it."""


def edges(sld: np.ndarray, kalpha2_scale: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Critical q [Å⁻¹] of every medium denser than the ambient and its edge width 8π Im ρ / q_c,
    for one curve (sld (N,) complex, 1e-6 Å⁻²). With `kalpha2_scale` = λ₂/λ₁ the Kα₂ edges (same
    angle, so at q_c λ₂/λ₁ in nominal q) are added."""
    rho = (sld[1:] - sld[0]) * 1e-6
    dense = rho.real > 0
    qc = 4.0 * np.sqrt(np.pi * rho.real[dense])
    width = np.maximum(8.0 * np.pi * rho.imag[dense] / qc, EDGE_FLOOR)
    if kalpha2_scale is not None and qc.size:
        qc = np.concatenate([qc, qc * kalpha2_scale])
        width = np.concatenate([width, width * kalpha2_scale])
    return qc, width


def point_nodes(
    q: np.ndarray, sigma: np.ndarray, thickness: float, qc: np.ndarray, width: np.ndarray
) -> np.ndarray:
    """Odd node count per point (q, sigma (Q,), one curve) by the rules of the module docstring."""
    lo, hi = (q - SPAN * sigma)[:, None], (q + SPAN * sigma)[:, None]
    h = np.full(q.shape, 0.5)
    stretch = np.ones(q.shape)
    if qc.size:
        # between the edges: waveguide resonances, about as narrow as the narrowest edge
        between = (lo[:, 0] <= qc.max()) & (hi[:, 0] >= qc.min())
        h = np.where(between, np.minimum(h, RESONANCE * width.min() / sigma), h)
        # an edge inside the window: its own kink, smoothed over its own width
        inside = (lo <= qc) & (qc <= hi)
        h = np.minimum(h, np.where(inside, KINK * width / sigma[:, None], np.inf).min(axis=1))
        # fringe frequency just above an edge, at the lowest x of the window above it
        x = np.maximum(lo, qc + width)
        s = x / np.sqrt(np.maximum(x * x - qc * qc, 2.0 * qc * width))
        stretch = np.maximum(np.where(qc < hi, s, 1.0).max(axis=1), 1.0)
    if thickness > 0:
        h = np.minimum(h, 2.0 * np.pi / (2.0 * sigma * thickness * stretch + 8.0))
    n = 2 * np.ceil(SPAN / h).astype(int) + 1
    return np.minimum(n, MAX_POINT_NODES)


def layout(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat node list for one curve: owner point of each node, its position u in units of σ, and
    its weight (Gaussian, normalised per point)."""
    owner = np.repeat(np.arange(counts.size), counts)
    starts = np.cumsum(counts) - counts
    k = np.arange(owner.size) - starts[owner]
    u = -SPAN + 2.0 * SPAN * k / (counts[owner] - 1)
    w = np.exp(-0.5 * u * u)
    w /= np.bincount(owner, w, minlength=counts.size)[owner]
    return owner, u, w
