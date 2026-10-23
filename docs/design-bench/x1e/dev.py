"""X1e development grid (constants may be tuned here): the ReflectoLearn default distribution at
seed 0 and a few real-SLD stacks; default rule against brute force, node counts and time.

Usage: uv run python docs/design-bench/x1e/dev.py [n_error_curves [GAUSS_PER_PANEL PHASE_PER_PANEL]]
"""

import sys
import time

import numpy as np
from rl_distribution import WL, draw

from reflectofit import Instrument, Layer, Stack, forward, resolution, tth_to_q


def brute(q, packed, inst: Instrument, sigma, n=24001, chunk=100):
    """Uniform nodes over ±9 σ per point, in chunks of points."""
    u = np.linspace(-9, 9, n)
    w = np.exp(-0.5 * u**2)
    w /= w.sum()
    bare = inst.replace(sigma_q=0.0, dq_over_q=0.0, i0=1.0, background=0.0)
    out = np.empty(q.size)
    for i in range(0, q.size, chunk):
        nodes = np.clip(q[i : i + chunk, None] + sigma[i : i + chunk, None] * u, 1e-9, None)
        r = forward(nodes.ravel(), packed, bare).reshape(nodes.shape)
        out[i : i + chunk] = (r * w).sum(1)
    return out


def nodes_per_point(q, packed, inst: Instrument):
    sld, thk = packed[0], packed[1]
    ka2 = inst.kalpha2_wavelength / inst.wavelength if np.any(inst.kalpha2_ratio) else None
    sigma = np.full(q.size, inst.sigma_q)
    owner, _, _ = resolution.point_layout(q, sigma, resolution.edges(sld, thk, ka2))
    return owner.size / q.size


def main():
    n_err = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    if len(sys.argv) > 3:
        resolution.GAUSS_PER_PANEL, resolution.PHASE_PER_PANEL = map(float, sys.argv[2:4])
    print(
        f"GAUSS_PER_PANEL {resolution.GAUSS_PER_PANEL}, PHASE_PER_PANEL {resolution.PHASE_PER_PANEL}"
    )
    q0, p0, i0 = draw(64, seed=0)
    ka2 = i0.kalpha2_wavelength / i0.wavelength
    t = time.perf_counter()
    for j in range(64):
        e = resolution.edges(p0[0][j], p0[1][j], ka2)
        resolution.point_layout(q0, np.full(q0.size, i0.sigma_q[j]), e)
    print(f"layouts alone: {time.perf_counter() - t:.2f} s")
    q, packed, inst = draw(64, seed=0)
    forward(q, tuple(a[:2] for a in packed), Instrument(wavelength=WL, sigma_q=1e-3))  # compile
    t = time.perf_counter()
    forward(q, packed, inst)
    print(f"RL batch 64 x {q.size}, sigma_q per curve: numba {time.perf_counter() - t:.2f} s")
    for s in (1e-4, 6e-3):
        t = time.perf_counter()
        forward(q, packed, inst.replace(sigma_q=s))
        print(f"  all at sigma_q {s:g}: {time.perf_counter() - t:.2f} s")

    worst = 0.0
    for j in range(n_err):
        p = tuple(a[j] for a in packed)
        for s in (1e-4, 6e-3):
            one = Instrument(
                wavelength=WL,
                kalpha2_wavelength=inst.kalpha2_wavelength,
                tth_offset=float(inst.tth_offset[j]),
                kalpha2_ratio=float(inst.kalpha2_ratio[j]),
                sigma_q=s,
            )
            got = forward(q, p, one)
            q_true = tth_to_q(
                np.clip(
                    np.degrees(2 * np.arcsin(q * WL / (4 * np.pi))) + one.tth_offset, 1e-9, None
                ),
                WL,
            )
            ref = brute(q_true, p, one.replace(tth_offset=0.0), np.full(q.size, s))
            rel = np.abs(got / ref - 1)
            worst = max(worst, rel.max())
            print(
                f"curve {j} sigma {s:g}: max rel {rel.max():.1e} at q {q[rel.argmax()]:.4f}, "
                f"nodes/point {nodes_per_point(q, p, one):.0f}"
            )
    print(f"worst {worst:.1e}")

    real = {
        "bare Si real": Stack([], Layer(20.07)),
        "PS 1000 on real Si": Stack([Layer(9.6, 1000, 3)], Layer(20.07, roughness=3)),
        "60 / 800 on real Si": Stack([Layer(60.0, 800, 5)], Layer(20.0, roughness=3)),
    }
    qq = np.linspace(0.01, 0.2, 600)
    for name, st in real.items():
        for s in (1e-4, 5e-4, 2e-3):
            inst1 = Instrument(sigma_q=s)
            got = forward(qq, st, inst1)
            ref = brute(qq, st.packed(), inst1, np.full(qq.size, s))
            rel = np.abs(got / ref - 1)
            print(f"{name} sigma {s:g}: max rel {rel.max():.1e} at q {qq[rel.argmax()]:.4f}")


if __name__ == "__main__":
    main()
