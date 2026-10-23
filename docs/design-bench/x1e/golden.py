"""Write tests/reference/layout_golden.npz: the resolution layout of a few stacks as computed by the
rule under test (75c7b1c), so that faster implementations can be checked to give the same nodes,
bit for bit (tests/test_layout_golden.py).

Usage: uv run python docs/design-bench/x1e/golden.py
"""

from pathlib import Path

import numpy as np

from reflectofit import Layer, Stack, resolution

OUT = Path(__file__).parents[3] / "tests" / "reference" / "layout_golden.npz"
WL, WL2 = 1.5406, 1.54439


def cases():
    si = Layer(20.07, roughness=3)
    yield "film with oxide", Stack([Layer(18.0, 15, 3), Layer(60.0, 800, 5)], si), 2e-3, WL2
    yield (
        "real superlattice",
        Stack([Layer(30.0, 25, 3), Layer(55.0, 35, 3.5)] * 20, si),
        5e-4,
        None,
    )
    yield (
        "absorbing waveguide",
        Stack(
            [Layer(64 + 1.35j, 100, 2), Layer(18 + 0.02j, 800, 2)], Layer(64 + 1.35j, roughness=2)
        ),
        1e-4,
        WL2,
    )
    yield "real waveguide", Stack([Layer(25.0, 30, 3), Layer(5.0, 300, 3)], si), 6e-3, WL2


def main():
    q = np.linspace(0.005, 0.15, 60)
    arrays = {}
    for k, (name, stack, sigma, wl2) in enumerate(cases()):
        packed = tuple(a[None] for a in stack.packed())
        sig = np.full((1, q.size), sigma)
        counts, u, w, err, capped = resolution.layout(q[None], sig, packed, WL, wl2)
        for key, value in (
            ("counts", counts),
            ("u", u),
            ("w", w),
            ("err", err),
            ("capped", capped),
        ):
            arrays[f"{k}_{key}"] = value
        print(f"{name}: {counts.sum()} nodes, capped {int(capped.sum())}")
    np.savez_compressed(OUT, q=q, **arrays)


if __name__ == "__main__":
    main()
