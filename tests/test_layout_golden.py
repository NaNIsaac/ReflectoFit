"""The resolution layout is the one of the registered rule (75c7b1c): a faster implementation must
not change a node (docs/design-bench/x1e/golden.py wrote the reference on Windows).

The refinement decisions, hence the node counts and the depth-limit flags, must be identical. Node
positions and weights go through sin and exp, whose last bit differs between platform math
libraries (the Ubuntu runners of ReflectoFit#8 differed from Windows in u and w only), so they are
compared to 1e-13 relative; on the platform that wrote the reference they are equal bit for bit."""

from pathlib import Path

import numpy as np
import pytest

from reflectofit import resolution

GOLDEN = np.load(Path(__file__).parent / "reference" / "layout_golden.npz")


def _cases():
    """The stacks of docs/design-bench/x1e/golden.py, in its order."""
    from reflectofit import Layer, Stack

    si = Layer(20.07, roughness=3)
    wl2 = 1.54439
    return [
        (Stack([Layer(18.0, 15, 3), Layer(60.0, 800, 5)], si), 2e-3, wl2),
        (Stack([Layer(30.0, 25, 3), Layer(55.0, 35, 3.5)] * 20, si), 5e-4, None),
        (
            Stack(
                [Layer(64 + 1.35j, 100, 2), Layer(18 + 0.02j, 800, 2)],
                Layer(64 + 1.35j, roughness=2),
            ),
            1e-4,
            wl2,
        ),
        (Stack([Layer(25.0, 30, 3), Layer(5.0, 300, 3)], si), 6e-3, wl2),
    ]


@pytest.mark.parametrize("k", range(4))
def test_layout_is_the_registered_rule(k):
    stack, sigma, wl2 = _cases()[k]
    q = GOLDEN["q"]
    packed = tuple(a[None] for a in stack.packed())
    got = resolution.layout(q[None], np.full((1, q.size), sigma), packed, 1.5406, wl2)
    counts, u, w, err, capped = got
    assert np.array_equal(counts, GOLDEN[f"{k}_counts"])
    assert np.array_equal(capped, GOLDEN[f"{k}_capped"])
    # u is in units of σ and can be 0; err is a difference of nearly equal panel values, so its
    # rounding is amplified (1e-16 estimates are rounding noise); 1e-12 is far below REFINE_TOL
    for key, value, rtol, atol in (
        ("u", u, 1e-13, 1e-14),
        ("w", w, 1e-13, 0.0),
        ("err", err, 1e-6, 1e-12),
    ):
        np.testing.assert_allclose(value, GOLDEN[f"{k}_{key}"], rtol=rtol, atol=atol, err_msg=key)


@pytest.mark.parametrize("k", range(4))
def test_points_refined_again_give_the_same_layout(monkeypatch, k):
    monkeypatch.setattr(resolution, "LEAF_ROOM", 0)  # no leaf is kept: every refined point replays
    test_layout_is_the_registered_rule(k)
