"""The resolution layout is the one of the registered rule (75c7b1c), bit for bit: a faster
implementation must not change a node (docs/design-bench/x1e/golden.py wrote the reference)."""

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
def test_layout_is_bit_identical_to_the_registered_rule(k):
    stack, sigma, wl2 = _cases()[k]
    q = GOLDEN["q"]
    packed = tuple(a[None] for a in stack.packed())
    got = resolution.layout(q[None], np.full((1, q.size), sigma), packed, 1.5406, wl2)
    for key, value in zip(("counts", "u", "w", "err", "capped"), got, strict=True):
        assert np.array_equal(value, GOLDEN[f"{k}_{key}"]), key


@pytest.mark.parametrize("k", range(4))
def test_points_refined_again_give_the_same_layout(monkeypatch, k):
    monkeypatch.setattr(resolution, "LEAF_ROOM", 0)  # no leaf is kept: every refined point replays
    test_layout_is_bit_identical_to_the_registered_rule(k)
