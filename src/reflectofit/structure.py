"""Layer stacks: ambient on top, substrate at the bottom.

A medium is (SLD [1e-6 Å⁻², complex, Im > 0 absorbs], thickness [Å], roughness [Å] of the interface
above it). `Stack.packed()` gives the arrays every backend takes, ambient first.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Packed = tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.float64]]


@dataclass(frozen=True)
class Layer:
    sld: complex
    thickness: float = 0.0
    roughness: float = 0.0
    name: str = ""


@dataclass(frozen=True)
class Stack:
    layers: list[Layer]
    substrate: Layer
    ambient: complex = 0j

    @property
    def names(self) -> list[str]:
        """Names of the layers and the substrate, top first; unnamed ones are layer0, layer1, ..."""
        names = [lay.name or f"layer{i}" for i, lay in enumerate(self.layers)]
        return [*names, self.substrate.name or "substrate"]

    def packed(self) -> Packed:
        media = [Layer(self.ambient), *self.layers, self.substrate]
        sld = np.array([m.sld for m in media], dtype=complex)
        thickness = np.array([0.0, *(lay.thickness for lay in self.layers), 0.0])
        roughness = np.array([0.0, *(m.roughness for m in media[1:])])
        return sld, thickness, roughness


def as_batch(model) -> tuple[Packed, bool]:
    """(B, N) arrays from a Stack, a packed tuple, or a sequence of either; and whether it was one."""
    if isinstance(model, Stack):
        return as_batch(model.packed())
    if isinstance(model, tuple) and len(model) == 3 and not isinstance(model[0], (Stack, tuple)):
        sld, thickness, roughness = (np.asarray(a) for a in model)
        single = sld.ndim == 1
        packed = (
            np.atleast_2d(sld).astype(complex),
            np.atleast_2d(thickness).astype(float),
            np.atleast_2d(roughness).astype(float),
        )
        check_packed(*packed)
        return packed, single
    rows = [as_batch(m)[0] for m in model]
    packed = tuple(np.concatenate(parts) for parts in zip(*rows, strict=True))
    check_packed(*packed)
    return packed, False


def check_packed(sld: NDArray, thickness: NDArray, roughness: NDArray) -> None:
    """Shapes (B, N), at least ambient and substrate, no gain, no negative lengths."""
    if sld.ndim != 2 or sld.shape != thickness.shape or sld.shape != roughness.shape:
        raise ValueError(
            "sld, thickness and roughness must share a (B, N) shape, got "
            f"{sld.shape}, {thickness.shape}, {roughness.shape}"
        )
    if sld.shape[1] < 2:
        raise ValueError("a stack needs at least the ambient and the substrate")
    if np.any(sld.imag < 0):
        raise ValueError("negative imaginary SLD is gain; absorption is Im SLD > 0")
    if np.any(sld[:, 0].imag != 0):
        # the kernel works with SLD relative to the ambient; an absorbing ambient makes every less
        # absorbing medium a gain medium (R > 1). XRR ambients are air or vacuum.
        raise ValueError("the ambient must not absorb (Im SLD of the ambient must be 0)")
    if np.any(thickness < 0) or np.any(roughness < 0):
        raise ValueError("thickness and roughness must be non-negative")
