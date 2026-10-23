"""Measured curves: reading, stitching attenuated segments, held-out splits.

Measured files are never part of this repository (DESIGN.md section 6); tests use synthetic files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from reflectofit.units import CU_KALPHA1, q_to_tth, tth_to_q


@dataclass
class Curve:
    q: NDArray[np.float64]
    y: NDArray[np.float64]
    sigma: NDArray[np.float64] | None = None
    meta: dict = field(default_factory=dict)

    def tth(self, wavelength: float | None = None) -> NDArray[np.float64]:
        return q_to_tth(self.q, wavelength or self.meta.get("wavelength", CU_KALPHA1))


def read(path: str | Path, x: Literal["tth", "q"] = "tth", wavelength: float = CU_KALPHA1) -> Curve:
    """Read 2- or 3-column text (x, intensity[, error]); whitespace or commas, comment and header
    lines skipped. Sorted by q; equal q values are kept."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line[0] in "#!;%/":
            continue
        parts = line.replace(",", " ").split()
        try:
            rows.append([float(p) for p in parts[:3]])
        except ValueError:
            continue
    rows = [r for r in rows if len(r) >= 2]
    if not rows:
        raise ValueError(f"no numeric data with at least two columns in {path}")
    width = min(len(r) for r in rows)
    arr = np.array([r[:width] for r in rows])
    q = tth_to_q(arr[:, 0], wavelength) if x == "tth" else arr[:, 0]
    order = np.argsort(q, kind="stable")
    sigma = arr[order, 2] if width >= 3 else None
    meta = {"source": str(path), "x": x, "wavelength": wavelength}
    return Curve(np.asarray(q)[order], arr[order, 1], sigma, meta)


def write(path: str | Path, curve: Curve, x: Literal["tth", "q"] = "tth") -> None:
    cols = [curve.tth() if x == "tth" else curve.q, curve.y]
    if curve.sigma is not None:
        cols.append(curve.sigma)
    header = f"{'2theta_deg' if x == 'tth' else 'q_invA'} intensity" + (
        " error" if curve.sigma is not None else ""
    )
    np.savetxt(path, np.column_stack(cols), header=header)


def stitch(
    scans: list[NDArray], method: Literal["boundary", "median"] = "boundary"
) -> NDArray[np.float64]:
    """Join scans (N, 2+) of [x, intensity, (error, ...)] measured through different attenuators.

    boundary (xrr_combine, ReflectoLearn): scans in increasing x share exactly one boundary point;
    the last scan is the reference, each earlier scan is scaled so its last point equals the next
    scan's first point, and the duplicate is dropped.
    median (xray_ooguy): the first scan is the reference; each later scan is scaled by the median of
    the point-by-point ratio to the curve assembled so far (linear interpolation) over their overlap,
    which needs at least two points; all points are kept. A scan without such an overlap raises.
    Every column after x is scaled together.
    """
    scans = [np.asarray(s, dtype=float) for s in scans]
    if not scans:
        return np.empty((0, 3))
    if method == "boundary":
        out = [scans[-1]]
        head = scans[-1][0, 1]
        for scan in reversed(scans[:-1]):
            scan = scan.copy()
            scan[:, 1:] *= head / scan[-1, 1]
            scan = scan[:-1]
            head = scan[0, 1]
            out.append(scan)
        return np.concatenate(out[::-1])
    if method != "median":
        raise ValueError(f"unknown method {method!r}; use 'boundary' or 'median'")
    order = sorted(scans, key=lambda s: s[0, 0])
    acc = order[0].copy()
    for scan in order[1:]:
        inside = scan[:, 0] <= acc[-1, 0]
        prev = np.interp(scan[inside, 0], acc[:, 0], acc[:, 1], left=np.nan)
        ok = np.isfinite(prev) & (prev > 0) & (scan[inside, 1] > 0)
        if ok.sum() < 2:
            raise ValueError(
                f"scan starting at x = {scan[0, 0]} overlaps the curve in fewer than two points"
            )
        scaled = scan.copy()
        scaled[:, 1:] *= np.median(prev[ok] / scan[inside, 1][ok])
        acc = np.concatenate([acc, scaled])
        acc = acc[np.argsort(acc[:, 0], kind="stable")]
    return acc


def block_split(
    q: NDArray, n_blocks: int, held: tuple[int, ...], q_min: float = 0.0
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Training and held-out masks: points with q < q_min always train; the rest are cut into
    `n_blocks` contiguous blocks of equal size (np.array_split) and the blocks in `held` (0-based)
    are held out (ReflectoLearn#35 section 3.2)."""
    q = np.asarray(q)
    upper = np.flatnonzero(q >= q_min)
    held_mask = np.zeros(q.shape, bool)
    for i, block in enumerate(np.array_split(upper, n_blocks)):
        if i in held:
            held_mask[block] = True
    return ~held_mask, held_mask
