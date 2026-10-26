"""Figure of throughput.json for the README: curves per second against batch size, per engine.

    uv run --with matplotlib python -I docs/design-bench/plot_throughput.py docs/design-bench/throughput.json docs/media/throughput.png

Hollow markers are values the benchmark extrapolated or scaled from a smaller batch ("how" in the JSON).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ENGINES = [
    ("numba-f64-par", "ReflectoFit (numba, float64)", "o"),
    ("torch-cpu-c128", "torch CPU, complex128", "s"),
    ("torch-cuda-c64", "torch CUDA, complex64", "D"),
    ("jax-cpu-f64", "JAX CPU, float64", "^"),
    ("numpy-f64", "NumPy, float64", "v"),
    ("refnx", "refnx", "P"),
    ("genx", "GenX", "X"),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
CASES = [
    ("single", "1 film (3 media)"),
    ("five_layer", "5 layers"),
    ("superlattice50", "50-layer superlattice"),
]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main() -> None:
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    rows = [v for v in data.values() if "engine" in v]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True, facecolor=SURFACE)
    for ax, (case, title) in zip(axes, CASES, strict=True):
        ax.set_facecolor(SURFACE)
        for (engine, label, marker), color in zip(ENGINES, COLORS, strict=True):
            pts = sorted(
                (
                    r["batch"],
                    r["curves_per_s"],
                    r["how"].startswith("measured") and "scaled" not in r["how"],
                )
                for r in rows
                if r["engine"] == engine and r["case"] == case
            )
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            ax.plot(
                x,
                y,
                color=color,
                lw=2.6 if engine == "numba-f64-par" else 1.6,
                label=label,
                zorder=3,
            )
            for xi, yi, measured in pts:
                ax.plot(
                    xi,
                    yi,
                    marker=marker,
                    ms=7,
                    color=color,
                    mfc=color if measured else SURFACE,
                    mew=1.6,
                    zorder=4,
                )
        ax.set(xscale="log", yscale="log", title=title)
        ax.title.set_color(INK)
        ax.set_xlabel("curves per batch", color=MUTED)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("curves per second", color=MUTED)
    handles = [
        Line2D([], [], color=c, marker=m, ms=6, lw=1.6, label=label)
        for (_, label, m), c in zip(ENGINES, COLORS, strict=True)
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(sys.argv[2], dpi=110, facecolor=SURFACE)


if __name__ == "__main__":
    main()
