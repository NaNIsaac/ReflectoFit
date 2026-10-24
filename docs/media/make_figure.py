"""Draw docs/media/fit.png: the README's first run, from the starting guess to the fit.

uv run --with matplotlib docs/media/make_figure.py
"""

import json
import tempfile
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reflectofit.cli import main

ROOT = Path(__file__).resolve().parents[2]
TRUE, START = ROOT / "examples/film.toml", ROOT / "examples/film_start.toml"
COUNTS = 1e8


def run(*args) -> None:
    main([str(a) for a in args])


with tempfile.TemporaryDirectory() as tmp:
    data, start_curve, out = Path(tmp, "film.dat"), Path(tmp, "start.dat"), Path(tmp, "fit")
    run("calc", "--model", TRUE, "--counts", COUNTS, "-o", data)
    run("calc", "--model", START, "-o", start_curve)
    run("fit", data, "--model", START, "--loss", "poisson", "--counts-scale", COUNTS, "-o", out)
    tth, counts = np.loadtxt(data, unpack=True)
    start = np.loadtxt(start_curve)[:, 1] * COUNTS
    fitted = np.loadtxt(out / "fitted.dat")[:, 1]
    result = json.loads((out / "result.json").read_text())

film = tomllib.loads(TRUE.read_text(encoding="utf-8"))["layer"][0]
values, errors = result["values"], result["errors"]
summary = "\n".join(
    f"{name} {values[f'film.{key}']:.2f} ± {errors[f'film.{key}']:.1g} (true {true:g}) {unit}"
    for name, key, true, unit in [
        ("thickness", "thickness", film["thickness"], "Å"),
        ("roughness", "roughness", film["roughness"], "Å"),
        ("SLD", "sld", film["sld"][0], "10$^{-6}$ Å$^{-2}$"),
    ]
)

fig, (top, bottom) = plt.subplots(
    2, 1, figsize=(7.2, 4.6), dpi=150, sharex=True, height_ratios=(3, 1), layout="constrained"
)
top.semilogy(tth, counts, ".", ms=3, color="0.35", label="simulated data (examples/film.toml)")
top.semilogy(tth, start, "--", lw=1, color="tab:orange", label="start (examples/film_start.toml)")
top.semilogy(tth, fitted, "-", lw=1.2, color="tab:blue", label="fit, Poisson loss")
top.set_ylabel("counts")
top.legend(fontsize=8, frameon=False, loc="upper right")
top.text(0.98, 0.6, f"film\n{summary}", transform=top.transAxes, ha="right", va="top", fontsize=8)
bottom.plot(tth, (counts - fitted) / np.sqrt(fitted), ".", ms=2.5, color="tab:blue")
bottom.axhline(0, color="0.5", lw=0.8)
bottom.set_ylabel("(data − fit) / √fit", fontsize=8)
bottom.set_xlabel("2θ (°)")
bottom.set_xlim(tth[0], tth[-1])
fig.savefig(ROOT / "docs/media/fit.png")
