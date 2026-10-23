"""The command line on the bundled example: calc, fit, score, stitch."""

import json
from pathlib import Path

import numpy as np
import pytest

from reflectofit.cli import main

EXAMPLES = Path(__file__).parents[1] / "examples"


def test_calc_then_fit_recovers_the_example(tmp_path, capsys):
    data = tmp_path / "film.dat"
    assert (
        main(["calc", "--model", str(EXAMPLES / "film.toml"), "--counts", "1e8", "-o", str(data)])
        == 0
    )
    out = tmp_path / "fit"
    code = main(
        [
            "fit", str(data), "--model", str(EXAMPLES / "film_start.toml"),
            "--loss", "poisson", "--counts-scale", "1e8", "-o", str(out),
        ]
    )  # fmt: skip
    assert code == 0
    result = json.loads((out / "result.json").read_text())
    assert result["values"]["film.thickness"] == pytest.approx(240, abs=0.1)
    assert (out / "model.toml").exists() and (out / "fitted.dat").exists()
    capsys.readouterr()
    assert (
        main(["score", str(data), "--model", str(out / "model.toml"), "--counts-scale", "1e8"]) == 0
    )
    scores = json.loads(capsys.readouterr().out)
    assert scores["poisson"] < 400


def test_stitch_command(tmp_path):
    a, b = tmp_path / "1.dat", tmp_path / "2.dat"
    np.savetxt(a, [[0.1, 1000.0], [0.2, 500.0]])
    np.savetxt(b, [[0.2, 50.0], [0.3, 20.0]])
    out = tmp_path / "s.dat"
    assert main(["stitch", str(a), str(b), "-o", str(out)]) == 0
    np.testing.assert_allclose(np.loadtxt(out)[:, 1], [100.0, 50.0, 20.0])
