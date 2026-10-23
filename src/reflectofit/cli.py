"""Command line: reflectofit calc | fit | score | stitch."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from reflectofit import __version__, data, fit, model
from reflectofit.units import tth_to_q


def _grid(spec: str, wavelength: float) -> np.ndarray:
    """'tth:START:STOP:N' (degrees) or 'q:START:STOP:N' (Å⁻¹)."""
    axis, start, stop, n = spec.split(":")
    values = np.linspace(float(start), float(stop), int(n))
    if axis == "tth":
        return tth_to_q(values, wavelength)
    if axis == "q":
        return values
    raise SystemExit(f"grid axis must be tth or q, got {axis!r}")


def _calc(args) -> int:
    m = model.load(args.model)
    q = _grid(args.grid, m.instrument.wavelength)
    r = m.curves(q, m.start())[0]
    y = (
        np.random.default_rng(args.seed).poisson(args.counts * r).astype(float)
        if args.counts
        else r
    )
    data.write(args.out, data.Curve(q, y, meta={"wavelength": m.instrument.wavelength}), x=args.x)
    print(f"wrote {args.out} ({q.size} points{', Poisson counts' if args.counts else ''})")
    return 0


def _fit(args) -> int:
    m = model.load(args.model)
    curve = data.read(args.data, x=args.x, wavelength=m.instrument.wavelength)
    common = {"loss": args.loss, "counts_scale": args.counts_scale}
    if args.method == "lsq":
        res = fit.lsq(m, curve, trust=args.trust, max_nfev=args.max_nfev, **common)
    else:
        res = fit.de(m, curve, seed=args.seed, **common)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(res.to_json(), encoding="utf-8")
    (out / "model.toml").write_text(model.dumps(res.model), encoding="utf-8")
    lam = res.model.curves(curve.q, res.model.start())[0] * (args.counts_scale or 1.0)
    data.write(out / "fitted.dat", data.Curve(curve.q, lam, meta=curve.meta), x=args.x)
    width = max(len(k) for k in res.values)
    for k, v in res.values.items():
        print(f"{k:<{width}}  {v:.6g} ± {res.errors[k]:.2g}")
    if res.at_bound:
        print(f"on a bound (errors not meaningful): {', '.join(res.at_bound)}")
    print(f"{res.loss} objective {res.objective:.6g} over {res.n_points} points; {res.message}")
    print(f"wrote {out / 'result.json'}, {out / 'model.toml'}, {out / 'fitted.dat'}")
    return 0 if res.success else 2


def _score(args) -> int:
    m = model.load(args.model)
    curve = data.read(args.data, x=args.x, wavelength=m.instrument.wavelength)
    print(json.dumps(fit.score(m, curve, counts_scale=args.counts_scale), indent=1))
    return 0


def _stitch(args) -> int:
    scans = [np.loadtxt(p, ndmin=2) for p in args.scans]
    out = data.stitch(scans, method=args.method)
    np.savetxt(args.out, out)
    print(f"wrote {args.out} ({len(out)} points from {len(scans)} scans)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="reflectofit", description="X-ray reflectivity fitting")
    p.add_argument("--version", action="version", version=f"reflectofit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("calc", help="compute a model curve, optionally with Poisson counts")
    c.add_argument("--model", required=True, help="model TOML")
    c.add_argument("--grid", default="tth:0.1:6:300", help="tth:START:STOP:N or q:START:STOP:N")
    c.add_argument(
        "--counts", type=float, default=None, help="counts at R = 1; draws Poisson counts"
    )
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--x", choices=("tth", "q"), default="tth", help="x column of the output")
    c.add_argument("-o", "--out", required=True)
    c.set_defaults(run=_calc)

    for name, helptext, run in (
        ("fit", "fit a model to a curve", _fit),
        ("score", "losses of a model", _score),
    ):
        f = sub.add_parser(name, help=helptext)
        f.add_argument("data", help="2- or 3-column text: x, intensity[, error]")
        f.add_argument("--model", required=True, help="model TOML with [fit] bounds")
        f.add_argument("--x", choices=("tth", "q"), default="tth", help="x column of the data")
        f.add_argument(
            "--counts-scale", type=float, default=None, help="data are counts; counts at R = 1"
        )
        f.set_defaults(run=run)
        if name == "fit":
            f.add_argument("--method", choices=("lsq", "de"), default="lsq")
            f.add_argument("--loss", choices=("log", "raw", "poisson", "chi2"), default="log")
            f.add_argument(
                "--trust", type=float, default=None, help="limit each start to ±fraction"
            )
            f.add_argument("--max-nfev", type=int, default=200)
            f.add_argument("--seed", type=int, default=0)
            f.add_argument("-o", "--out", default="fit", help="output directory")

    s = sub.add_parser("stitch", help="join scans measured through different attenuators")
    s.add_argument("scans", nargs="+", help="scan files (x, intensity[, error]) in increasing x")
    s.add_argument("--method", choices=("boundary", "median"), default="boundary")
    s.add_argument("-o", "--out", required=True)
    s.set_defaults(run=_stitch)

    args = p.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
