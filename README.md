# ReflectoFit

[![Tests](https://github.com/NaNIsaac/ReflectoFit/actions/workflows/ci.yml/badge.svg)](https://github.com/NaNIsaac/ReflectoFit/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

<img src="https://raw.githubusercontent.com/NaNIsaac/ReflectoFit/main/docs/media/fit.png" width="720" alt="The first run below: Poisson counts from examples/film.toml, the starting model and the fit, with normalised residuals">

A fast X-ray reflectivity fitting engine and command line: layer models, an instrument model, least
squares, differential evolution and MCMC, model selection, and fit reports.

It is for anyone with a measured specular XRR curve of a thin film who wants the thickness, roughness
and density of each layer. The forward model is a float64 Parratt recursion (compiled with numba)
with Névot-Croce roughness, Gaussian resolution, beam footprint, Cu Kα₂ and a 2θ offset; it agrees
with refnx to 1e-10. A torch backend gives the same curves with gradients. The companion simulator,
ReflectoSim, generates synthetic XRR data for machine learning on top of this forward model.

Status: early development. The design is in [`docs/DESIGN.md`](docs/DESIGN.md); model selection
follows.

<img src="https://raw.githubusercontent.com/NaNIsaac/ReflectoFit/main/docs/media/throughput.png" width="720" alt="Forward-model curves per second against batch size for seven engines, for one film, five layers and a 50-layer superlattice">

Forward-model throughput on the benchmark machine (`docs/design-bench/throughput.json`, DESIGN.md section 5). ReflectoFit's numba backend computes about 57 000 five-layer curves per second in batches of 100 000 on CPU; hollow markers are values scaled or extrapolated from a smaller batch.

## Install

Not on PyPI yet. From a clone, with [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/NaNIsaac/ReflectoFit
cd ReflectoFit
uv sync
```

## First run

Simulate a measured curve of a 240 Å film on SiO₂ on Si with Poisson counts, then fit it from a wrong
starting guess:

<!-- test -->
```sh
uv run reflectofit calc --model examples/film.toml --counts 1e8 -o film.dat
uv run reflectofit fit film.dat --model examples/film_start.toml --loss poisson --counts-scale 1e8 -o fit
```

The fit prints each free parameter with its standard error and writes `fit/result.json`,
`fit/model.toml` (the fitted model, reusable as a start) and `fit/fitted.dat`.

Model files, every command and the Python API: [docs/usage.md](docs/usage.md).

## Development

```sh
uv run pytest
```

## License

MIT
