# ReflectoFit

ReflectoFit fits specular X-ray reflectivity (XRR) curves of thin films: you give it a measured curve
and a layer model with bounds, and it returns the thickness, roughness and scattering length density
of each layer with their standard errors. It is a library and a command line.

The forward model is a float64 Parratt recursion compiled with numba, with Névot-Croce roughness.
The instrument model adds Gaussian resolution (constant and dq/q terms), the beam footprint, Cu Kα₂,
a 2θ offset, I0 and background. It agrees with refnx, a JAX implementation and the ReflectoLearn
engine to 1e-10, and the resolution integral to within 1e-5 of a brute-force convolution, including
at the critical edges of weakly absorbing films. A torch backend gives the same curves with
gradients.

Fitting: least squares (optionally inside a trust window around the start), differential evolution,
and losses on log R, raw R, Poisson deviance or χ², with AIC and BIC for the likelihood losses.
MCMC, the staged refinement of ReflectoLearn and model selection follow.

[ReflectoSim](https://github.com/NaNIsaac/ReflectoSim) builds synthetic training data on this
forward model, so a network trained on it and a fit of the same measurement use the same physics.

## Install

Not on PyPI yet. From a clone, with [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/NaNIsaac/ReflectoFit
cd ReflectoFit
uv sync
```

## First run

Simulate a curve of a 240 Å film on SiO₂ on Si with Poisson counts, then fit it from a wrong start:

```sh
uv run reflectofit calc --model examples/film.toml --counts 1e8 -o film.dat
uv run reflectofit fit film.dat --model examples/film_start.toml --loss poisson --counts-scale 1e8 -o fit
```

The fit prints each free parameter with its standard error and writes `fit/result.json`,
`fit/model.toml` and `fit/fitted.dat`. The model file format, the command line and the Python API
are in the [README](https://github.com/NaNIsaac/ReflectoFit#readme).

## Conventions

Layers are listed from the top: ambient first, substrate last, and a layer's roughness belongs to
the interface above it. SLD is in 1e-6 Å⁻², complex, with a positive imaginary part for absorption;
lengths in Å, q in Å⁻¹, angles in degrees. The ambient must not absorb. These are pinned by tests
written before the engine, and the reasons are in the [design notes](DESIGN.md).
