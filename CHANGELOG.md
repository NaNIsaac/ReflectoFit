# Changelog

## Unreleased

- Float64 Parratt kernel (numba) and a differentiable torch backend, matching refnx, the
  ReflectoLearn kernel and JAX to 1e-10.
- Instrument model: Gaussian resolution (constant and dq/q terms), footprint, Cu Kα2, 2θ offset,
  I0, background.
- Materials from composition and density, losses (log, raw, Poisson, chi2) with AIC/BIC, data
  reading, stitching, held-out splits, critical edge and Kiessig thickness estimators.
- Least squares in a trust window and differential evolution; command line `calc`, `fit`,
  `score`, `stitch`.
- Staged gradient fit (`fit.staged`, `reflectofit.staged`): the refine strategy of ReflectoLearn
  (stages, sigmoid bounds, Adam, its losses, Occam penalty) for any differentiable model.
