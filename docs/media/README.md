# README media

`fit.png` is drawn by `make_figure.py` from this commit's code, with no measured data:

```sh
uv run --with matplotlib docs/media/make_figure.py
```

The script runs the README's first run through `reflectofit.cli.main` in a temporary directory:
`calc` of `examples/film.toml` with Poisson counts (1e8 at R = 1, seed 0), `calc` of
`examples/film_start.toml` without noise, and `fit --loss poisson`. It plots the counts, the
starting curve and `fit/fitted.dat`, and the residuals scaled by the square root of the fit. Running
it again gives the same file byte for byte (checked on Windows, matplotlib 3.11.2).
