# Usage

Model files, the command line and the Python API. Install and a first run are in the README.

## Models

A model is a TOML file: the instrument, the layers from the top, the substrate, and the free
parameters with their bounds.

```toml
[instrument]
wavelength = 1.5406    # Å
i0 = 1.0
background = 3e-8
sigma_q = 5e-4         # Gaussian resolution σ in q [Å⁻¹]; dq_over_q adds a term ∝ q
sample_length = 10.0   # mm, with beam_sigma (mm) for the footprint
beam_sigma = 0.04

[[layer]]
name = "film"
sld = [45.0, 1.5]      # 1e-6 Å⁻², [real, imaginary]; imaginary > 0 is absorption
thickness = 240.0      # Å
roughness = 6.0        # Å, of the interface above this layer

[[layer]]
name = "oxide"
material = "SiO2"      # or give the SLD from composition and density
density = 2.20         # g/cm³
thickness = 12.0
roughness = 3.0

[substrate]
material = "Si"
density = 2.33
roughness = 3.0

[fit]
"film.thickness" = [150.0, 350.0]
"film.sld" = [20.0, 80.0]
i0 = [0.5, 2.0]
log10_background = [-10.0, -5.0]
```

Free parameters: `<layer>.thickness`, `.roughness`, `.sld`, `.isld` for each named layer and
`substrate`; the instrument fields `i0`, `background`, `log10_background`, `tth_offset`, `sigma_q`,
`dq_over_q`, `kalpha2_ratio`, `sample_length`, `beam_sigma`.

## Command line

| Command | What it does |
|---|---|
| `reflectofit calc --model M -o out.dat` | model curve on a grid (`--grid tth:0.1:6:300`), optionally with Poisson counts |
| `reflectofit fit data.dat --model M` | least squares (`--method lsq`, `--trust 0.2` limits each start to ±20 %) or differential evolution (`--method de`); losses `log`, `raw`, `poisson`, `chi2` |
| `reflectofit score data.dat --model M` | every loss the data allow, with AIC and BIC for the likelihood ones |
| `reflectofit stitch 1.dat 2.dat -o all.dat` | join scans measured through different attenuators (`--method boundary` or `median`) |

Data files are text with 2θ in degrees (or q in Å⁻¹ with `--x q`), intensity and an optional error
column. With `--counts-scale S` the intensity is read as counts, S counts at R = 1; the Poisson loss
needs it.

## Python

```python
import numpy as np
from reflectofit import Instrument, Layer, Stack, forward, materials, tth_to_q

stack = Stack(
    [Layer(45 + 1.5j, 240, 6, "film"), Layer(materials.sld("SiO2", 2.20), 12, 3, "oxide")],
    Layer(materials.sld("Si", 2.33), roughness=3),
)
q = tth_to_q(np.linspace(0.1, 6, 300))
r = forward(q, stack, Instrument(sigma_q=5e-4, background=3e-8))
r_torch = forward(q, stack, backend="torch")  # differentiable, needs the torch extra
```

