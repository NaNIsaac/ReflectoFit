# Design: an XRR fitter and an XRR simulator

Status: design for review (BACKLOG X1a). Nothing here is implemented. Names decided 2026-10-24
(NaNIsaac DECISIONS #23): the fitter is **ReflectoFit** (this repository, formerly the temporary
`xrrkit`), the simulator is **ReflectoSim** (repository created with the simulator's first PR).

Labels: **[fact]** is read from code or measured here (script and output in `docs/design-bench/`),
**[inference]** is a reasoned conclusion that has not been tested, **[proposal]** is a design choice
open for review.

## 0. Identity and names

### 0.1 Two projects, not one toolkit

The XRR research repositories each carry their own forward model, instrument effects, generator,
noise, losses, fitting and file reading (section 1). One "toolkit" repository that absorbs all of it
would be a store room, not a project. Instead the shared code splits into two projects, each usable on
its own by someone outside this organisation and each described by one sentence:

| | Fitter | Simulator |
|---|---|---|
| One sentence | A fast X-ray reflectivity fitting engine and command line: layer models, an instrument model, least squares, differential evolution and MCMC, model selection, and fit reports. | Realistic synthetic X-ray reflectivity data for machine learning: structure priors, instrument and counting-noise randomisation, and a dataset format, built on the fitter's forward model. |
| Level | general fitting program of the GenX 3 / refl1d class | data-generation library |
| Users | anyone with a measured specular XRR curve | anyone training or testing an inverse model |
| Owns | the physics: forward model, instrument model, materials, conventions, metrics, measured-file readers, stitching, optimisers, model selection, reports | the data distribution: priors over structures and instruments, noise, dataset files and their metadata |
| CLI | `fit`, `score`, `stitch`, `convert` | `sim`, `describe` |
| Depends on | numpy, numba, periodictable; scipy for fitting; extras: torch, emcee | the fitter (core only), numpy; extra: h5py |

Out of scope for both, and where it stays:

| Out of scope | Stays in |
|---|---|
| neural networks, training, tuning, predict-from-network | ReflectoLearn (and the archived ReflectoExp, Reflecto, XRefine) |
| experiment series, ablations, notebooks, research write-ups | ReflectoLearn `research/*`, ReflectoExp |
| GUI, desktop product | ReflexAuto (ended), Reflecto app |
| measured data, reference fits of measured samples | `$DATA` and the data repositories pending DotFiles#83/#84 |
| beamline-specific scripts (scan lists, attenuator bookkeeping) | xrr_combine (to be renamed xrr_tools), xray_ooguy |

### 0.2 Where the shared forward model lives

Both projects need the same forward model. Three placements were compared:

| Option | For | Against |
|---|---|---|
| **A. In the fitter; the simulator depends on the fitter's core** | one owner of the physics; every change to the model is checked by fitting tests; one version pin (`reflectosim` requires `reflectofit>=x.y`) | installing the simulator pulls the fitter; kept light by importing scipy, emcee and torch only inside the modules that need them |
| B. A small core package | cleanest dependency graph | a third repository, which the brief forbids; or a second distribution from the fitter repository, which is A with more release work |
| C. Both vendor the same file | no dependency | the duplication this work exists to remove comes back on the first edit |

**[proposal] A.** The fitter is the authority on the physics because it is the one checked against
measured curves; the simulator is a consumer that adds a data distribution. The simulator never defines
its own forward model; its tests assert that its curves equal the fitter's forward model for the same
parameters before noise.

### 0.3 Names

Constraints from the brief: toolkit-style names (`xrrkit`, `xrrtoolkit`) are out because they read
like `xrr_tools`; any name starting with `xrr` is out for the same reason. A proper-noun brand signals a
large feature project (the Qora* family here), so it fits the fitter at most; descriptive lowercase names
fit tools and libraries (codigest, clippath, h5explorer, xrd_tools). At most four syllables. The XRR
family here is Reflecto* (Reflecto, ReflectoExp, ReflectoLearn) plus ReflexAuto.

How tools in this field are named (read from the names themselves, not from their authors):

| Tool | Kind | Name strategy |
|---|---|---|
| GenX | general fitter | coinage from the method (genetic / differential evolution) + X |
| BornAgain | general simulation and fitting | pun on the Born approximation; proper-noun brand |
| Motofit | general fitter | coined proper noun |
| Parratt32 | fitter | person (Parratt) + platform |
| anaklasis | fitter | Greek word for reflection |
| refl1d | fitting library | descriptive abbreviation (1-D reflectometry) |
| refnx | fitting library | abbreviation of reflectometry + suffix |
| reflectorch | ML library | portmanteau reflectometry + PyTorch |
| mlreflect | ML library | descriptive (machine learning + reflectivity) |

Long-lived general fitters tend to carry proper nouns; libraries and ML data tools are descriptive.

Collision checks, 2026-10-24: PyPI by `https://pypi.org/pypi/<name>/json` (404 = free; the
`pypi_status` tool named in the brief is not installed on this machine, the JSON API is what it would
query), GitHub by `gh search repos "<name> in:name"`.

Fitter, three candidates:

| Name | Strategy | Syllables | PyPI | GitHub repos named so | Notes |
|---|---|---|---|---|---|
| **reflectofit** | family + job | 4 | free | 0 | reads as the Reflecto family's fitter; pairs with reflectosim |
| Kiessig | person name, proper noun (as Parratt32, BornAgain) | 2 | free | 4, all personal or unrelated (a user `kiessig` exists) | the fringes an XRR fit reads first; brand-like; search results are dominated by "Kiessig fringes", which helps users in the field find it and hurts plain search |
| reflfit | descriptive abbreviation (as refl1d) | 2 | free | 1 (`chelberserker/ReflFIT`) | short; a GitHub namesake exists |

Simulator, three candidates:

| Name | Strategy | Syllables | PyPI | GitHub repos named so | Notes |
|---|---|---|---|---|---|
| **reflectosim** | family + job | 4 | free | 0 | sister of reflectofit; says what it does |
| reflsim | descriptive abbreviation | 2 | free | 0 | no family tie |
| synthrefl | descriptive | 2 | free | 0 | "synth" reads as audio; no family tie |

Rejected: `xrrkit`, `xrrsim`, `xrrsynth`, `xrrfit` (xrr prefix, confusable with xrr_tools);
`reflectogen` ("gen" reads as GenX, whose port lives in the fitter); `reflex*` (ReflexAuto and several
unrelated projects).

Proposed: reflectofit and reflectosim. Same family, each name says the job, the pair makes the
dependency obvious, both free on PyPI and GitHub. If the fitter should stand as a brand outside the
organisation, Kiessig is the proper-noun alternative. Reflecto (the existing CNN + GenX app) keeps its
name; once ReflectoFit exists its README points there for fitting.

**Decided (DECISIONS #23):** repositories `NaNIsaac/ReflectoFit` and `NaNIsaac/ReflectoSim`, matching
ReflectoLearn and ReflectoExp. Distribution and import names are lowercase, as `reflectolearn`:
`reflectofit` and `reflectosim` on PyPI, `import reflectofit`, `import reflectosim`, commands
`reflectofit` and `reflectosim`.

### 0.4 Distribution

Both are published to PyPI (Isaac, 2026-10-24: repositories many people use go to PyPI, as CXD).
Following `.github/CONVENTIONS.md` and the existing `publish.yml` (codigest, QoraPhase, TraTT): tag
`vX.Y.Z`, GitHub release notes, `CHANGELOG.md`, Trusted Publishing from `publish.yml` with environment
`release`, `uv_build` backend, `uv.lock` committed, README install and first-run blocks marked
`<!-- test -->`. `requires-python >= 3.12` (lowest CI-tested version). The names must still be free on
PyPI when the first release is tagged; register the Trusted Publisher as a pending publisher before
that.

## 1. Duplicate code

Read at the commits below; private repositories from local clones, paths and functions only.
Line counts are whole files. "Last change" is the last commit touching the file; many files show
2026-10-22/23 because of repository-wide ruff, conventions and lineage passes, so it is not the date of
the last change in behaviour.

| Repository | Commit | Note |
|---|---|---|
| ReflectoLearn | `b6fb0c4` (origin/main) | active library |
| Reflecto | `635b9bf` (main), `fb5de5c` (local-snapshot-20261022) | snapshot holds split and multilayer fitting, no tests |
| ReflectoExp | `752a601` | exp07 and `src/reflecto_exp`; exp01-exp06 carry their own copies (exp02, exp04, exp05, exp06), not listed per file |
| XRefine | `47956ab` | private |
| ReflexAuto | `a6b91c3` | private; documents and example data only, **no engine source** in any repository (ReflectoLearn `docs/xrr-research-index.md`) |
| xrr_combine | `d61b2bb` | private |
| DiffractoLearn | `f868dc0` | private; a powder-XRD notebook, **no XRR code** |

Column "To": **F** fitter, **S** simulator, **keep** stays where it is (section 0.1).

### 1.1 Forward model and instrument effects

| Function | Implementations (file, lines, tests, last change) | Differences that matter | To |
|---|---|---|---|
| Abeles / Parratt kernel | RL `physics/abeles.py` 167, tested (`test_abeles`, `test_engine_parity` vs Parratt and the two archived engines, `test_golden_xrr`), 2026-10-22 · RL `benchmark.py::parratt` (numpy f64 Parratt) 353, `test_benchmark` · RExp `simulate/torch_engine/abeles.py` 82, `numpy_implementations.py` 128, `memory_eff.py` 104, `kinematical.py` 61, `tests/test_abeles.py` vs Parratt · XRefine `core/abeles.py` 133, `abeles_enhanced.py` 152, `numpy_implementations.py` 136, `memory_eff.py` 116, `kinematical.py` 132, `tests/test_forward_consistency.py` (indirect), 2026-01-22 · GenX used through `spec_nx` in RExp `simulate/simul_genx.py` 216, Reflecto `simulate/simulate.py` 225 / 234, RL `genx_fit.py` 153 · refnx in RExp `simulate/simul_refnx.py` 201 | absorption guard 1e-9j (RL, XRefine), 1e-30j / 1e-36j (RExp); complex64 (RL, XRefine) vs complex128 (RExp when q is f64); GenX uses exact n² (section 5); RExp `simul_genx` sets Im f = 0 (no absorption); RExp `simul_refnx` repeats the whole stack n times (ReflectoExp#12) | F (one f64 Parratt kernel, numba; torch backend) |
| Roughness | Névot-Croce everywhere listed above | none found | F |
| Resolution | RL `refine/pipeline.py::make_forward` (`_smear` constant σ_q on the grid; `_smear_varying` σ(q) = √(σ₀² + (r·q)²)), `test_two_term_resolution` · RL `data/augmentations.py::_smear` (conv1d on a uniform grid, one σ per batch) · RL `benchmark.py::smeared` (dq/q, 17-point quadrature) · RExp `torch_engine/smearing.py` 133 (FWHM, ±1.7 FWHM, 51 taps; constant dq or dq/q on a log grid) · exp07 `augmentations.py` 133 (`gaussian_filter1d`, σ in q) · XRefine `core/smearing.py` 238 (FWHM, dense grid + interpolation) · GenX `res` (σ) | σ vs FWHM; constant vs ∝ q; grid convolution (needs a uniform grid, edge padding) vs pointwise quadrature | F |
| Footprint | RL `abeles.py` and `augmentations.py::_footprint` (erf(L sin θ / (2√2 σ_b)), σ_b = FWHM/2.3548) · RL refine (`erf(s sin θ)`, one scale s) · exp07 `augmentations.py` (erf((L/2) sin θ / (√2 w)), w used as σ) · RExp `instrument_uitls.py` 14 and XRefine `core/corrections.py` 28 (GenX Gaussian form) · GenX `FootType.gauss` | same formula up to parameterisation; XRefine enhanced path scales sin α by 5 (XRefine#7) | F |
| I0, background | RL augmentor (R·i0 + 10^bkg), refine (i0, log10 bkg) · XRefine `models/common.py` (after normalisation) · GenX `I0`, `Ibkg` (background not applied in the Reflecto and exp07 GenX fits, Reflecto#2, ReflectoExp#9; fixes in open PRs) | order relative to smearing and normalisation differs | F |
| Kα2 | RL `data/generator.py` 186 and refine only (weighted sum at q·λ₁/λ₂, λ₂ = 1.54439 Å) | absent elsewhere | F |
| 2θ offset | RL `physics/utils.py::apply_tth_offset` 42 (2θ → q exactly) · exp07 q shift N(0, 0.002) Å⁻¹ by interpolation · RExp `torch_engine` q + q_offset · XRefine `core/smearing.py` q + 0.01·δ (about 7× too small, XRefine#6) | angle vs q parameterisation | F |
| Wavelength | 1.5406 Å (RL, XRefine), 1.54 Å (RExp, Reflecto scripts) | 0.04 % in q | F (one constant, explicit) |
| Materials / constants | RL `benchmark.py` (Si 20.07+0.46j, SiO₂ 18.9+0.24j) · RExp `consts_genx.py` 13 and Reflecto `utils/consts_genx.py` 20 / 35 (Si 19.02, SiO₂ 19.73) · Reflecto `utils/materials.py` 52, `physics.py` 96 (density, electron density) · XRefine `constants.py` 34 (no material table) | Si/SiO₂ values disagree (ReflectoLearn#8, DECISIONS #8) | F (`materials`: SLD from composition and density) |
| SLD profile | Reflecto `simulate/profile.py` 46 (erf profile) | | F (report) |

### 1.2 Generators and noise

| Function | Implementations | Differences | To |
|---|---|---|---|
| Structure generator / domain randomisation | RL `data/generator.py` (film + optional top oxide, substrate range, Kα2, offset; streaming), 2026-10-22, tested through `test_benchmark` only · RExp `simul_genx.py::XRRSimulator` (SiO₂ under film) and `simul_refnx.py` (Dirichlet thickness split) · exp07 `simulate.py` 59 · Reflecto `simulate/simulate.py` · XRefine `data/generator.py` 183 (top layer + film + substrate, complex SLD, instrument ranges), `test_forward_consistency` | oxide on top (RL) vs under (RExp, Reflecto), DECISIONS #19; RL draws roughness without σ < d constraint, RExp uses σ ≤ 0.3 d | S |
| Noise | RL `augmentations.py` (Poisson at s = 10^U[5,8]) · RExp and Reflecto `simulate/noise.py` 23 (Poisson at 1e8 × multiplicative × Gaussian background) · exp07 `augmentations.py` · XRefine `data/augmentations.py` 162 (resampling, Poisson, masks) | none tested; count scale conventions differ | S |
| Masks, resampling | RL augmentor (end masks), XRefine (low/high-q cut, resampling) | | S |
| Normalisation for networks | RL generator (max-norm, log10, floor −15, mask channel) · XRefine `data/normalization.py` 151 (plateau, max, anchor) · exp07 `dataset.py` | | keep (consumer side, ReflectoLearn) |

### 1.3 Fitting

| Function | Implementations | Differences | To |
|---|---|---|---|
| Losses / metrics | RL `refine/losses.py` 124 (log-MSE, linear, correlation, log-gradient, hybrid, Poisson deviance), tested through refine tests · XRefine `refine/losses.py` 81 (same family, fringe-weighted) · RExp `math_utils.py::fom_log` 54, tested · GenX FOMs `logR1`/`logR2` (Reflecto, exp07) | no χ² with errors anywhere; one signature already in RL | F (`metrics`) |
| Gradient refine, trust region | RL `refine/pipeline.py` 816 (`refine`, staged Adam, `RefineConfig.likelihood`, `fft_thickness`, `refine_multistart`), tests `test_refine_alias_mcmc`, `test_refine_likelihood`, `test_offset_last_stage`, `test_example_synthetic_film_fit`, 2026-10-23 · XRefine `refine/optimizer_robust.py` 189 | RL default stops short of the optimum (ReflectoLearn#25, DECISIONS #17) | F (`fit.staged`) in X1c |
| DE, grid search | XRefine `refine/optimizer.py` 144, `optimizer_enhanced.py` 273, `pipeline.py` 378 (I0 pre-fit, thickness grid, DE stages), `pipeline_next.py` 588; no tests | `pipeline_next` grid replaces the stage fit; GD stages run under `no_grad` (XRefine agent reading, not verified here) | F (`fit.de` from scipy, strategy ideas only) |
| GenX fitting | RL `genx_fit.py` 153 (`genx_fit`, two-stage DE), `test_genx_fit` · Reflecto `engines/fitting.py` 179 / 229, `script_builder.py` 95, `tests/test_script_builder.py` · exp07 `engine/fitting.py` 168, `script_builder.py` 95, `test_exp07_genx_script.py` · Reflecto snapshot `fitting_split.py` 319, `script_builder_multi.py` 122 | oxide put on top in every GenX fitter; duplicate `s_d`, missing `s_sld` in the exp07/Reflecto substrate stage | F (`fit.genx`, optional extra, adapter that flips layer order) |
| Least squares | Reflecto snapshot `fit_manage.py` 109 (scipy `least_squares`, linear residual, broken `__main__`); ReflectoLearn `research/identifiability` `scan.py` (trf + `jacfwd`, research branch) | | F (`fit.lsq`, the default method) |
| MCMC | RL `refine/mcmc.py` 183 (emcee, `sample_posterior`), `test_refine_alias_mcmc`; convergence open (ReflectoLearn#29) | only one | F (`fit.mcmc`, extra `emcee`) |
| Model selection, superlattice detection | none in code. ReflexAuto describes both in its documents; source not found. RL `fft_thickness` and Reflecto snapshot `_estimate_thickness_from_fft` (FFT of R·q⁴) are the only estimators | new code, specified by ReflexAuto docs and RL#35 | F (`select`, `estimate`) |

### 1.4 Data handling

| Function | Implementations | Differences | To |
|---|---|---|---|
| Stitch attenuated ranges | xrr_combine `src/xrr_combine.py` 75 (one shared boundary point, last scan is reference), `tests/test_xrr_combine.py`, no CLI arguments · RL `stitch.py` 35 (NumPy port), `test_stitch` · xray_ooguy DATA view (median ratio over ≥ 2 overlapping points) | two algorithms | F (`data.stitch`, both methods) ; beamline script keeps calling it |
| Measured-file reading | RL `predict.py::load_measured_curve` (2 columns, 2θ or q, comment skipping, commas) · RExp `io_utils.py::load_xrr_dat` 82, tested · Reflecto `utils/io.py` 8 / 13 · XRefine `load_real_data` ×3 (predict_real, pipeline, pipeline_next) · xrr_combine `read_data` (x, y, error) | error column ignored by most; no `.ras`/`.brml`/`.xy` reader anywhere | F (`data.read`) |
| Dataset files | RExp `simul_genx.py::save_hdf5` (keys q, thickness, roughness, sld, sio2_*, R, all f4) · Reflecto same · exp07 `dataset.py` · RL streams, no files | no metadata on instrument, seed, engine or conventions | S (`dataset`) |
| Units, q from 2θ | RL `physics/utils.py::tth2q`, RExp `physics_utils.py`, Reflecto `utils/physics.py`, XRefine `physics_utils.py` | λ default 1.54 vs 1.5406 | F |

Defects found while reading, filed so the port does not carry them: XRefine#6 (offset factor),
XRefine#7 (enhanced clamps), ReflectoExp#12 (refnx stack repeated). Already filed: ReflectoExp#9,
Reflecto#2, Reflecto#3, ReflectoLearn#8, #25, #29, #32.

## 2. Conventions

These are fixed in the fitter and every other module uses them. **[proposal]** unless marked.

| Item | Convention |
|---|---|
| Layer order | ambient first, substrate last (refnx, ReflectoLearn). Each medium j > 0 has (SLD, thickness, roughness of the interface above it). The GenX adapter reverses the order; GenX is substrate-first, `Stack.Layers[0]` lies on the substrate [fact, genx 3.8 `refl_base.py`]. |
| Oxide | no implicit oxide. A structure lists every medium. The generator's templates name the order: `film_on_oxide_on_si` (SiO₂ under the film, DECISIONS #19 recommendation b) and `oxide_on_film_on_si` (surface oxide, option c's second half). The default template follows the Isaac decision on DECISIONS #19; until then the generator requires the template to be chosen explicitly. |
| Units | thickness, roughness, wavelength Å; SLD 1e-6 Å⁻² complex; q Å⁻¹; 2θ and offsets degrees; sample length and beam width mm. |
| Absorption sign | Im SLD > 0 is absorption in every public API. The Parratt kernel conjugates internally; the Abeles form used by ReflectoLearn takes it as is. [fact] Mixing the two conventions gives wrong curves without any error (relative errors 8 to 600 on the reference structures in this audit's first, wrong, run); a test pins it. The ambient must not absorb (air or vacuum): the kernel uses SLD relative to the ambient, so an absorbing ambient turns every less absorbing medium into gain (R > 1, physics review on ReflectoFit#2). |
| Kernel | float64 Parratt with k² = (q/2)² − 4π(ρ − ρ_ambient), Névot-Croce roughness. No absorption guard; absorption comes from the materials (DECISIONS #9). This form is exact within the model, with no small-angle step when q is measured in the ambient: tabulated δ is r_e λ² n_a f₁ / 2π, so 1 − 2δ (+ 2iβ) is the model's ε. Exact n² (GenX) squares the first-order root n = 1 − δ and puts back a δ² term the model does not have (physics review on ReflectoFit#1). |
| Materials | SLD from composition and mass density with a named f₁, f₂ table (DECISIONS #8): the fitter uses the X-ray tables of `periodictable`, which give at Cu Kα₁ Si 2.33 g/cm³ → 20.071 + 0.457i and SiO₂ 2.20 g/cm³ → 18.865 + 0.244i (DECISIONS #8's 20.06 for Si came from another table, 5e-4 lower). No copied constants. |
| Wavelength | explicit; default Cu Kα₁ 1.5406 Å. Kα₂ 1.54439 Å, intensity ratio a (physical ≈ 0.5) as a parameter. |
| Resolution | Gaussian in q, given as a standard deviation (not FWHM): σ_q(q) = √(σ₀² + (r·q)²). σ₀ constant term, r = dq/q term. Evaluated pointwise at each measured q, so non-uniform grids and masks need no resampling, by a Gaussian-weighted trapezoid over ±6.5 σ with spacing h (in σ) = min(0.5, 3e-4 Å⁻¹ / σ_max, 2π / (2 σ_max T + 8)), T the total thickness: the second term resolves the square-root kink of R at a critical edge, the third the fringes. A fixed 17-node Gauss-Hermite rule, the first choice, was off by 1.4e-3 at σ_q 1e-3 and 2000 Å and returned spurious fringes beyond σ·d ≈ 5 (physics review on ReflectoFit#2); the uniform rule is within 1e-6 of a brute-force convolution on that grid. Typical lab resolution (σ_q 5e-4) takes 27 nodes. Conversions from FWHM inputs are explicit (`fwhm_to_sigma`). |
| Footprint | factor erf(L sin θ / (2√2 σ_b)) for a Gaussian beam centred on the sample: σ_b the beam's Gaussian σ in mm (FWHM / 2.3548), L the sample length in mm, θ the true incidence angle (after the 2θ offset). A one-parameter form erf(s sin θ) is the fitting reparameterisation. |
| Order of effects | the true angle is the nominal 2θ + offset; the resolution is a Gaussian in q around the q of that angle (as ReflectoLearn and the X2 environment). For each node: R(q) × footprint(θ of the node) → Kα₂ mix at the same angle; then the weighted sum over nodes → × I0 → + background. The footprint sits inside the angular integral because it depends on angle; multiplying it after the convolution is an approximation that fails first at the lowest angles, and inside costs the same. |
| Background, I0 | R_obs = I0 · (...) + b, b in R units; fitting may parameterise log10 b. |
| Counts | expected counts λ = S · R_obs with S the counts at R = 1 for that point (counting time and attenuators folded in, per point when stitched). Noise model: Poisson(λ) + optional dark counts and Gaussian read noise σ_read. Stored data keep the counts; R = y / S is derived. |
| Precision | float64 for fitting and for reference curves. complex64 allowed only for the simulator's GPU bulk path, with its error recorded in the dataset metadata (section 5: max relative 7e-3 in deep minima, median 3-7e-5 for the ReflectoLearn kernel). |

## 3. API

Sketch for X1b; names follow section 0's proposal. Arrays are batched over parameter sets: leading
dimension B, q dimension Q.

### 3.1 Fitter core (no scipy, emcee or torch import at module load)

```python
from reflectofit import Stack, Layer, Instrument, forward, materials

stack = Stack.from_layers(
    ambient=0,
    layers=[
        Layer(sld=45 + 1.5j, thickness=240, roughness=6),
        Layer(sld=materials.sld("SiO2", density=2.20), thickness=10, roughness=3),
    ],
    substrate=Layer(sld=materials.sld("Si", density=2.33), roughness=3),
)
inst = Instrument(
    wavelength=1.5406,
    sigma_q=5e-4,
    dq_over_q=0.0,
    tth_offset=0.0,
    kalpha2_ratio=0.0,
    sample_length=10,
    beam_sigma=0.04,
    i0=1.0,
    background=1e-8,
)
r = forward(q, stack, inst)  # (Q,) or (B, Q); numba float64
r = forward(q, params, inst, backend="torch")  # differentiable, complex128 by default
```

`params` is the packed form `(sld[B, N], thickness[B, N], roughness[B, N])` that every backend
accepts; `Stack` and `Instrument` are thin dataclasses that pack to it.

### 3.2 Metrics (one signature; X2 uses these to choose the loss)

```python
from reflectofit import metrics

metrics.residuals(kind, lam, y, scale=None, sigma=None)  # kind: "log" | "raw" | "poisson" | "chi2"
metrics.loss(
    kind, lam, y, mask=None, scale=None, sigma=None
)  # sum of squared residuals over the mask
metrics.deviance(lam, y)  # Poisson deviance, zero counts included
metrics.information_criteria(deviance, k, n)  # -> (aic, bic)
```

`log`: log10 λ − log10 max(y, 0.5); `raw`: (λ − y)/S; `poisson`: sign(λ − y)·√(2[λ − y + y ln(y/λ)]);
`chi2`: (λ − y)/σ. These are the residuals of ReflectoLearn#35 section 4.3, so a least-squares solver
minimises the matching loss. The existing ReflectoLearn presets (log-MSE, linear, correlation,
log-gradient, hybrid) are kept as named losses for the staged refine port.

### 3.3 Data

```python
from reflectofit import data

curve = data.read(path, x="tth", wavelength=1.5406)  # -> Curve(q, y, sigma, meta); 2-3 column text
curve = data.stitch([a, b, c], method="boundary")  # xrr_combine; "median" = xray_ooguy rule
train, held = data.block_split(q, n_blocks=8, held=(2, 6), q_min=tth2q(0.5))
```

Format adapters for instrument files (.ras, .brml, .xy) are new code and come after X1b, each with a
synthetic test file; measured files are never committed (section 6).

### 3.4 Fit

```python
from reflectofit import fit, estimate, select

model = fit.Model(
    stack, inst, free={"film.thickness": (180, 300), "film.sld": (20, 80), "i0": (0.5, 2)}
)
res = fit.lsq(model, curve, loss="poisson", max_nfev=200)  # scipy trf, Jacobian from the backend
res = fit.de(model, curve, loss="log", popsize=40, maxiter=400)  # scipy differential_evolution
post = fit.mcmc(model, curve, res, steps=20000)  # emcee extra; reports tau and warns < 50 tau
res = fit.staged(
    model, curve, preset="likelihood", counts_scale=S
)  # ReflectoLearn refine port (X1c)
res = fit.genx(model, curve)  # genx3 extra; checks against our forward model
best = select.layer_count(curve, candidates=[1, 2, 3], criterion="bic")
estimate.critical_q(q, y)
estimate.kiessig_thickness(q, y)  # port of fft_thickness
```

`res` carries the parameters, the loss, the deviance, AIC/BIC, the covariance from the Jacobian, the
stopping reason and the settings; `res.to_json()` and `report.figure(res)` write the outputs.

### 3.5 Simulator

```python
from reflectosim import priors, simulate, noise, dataset

prior = priors.stack(
    n_layers=(1, 3),
    thickness=(20, 250),
    roughness_max_ratio=0.25,
    sld=(5, 80),
    min_contrast=4.0,
    template="film_on_oxide_on_si",
)
inst = priors.instrument(
    sigma_q=(5e-4, 3e-3),
    tth_offset=(-0.01, 0.01),
    i0=(0.8, 1.2),
    log_background=(-8.5, -7.0),
    kalpha2_ratio=(0, 0.6),
    counts=(1e6, 1e8),
)
batch = simulate(prior, inst, q, n=100_000, seed=(20261024, 1))  # truth params + R + counts
dataset.write("train.h5", batch, prior=prior, instrument=inst)
```

`priors.stack` is the constrained stack generator X2 asks for (adjacent contrast, σ ≤ d/4, seeded
by a tuple as in RL#35). Every array in a dataset has a named parameter table; the file's attributes
hold the prior and instrument specifications as TOML text, the seed, the package versions, the backend
and precision, and the conventions version, so a dataset can be regenerated.

### 3.6 What X2 asked for (ReflectoLearn#35 section 10)

| Request | Where |
|---|---|
| N-layer forward, vectorised, float64, torch-differentiable, i0, background, resolution, 2θ offset (Kα2, footprint optional), checked against Parratt | `forward(..., backend="torch")`, 3.1; tests in X1b |
| `deviance`, `residuals(kind, lam, y, scale)` for log, raw, poisson, `information_criteria` | `metrics`, 3.2 |
| `block_split(q, n_blocks, held, q_min)` | `data.block_split`, 3.3 |
| seeded constrained stack generator | `reflectosim.priors.stack`, 3.5 |
| `critical_q`, `kiessig_thickness` | `estimate`, 3.4 |

## 4. Command line

```console
reflectofit fit scan.dat --x tth --model model.toml --method lsq --loss poisson --out fit/
reflectofit score scan.dat --model model.toml --loss log raw poisson   # metrics of a given model
reflectofit stitch 1.dat 2.dat 3.dat --method boundary -o stitched.dat
reflectofit convert scan.ras -o scan.dat
reflectosim sim --prior prior.toml --n 1000000 --q "tth:0.05:8:500" --seed 1 -o train.h5
reflectosim describe train.h5
```

`model.toml` lists media top-down with values and bounds, the instrument, and which parameters are
free; `fit` writes `result.json` (parameters, uncertainties, loss, AIC/BIC, settings, versions), the
fitted curve as text, and a figure. Exit code non-zero when the fit did not converge, so scripts and
the X2 environment can check it.

How each repository would switch:

| Repository | Change | When |
|---|---|---|
| ReflectoLearn | `physics/abeles.py` → `reflectofit.forward(backend="torch")`; `data/generator.py`, `augmentations.py` → `reflectosim`; `refine/losses.py` → `reflectofit.metrics`; `refine/pipeline.py`, `mcmc.py` → `reflectofit.fit.staged`, `fit.mcmc`; `stitch.py`, `genx_fit.py` → fitter. Networks, training, predict stay. Equivalence tests: same seed, same curves | X1c |
| ReflectoLearn `research/fit-policy` (X2) | imports `metrics`, `data.block_split`, `estimate`, `priors.stack` | after X1b |
| ReflectoExp | nothing: archived experiments keep their code for reproducibility | proposal only |
| Reflecto | its GenX fitter could become `reflectofit fit --method de` after DECISIONS #19 | proposal only |
| XRefine | nothing; its strategies (I0 pre-fit, thickness grid, DE then gradient) are candidates for `fit.staged` presets after review | proposal only |
| xrr_combine | its CLI calls `reflectofit.data.stitch` | proposal only |
| ReflexAuto, DiffractoLearn | nothing to switch (no engine source, no XRR code) | — |

## 5. Engine choice

All numbers measured on 2026-10-24 on the lab PC (Ryzen 5 7500F 6 cores / 12 threads, RTX 5070,
Windows 11, Python 3.13.14, numpy 2.5.4, numba 0.68.0, jax 0.12.0.1, torch 2.14.1 CPU and 2.11.0+cu128
for CUDA, refnx 0.1.67, genx3 3.8.13, rustc 1.99.1). Section 5.3 is the quiet-window run (01:15-01:20
local clock, no other heavy job; CPU load 25 % just before and 14 % just after, which includes starting
the runs). The first, non-quiet pilot is kept as `throughput_pilot.json`; it understated numba's batch
throughput by up to 6× (thread contention). Scripts and raw JSON are in `docs/design-bench/`.

### 5.1 Agreement

Baseline: float64 numpy Parratt (the recursion of ReflectoLearn `benchmark.parratt`). Structures:
film 240 Å on Si; film + 10 Å SiO₂ under it; bilayer + SiO₂; 10-period superlattice + cap + SiO₂ (24
media). 2000 q points, 2θ 0.05-8°, no resolution.

| Engine | max relative difference (four structures) | median | Verdict |
|---|---|---|---|
| numpy Abeles, complex128 | 6e-15 – 2.2e-13 | ≤ 2e-15 | same |
| ReflectoLearn Abeles kernel in complex128, no guard | 2.8e-13 – 5.1e-13 | ≤ 2.4e-15 | same |
| refnx 0.1.67 (`dq=0`) | 2.8e-13 – 5.1e-13 | ≤ 2.6e-15 | same |
| JAX Parratt float64 | 2.8e-13 – 1.6e-12 | ≤ 1.1e-15 | same |
| Rust Parratt float64 (written for this test; two structures) | 3.6e-13 – 2.7e-12 | ≤ 1.9e-15 | same |
| **GenX 3.8.13 `ReflQ`** | **2.6e-4 – 5.7e-3** | 2.4e-5 – 2.6e-5 | **> 1e-6: an SLD relabelling (physics review)** |
| numpy Parratt with exact n² | identical to GenX to all printed digits | | explains GenX |
| **ReflectoLearn `AbelesMatrix` as shipped (complex64, +1e-9j)** | **1.0e-3 – 7.3e-3** | 3.3e-5 – 6.8e-5 | **> 1e-6: precision, known (DECISIONS #9)** |

[fact] Five independent implementations agree to ~1e-12, so the float64 Parratt is a well-defined
reference. GenX builds k from n² with n = 1 − δ. The physics review (ReflectoFit#1, reproduced
independently) showed this is exact: the exact-n² form equals the linear form with every SLD scaled by
1 − λ²ρ/4π (≤ 2.5e-5 at Cu Kα, 3.8e-6 for Si), at every q; feeding that map into the linear kernel
reproduces GenX to 4e-10. Fits therefore differ only in SLD, by that factor, with d and σ unchanged.
The relative difference in R is largest in deep minima, but in counts the minima carry almost nothing:
at 10⁸ counts and fixed parameters the difference is ΔD ≈ 5 over 2000 points, 82-88 % of it below
1.2 q_c (a critical-edge shift Δq_c/q_c = −λ²ρ/8π). The linear form is the model's; GenX can serve as a
third cross-check at about 1e-9 when its SLDs are given through the inverse map. ReflectoLearn's
shipped engine differs by float32 rounding and the guard, not by algorithm.

Consequence for X1b's acceptance test ("matches ReflectoLearn physics and XRefine Abeles to 1e-6"):
it cannot hold against the shipped complex64 engines. **Adopted (BACKLOG X1b, 2026-10-24):** the new
kernels are tested against the float64 Parratt and refnx at 1e-10 relative, against ReflectoLearn's
kernel run in complex128 at 1e-10, and against the shipped complex64 engines at the measured float32
level (max relative 1e-2, median 1e-4), stated as such.

A side finding from writing the Rust kernel: computing Im √z as √((|z| − Re z)/2) loses digits
when Re z > 0 and Im z is small (here 5e-8 relative error at high q). The stable form takes the large
component from √((|z| + |Re z|)/2) and the small one from Im z divided by twice the large one: above
the critical edge (Re z ≥ 0) the large one is Re √z, below it (the plateau, Re z < 0) it is Im √z, where
Im z / (2 Re √z) would divide by a rounded zero. `rs/src/main.rs` branches this way. Every backend's
complex square root gets a test on both branches at small absorption.

### 5.2 Targets

Set before choosing:

- Simulator: 1 000 000 curves, 5 layers, 500 q points in under 1 minute on this PC's CPU
  (≥ 17 000 curves/s), without requiring a GPU.
- Fitter: one forward evaluation plus Jacobian for a single film at 500 q points in under 1 ms, so a
  200-evaluation least-squares fit takes well under a second and a 40 × 400 differential evolution
  under 2 s at batch throughput.

### 5.3 Throughput

Structures: single film (3 media), 5 layers + Si (7 media), 50-layer superlattice + Si (52 media);
500 q points; float64 unless noted. Curves per second; batch 1 is the median per call (fit latency),
1k best of 3, 100k measured where under 20 s, otherwise extrapolated from 1k (marked *).

| Engine | single B=1 | 1k | 100k | 5-layer B=1 | 1k | 100k | SL50 B=1 | 1k | 100k |
|---|---|---|---|---|---|---|---|---|---|
| numpy (vectorised over batch) | 10 500 | 9 080 | 9 290 | 3 930 | 3 120 | 3 120* | 418 | 355 | 355* |
| **numba, parallel** | 15 800 | 175 000 | 161 000 | 6 010 | 60 000 | **57 100** | 728 | 6 950 | 6 930 |
| Rust, 12 threads (B=1 single thread) | 32 500 | 191 000 | 196 000 | 10 900 | 77 400 | 78 200 | 1 270 | 9 250 | 9 030 |
| JAX CPU float64, jit | 19 600 | 63 300 | 50 400 | 7 320 | 18 900 | 15 500 | 1 490 | 1 260 | 1 260* |
| JAX CPU float32 | 19 100 | 141 000 | 98 000 | 10 700 | 33 600 | 28 000 | 2 160 | 1 780 | 1 780* |
| torch CPU complex128 (RL kernel) | 4 380 | 21 400 | 20 600 | 2 130 | 5 590 | 5 310 | 415 | 639 | 639* |
| torch CPU complex64 (RL as shipped) | 4 000 | 27 200 | 27 500 | 2 260 | 7 930 | 8 380 | 451 | 921 | 921* |
| torch CUDA complex128 | 2 290 | 361 000 | 361 000 | 1 010 | 114 000 | 114 000 | 142 | 11 500 | 11 500 |
| torch CUDA complex64 | 1 620 | 1 850 000 | 1 640 000 | 874 | 342 000 | 338 000 | 139 | 28 800 | 28 800 |
| GenX 3.8 `ReflQ` (numba, one thread, loop) | 46 900 | 47 000† | 47 000* | 22 800 | 21 800† | 21 800* | 3 330 | 3 210† | 3 210* |
| refnx 0.1.67 (loop) | 839 | 850† | 850* | 746 | 826† | 826* | 789 | 795† | 795* |

† measured on 100 curves and scaled. torch CUDA 100k runs as 100 chunks of 1k; the shipped kernel holds
(B, Q, N) intermediates at once and ran out of device memory at B = 4096 with 24 media and 2000 q in
the pilot: throughput fell to about 1 200 curves/s and the complex128 run did not finish within 60 s, consistent with exceeding device memory (not confirmed). Chunking is required either way.

Jacobian, single film, batch 1, 9 parameters (d, σ, Re ρ of each medium): numba central differences
(18 curves in one call) **0.178 ms**; JAX `jacfwd` float64 0.52 ms; torch forward-mode
(`autograd.functional.jacobian`, vectorised) 2.6 ms, reverse-mode 16 ms (222 ms without
`vectorize=True`).

### 5.4 Decision

- **Reference**: float64 Parratt in numpy, kept readable in the test suite as the specification;
  refnx as an independent cross-check in the test dependency group.
- **CPU backend (default, fitter and simulator): numba float64.** Meets both targets with margin
  (5 layers: 57 000 curves/s, 1M in 18 s; Jacobian 0.178 ms). Pure-Python wheels on every platform, no
  build matrix, numba is already GenX's accelerator.
- **GPU and autodiff backend: torch** (complex128 for fitting; complex64 allowed for bulk simulation
  with the recorded error), chunked. ReflectoLearn trains in torch and X2 uses `torch.func.jacfwd`, so
  this keeps their switch an import change. 1M 5-layer curves take about 9 s (complex128) on the RTX
  5070.
- **Not now: Rust.** In the quiet window it is 1.2-1.4× faster than numba at batch 100k and 1.7-2×
  at batch 1; seltools shows the PyO3 build (maturin, five targets). Neither target needs it; it stays
  an option behind the same backend interface if a later target (for example 50-layer superlattice
  fitting) misses.
- **Not adopted: JAX.** Good CPU float64 numbers, but GPU on native Windows needs WSL2 (DECISIONS #18),
  and it would be a third array framework next to numpy and torch.

Falsification (pre-registered in ReflectoFit#1 before the quiet run): the numba choice is wrong if,
in the quiet re-measurement, 5-layer batch-100k throughput is below 17 000 curves/s or the single-film
Jacobian above 1 ms; then Rust is built behind the backend interface. **Result: 57 100 curves/s and
0.178 ms; numba float64 stands.**

## 6. Measured data

Measured curves and reference fits are not committed to either repository until DotFiles#83/#84 decide
what may be published. The repositories hold readers, a metadata schema (`Curve.meta`: instrument,
wavelength, x axis, attenuator segments, counting time, source path) and synthetic test files only.
Benchmarks against measured data read an external path (`$DATA/...`) given on the command line and
are skipped in CI when the path is missing. No credentials or personal data appear in either.

## 7. Plan

1. X1a (this): design; physics review (conventions, section 5), xrr-policy review (API sufficiency).
2. Names decided (DECISIONS #23); `xrrkit` renamed to ReflectoFit. ReflectoSim is created with its
   first PR.
3. X1b: fitter core (`forward` numba + torch, `Stack`, `Instrument`, `materials`, `metrics`, `data`,
   `estimate`), simulator (`priors`, `simulate`, `noise`, `dataset`), CLI `sim` and `score`, tests per
   section 5.1, README with tested first-run blocks, CI from the shared workflows, quiet re-measurement.
4. X1c: ReflectoLearn switches; `fit.staged`, `fit.mcmc`, `fit.genx`, CLI `fit`.
