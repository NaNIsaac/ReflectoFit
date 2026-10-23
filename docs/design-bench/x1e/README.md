# X1e: resolution quadrature at critical edges (ReflectoFit#6)

Pre-registration, committed before the new rule is written. The old rule spent up to 8001 nodes per
point when a medium has a real SLD, because the edge width it resolved went to the floor
`EDGE_FLOOR` (85 s per batch of 64 curves × 2000 points on the GPU, 17 GB of RAM).

## Idea

R is an analytic function of the normal wavevectors, and kz of a medium is √(q² − q_c²)/2: a
square root at its critical edge. Above an edge R − R(q_c) ∝ √(q − q_c), below it the same with an
imaginary kz; R is continuous with a square-root kink, which uniform nodes integrate slowly. Splitting
each point's Gaussian window at the edges inside it, and mapping each piece so that q − q_c ∝ s² at
an edge end (sin² when both ends are edges), makes the integrand smooth in s. Gauss–Legendre panels
in s then converge fast whatever the absorption. Node counts follow the fringe phase change across
each piece and the Gaussian; the waveguide-resonance rule is kept only for absorbing media, whose
resonance width it bounds.

## Criteria

Development grid (constants may be tuned on it): `tests/test_review_physics.py`,
`tests/test_instrument.py`, and the ReflectoLearn default distribution at seed 0.

Confirmation grid (`confirm.py`, new seed 20261025; nothing tuned on it):

- ReflectoLearn's default training distribution: 16 curves, 2000 points from 2θ = 0.05 to 15°
  at 1.54 Å, real film SLD 5-100, real substrate 18-22, oxide with probability 0.7, 2θ offset
  ±0.02°, Kα₂ ratio 0-0.5; σ_q drawn from 1e-4 to 6e-3, and the same 16 stacks again at
  σ_q = 1e-4 and at 6e-3.
- Real-SLD structures: bare Si 20.07; film 9.6 / 1000 Å on Si 20.07; Ni 64 / C 18 / Ni with real
  SLDs (top 100 Å, C 800 Å); a real-SLD waveguide, oxide 25 / 30 Å over a film 5 / 300 Å on Si 20;
  a real 20-period superlattice 30 / 25 Å and 55 / 35 Å on Si. Each at σ_q 1e-4, 5e-4, 2e-3 and at
  dq/q = 0.02, on 600 points from q = 0.01 to 0.2 Å⁻¹.
- The absorbing cases of `tests/test_review_physics.py`, unchanged.

Reference: uniform nodes over ±9 σ (24001 per point; 96001 for points whose window contains an
edge). Where those two references differ by more than 2e-5 at a point, the point is reported as
reference-limited rather than counted.

1. **Accuracy.** Default rule against the reference: max relative error ≤ 1e-4 over every point
   of every confirmation case, including the points within 5 σ of each critical edge.
2. **Cost.** ReflectoLearn's distribution, batch 64 × 2000, σ_q per curve as above: ≤ 1 s and
   ≤ 2 GB peak job memory with the numba backend, measured in a quiet window (or with the load
   stated). The torch backend on the GPU is reported, with the same limits as a target.
3. **No regression.** The existing tests keep their tolerances (`test_review_physics.py` 1e-5,
   `test_instrument.py` 1e-6) and pass.

Falsified if any confirmation case exceeds 1e-4 at a point that is not reference-limited, or the
numba cost exceeds 1 s or 2 GB. A falsified criterion is reported as such and the thresholds are
not moved.

## Result of the first registration (252cc2f)

Falsified (`confirm_accuracy.json`, f207fd5): the real 20-period superlattice was off by up to
2.7e-2 (σ_q 5e-4), 1.3e-2 (dq/q 0.02) and 9.7e-4 (σ_q 2e-3), all near q = 0.048, between the edges
of its two materials and more than 5 σ from either. Every other case held (ReflectoLearn
distribution ≤ 1.2e-6, the other real-SLD stacks and the absorbing cases ≤ 2.4e-6). Inference, not
tested: resonant tunnelling through the evanescent layers gives narrow features whose width is set
by the barriers, not by absorption, and the resonance rule covered absorbing media only.

## Second registration (written before the change below is run)

Change: inside the range between the lowest and highest critical q, each panel is compared with
its two halves (16 Gauss–Legendre nodes each) and split while the difference exceeds 1e-5 of the
point's integral over that range (R ≥ 0, so this bounds the point's relative error), up to a depth
limit. A point that stops at the limit carries its estimated error, `forward` warns, and
`resolution_report` returns the estimate and the flag per point. Everything else stays as above.

Development grid, now including the first confirmation grid (with the 20-period superlattice).

Confirmation grid 2 (`confirm.py --grid 2`, nothing tuned on it):

- ReflectoLearn's default distribution at a new seed, 20261026: 16 curves, σ_q drawn, at 1e-4 and
  at 6e-3.
- New real-SLD multilayers on Si 20.07, each at σ_q 1e-4, 5e-4, 2e-3 and dq/q 0.02 on 600 points
  from q = 0.01 to 0.2 Å⁻¹: a 10-period superlattice 15 / 40 Å and 45 / 30 Å; a 40-period
  superlattice 25 / 20 Å and 60 / 20 Å; a triple waveguide 40 / 50 Å, 10 / 200 Å, 40 / 50 Å,
  10 / 200 Å, 40 / 50 Å; a graded stack of 10 layers, 20 Å each, SLD 10 to 60 in equal steps.
- The absorbing cases of `tests/test_review_physics.py`, unchanged.

Reference and criteria as in the first registration: max relative error ≤ 1e-4 at every point that
is not reference-limited, including points that stopped at the depth limit (the flag does not
excuse them); ReflectoLearn's distribution, batch 64 × 2000, ≤ 1 s and ≤ 2 GB with numba in a quiet
window; the existing tests keep their tolerances. Falsified on any breach; thresholds not moved.

## Third registration (before confirmation grid 2 is run)

Two changes to the code after the second registration, found by the test suite, before any run of
confirmation grid 2:

- The refinement criterion is applied to Kα₁ and Kα₂ each on its own (each line's panel against
  its halves, relative to that line's integral over the range), not to their mix. Reason: the
  intensity ratio can be a traced value (`torch.func.vmap`), so node choice must not depend on it;
  the mix is a convex combination of the two lines, so bounding each bounds the mix. This changes
  the nodes for curves with Kα₂, so it is a change to the rule.
- `REFINE_TOL = inf` switches the refinement off (used only by a regression test); the default is
  unchanged.

The rule as committed in 75c7b1c is the one under test. Development data now includes the rerun of
grid 1 with this rule (`dev1_accuracy.json`, worst 1.7e-5). Confirmation grid 2 and the criteria
are those of the second registration, unchanged; grid 2 has not been run.
