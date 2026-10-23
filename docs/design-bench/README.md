# Engine agreement and throughput (DESIGN.md section 5)

Scripts used for section 5 of `docs/DESIGN.md`. They are measurement scripts, not part of a package.

Environment: a uv virtualenv with Python 3.13 and `numpy scipy numba jax refnx genx3 torch`; CUDA runs
used a separate environment with torch 2.11.0+cu128. `rl_abeles.py` is ReflectoLearn's engine at
`b6fb0c4`:

```console
git -C ReflectoLearn show b6fb0c4:src/reflectolearn/physics/abeles.py > rl_abeles.py
cargo build --release --manifest-path rs/Cargo.toml
python -I compare.py . agreement.json        # section 5.1, 2000 q points
python -I bench2.py . throughput.json numba-f64-par --jac   # section 5.3, one engine per process
python -I bench2.py . throughput.json rust-12
python -I jac_torch.py . throughput.json x
```

Engines for `bench2.py`: `numpy-f64 numba-f64-par rust-12 jax-cpu-f64 jax-cpu-f32 torch-cpu-c128
torch-cpu-c64 torch-cuda-c128 torch-cuda-c64 refnx genx` (`--jac` for numba, JAX f64, torch c128).
`bench.py` holds the engine definitions; its own `main` is the earlier pilot at 2000 q points.

`agreement.json` and `throughput.json` are the outputs quoted in DESIGN.md. `compare.py` reads the
Rust curves from `rust_out/`, which `bench.py` writes.
