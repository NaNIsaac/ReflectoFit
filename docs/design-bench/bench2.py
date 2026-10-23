"""Forward-model throughput on the design spec: single film, 5 layers, 50-layer superlattice; q 500 points;
batch 1, 1k, 100k. Float64 unless the engine name says c64/f32.

Usage: python -I bench2.py <rl_abeles dir> <out.json> <engine> [--jac]

Batch 1 is the median of many calls (fit latency). Batch 1k is best of 3. Batch 100k is measured when the
1k rate predicts under 20 s, otherwise extrapolated linearly from 1k (the array engines run 100k as 100
chunks of 1k anyway) and marked as such.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, sys.argv[1])

from bench import ENGINES
from structures import SI, SIO2, WAVELENGTH, arrays

NQ = 500
BATCHES = (1, 1_000, 100_000)
CHUNK = 1_000


def q_grid():
    tth = np.linspace(0.05, 8.0, NQ)
    return 4 * np.pi / WAVELENGTH * np.sin(np.radians(tth / 2))


def spec_structures():
    amb = (0j, 0.0, 0.0)
    five = [
        (40 + 1.0j, 30.0, 4.0),
        (55 + 2.0j, 80.0, 5.0),
        (30 + 0.5j, 120.0, 4.0),
        (45 + 1.5j, 60.0, 3.0),
        (SIO2, 15.0, 3.0),
    ]
    sl = []
    for _ in range(25):
        sl += [(30 + 0.5j, 25.0, 3.0), (55 + 2.0j, 35.0, 3.5)]
    return {
        "single": [amb, (45 + 1.5j, 240.0, 6.0), (SI, 0.0, 3.0)],
        "five_layer": [amb, *five, (SI, 0.0, 3.0)],
        "superlattice50": [amb, *sl, (SI, 0.0, 3.0)],
    }


def batch(layers, b):
    sld, d, sig = arrays(layers)
    rng = np.random.default_rng(0)
    d = np.tile(d, (b, 1)) * rng.uniform(0.9, 1.1, (b, 1))
    return np.tile(sld, (b, 1)), d, np.tile(sig, (b, 1))


def rust(name, layers, b, threads, reps):
    f = HERE / "rust_out" / f"{name}.layers"
    f.parent.mkdir(exist_ok=True)
    f.write_text("\n".join(f"{s.real} {s.imag} {t} {r}" for s, t, r in layers))
    exe = HERE / "rs" / "target" / "release" / "parratt_bench.exe"
    res = subprocess.run(
        [str(exe), str(f), str(b), str(threads), "-", str(NQ), str(reps)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(res.stdout.strip())


def per_call(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def jacobian_latency(eng, q, layers):
    """Single film, batch 1: forward + Jacobian wrt (d, sigma, Re SLD) of each medium (3N params)."""
    sld, d, sig = arrays(layers)
    if eng == "torch-cpu-c128":
        import torch
        from rl_abeles import abeles_core

        qq = torch.tensor(q, dtype=torch.complex128)[None, :, None]

        def f(dd, ss, rr):
            rho = (torch.complex(rr, torch.tensor(sld.imag)) - complex(sld[0])) * 1e-6
            return abeles_core(qq, dd[None, None, :-1], ss[None, None, 1:], rho[None, None])[0]

        args = (torch.tensor(d), torch.tensor(sig), torch.tensor(sld.real))
        return per_call(lambda: torch.autograd.functional.jacobian(f, args), 50)
    if eng == "jax-cpu-f64":
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        n = len(sld)

        def f(dd, ss, rr):
            rho = jnp.conj(rr + 1j * sld.imag - sld[0]) * 1e-6
            k = jnp.sqrt((jnp.asarray(q)[:, None] / 2) ** 2 - 4 * jnp.pi * rho[None, :] + 0j)
            x = jnp.zeros(len(q), complex)
            for j in range(n - 2, -1, -1):
                k1, k2 = k[:, j], k[:, j + 1]
                r = (k1 - k2) / (k1 + k2) * jnp.exp(-2 * k1 * k2 * ss[j + 1] ** 2)
                p = jnp.exp(2j * k2 * (dd[j + 1] if j + 1 < n - 1 else 0.0))
                x = (r + x * p) / (1 + r * x * p)
            return jnp.abs(x) ** 2

        jf = jax.jit(jax.jacfwd(f, argnums=(0, 1, 2)))
        args = (jnp.asarray(d), jnp.asarray(sig), jnp.asarray(sld.real))
        return per_call(lambda: jax.block_until_ready(jf(*args)), 200)
    if eng == "numba-f64-par":
        # central differences: 2 * 3N forward curves in one batch call
        p = 3 * len(sld)
        sl, dl, sg = batch(layers, 2 * p)
        fn = ENGINES[eng](q, sl, dl, sg)
        return per_call(fn, 200)
    return None


def main():
    eng = sys.argv[3]
    out_path = Path(sys.argv[2])
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    q = q_grid()
    for name, layers in spec_structures().items():
        rate_1k = None
        for b in BATCHES:
            key = f"{eng}|{name}|{b}"
            how = "measured"
            if eng.startswith("rust"):
                threads = 1 if b == 1 else int(eng.split("-")[1])
                sec = rust(name, layers, b, threads, 2000 if b == 1 else 1)
            elif b == 1:
                reps = 20 if eng in ("numpy-f64", "torch-cpu-c64", "torch-cpu-c128") else 200
                sec = per_call(ENGINES[eng](q, *batch(layers, 1)), reps)
            elif b == CHUNK:
                bb = 100 if eng in ("refnx", "genx") else b
                fn = ENGINES[eng](q, *batch(layers, bb))
                sec = min(per_call(fn, 1) for _ in range(3)) * (b / bb)
                how = "measured" if bb == b else f"measured on {bb}, scaled"
                rate_1k = b / sec
            else:
                if rate_1k and b / rate_1k < 20 and eng not in ("refnx", "genx"):
                    if eng in ("numba-f64-par",):
                        fn = ENGINES[eng](q, *batch(layers, b))
                        sec = per_call(fn, 1)
                    else:
                        fn = ENGINES[eng](q, *batch(layers, CHUNK))
                        t0 = time.perf_counter()
                        for _ in range(b // CHUNK):
                            fn()
                        sec = time.perf_counter() - t0
                        how = "measured, 1k chunks"
                else:
                    sec = b / rate_1k
                    how = "extrapolated from 1k"
            results[key] = {
                "engine": eng,
                "case": name,
                "n_media": len(layers),
                "batch": b,
                "seconds": sec,
                "curves_per_s": b / sec,
                "how": how,
            }
            print(
                f"{eng:16s} {name:15s} N={len(layers):2d} B={b:6d} {sec:10.5f} s  {b / sec:10.0f} curves/s  {how}",
                flush=True,
            )
            out_path.write_text(json.dumps(results, indent=1))
    if "--jac" in sys.argv:
        lat = jacobian_latency(eng, q, spec_structures()["single"])
        results[f"{eng}|jacobian_single"] = {"seconds": lat}
        print(f"{eng} jacobian single film batch 1: {lat * 1e3:.3f} ms")
        out_path.write_text(json.dumps(results, indent=1))
    print("cpu_count", os.cpu_count())


if __name__ == "__main__":
    main()
