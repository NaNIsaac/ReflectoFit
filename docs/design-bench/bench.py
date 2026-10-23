"""Forward-model throughput: curves per second for B curves x 2000 q points.

Usage: python -I bench.py <rl_abeles dir> <out.json> <engine,engine,...>
Each engine is timed best-of-3 after one warm-up call (JIT compile excluded and reported separately).
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, sys.argv[1])

from structures import WAVELENGTH, arrays, parratt, q_grid, structures

SCALE = int(os.environ.get("BENCH_SCALE", "1"))
CASES = {"oxide_under": 1024 * SCALE, "superlattice": 256 * SCALE}


def batch(name, b):
    sld, d, sig = arrays(structures()[name])
    rng = np.random.default_rng(0)
    d = np.tile(d, (b, 1)) * rng.uniform(0.9, 1.1, (b, 1))
    return np.tile(sld, (b, 1)), d, np.tile(sig, (b, 1))


def timed(fn, reps=3):
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    best = min(_once(fn) for _ in range(reps))
    return first, best


def _once(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def eng_numpy(q, sld, d, sig):
    return lambda: parratt(q, sld, d, sig)


def eng_numba(q, sld, d, sig):
    import numba

    @numba.njit(parallel=True, fastmath=False, cache=False)
    def kern(q, sld, d, sig, out):
        b, n = sld.shape
        for i in numba.prange(b):
            kz = np.empty(n, np.complex128)
            for iq in range(q.shape[0]):
                h = q[iq] / 2
                for j in range(n):
                    rho = np.conj(sld[i, j] - sld[i, 0]) * 1e-6
                    kz[j] = np.sqrt(h * h - 4 * np.pi * rho + 0j)
                x = 0j
                for j in range(n - 2, -1, -1):
                    k1, k2 = kz[j], kz[j + 1]
                    r = (k1 - k2) / (k1 + k2) * np.exp(-2 * k1 * k2 * sig[i, j + 1] ** 2)
                    dj = d[i, j + 1] if j + 1 < n - 1 else 0.0
                    p = np.exp(2j * k2 * dj)
                    x = (r + x * p) / (1 + r * x * p)
                out[i, iq] = x.real**2 + x.imag**2

    out = np.empty((sld.shape[0], q.shape[0]))
    return lambda: kern(q, sld, d, sig, out)


def eng_torch(q, sld, d, sig, dtype="c64", device="cpu"):
    import torch
    from rl_abeles import AbelesMatrix, abeles_core

    dev = torch.device(device)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    if dtype == "c64":
        args = [
            torch.tensor(a, device=dev)
            for a in (np.tile(q, (sld.shape[0], 1)), d[:, :-1], sig[:, 1:], sld)
        ]
        model = AbelesMatrix(device=dev)

        def run():
            with torch.no_grad():
                model(*args, wavelength=WAVELENGTH)
            sync()

        return run
    qq = torch.tensor(q, dtype=torch.complex128, device=dev)[None, :, None]
    t = torch.tensor(d[:, :-1], device=dev)[:, None]
    s = torch.tensor(sig[:, 1:], device=dev)[:, None]
    rho = torch.tensor((sld - sld[:, :1]) * 1e-6, dtype=torch.complex128, device=dev)[:, None]

    def run():
        with torch.no_grad():
            abeles_core(qq, t, s, rho)
        sync()

    return run


def eng_jax(q, sld, d, sig, f64=True):
    import jax

    jax.config.update("jax_enable_x64", f64)
    import jax.numpy as jnp

    ct = jnp.complex128 if f64 else jnp.complex64
    n = sld.shape[1]

    @jax.jit
    def kern(q, sld, d, sig):
        rho = jnp.conj(sld - sld[:, :1]).astype(ct) * 1e-6
        k = jnp.sqrt(((q[None, :, None] / 2) ** 2).astype(ct) - 4 * jnp.pi * rho[:, None, :])
        x = jnp.zeros(k.shape[:2], ct)
        for j in range(n - 2, -1, -1):
            k1, k2 = k[..., j], k[..., j + 1]
            r = (k1 - k2) / (k1 + k2) * jnp.exp(-2 * k1 * k2 * sig[:, None, j + 1] ** 2)
            dj = d[:, None, j + 1] if j + 1 < n - 1 else 0.0
            p = jnp.exp(2j * k2 * dj)
            x = (r + x * p) / (1 + r * x * p)
        return jnp.abs(x) ** 2

    fd = np.float64 if f64 else np.float32
    args = [
        jnp.asarray(q.astype(fd)),
        jnp.asarray(sld.astype(ct)),
        jnp.asarray(d.astype(fd)),
        jnp.asarray(sig.astype(fd)),
    ]
    return lambda: kern(*args).block_until_ready()


def eng_refnx(q, sld, d, sig):
    from refnx.reflect import reflectivity

    ws = [np.column_stack([d[i], sld[i].real, sld[i].imag, sig[i]]) for i in range(sld.shape[0])]
    return lambda: [reflectivity(q, w, dq=0.0) for w in ws]


def eng_genx(q, sld, d, sig):
    from genx.models.lib import paratt

    n = 1 - WAVELENGTH**2 * np.conj(sld) * 1e-6 / (2 * math.pi)
    items = [
        (n[i, ::-1].copy(), d[i, ::-1].copy(), np.append(sig[i, ::-1][:-1], 0.0))
        for i in range(sld.shape[0])
    ]
    return lambda: [paratt.ReflQ(q, WAVELENGTH, a, b, c) for a, b, c in items]


def rust(name, b, threads):
    layers = HERE / "rust_out" / f"{name}.layers"
    layers.parent.mkdir(exist_ok=True)
    layers.write_text("\n".join(f"{s.real} {s.imag} {t} {r}" for s, t, r in structures()[name]))
    exe = HERE / "rs" / "target" / "release" / "parratt_bench.exe"
    out = HERE / "rust_out" / f"{name}.txt"
    res = subprocess.run(
        [str(exe), str(layers), str(b), str(threads), str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(res.stdout.strip())


ENGINES = {
    "numpy-f64": eng_numpy,
    "numba-f64-par": eng_numba,
    "torch-cpu-c64": lambda *a: eng_torch(*a, "c64", "cpu"),
    "torch-cpu-c128": lambda *a: eng_torch(*a, "c128", "cpu"),
    "torch-cuda-c64": lambda *a: eng_torch(*a, "c64", "cuda"),
    "torch-cuda-c128": lambda *a: eng_torch(*a, "c128", "cuda"),
    "jax-cpu-f64": lambda *a: eng_jax(*a, True),
    "jax-cpu-f32": lambda *a: eng_jax(*a, False),
    "refnx": eng_refnx,
    "genx": eng_genx,
}


def main():
    out_path = Path(sys.argv[2])
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    q = q_grid()
    for eng in sys.argv[3].split(","):
        for name, b in CASES.items():
            n_media = len(structures()[name])
            if eng.startswith("rust"):
                threads = int(eng.split("-")[1])
                rust(name, 8, threads)
                first, best = float("nan"), rust(name, b, threads)
            else:
                bb = b // 8 if eng in ("refnx", "genx") else b
                sld, d, sig = batch(name, bb)
                first, best = timed(ENGINES[eng](q, sld, d, sig))
                b = bb
            rate = b / best
            results[f"{eng}|{name}"] = {
                "engine": eng,
                "case": name,
                "n_media": n_media,
                "batch": b,
                "first_s": first,
                "best_s": best,
                "curves_per_s": rate,
            }
            print(
                f"{eng:16s} {name:13s} N={n_media:2d} B={b:5d} first {first:7.3f}s best {best:7.4f}s  {rate:10.0f} curves/s",
                flush=True,
            )
            out_path.write_text(json.dumps(results, indent=1))
    print(
        "threads: torch", os.environ.get("OMP_NUM_THREADS", "default"), "cpu_count", os.cpu_count()
    )


if __name__ == "__main__":
    main()
