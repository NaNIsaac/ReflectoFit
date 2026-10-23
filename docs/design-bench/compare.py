"""Engine agreement on the reference structures: numpy Parratt (float64) as baseline.

Usage: python -I compare.py <dir with rl_abeles.py> <out.json>
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, sys.argv[1])

from structures import (
    WAVELENGTH,
    abeles_np,
    arrays,
    parratt,
    parratt_exact,
    q_grid,
    structures,
)


def rl_torch(q, sld, d, sig, double=False):
    """ReflectoLearn physics/abeles.py as shipped (complex64, +1e-9j), or the same kernel in complex128 without the guard."""
    import torch
    from rl_abeles import AbelesMatrix, abeles_core

    if not double:
        out = AbelesMatrix()(
            torch.tensor(q)[None],
            torch.tensor(d[:-1])[None],
            torch.tensor(sig[1:])[None],
            torch.tensor(sld)[None],
            wavelength=WAVELENGTH,
        )
        return out[0].double().numpy()
    rho = torch.tensor((sld - sld[0]) * 1e-6, dtype=torch.complex128)
    return abeles_core(
        torch.tensor(q, dtype=torch.complex128)[None, :, None],
        torch.tensor(d[:-1])[None, None],
        torch.tensor(sig[1:])[None, None],
        rho[None, None],
    )[0].numpy()


def refnx_r(q, sld, d, sig):
    from refnx.reflect import reflectivity

    w = np.column_stack([d, sld.real, sld.imag, sig])
    return reflectivity(q, w, dq=0.0)


def genx_r(q, sld, d, sig):
    """GenX 3.8 Parratt (lib.paratt.ReflQ), substrate first. n = 1 - lambda^2 SLD / (2 pi)."""
    from genx.models.lib import paratt

    n = 1 - WAVELENGTH**2 * np.conj(sld) * 1e-6 / (2 * math.pi)
    # ReflQ drops d[0] and d[-1] and uses sigma[j] for the interface above medium j (substrate first).
    nr, dr = n[::-1].copy(), d[::-1].copy()
    sr = np.append(sig[::-1][:-1], 0.0)
    return paratt.ReflQ(q, WAVELENGTH, nr, dr, sr)


def jax_r(q, sld, d, sig):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    rho = jnp.asarray(np.conj(sld - sld[0]) * 1e-6)
    k = jnp.sqrt((jnp.asarray(q)[:, None] / 2) ** 2 - 4 * jnp.pi * rho[None, :] + 0j)
    x = jnp.zeros(len(q), complex)
    for j in range(len(sld) - 2, -1, -1):
        k1, k2 = k[:, j], k[:, j + 1]
        r = (k1 - k2) / (k1 + k2) * jnp.exp(-2 * k1 * k2 * sig[j + 1] ** 2)
        p = jnp.exp(2j * k2 * (d[j + 1] if j + 1 < len(sld) - 1 else 0.0))
        x = (r + x * p) / (1 + r * x * p)
    return np.asarray(jnp.abs(x) ** 2)


def main():
    q = q_grid()
    out = {}
    for name, layers in structures().items():
        sld, d, sig = arrays(layers)
        base = parratt(q, sld, d, sig)[0]
        engines = {
            "numpy Abeles c128": abeles_np(q, sld, d, sig)[0],
            "numpy Parratt, exact n^2": parratt_exact(q, sld, d, sig)[0],
            "ReflectoLearn torch c64 (as shipped)": rl_torch(q, sld, d, sig),
            "ReflectoLearn torch c128, no guard": rl_torch(q, sld, d, sig, double=True),
            "refnx 0.1.67 (dq=0)": refnx_r(q, sld, d, sig),
            "GenX 3.8.13 ReflQ (exact n^2)": genx_r(q, sld, d, sig),
            "JAX Parratt f64": jax_r(q, sld, d, sig),
        }
        rust = Path(__file__).parent / "rust_out" / f"{name}.txt"
        if rust.exists():
            engines["Rust Parratt f64"] = np.loadtxt(rust)
        row = {}
        for key, r in engines.items():
            rel = np.abs(r - base) / base
            row[key] = {
                "max_rel": float(rel.max()),
                "median_rel": float(np.median(rel)),
                "max_dlog10": float(np.abs(np.log10(r) - np.log10(base)).max()),
                "q_at_max": float(q[rel.argmax()]),
            }
        out[name] = {"n_media": len(layers), "engines": row}
        print(name, len(layers))
        for key, v in row.items():
            print(
                f"  {key:40s} max_rel {v['max_rel']:.2e}  median {v['median_rel']:.2e}  max|dlog10| {v['max_dlog10']:.2e}  q@max {v['q_at_max']:.4f}"
            )
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
