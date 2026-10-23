"""X1e confirmation grid (README.md in this folder; nothing here was tuned).

Usage: uv run python docs/design-bench/x1e/confirm.py --accuracy | --cost [--torch]
Writes confirm_accuracy.json or confirm_cost.json next to this file.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from rl_distribution import WL, draw

from reflectofit import Instrument, Layer, Stack, forward, resolution, tth_to_q

HERE = Path(__file__).parent
SEED = 20261025
SI = 20.07 + 0.46j


def brute(q_true, packed, inst: Instrument, sigma, n, chunk=25):
    u = np.linspace(-9, 9, n)
    w = np.exp(-0.5 * u**2)
    w /= w.sum()
    bare = inst.replace(sigma_q=0.0, dq_over_q=0.0, tth_offset=0.0, i0=1.0, background=0.0)
    out = np.empty(q_true.size)
    for i in range(0, q_true.size, chunk):
        nodes = np.clip(q_true[i : i + chunk, None] + sigma[i : i + chunk, None] * u, 1e-9, None)
        out[i : i + chunk] = (forward(nodes.ravel(), packed, bare).reshape(nodes.shape) * w).sum(1)
    return out


def score(q, packed, inst: Instrument):
    """Default rule against the reference; returns max error over counted points and the counts."""
    got = forward(q, packed, inst)
    tth = np.degrees(2 * np.arcsin(np.clip(q * inst.wavelength / (4 * np.pi), 0, 1)))
    q_true = tth_to_q(np.clip(tth + float(inst.tth_offset), 1e-9, None), inst.wavelength)
    sigma = np.sqrt(float(inst.sigma_q) ** 2 + (float(inst.dq_over_q) * q_true) ** 2)
    ref = brute(q_true, packed, inst, sigma, 24001)
    ka2 = inst.kalpha2_wavelength / inst.wavelength if float(inst.kalpha2_ratio) else None
    qc = resolution.edges(packed[0], packed[1], ka2).qc
    near = (np.abs(qc[None, :] - q_true[:, None]) < 9 * sigma[:, None]).any(axis=1)
    limited = np.zeros(q.size, bool)
    if near.any():
        fine = brute(q_true[near], packed, inst, sigma[near], 96001)
        limited[near] = np.abs(fine / ref[near] - 1) > 2e-5
        ref[near] = fine
    rel = np.abs(got / ref - 1)
    edge5 = (np.abs(qc[None, :] - q_true[:, None]) < 5 * sigma[:, None]).any(axis=1)
    counted = ~limited
    return {
        "max_rel": float(rel[counted].max()),
        "q_at_max": float(q[counted][rel[counted].argmax()]),
        "max_rel_within_5_sigma_of_an_edge": float(rel[counted & edge5].max())
        if (counted & edge5).any()
        else None,
        "max_rel_all_points": float(rel.max()),
        "reference_limited_points": int(limited.sum()),
    }


def structures():
    real = {
        "bare Si": Stack([], Layer(20.07)),
        "film 9.6 / 1000 on Si": Stack([Layer(9.6, 1000, 3)], Layer(20.07, roughness=3)),
        "Ni / C / Ni real": Stack(
            [Layer(64.0, 100, 2), Layer(18.0, 800, 2)], Layer(64.0, roughness=2)
        ),
        "real waveguide": Stack([Layer(25.0, 30, 3), Layer(5.0, 300, 3)], Layer(20.0, roughness=3)),
        "real superlattice": Stack(
            [Layer(30.0, 25, 3), Layer(55.0, 35, 3.5)] * 20, Layer(20.07, roughness=3)
        ),
    }
    absorbing = {  # the cases of tests/test_review_physics.py
        "45+1.5i 2000 A": (Stack([Layer(45 + 1.5j, 2000, 4)], Layer(SI)), 1e-3, 0.0),
        "45+1.5i 4000 A": (Stack([Layer(45 + 1.5j, 4000, 4)], Layer(SI)), 2e-3, 0.0),
        "45+1.5i 1000 A dq/q": (Stack([Layer(45 + 1.5j, 1000, 4)], Layer(SI)), 0.0, 0.02),
        "PS-like 9.6+0.012i": (
            Stack([Layer(9.6 + 0.012j, 1000, 4)], Layer(SI, roughness=3)),
            5e-4,
            0.0,
        ),
        "45+0.05i 1000 A": (Stack([Layer(45 + 0.05j, 1000, 4)], Layer(SI, roughness=3)), 5e-4, 0.0),
        "Ni/C/Ni waveguide": (
            Stack(
                [Layer(64 + 1.35j, 100, 2), Layer(18 + 0.02j, 800, 2)],
                Layer(64 + 1.35j, roughness=2),
            ),
            1e-4,
            0.0,
        ),
    }
    return real, absorbing


def accuracy():
    results = {}
    q, packed, inst = draw(16, seed=SEED)
    for label, sig in (("drawn", None), ("1e-4", 1e-4), ("6e-3", 6e-3)):
        for j in range(16):
            one = Instrument(
                wavelength=WL,
                kalpha2_wavelength=inst.kalpha2_wavelength,
                tth_offset=float(inst.tth_offset[j]),
                kalpha2_ratio=float(inst.kalpha2_ratio[j]),
                sigma_q=float(inst.sigma_q[j]) if sig is None else sig,
            )
            results[f"RL {j} sigma {label}"] = score(q, tuple(a[j] for a in packed), one)
            print(
                f"RL {j} sigma {label}: {results[f'RL {j} sigma {label}']['max_rel']:.1e}",
                flush=True,
            )
    qq = np.linspace(0.01, 0.2, 600)
    real, absorbing = structures()
    for name, st in real.items():
        for res in ({"sigma_q": 1e-4}, {"sigma_q": 5e-4}, {"sigma_q": 2e-3}, {"dq_over_q": 0.02}):
            key = f"{name} {res}"
            results[key] = score(qq, st.packed(), Instrument(**res))
            print(f"{key}: {results[key]['max_rel']:.1e}", flush=True)
    qa = np.concatenate([np.linspace(0.015, 0.07, 200), tth_to_q(np.linspace(1.0, 6.0, 150))])
    for name, (st, sq, dqq) in absorbing.items():
        results[name] = score(qa, st.packed(), Instrument(sigma_q=sq, dq_over_q=dqq))
        print(f"{name}: {results[name]['max_rel']:.1e}", flush=True)
    worst = max(r["max_rel"] for r in results.values())
    out = {"criterion": "max_rel <= 1e-4 at every counted point", "worst": worst, "cases": results}
    (HERE / "confirm_accuracy.json").write_text(json.dumps(out, indent=1))
    print(f"worst {worst:.2e}: {'held' if worst <= 1e-4 else 'FALSIFIED'}")


def cost(backend: str):
    q, packed, inst = draw(64, seed=SEED)
    if backend == "torch":
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        q_in = torch.as_tensor(q, device=dev)
        packed = tuple(torch.as_tensor(a, device=dev) for a in packed)
        sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    else:
        q_in, sync = q, (lambda: None)
    times = []
    for _ in range(6):
        t = time.perf_counter()
        forward(q_in, packed, inst, backend=backend)
        sync()
        times.append(time.perf_counter() - t)
    e = [
        resolution.edges(packed[0][j], packed[1][j], inst.kalpha2_wavelength / WL)
        for j in range(64)
    ]
    tth = np.degrees(2 * np.arcsin(q * WL / (4 * np.pi)))
    q_true = np.stack([tth_to_q(tth + o, WL) for o in inst.tth_offset])
    counts, _, _ = resolution.layout(q_true, np.repeat(inst.sigma_q[:, None], q.size, 1), e)
    out = {
        "backend": backend,
        "first_s": times[0],
        "steady_s": times[1:],
        "median_steady_s": float(np.median(times[1:])),
        "nodes_per_point_mean": float(counts.mean()),
        "nodes_per_point_max": int(counts.max()),
        "criterion": "<= 1 s and <= 2 GB (memory from the job cap report)",
    }
    (HERE / f"confirm_cost_{backend}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    if "--accuracy" in sys.argv:
        accuracy()
    if "--cost" in sys.argv:
        cost("torch" if "--torch" in sys.argv else "numba")
