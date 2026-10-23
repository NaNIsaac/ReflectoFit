"""X1e cost: ReflectoLearn's distribution, batch 64 x 2000, default rule, repeated (numba; torch with
--torch). Reports the load of the machine as seen by psutil when available."""

import sys
import time

import numpy as np
from rl_distribution import draw

from reflectofit import forward


def main():
    backend = "torch" if "--torch" in sys.argv else "numba"
    seed = 20261025 if "--confirm" in sys.argv else 0
    q, packed, inst = draw(64, seed=seed)
    if backend == "torch":
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        q_in = torch.as_tensor(q, device=dev)
        packed = tuple(torch.as_tensor(a, device=dev) for a in packed)
        sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    else:
        q_in, sync = q, (lambda: None)
    times = []
    for _ in range(4):
        t = time.perf_counter()
        forward(q_in, packed, inst, backend=backend)
        sync()
        times.append(time.perf_counter() - t)
    print(f"{backend}: first {times[0]:.2f} s, then {np.round(times[1:], 2).tolist()} s")


if __name__ == "__main__":
    main()
