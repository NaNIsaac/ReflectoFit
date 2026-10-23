"""torch Jacobian latency, single film, batch 1, vectorised reverse mode and forward mode.

Usage: python -I jac_torch.py <rl_abeles dir> <unused> <unused>
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, sys.argv[1])

import numpy as np  # noqa: E402
import torch  # noqa: E402
from bench2 import q_grid, spec_structures  # noqa: E402
from rl_abeles import abeles_core  # noqa: E402
from structures import arrays  # noqa: E402

sld, d, sig = arrays(spec_structures()["single"])
qq = torch.tensor(q_grid(), dtype=torch.complex128)[None, :, None]


def f(dd, ss, rr):
    rho = (torch.complex(rr, torch.tensor(sld.imag)) - complex(sld[0])) * 1e-6
    return abeles_core(qq, dd[None, None, :-1], ss[None, None, 1:], rho[None, None])[0]


args = (torch.tensor(d), torch.tensor(sig), torch.tensor(sld.real))
for strategy in ("reverse-mode", "forward-mode"):

    def jac(strategy=strategy):
        return torch.autograd.functional.jacobian(f, args, vectorize=True, strategy=strategy)

    jac()
    ts = []
    for _ in range(50):
        t0 = time.perf_counter()
        jac()
        ts.append(time.perf_counter() - t0)
    print(strategy, f"{np.median(ts) * 1e3:.3f} ms")
