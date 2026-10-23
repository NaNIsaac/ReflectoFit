"""ReflectoLearn's default training distribution (config.py and data/generator.py at f346d3f), drawn
with numpy: one film on a real-SLD substrate, an oxide on top with probability 0.7, 2θ offset, Kα₂,
σ_q per curve."""

import numpy as np

from reflectofit import Instrument, tth_to_q

WL = 1.54
KALPHA2 = 1.54439


def draw(n: int, seed: int, sigma_q: tuple[float, float] = (1e-4, 6e-3)):
    rng = np.random.default_rng(seed)
    q = tth_to_q(np.linspace(0.05, 15.0, 2000), WL)

    def u(lo, hi):
        return rng.uniform(lo, hi, n)

    thk, rgh, sld = u(10, 1200), u(0, 15), u(5, 100)
    sub_sld, sub_rgh = u(18, 22), u(1, 6)
    ox = rng.uniform(size=n) < 0.7
    ox_thk = np.where(ox, u(0, 40), 0.0)
    ox_sld = np.where(ox, u(5, 25), sld)
    ox_rgh = u(0, 8)
    top = np.where(ox, ox_rgh, rgh)
    mid = np.where(ox, rgh, 0.0)
    z = np.zeros(n)
    packed = (
        np.stack([z, ox_sld, sld, sub_sld], 1).astype(complex),
        np.stack([z, ox_thk, thk, z], 1),
        np.stack([z, top, mid, sub_rgh], 1),
    )
    inst = Instrument(
        wavelength=WL,
        kalpha2_wavelength=KALPHA2,
        tth_offset=u(-0.02, 0.02),
        kalpha2_ratio=u(0, 0.5),
        sigma_q=u(*sigma_q),
    )
    return q, packed, inst
