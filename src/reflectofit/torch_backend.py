"""Differentiable backend: the same Parratt recursion and instrument model in torch.

complex128 when q is float64 (the default), complex64 when q is float32. Large batches run in chunks
so the (B, Q, N) wavevector array stays bounded.
"""

import math

import numpy as np
import torch
from torch import Tensor

from reflectofit.forward import Instrument, _observed
from reflectofit.structure import Stack, as_batch

CHUNK = 1024


def parratt(q: Tensor, sld: Tensor, thickness: Tensor, roughness: Tensor) -> Tensor:
    """|r|² for q (B, Q) and media (B, N), ambient first; Im SLD > 0 absorbs. Returns (B, Q)."""
    ctype = torch.complex128 if q.dtype == torch.float64 else torch.complex64
    rho = (4e-6 * math.pi) * torch.conj(sld - sld[:, :1]).to(ctype)
    kz = torch.sqrt((0.5 * q).to(ctype).unsqueeze(-1) ** 2 - rho.unsqueeze(1))
    n = sld.shape[1]
    x = torch.zeros(q.shape, dtype=ctype, device=q.device)
    for j in range(n - 2, -1, -1):
        k1, k2 = kz[..., j], kz[..., j + 1]
        s = roughness[:, j + 1 : j + 2]
        r = (k1 - k2) / (k1 + k2) * torch.exp(-2.0 * k1 * k2 * s**2)
        if j + 1 < n - 1:
            xp = x * torch.exp(2j * k2 * thickness[:, j + 1 : j + 2])
        else:
            xp = torch.zeros_like(x)
        x = (r + xp) / (1.0 + r * xp)
    return x.real**2 + x.imag**2


class _TorchOps:
    def __init__(self, dtype: torch.dtype, device: torch.device):
        self.dtype, self.device = dtype, device

    def asarray(self, x):
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def column(self, x, b):
        t = self.asarray(x)
        return (t.expand(b) if t.ndim == 0 else t).reshape(b, 1, 1)

    sqrt = staticmethod(torch.sqrt)
    sin = staticmethod(torch.sin)
    arcsin = staticmethod(torch.asin)
    erf = staticmethod(torch.erf)
    exp = staticmethod(torch.exp)
    cat = staticmethod(torch.cat)

    def take_rows(self, values, idx):
        """values (B, Q) gathered at idx (B, M)."""
        return values.gather(1, torch.as_tensor(idx, device=self.device))

    def sum_rows(self, values, idx, width):
        """(B, width) sums of values (B, M) by idx (B, M)."""
        out = torch.zeros(values.shape[0], width, dtype=values.dtype, device=self.device)
        return out.scatter_add(1, torch.as_tensor(idx, device=self.device), values)

    @staticmethod
    def clip(x, lo, hi):
        return torch.clamp(x, lo, hi)


def _to_tensors(model, dtype, device):
    if isinstance(model, tuple) and len(model) == 3 and isinstance(model[0], Tensor):
        sld, thickness, roughness = model
        single = sld.ndim == 1
        if single:
            sld, thickness, roughness = sld[None], thickness[None], roughness[None]
        return (sld, thickness.to(dtype), roughness.to(dtype)), single
    if not isinstance(model, Stack | tuple | list):
        raise TypeError("model must be a Stack, a packed tuple or a sequence of them")
    (sld, thickness, roughness), single = as_batch(model)
    ctype = torch.complex128 if dtype == torch.float64 else torch.complex64

    def as_t(a, t):
        return torch.as_tensor(np.asarray(a), dtype=t, device=device)

    return (as_t(sld, ctype), as_t(thickness, dtype), as_t(roughness, dtype)), single


def forward(q, model, instrument: Instrument | None = None, chunk: int = CHUNK) -> Tensor:
    """Observed reflectivity as a differentiable tensor; arguments as `reflectofit.forward`.

    Parameters given as tensors (packed tuple, instrument fields) keep their graph and work under
    torch.func transforms; for tensors no value is checked (Im SLD >= 0 is the caller's to keep),
    and a tensor instrument field always counts as switched on.
    """
    q = torch.as_tensor(q) if isinstance(q, Tensor) else torch.as_tensor(np.asarray(q, float))
    dtype = q.dtype if q.dtype in (torch.float32, torch.float64) else torch.float64
    q = q.to(dtype)
    packed, single = _to_tensors(model, dtype, q.device)
    b = packed[0].shape[0]
    ops = _TorchOps(dtype, q.device)
    out = []
    for start in range(0, b, chunk):
        part = tuple(a[start : start + chunk] for a in packed)
        bb = part[0].shape[0]
        if instrument is None:
            out.append(parratt(q.expand(bb, -1), *part))
        else:
            inst = _slice_instrument(instrument, start, bb, b)
            out.append(_observed(ops, parratt, q, bb, part, inst))
    r = torch.cat(out) if len(out) > 1 else out[0]
    return r[0] if single else r


def _slice_instrument(inst: Instrument, start: int, size: int, total: int) -> Instrument:
    if size == total:
        return inst
    changes = {}
    for name in (
        "i0",
        "background",
        "tth_offset",
        "sigma_q",
        "dq_over_q",
        "kalpha2_ratio",
        "sample_length",
        "beam_sigma",
    ):
        v = getattr(inst, name)
        if v is not None and np.ndim(v) == 1:
            changes[name] = v[start : start + size]
    return inst.replace(**changes) if changes else inst
