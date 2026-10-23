"""Staged gradient fitting: the refine strategy of ReflectoLearn, for any differentiable model.

A fit runs stages in order. Each stage frees some parameters and minimises one loss with Adam for a
fixed number of steps; the other parameters are held at their current values. Parameters live on
bounds through a sigmoid of a latent variable, so they never leave [lo, hi]. Keys freed by any stage
so far are `active`; the model receives them, so it can switch an effect on only once a stage has
fitted it (ReflectoLearn turns resolution and the oxide on that way). An optional Occam penalty pulls
chosen parameters towards a neutral value, scaled by their bound range.

The model is a callable `simulate(values, active) -> R` with `values` a dict of (B,) tensors and R
(B, Q) linear reflectivity; `reflectofit.fit.staged` builds one from a `Model`. Observed data are
given as log10 R (`log_y`), as in ReflectoLearn; with counts, log10(counts / counts_scale).

The losses are those of ReflectoLearn `refine/losses.py`: per curve, a mean over the masked points,
then the mean over the batch.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

LossFn = Callable[[Tensor, Tensor, Tensor], Tensor]
Simulate = Callable[[dict[str, Tensor], set[str]], Tensor]

_EPS = 1e-12


def _masked_mean(err: Tensor, mask: Tensor) -> Tensor:
    m = mask.to(err.dtype)
    return ((err * m).sum(-1) / m.sum(-1).clamp_min(1.0)).mean()


def log_mse(r: Tensor, log_y: Tensor, mask: Tensor) -> Tensor:
    """MSE of log10 R: sensitive to the fringes and the dynamic range."""
    return _masked_mean((torch.log10(r.clamp_min(_EPS)) - log_y) ** 2, mask)


def linear_mse(r: Tensor, log_y: Tensor, mask: Tensor) -> Tensor:
    """MSE of R: dominated by the plateau and the critical edge (scale, alignment)."""
    return _masked_mean((r - 10.0**log_y) ** 2, mask)


def correlation(r: Tensor, log_y: Tensor, mask: Tensor) -> Tensor:
    """1 - Pearson correlation of log10 R: blind to scale and attenuation, so it matches the
    fringe period."""
    lr = torch.log10(r.clamp_min(_EPS))
    m = mask.to(lr.dtype)
    n = m.sum(-1).clamp_min(1.0)
    vx = (lr - ((lr * m).sum(-1) / n).unsqueeze(-1)) * m
    vy = (log_y - ((log_y * m).sum(-1) / n).unsqueeze(-1)) * m
    den = torch.sqrt((vx**2).sum(-1)) * torch.sqrt((vy**2).sum(-1)) + 1e-9
    return (1.0 - (vx * vy).sum(-1) / den).mean()


def log_gradient(r: Tensor, log_y: Tensor, mask: Tensor) -> Tensor:
    """log10 MSE plus 10 x the MSE of its point-to-point slope."""
    lr = torch.log10(r.clamp_min(_EPS))
    md = mask[:, 1:].to(r.dtype) * mask[:, :-1].to(r.dtype)
    slope = (lr[:, 1:] - lr[:, :-1]) - (log_y[:, 1:] - log_y[:, :-1])
    return log_mse(r, log_y, mask) + 10.0 * _masked_mean(slope**2, md)


def hybrid(r: Tensor, log_y: Tensor, mask: Tensor) -> Tensor:
    """20 x linear MSE + log10 MSE."""
    return 20.0 * linear_mse(r, log_y, mask) + log_mse(r, log_y, mask)


LOSSES: dict[str, LossFn] = {
    "log_mse": log_mse,
    "linear_mse": linear_mse,
    "correlation": correlation,
    "log_gradient": log_gradient,
    "hybrid": hybrid,
}


def poisson_deviance(counts_scale: float) -> LossFn:
    """Poisson deviance per point for counts = counts_scale · R, zero counts included. It is the
    likelihood only on the measured points with their raw counts."""

    def loss(r: Tensor, log_y: Tensor, mask: Tensor) -> Tensor:
        lam = counts_scale * r.double().clamp_min(_EPS)
        y = counts_scale * torch.pow(10.0, log_y.double())
        dev = 2.0 * (lam - y + torch.special.xlogy(y, y / lam))
        return _masked_mean(dev, mask).to(r.dtype)

    return loss


def loss_fn(kind: str, counts_scale: float | None = None) -> LossFn:
    """A loss by name: one of LOSSES, or "poisson" (needs counts_scale)."""
    if kind == "poisson":
        if counts_scale is None:
            raise ValueError("loss 'poisson' needs counts_scale")
        return poisson_deviance(counts_scale)
    if kind not in LOSSES:
        raise ValueError(f"unknown loss {kind!r}; use one of {[*LOSSES, 'poisson']}")
    return LOSSES[kind]


@dataclass(frozen=True)
class Param:
    """Start value and bounds; each a number or a (B,) tensor, one value per curve."""

    start: float | Tensor
    lo: float | Tensor
    hi: float | Tensor


@dataclass(frozen=True)
class Stage:
    """Which parameters to free, with which loss, for how many Adam steps."""

    name: str
    free: tuple[str, ...]
    loss: str = "log_mse"
    steps: int = 200
    lr: float = 0.03
    q_min: float | None = None
    """[Å⁻¹] the stage sees only points above it (needs the q grid)."""
    q_max: float | None = None


class StagedFit:
    """The state of a staged fit of B curves: values, bounds and active keys.

    `log_y` (B, Q) is log10 of the observed reflectivity; points where it is not finite (zero
    counts) are left out of every loss except the Poisson deviance, which counts them.
    """

    def __init__(
        self,
        simulate: Simulate,
        params: dict[str, Param],
        log_y: Tensor,
        mask: Tensor | None = None,
        *,
        q: Tensor | None = None,
        counts_scale: float | None = None,
        penalty: dict[str, float] | None = None,
        penalty_weight: float = 0.0,
    ):
        self.simulate = simulate
        self.log_y = log_y
        self.dtype, self.device = log_y.dtype, log_y.device
        b = log_y.shape[0]
        self.mask = torch.ones_like(log_y) if mask is None else mask.to(log_y.dtype)
        self.q = q
        self.counts_scale = counts_scale
        self.penalty = dict(penalty or {})
        self.penalty_weight = penalty_weight

        def full(v) -> Tensor:
            return torch.as_tensor(v, dtype=self.dtype, device=self.device).expand(b).clone()

        self.bounds = {k: (full(p.lo), full(p.hi)) for k, p in params.items()}
        self.state = {k: full(p.start) for k, p in params.items()}
        for k, (lo, hi) in self.bounds.items():
            if not bool((lo < hi).all()):
                raise ValueError(f"{k}: lower bound is not below the upper bound")
            self.state[k] = torch.minimum(torch.maximum(self.state[k], lo), hi)
        self.active: set[str] = set()

    def curve(self, active: set[str] | None = None) -> Tensor:
        """The model at the current state (B, Q)."""
        with torch.no_grad():
            return self.simulate(dict(self.state), self.active if active is None else active)

    def loss(self, kind: str = "log_mse", per_curve: bool = False) -> Tensor:
        """A loss at the current state; per_curve gives (B,) values instead of the batch mean."""
        fn = loss_fn(kind, self.counts_scale)
        r, log_y, mask = self.curve(), *self._data(kind)
        if not per_curve:
            return fn(r, log_y, mask)
        return torch.stack(
            [fn(r[i : i + 1], log_y[i : i + 1], mask[i : i + 1]) for i in range(r.shape[0])]
        )

    def _data(self, kind: str, q_min=None, q_max=None) -> tuple[Tensor, Tensor]:
        mask = self.mask
        if q_min is not None or q_max is not None:
            if self.q is None:
                raise ValueError("q_min / q_max need the q grid")
            if q_min is not None:
                mask = mask * (self.q > q_min)
            if q_max is not None:
                mask = mask * (self.q < q_max)
        if kind == "poisson":
            return self.log_y, mask
        finite = torch.isfinite(self.log_y)
        return torch.where(finite, self.log_y, 0.0), mask * finite

    def _penalty(self, vals: dict[str, Tensor], keys) -> Tensor:
        total = self.log_y.new_zeros(())
        if self.penalty_weight <= 0.0:
            return total
        for k, neutral in self.penalty.items():
            if k in keys:
                lo, hi = self.bounds[k]
                total = total + (((vals[k] - neutral) / (hi - lo)) ** 2).mean()
        return self.penalty_weight * total

    def optimize(
        self,
        keys,
        loss: str,
        steps: int,
        lr: float,
        q_min: float | None = None,
        q_max: float | None = None,
    ) -> None:
        """Adam on `keys` through their sigmoid latents; the result is written back to the state.
        Does not change `active` (a stage does, see `run`)."""
        keys = list(keys)
        if not keys or steps <= 0:
            return
        fn = loss_fn(loss, self.counts_scale)
        log_y, mask = self._data(loss, q_min, q_max)
        z = {}
        for k in keys:
            lo, hi = self.bounds[k]
            frac = ((self.state[k] - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
            z[k] = torch.log(frac / (1 - frac)).detach().clone().requires_grad_(True)
        opt = torch.optim.Adam(list(z.values()), lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            vals = dict(self.state)
            for k in keys:
                lo, hi = self.bounds[k]
                vals[k] = lo + (hi - lo) * torch.sigmoid(z[k])
            value = fn(self.simulate(vals, self.active), log_y, mask) + self._penalty(vals, keys)
            value.backward()
            opt.step()
        with torch.no_grad():
            for k in keys:
                lo, hi = self.bounds[k]
                self.state[k] = (lo + (hi - lo) * torch.sigmoid(z[k])).detach()

    def run(self, stage: Stage) -> None:
        """Free the stage's keys (they become active, and stay so) and optimise them."""
        unknown = [k for k in stage.free if k not in self.state]
        if unknown:
            raise ValueError(f"stage {stage.name!r}: unknown parameters {unknown}")
        if not stage.free:
            return
        self.active |= set(stage.free)
        self.optimize(stage.free, stage.loss, stage.steps, stage.lr, stage.q_min, stage.q_max)


def model_simulate(model, q) -> Simulate:
    """A `simulate` for a `reflectofit.model.Model`: values keyed by the model's parameter names
    (see `Model`), forward model on the torch backend in the dtype of the values."""
    from reflectofit.forward import forward
    from reflectofit.model import INSTRUMENT_FIELDS

    sld0, thickness0, roughness0 = model.stack.packed()
    names = model.stack.names

    def simulate(vals: dict[str, Tensor], active: set[str]) -> Tensor:
        ref = next(iter(vals.values()))
        b, dtype, device = ref.shape[0], ref.dtype, ref.device
        ctype = torch.complex128 if dtype == torch.float64 else torch.complex64

        def columns(a, t):
            return list(torch.as_tensor(a, dtype=t, device=device).expand(b, -1).unbind(1))

        sld, thk, rgh = columns(sld0, ctype), columns(thickness0, dtype), columns(roughness0, dtype)
        inst = {}
        for key, v in vals.items():
            if key == "log10_background":
                inst["background"] = 10.0**v
            elif key in INSTRUMENT_FIELDS:
                inst[key] = v
            else:
                medium, _, attr = key.partition(".")
                col = names.index(medium) + 1
                match attr:
                    case "thickness":
                        thk[col] = v
                    case "roughness":
                        rgh[col] = v
                    case "sld":
                        sld[col] = torch.complex(v, sld[col].imag)
                    case "isld":
                        sld[col] = torch.complex(sld[col].real, v)
        packed = (torch.stack(sld, 1), torch.stack(thk, 1), torch.stack(rgh, 1))
        qt = torch.as_tensor(q, dtype=dtype, device=device)
        return forward(qt, packed, model.instrument.replace(**inst), backend="torch")

    return simulate
