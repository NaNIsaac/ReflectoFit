"""A fit model: a stack, an instrument and the free parameters with their bounds.

Parameter names: `<layer>.thickness`, `<layer>.roughness`, `<layer>.sld` (real part),
`<layer>.isld` (imaginary part) for each named layer and `substrate`; instrument fields by name
(`i0`, `background`, `tth_offset`, `sigma_q`, `dq_over_q`, `kalpha2_ratio`, `sample_length`,
`beam_sigma`); and `log10_background`, which sets `background` to 10**value.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from reflectofit import materials
from reflectofit.forward import Instrument, forward
from reflectofit.structure import Layer, Stack

INSTRUMENT_FIELDS = (
    "i0",
    "background",
    "tth_offset",
    "sigma_q",
    "dq_over_q",
    "kalpha2_ratio",
    "sample_length",
    "beam_sigma",
)
LAYER_FIELDS = ("thickness", "roughness", "sld", "isld")


@dataclass
class Model:
    stack: Stack
    instrument: Instrument = field(default_factory=Instrument)
    free: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = self.stack.names
        for key, (lo, hi) in self.free.items():
            if not lo < hi:
                raise ValueError(f"{key}: lower bound {lo} is not below upper bound {hi}")
            if key in INSTRUMENT_FIELDS or key == "log10_background":
                continue
            medium, _, attr = key.partition(".")
            if medium not in names or attr not in LAYER_FIELDS:
                raise ValueError(f"unknown parameter {key!r}; layers are {names}")
            if medium == "substrate" and attr == "thickness":
                raise ValueError("the substrate has no thickness")

    @property
    def names(self) -> list[str]:
        return list(self.free)

    @property
    def bounds(self) -> tuple[NDArray, NDArray]:
        lo, hi = zip(*self.free.values(), strict=True) if self.free else ((), ())
        return np.array(lo, float), np.array(hi, float)

    def value(self, key: str) -> float:
        if key == "log10_background":
            return float(np.log10(self.instrument.background))
        if key in INSTRUMENT_FIELDS:
            return float(getattr(self.instrument, key))
        medium, _, attr = key.partition(".")
        lay = self._layer(medium)
        match attr:
            case "sld":
                return complex(lay.sld).real
            case "isld":
                return complex(lay.sld).imag
        return float(getattr(lay, attr))

    def start(self) -> NDArray:
        """Starting values, clipped into the bounds."""
        lo, hi = self.bounds
        return np.clip([self.value(k) for k in self.free], lo, hi)

    def _layer(self, medium: str) -> Layer:
        if medium == "substrate":
            return self.stack.substrate
        return self.stack.layers[self.stack.names.index(medium)]

    def batch(self, theta: NDArray) -> tuple[tuple[NDArray, NDArray, NDArray], Instrument]:
        """Packed (B, N) arrays and a per-curve instrument for parameter rows theta (B, P)."""
        theta = np.atleast_2d(np.asarray(theta, float))
        b = theta.shape[0]
        sld, thickness, roughness = (np.tile(a, (b, 1)) for a in self.stack.packed())
        names = self.stack.names
        inst = {}
        for j, key in enumerate(self.free):
            v = theta[:, j]
            if key == "log10_background":
                inst["background"] = 10.0**v
            elif key in INSTRUMENT_FIELDS:
                inst[key] = v
            else:
                medium, _, attr = key.partition(".")
                col = names.index(medium) + 1
                match attr:
                    case "thickness":
                        thickness[:, col] = v
                    case "roughness":
                        roughness[:, col] = v
                    case "sld":
                        sld[:, col] = v + 1j * sld[:, col].imag
                    case "isld":
                        sld[:, col] = sld[:, col].real + 1j * v
        return (sld, thickness, roughness), self.instrument.replace(**inst)

    def curves(self, q: NDArray, theta: NDArray) -> NDArray:
        """Observed reflectivity (B, Q) for parameter rows theta (B, P)."""
        packed, inst = self.batch(theta)
        return forward(q, packed, inst)

    def with_values(self, theta: NDArray) -> "Model":
        """A copy with the free parameters set to theta (P,)."""
        packed, inst = self.batch(np.asarray(theta)[None])
        sld, thickness, roughness = (a[0] for a in packed)
        inst = inst.replace(**{k: float(np.ravel(getattr(inst, k))[0]) for k in self._inst_keys()})
        layers = [
            Layer(complex(sld[i + 1]), float(thickness[i + 1]), float(roughness[i + 1]), lay.name)
            for i, lay in enumerate(self.stack.layers)
        ]
        sub = self.stack.substrate
        substrate = Layer(complex(sld[-1]), 0.0, float(roughness[-1]), sub.name)
        return Model(Stack(layers, substrate, self.stack.ambient), inst, dict(self.free))

    def select(self, names) -> "Model":
        """The same model with only `names` free (others held at their current values); `names`
        is a sequence of parameter names or a boolean mask over `self.names`."""
        names = list(names)
        if names and isinstance(names[0], (bool, np.bool_)):
            if len(names) != len(self.free):
                raise ValueError(f"mask of length {len(names)} for {len(self.free)} parameters")
            names = [k for k, keep in zip(self.free, names, strict=True) if keep]
        unknown = [k for k in names if k not in self.free]
        if unknown:
            raise ValueError(f"not free in this model: {unknown}")
        return Model(self.stack, self.instrument, {k: self.free[k] for k in names})

    def _inst_keys(self) -> list[str]:
        keys = [k for k in self.free if k in INSTRUMENT_FIELDS]
        return [*keys, "background"] if "log10_background" in self.free else keys


def _layer_from(spec: dict, wavelength: float) -> Layer:
    if "material" in spec:
        s = materials.sld(spec["material"], spec["density"], wavelength)
    else:
        v = spec["sld"]
        s = complex(*v) if isinstance(v, list) else complex(v)
    return Layer(
        s,
        float(spec.get("thickness", 0.0)),
        float(spec.get("roughness", 0.0)),
        spec.get("name", ""),
    )


def load(path: str | Path) -> Model:
    """Read a model TOML: [instrument], [[layer]] top first, [substrate], [fit] bounds.

    A layer gives `sld = [re, im]` (1e-6 Å⁻²) or `material` and `density` (g/cm³).
    """
    spec = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    inst = Instrument(**spec.get("instrument", {}))
    layers = [_layer_from(s, inst.wavelength) for s in spec.get("layer", [])]
    substrate = _layer_from({"name": "substrate"} | spec["substrate"], inst.wavelength)
    ambient = spec.get("ambient", {}).get("sld", 0.0)
    ambient = complex(*ambient) if isinstance(ambient, list) else complex(ambient)
    free = {k: (float(v[0]), float(v[1])) for k, v in spec.get("fit", {}).items()}
    return Model(Stack(layers, substrate, ambient), inst, free)


def dumps(model: Model) -> str:
    """TOML text for a model (values as given or fitted)."""
    inst = model.instrument
    lines = ["[instrument]"]
    for f in ("wavelength", "kalpha2_wavelength", "nodes", *INSTRUMENT_FIELDS):
        v = getattr(inst, f)
        if v is not None:
            lines.append(f"{f} = {_num(v)}")
    for lay in model.stack.layers:
        s = complex(lay.sld)
        lines += [
            "",
            "[[layer]]",
            f'name = "{lay.name}"',
            f"sld = [{_num(s.real)}, {_num(s.imag)}]",
        ]
        lines += [f"thickness = {_num(lay.thickness)}", f"roughness = {_num(lay.roughness)}"]
    s = complex(model.stack.substrate.sld)
    lines += ["", "[substrate]", f"sld = [{_num(s.real)}, {_num(s.imag)}]"]
    lines.append(f"roughness = {_num(model.stack.substrate.roughness)}")
    if model.free:
        lines += ["", "[fit]"]
        lines += [f'"{k}" = [{_num(lo)}, {_num(hi)}]' for k, (lo, hi) in model.free.items()]
    return "\n".join(lines) + "\n"


def _num(v) -> str:
    v = float(np.ravel(v)[0]) if np.ndim(v) else v
    return repr(int(v)) if isinstance(v, int) else repr(float(v))
