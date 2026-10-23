"""X-ray reflectivity fitting engine."""

from importlib.metadata import version

from reflectofit import materials
from reflectofit.forward import Instrument, forward
from reflectofit.structure import Layer, Stack
from reflectofit.units import CU_KALPHA1, CU_KALPHA2, fwhm_to_sigma, q_to_tth, tth_to_q

__version__ = version("reflectofit")

__all__ = [
    "CU_KALPHA1",
    "CU_KALPHA2",
    "Instrument",
    "Layer",
    "Stack",
    "__version__",
    "forward",
    "fwhm_to_sigma",
    "materials",
    "q_to_tth",
    "tth_to_q",
]
