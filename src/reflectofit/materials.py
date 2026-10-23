"""X-ray SLD from composition and mass density (DECISIONS #8: computed, never copied).

Uses the X-ray scattering factor tables of `periodictable`; at Cu Kα₁ Si at 2.33 g/cm³ gives
20.071 + 0.457i and SiO₂ at 2.20 g/cm³ gives 18.865 + 0.244i (1e-6 Å⁻²).
"""

from periodictable import xsf

from reflectofit.units import CU_KALPHA1


def sld(formula: str, density: float, wavelength: float = CU_KALPHA1) -> complex:
    """Complex SLD [1e-6 Å⁻²] of a compound at a mass density [g/cm³]; Im > 0 is absorption."""
    re, im = xsf.xray_sld(formula, density=density, wavelength=wavelength)
    return complex(re, im)
