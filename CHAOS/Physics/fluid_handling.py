"""
Fluid property models shared across the package.

Currently provides saturated N2O (nitrous oxide) properties from CoolProp

All quantities are SI:
    temperature [K], pressure [Pa], density [kg/m^3], energy/enthalpy [J/kg]
"""

# Dependency imports
from dataclasses import dataclass
import CoolProp.CoolProp as CP

# Fluid name understood by CoolProp for the self-pressurizing oxidizer
N2O = "NitrousOxide"

# Triple-point pressure: an equilibrium flash below this is undefined (the fluid would
# freeze), and CoolProp's P,S flash throws just under it. The flashing-injector flash is
# floored here so a low ambient / chamber pressure cannot push a trial state below it.
P_TRIPLE = CP.PropsSI("ptriple", N2O)   # Pa  (~87.8 kPa / 12.74 psi for N2O)


@dataclass
class FluidProperties:
    """
    Saturated (two-phase) thermodynamic state of the oxidizer at a single temperature.

    Liquid quantities are taken at quality Q=0 and vapor quantities at Q=1. This is the
    natural description for a self-pressurizing N2O tank sitting on its saturation dome.
    """
    temperature: float      # K
    pressure: float         # Pa  (saturation pressure at `temperature`)
    density_liquid: float   # kg/m^3
    density_vapor: float    # kg/m^3
    energy_liquid: float    # J/kg  (specific internal energy)
    energy_vapor: float     # J/kg
    enthalpy_liquid: float  # J/kg
    enthalpy_vapor: float   # J/kg
    entropy_liquid: float   # J/(kg K)
    entropy_vapor: float    # J/(kg K)

    @classmethod
    def update(cls, temperature: float, fluid: str = N2O) -> "FluidProperties":
        """Build a saturated property set at `temperature` [K] for `fluid`."""
        pressure = CP.PropsSI("P", "T", temperature, "Q", 0, fluid)
        rho_liquid = CP.PropsSI("D", "T", temperature, "Q", 0, fluid)
        rho_vapor = CP.PropsSI("D", "T", temperature, "Q", 1, fluid)
        u_liquid = CP.PropsSI("U", "T", temperature, "Q", 0, fluid)
        u_vapor = CP.PropsSI("U", "T", temperature, "Q", 1, fluid)
        h_liquid = CP.PropsSI("H", "T", temperature, "Q", 0, fluid)
        h_vapor = CP.PropsSI("H", "T", temperature, "Q", 1, fluid)
        s_liquid = CP.PropsSI("S", "T", temperature, "Q", 0, fluid)
        s_vapor = CP.PropsSI("S", "T", temperature, "Q", 1, fluid)
        return cls(temperature, pressure, rho_liquid, rho_vapor,
                   u_liquid, u_vapor, h_liquid, h_vapor, s_liquid, s_vapor)


def liquid_density(temperature: float, fluid: str = N2O) -> float:
    """Saturated liquid density [kg/m^3] at `temperature` [K]."""
    return CP.PropsSI("D", "T", temperature, "Q", 0, fluid)


def two_phase_downstream(entropy_upstream: float, p_downstream: float, fluid: str = N2O):
    """Isentropic expansion of the oxidizer to `p_downstream` [Pa] at fixed entropy.

    Used by the HEM / NHNE injector models: a saturated liquid entering an injector flashes
    to a two-phase mixture as it drops to the (lower) downstream pressure. Returns
    ``(enthalpy [J/kg], density [kg/m^3])`` of that downstream two-phase state.

    The flash pressure is floored at the triple point: below it the equilibrium state is
    undefined (and CoolProp throws). A real chamber never runs there, so this only guards the
    root-find's low bracket when ambient/chamber is driven near or below ~12.7 psi.
    """
    p_flash = max(p_downstream, P_TRIPLE * 1.02)
    h = CP.PropsSI("H", "P", p_flash, "S", entropy_upstream, fluid)
    rho = CP.PropsSI("D", "P", p_flash, "S", entropy_upstream, fluid)
    return h, rho
