"""
Engine components: the physical hardware, each sized by a design function and advanced by
`simulate()`.

    Tank        self-pressurized oxidizer tank, blown down by mass/energy balance
    Injector    flashing-flow oxidizer injector (SPI / HEM / NHNE)
    FuelGrain   single circular-port cylindrical grain regressing under r = a * G_ox^n
    Nozzle      converging-diverging nozzle, choked throat and isentropic expansion

Each module also exposes the physics as plain functions, so the design path and the
simulation path share one implementation rather than each having their own.
"""

from .tank import Tank, design_tank
from .injector import Injector, design_injector, mass_flux
from .nozzle import Nozzle, design_nozzle, exit_mach, thrust_coefficient, G0
from .fuel_grain import FuelGrain, design_fuel_grain, fuel_flow

__all__ = [
    "Tank", "design_tank",
    "Injector", "design_injector", "mass_flux",
    "Nozzle", "design_nozzle", "exit_mach", "thrust_coefficient", "G0",
    "FuelGrain", "design_fuel_grain", "fuel_flow",
]
