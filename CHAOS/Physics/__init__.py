"""
Physics backends: fluid properties, the propellant registry, and combustion.

    fluid_handling  saturated N2O properties from CoolProp, and the isentropic flash the
                    injector's HEM / NHNE models expand through
    fuel_handling   fuel and additive definitions loaded from fuels.csv, single or composite
    nasa_cea        chamber equilibrium via NASA GRC's `cea` package

`CEAResults` and `FluidProperties` are the two value objects the components are written
against; anything exposing the same attributes can stand in for either.
"""

from .fluid_handling import FluidProperties, liquid_density, two_phase_downstream, N2O
from .fuel_handling import Fuel, FuelComponent, FuelRegistry, fuel_registry
from .nasa_cea import NasaCEA, CEAResults

__all__ = [
    "FluidProperties", "liquid_density", "two_phase_downstream", "N2O",
    "Fuel", "FuelComponent", "FuelRegistry", "fuel_registry",
    "NasaCEA", "CEAResults",
]
