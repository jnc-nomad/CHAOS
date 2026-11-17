"""
CHAOS -- hybrid rocket engine design and simulation.

Two halves, deliberately separate:

    EngineDesign      requirements in, nominal geometry out. A steady-state inverse solve:
                      given thrust, chamber pressure, O/F and a grain OD, it sizes the
                      nozzle, injector, fuel grain and tank. Fast and closed-form.

    HybridSimulation  as-built hardware in, real performance out. Marches the coupled
                      blowdown forward in time: tank, injector, chamber, grain, nozzle.
                      This is the only thing that predicts what an engine actually does.

Design reports nominal values at a single instant. Everything transient -- total impulse,
burn duration, O/F drift, thrust decay -- comes from the simulation. Simulation never calls
design: it takes the dimensions you measured, so validating against fired data bypasses the
design path entirely.

Typical use::

    import CHAOS

    # Size a new engine from requirements
    engine = CHAOS.EngineDesign(oxidizer="N2O", fuel="htpb", p_combustion=500.0,
                                thrust=1000.0, burn_time=8.0, OF=7.0, fuel_od=6.0)

    # Simulate hardware you can measure
    sim = CHAOS.HybridSimulation(oxidizer="N2O", fuel="htpb",
                                 oxidizer_mass=28.0, oxidizer_temperature=68.0,
                                 tank_ullage=0.15,
                                 injector_orifice_diameter=0.060, injector_orifice_count=36,
                                 fuel_port_diameter=2.507, fuel_outer_diameter=6.0,
                                 fuel_length=21.5,
                                 throat_diameter=1.292, expansion_ratio=5.50)
    results = sim.run()
    print(results)
    results.to_csv("burn.csv")

Every dimensional input is Imperial by default (in, lbm, lbf, psi, F); pass ``units="SI"``
for metres, kg, N, bar and K. All internal state is SI regardless.
"""

__version__ = "2.0.0"

# Conversions first: every submodule imports it, so it must be bound on the package before
# the subpackages are pulled in.
from . import conversions
from .conversions import G0

# Physics -- no dependency on Components
from . import Physics
from .Physics.fluid_handling import FluidProperties, liquid_density, two_phase_downstream, N2O
from .Physics.fuel_handling import Fuel, FuelComponent, FuelRegistry, fuel_registry
from .Physics.nasa_cea import NasaCEA, CEAResults

# Components -- built on Physics
from . import Components
from .Components.tank import Tank, design_tank
from .Components.injector import Injector, design_injector, mass_flux
from .Components.nozzle import Nozzle, design_nozzle, exit_mach, thrust_coefficient
from .Components.fuel_grain import FuelGrain, design_fuel_grain, fuel_flow

# Hybrid -- built on both
from . import Hybrid
from .Hybrid.design import EngineDesign
from .Hybrid.simulate import HybridSimulation, SimulationResults, solve_chamber

__all__ = [
    # Subpackages and helpers
    "__version__", "conversions", "Components", "Physics", "Hybrid", "G0",
    # Top-level entry points
    "EngineDesign", "HybridSimulation", "SimulationResults",
    # Components
    "Tank", "Injector", "Nozzle", "FuelGrain",
    "design_tank", "design_injector", "design_nozzle", "design_fuel_grain",
    # Physics
    "NasaCEA", "CEAResults", "Fuel", "FuelComponent", "FuelRegistry", "fuel_registry",
    "FluidProperties", "liquid_density", "two_phase_downstream", "N2O",
    # Shared physics functions (design and simulation both route through these)
    "mass_flux", "fuel_flow", "thrust_coefficient", "exit_mach", "solve_chamber",
]
