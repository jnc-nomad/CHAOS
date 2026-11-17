"""
Whole-engine entry points.

    EngineDesign      requirements -> nominal geometry (steady-state, one instant)
    HybridSimulation  as-built hardware -> transient blowdown performance

`simulate` never imports `design`: it is built from measured dimensions, so validating
against fired data does not pass through the design path.
"""

from .design import EngineDesign
from .simulate import HybridSimulation, SimulationResults, solve_chamber

__all__ = ["EngineDesign", "HybridSimulation", "SimulationResults", "solve_chamber"]
