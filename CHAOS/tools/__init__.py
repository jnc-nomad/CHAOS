"""
Shared tools used by both the design/simulation and the data-analysis sides of CHAOS.

    generate_thrustcurve  a (time, thrust) history -> RASP .eng (OpenRocket / RockSim)
"""

from .thrust_curve import generate_thrustcurve, IMPULSE_THRESHOLDS

__all__ = ["generate_thrustcurve", "IMPULSE_THRESHOLDS"]
