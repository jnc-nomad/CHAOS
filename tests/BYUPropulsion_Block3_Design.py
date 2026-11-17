"""
BYU Propulsion Block 3 -- point design from requirements.

This is the design authority for the Block 3 engine: state what the engine has to do, and
EngineDesign sizes the nozzle, injector, fuel grain and tank to meet it (steady state, at one
instant). The transient burn -- total impulse, burn time, O/F drift -- comes from
HybridSimulation once the hardware is built; this file only fixes the nominal geometry that
gets manufactured and later compared against static-fire data.

Run from anywhere with

    python tests/BYUPropulsion_Block3_Design.py
"""

# Dependency Imports
import sys
import pathlib

# Package Imports
# Make the CHAOS package importable no matter where this is run from
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import CHAOS


# ----------------------------------------------------------------------------------------
# Block 3 requirements  (Imperial: lbf, psi, in, s -- pass units="SI" for N/bar/m/s)
# ----------------------------------------------------------------------------------------
engine = CHAOS.EngineDesign(
    oxidizer="N2O",
    fuel="paraffin",                    # fuels.csv name

    p_combustion=500,                   # psi, chamber pressure
    thrust=650,                        # lbf
    burn_time=5,                       # s
    OF=7,                               # oxidizer / fuel mass ratio
    fuel_od=3.27,                        # in, grain outer diameter (fixed envelope)

    injector_orifice_diameter=0.17,     # in -- reports how many orifices make up the area
    units="Imperial",
)
