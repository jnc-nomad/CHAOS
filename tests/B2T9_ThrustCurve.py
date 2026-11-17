"""
Block 2 Test 9 -- generate a RASP .eng thrust curve from the fired data.

Reads the Block2Test9 static-fire log and writes a thrust curve for OpenRocket / RockSim,
using the measured thrust directly (no simulation). Run from anywhere with

    python tests/B2T9_ThrustCurve.py
"""

# Dependency Imports
import sys
import pathlib

# Package Imports
# Make the CHAOS package importable no matter where this is run from
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from CHAOS.DataAnalysis.build_thrustcurve import build_thrustcurve

HERE = pathlib.Path(__file__).resolve().parent

build_thrustcurve(
    data_path=str(HERE / "Block2Test9.csv"),

    # -- motor metadata for the .eng header (the CSV can't supply these) --------------------
    propellant_mass=24.2,       # lbm loaded: 22 lbm N2O + ~2.2 lbm paraffin grain
    motor_length=10.0,          # in, fuel grain length
    diameter=98,                # mm, motor casing OD

    # -- output -----------------------------------------------------------------------------
    path=str(HERE / "Block2Test9.eng"),
    units="Imperial",
)
