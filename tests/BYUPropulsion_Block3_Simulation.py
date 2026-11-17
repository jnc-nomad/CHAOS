"""
BYU Propulsion Block 2 -- as-built engine, simulated against fired test data.

Fill in the measured hardware below and point DATA_CSV at the test file. Everything here is
the real engine: no design estimates. Run from anywhere with

    python tests/BYUPropulsion_Block2.py
"""

# Dependency Imports
import csv
import sys
import pathlib
import numpy as np

# Package Imports
# Make the CHAOS package importable no matter where this is run from
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import CHAOS
from CHAOS import conversions as cv


# ----------------------------------------------------------------------------------------
# Test data
# ----------------------------------------------------------------------------------------
# DATA_CSV = pathlib.Path(__file__).parent / "Block2Test7.csv"
DATA_CSV = pathlib.Path(__file__).parent / "Block2Test9.csv"

# DAQ column -> the name used here. Pressures are psi, thrust lbf, temperatures C.
#
# The log's Liquid Mass / Vapor Mass / Fill % / Max Mass columns are fill-estimator
# placeholders, not weighed quantities, so they are deliberately absent from this map and
# nothing below reads them. That costs the comparison the one measurement that would pin
# oxidizer flow directly -- see the fitting note in compare().
CHANNELS = {
    "time": "Timestamp",
    "p_tank": "Tank P",
    "p_injector": "Injector P",     # manifold, downstream of the feed line
    "p_chamber": "Chamber P",
    "thrust": "Thrust",
    "T_tank": "Tank T",
}


# Test-day ambient pressure. Use the ABSOLUTE station pressure at the site, not the
# sea-level-corrected "altimeter" setting a METAR reports (~30 inHg). Provo (~4500 ft) sits
# near 25.4 inHg ~= 12.5 psi. 1 psi = 2.036 inHg.
p_ambient_inHg = 25.59                      # inHg, absolute station pressure // FROM PROVO AIRPORT DATA
# p_ambient_psi = p_ambient_inHg / 2.036      # -> psi
p_ambient_psi = 14

# ----------------------------------------------------------------------------------------
# As-built hardware  (Imperial: in, lbm, psi, F -- pass units="SI" to work in metres/kg/bar/K)
# ----------------------------------------------------------------------------------------
BLOCK2 = dict(
    oxidizer="N2O",
    fuel="paraffin",                            # fuels.csv name: "htpb", "paraffin", "pla", "abs"

    # -- Oxidizer tank ------------------------------------------------------------------
    oxidizer_mass = 10.1,                 # lbm of liquid N2O loaded
    oxidizer_temperature = 90,          # F at ignition
    tank_ullage = 0.15,                   # ft^3   (or give tank_inner_diameter + tank_internal_length,
                                        #         or tank_ullage to size it from the load)

    # -- Injector -----------------------------------------------------------------------
    injector_orifice_diameter=0.17,     # in, per orifice
    injector_orifice_count=4,           # -     (or give injector_area in in^2 instead)
    injector_cd = 0.53,                   # measured if you have it -- see the fitting note below
    injector_model="NHNE",              # "NHNE" | "HEM" | "SPI"

    # -- Fuel grain ---------------------------------------------------------------------
    fuel_port_diameter=2.5,             # in, initial
    fuel_outer_diameter=3.27,           # in
    fuel_length=10,                     # in

    # -- Nozzle -------------------------------------------------------------------------
    throat_diameter=1.25,                  # in
    exit_diameter=2.202,                # in    (or give expansion_ratio instead)

    # -- Run configuration --------------------------------------------------------------
    p_ambient = p_ambient_psi,          # psi, test-day ambient (converted from inHg above)
    combustion_efficiency=0.72,         # eta_c*, the usual thing to fit
    time_step=0.02,                     # s -- 0.1% impulse error; drop to 0.01 to confirm
    max_time=60.0,
    units="Imperial",
)


def run():
    """Simulate the as-built engine and report performance."""
    sim = CHAOS.HybridSimulation(**BLOCK2)
    print(sim)
    results = sim.run()
    print(results)
    # Written as "BYUPR Block3 <class>.eng", e.g. "BYUPR Block3 K1234.eng"
    eng_path = results.generate_thrustcurve(manufacturer="BYUPR Block3")
    print(f"Thrust curve written to {eng_path}")
    return sim, results


def load_test_data(path=DATA_CSV, ignition_psi: float = 50.0) -> dict:
    """Read the DAQ log and return the burn window with time re-zeroed to ignition.

    The stand logs for minutes on either side of a burn that lasts seconds, so the file is
    trimmed to the window where the chamber is actually lit. Chamber pressure is the marker:
    the igniter line stays high for a second after the propellant catches and the thrust cell
    reads a nonzero tare, but Pc is flat at the transducer floor until combustion starts.

    Dropouts (blank cells) and sensor FAULT strings come back as NaN rather than being
    removed, so every channel stays on one time base and a dead sensor cannot silently
    shift another channel's samples.

    Returns:
        dict of numpy arrays keyed by CHANNELS: time [s from ignition], p_tank, p_injector,
        p_chamber [psi], thrust [lbf], T_tank [C].
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    def numeric(column):
        values = np.empty(len(rows))
        for i, row in enumerate(rows):
            try:
                values[i] = float(row[column])
            except (TypeError, ValueError):     # blank cell, or a "FAULT" from the sensor
                values[i] = np.nan
        return values

    def clock_seconds(column):
        """Parse a timestamp column into seconds.

        Handles plain seconds ("12.34") and colon clock strings "[[HH:]MM:]SS.s" -- Test 9
        logs MM:SS.s (e.g. "34:25.9"), Test 7 logged HH:MM:SS.sss. Blanks/faults become NaN.
        """
        values = np.empty(len(rows))
        for i, row in enumerate(rows):
            try:
                seconds = 0.0
                for part in (row[column] or "").strip().split(":"):   # most significant first
                    seconds = seconds * 60.0 + float(part)
                values[i] = seconds
            except (TypeError, ValueError):
                values[i] = np.nan
        return values

    data = {name: (clock_seconds if name == "time" else numeric)(column)
            for name, column in CHANNELS.items()}

    lit = np.flatnonzero(data["p_chamber"] > ignition_psi)
    if lit.size == 0:
        raise ValueError(f"No chamber pressure above {ignition_psi} psi in {path} -- "
                         f"either the wrong file or the burn never lit")
    window = slice(lit[0], lit[-1] + 1)
    t_ignition = data["time"][lit[0]]
    data = {name: values[window] for name, values in data.items()}
    data["time"] = data["time"] - t_ignition
    return data

if __name__ == "__main__":
    sim, results = run()