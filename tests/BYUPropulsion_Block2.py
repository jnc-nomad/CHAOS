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
p_ambient_psi = p_ambient_inHg / 2.036      # -> psi

# ----------------------------------------------------------------------------------------
# As-built hardware  (Imperial: in, lbm, psi, F -- pass units="SI" to work in metres/kg/bar/K)
# ----------------------------------------------------------------------------------------
BLOCK2 = dict(
    oxidizer="N2O",
    fuel="paraffin",                            # fuels.csv name: "htpb", "paraffin", "pla", "abs"

    # -- Oxidizer tank ------------------------------------------------------------------
    oxidizer_mass = 22,                 # lbm of liquid N2O loaded
    oxidizer_temperature = 83,          # F at ignition
    tank_ullage = 0.15,                   # ft^3   (or give tank_inner_diameter + tank_internal_length,
                                        #         or tank_ullage to size it from the load)

    # -- Injector -----------------------------------------------------------------------
    injector_orifice_diameter=0.17,     # in, per orifice
    injector_orifice_count=4,           # -     (or give injector_area in in^2 instead)
    injector_cd = 0.53,                   # measured if you have it -- see the fitting note below
    injector_model="NHNE",              # "NHNE" | "HEM" | "SPI"

    # -- Fuel grain ---------------------------------------------------------------------
    fuel_port_diameter=1.125,             # in, initial
    fuel_outer_diameter=3.27,           # in
    fuel_length=10,                     # in

    # -- Nozzle -------------------------------------------------------------------------
    throat_diameter=1,                  # in
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


def compare(results, data=None, plot: bool = True):
    """Overlay the simulation against the fired data.

    Fitting note: eta_c*, injector Cd, and the grain's regression coefficients a/n all trade
    against each other, so fit them against different measurements or they are unidentifiable:

        chamber pressure trace  ->  eta_c*      (c* sets Pc for a given mass flow)
        oxidizer mass flow      ->  injector Cd (measurable cold-flow, fit it first)
        consumed fuel mass      ->  a / n       (weigh the grain before and after)

    Fitting eta_c* to absorb a Cd error will look right on this burn and mispredict the next.

    This test has no usable mass measurement -- the log's mass columns are placeholders --
    so the second line of that table is unavailable here. Cd has to come from a cold flow,
    or from the tank blowdown rate, which the pressure trace does constrain: with Cd wrong,
    the tank empties at the wrong rate and the burn ends at the wrong time regardless of how
    well eta_c* is tuned. Burn duration is the channel to read for that.

    Args:
        results: the SimulationResults from run().
        data: a load_test_data() dict, loaded from DATA_CSV when omitted.
        plot: draw the four-panel overlay.
    """
    if data is None:
        data = load_test_data()

    t = data["time"]
    measured_duration = float(t[-1] - t[0])

    def mean_over_burn(values):
        """Time-weighted mean, skipping dropouts, so gaps do not tilt the average."""
        good = np.isfinite(values)
        if good.sum() < 2:
            return np.nan
        return float(np.trapezoid(values[good], t[good]) / (t[good][-1] - t[good][0]))

    # Sim dP is tank-to-chamber, so the measured counterpart has to be too. The injector
    # manifold tap is reported separately: the tank-to-manifold gap is feed line loss the
    # simulation does not model, and it lands entirely in this comparison's dP error.
    dp_measured = data["p_tank"] - data["p_chamber"]
    dp_manifold = data["p_injector"] - data["p_chamber"]
    feed_loss = data["p_tank"] - data["p_injector"]

    impulse_measured = float(np.trapezoid(data["thrust"][np.isfinite(data["thrust"])],
                                          t[np.isfinite(data["thrust"])]))

    comparison = [
        # label,                measured,                        simulated
        ("Burn time [s]",       measured_duration,               results.BurnTime),
        ("Total impulse [lbf-s]", impulse_measured,              cv.n2lbf(results.TotalImpulse)),
        ("Peak thrust [lbf]",   float(np.nanmax(data["thrust"])), cv.n2lbf(results.PeakThrust)),
        ("Avg thrust [lbf]",    mean_over_burn(data["thrust"]),  cv.n2lbf(results.AverageThrust)),
        ("Peak Pc [psi]",       float(np.nanmax(data["p_chamber"])),
                                cv.pa2psi(results.PeakChamberPressure)),
        ("Avg Pc [psi]",        mean_over_burn(data["p_chamber"]),
                                cv.pa2psi(results.AverageChamberPressure)),
        ("Avg dP [psi]",        mean_over_burn(dp_measured),
                                cv.pa2psi(results.AveragePressureDrop)),
        ("Start tank P [psi]",  float(data["p_tank"][0]),        cv.pa2psi(results.p_tank[0])),
        ("End tank P [psi]",    float(data["p_tank"][-1]),       cv.pa2psi(results.p_tank[-1])),
    ]

    print(f"""
        Simulation vs {pathlib.Path(DATA_CSV).name}
        --------------------------------------------------------------""")
    print(f"        {'':22s}{'measured':>12s}{'sim':>12s}{'error':>10s}")
    for label, measured, simulated in comparison:
        error = (100.0 * (simulated - measured) / measured) if measured else float("nan")
        print(f"        {label:22s}{measured:12.1f}{simulated:12.1f}{error:9.1f}%")
    print(f"""
        Feed line loss (tank -> manifold, unmodelled): {mean_over_burn(feed_loss):.1f} psi avg
        Injector dP at the manifold tap:               {mean_over_burn(dp_manifold):.1f} psi avg
        Termination: {results.TerminationReason}
        """)

    if plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        t_sim = results.time

        axs[0, 0].plot(t, data["thrust"], label="measured")
        axs[0, 0].plot(t_sim, cv.n2lbf(results.thrust), label="sim")
        axs[0, 0].set(title="Thrust", ylabel="F [lbf]")

        axs[0, 1].plot(t, data["p_chamber"], label="measured")
        axs[0, 1].plot(t_sim, cv.pa2psi(results.p_chamber), label="sim")
        axs[0, 1].set(title="Chamber Pressure", ylabel="Pc [psi]")

        axs[1, 0].plot(t, data["p_tank"], label="measured (tank)")
        axs[1, 0].plot(t, data["p_injector"], label="measured (manifold)")
        axs[1, 0].plot(t_sim, cv.pa2psi(results.p_tank), label="sim")
        axs[1, 0].set(title="Feed Pressure", xlabel="time from ignition [s]", ylabel="P [psi]")

        axs[1, 1].plot(t, dp_measured, label="measured (tank - Pc)")
        axs[1, 1].plot(t, dp_manifold, label="measured (manifold - Pc)")
        axs[1, 1].plot(t_sim, cv.pa2psi(results.dp_injector), label="sim")
        axs[1, 1].set(title="Injector dP", xlabel="time from ignition [s]", ylabel="dP [psi]")

        for ax in axs.flat:
            ax.legend()
        fig.tight_layout()
        plt.show()
        return fig


if __name__ == "__main__":
    sim, results = run()
    compare(results, plot="--noplot" not in sys.argv)
