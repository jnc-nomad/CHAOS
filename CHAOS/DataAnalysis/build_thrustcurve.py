"""
Build a RASP .eng thrust curve from measured static-fire data.

The data counterpart to SimulationResults.generate_thrustcurve: both sides prepare a
(time, thrust) history and hand it to the one shared writer, CHAOS.tools.generate_thrustcurve,
so a fired motor and a designed one export identically. This module's job is only the data
prep -- read the log, trim to the burn window, de-tare, convert to SI -- then delegate.

Functions Summary
    build_thrustcurve: data file path -> RASP .eng motor file

All internal state is SI.
"""

# Dependency imports
import csv
import numpy as np

# Package imports (dual-mode: relative when imported, absolute when run as a script)
try:
    from ..tools.thrust_curve import generate_thrustcurve
    from .. import conversions as cv
except ImportError:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from CHAOS.tools.thrust_curve import generate_thrustcurve
    from CHAOS import conversions as cv


def _clock_seconds(cell: str) -> float:
    """Parse a timestamp cell into seconds: plain seconds, or "[[HH:]MM:]SS.s" clock strings."""
    try:
        seconds = 0.0
        for part in (cell or "").strip().split(":"):     # most significant first
            seconds = seconds * 60.0 + float(part)
        return seconds
    except (TypeError, ValueError):
        return np.nan


def _numeric(cell: str) -> float:
    """Parse a numeric cell, returning NaN for blanks and sensor FAULT strings."""
    try:
        return float(cell)
    except (TypeError, ValueError):
        return np.nan


def build_thrustcurve(data_path: str,
                      propellant_mass: float,
                      motor_length: float,
                      diameter: float = 98,
                      manufacturer: str = "BYUPropulsion",
                      datapoints: int = None,
                      path: str = None,
                      time_column: str = "Timestamp",
                      thrust_column: str = "Thrust",
                      chamber_column: str = "Chamber P",
                      ignition_psi: float = 50.0,
                      units: str = "Imperial") -> str:
    """Write a RASP .eng thrust curve from a fired static-fire log.

    The log is trimmed to the lit window (chamber pressure above `ignition_psi`), time is
    re-zeroed to ignition, a pre-ignition thrust tare is removed, and the resulting
    (time, thrust) history is handed to CHAOS.tools.generate_thrustcurve for export.

    Args:
        data_path: path to the DAQ CSV.
        propellant_mass: loaded propellant, oxidizer + fuel (lbm for Imperial, kg for SI).
        motor_length: motor length (in for Imperial, m for SI) -- dictated, like `diameter`.
        diameter: motor diameter [mm].
        manufacturer: used in the default output filename.
        datapoints: samples in the curve; defaults to 4 per second of burn.
        path: output file; defaults to "<manufacturer> <designation>.eng".
        time_column / thrust_column / chamber_column: DAQ column headers to read.
        ignition_psi: chamber pressure [psi] marking the lit window.
        units: "Imperial" (thrust lbf, mass lbm, length in) or "SI" (N, kg, m).

    Returns:
        The path written.
    """
    if units not in ("Imperial", "SI"):
        raise ValueError("Error: units must equal 'Imperial' or 'SI'")
    imperial = units == "Imperial"

    with open(data_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {data_path}")

    time = np.array([_clock_seconds(r.get(time_column)) for r in rows])
    thrust = np.array([_numeric(r.get(thrust_column)) for r in rows])
    chamber = np.array([_numeric(r.get(chamber_column)) for r in rows])

    # Trim to the lit window: chamber pressure is flat at the transducer floor until the
    # chamber actually lights, so it marks ignition and burnout cleanly.
    lit = np.flatnonzero(chamber > ignition_psi)
    if lit.size == 0:
        raise ValueError(f"No chamber pressure above {ignition_psi} psi in {data_path} -- "
                         f"wrong file, or check the '{chamber_column}' column name")
    i0, i1 = lit[0], lit[-1]

    # Remove the thrust-cell tare: the mean of whatever it read before ignition. Fall back to
    # no correction if the log starts already lit.
    pre = thrust[:i0][np.isfinite(thrust[:i0])]
    tare = float(pre.mean()) if pre.size else 0.0

    t = time[i0:i1 + 1] - time[i0]
    F = thrust[i0:i1 + 1] - tare
    good = np.isfinite(t) & np.isfinite(F)
    t, F = t[good], F[good]
    if t.size < 2:
        raise ValueError("Fewer than two valid (time, thrust) samples in the burn window")
    F = np.clip(F, 0.0, None)                       # a thrust curve cannot go negative
    if imperial:
        F = cv.lbf2n(F)                             # history is SI (N) for the shared writer

    propellant_kg = cv.lbm2kg(propellant_mass) if imperial else propellant_mass
    length_mm = (cv.in2m(motor_length) if imperial else motor_length) * 1000

    info = generate_thrustcurve(t, F, propellant_mass=propellant_kg, length=length_mm,
                                diameter=diameter, manufacturer=manufacturer,
                                datapoints=datapoints, path=path, source=data_path)

    print(f"Wrote {info['path']}: {info['designation']}  "
          f"({info['total_impulse']:.1f} N-s, {info['burn_time']:.3f} s, "
          f"avg {cv.n2lbf(info['average_thrust']):.1f} lbf)")
    return info["path"]


if __name__ == "__main__":
    import pathlib
    # Example: build Block 2 Test 9's curve. Propellant = 22 lbm N2O + ~2 lbm paraffin grain.
    data = pathlib.Path(__file__).resolve().parents[2] / "tests" / "Block2Test9.csv"
    build_thrustcurve(str(data), propellant_mass=24.0, motor_length=10.0,
                      path="Block2Test9.eng")
