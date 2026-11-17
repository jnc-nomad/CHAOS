"""
Converging-diverging nozzle: throat sizing (design) + choked-flow thrust (simulation).

Class Summary
    Nozzle: holds the fixed throat/exit geometry, evaluated each step by `simulate()`

Functions Summary
    exit_mach: supersonic branch of the area-Mach relation
    thrust_coefficient: Cf from the area ratio and the chamber/ambient pressures
    design_nozzle: sizes the throat from thrust (or mdot), Pc and c*

All internal state is SI
"""

# Dependency imports
import numpy as np
from scipy.optimize import brentq

# Package imports
from .. import conversions as cv

# Constants
G0 = cv.G0  # standard gravity [m/s^2]


def exit_mach(expansion_ratio: float, gamma: float) -> float:
    """Supersonic exit Mach number for an area ratio Ae/At.

    Inverts the isentropic area-Mach relation
        A/A* = (1/M) * [ (2/(g+1)) * (1 + (g-1)/2 * M^2) ] ^ ((g+1)/(2(g-1)))
    on the M > 1 branch, where A/A* increases monotonically.
    """
    if gamma <= 1.0:
        raise ValueError(f"gamma must be greater than 1 (got {gamma})")
    if expansion_ratio <= 1.0:
        raise ValueError(f"expansion_ratio must be greater than 1 for a supersonic "
                         f"nozzle (got {expansion_ratio})")

    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))

    def area_ratio(M):
        return (1.0 / M) * ((2.0 / (gamma + 1.0)) *
                            (1.0 + 0.5 * (gamma - 1.0) * M * M)) ** exponent - expansion_ratio

    return brentq(area_ratio, 1.0 + 1e-9, 100.0)


def thrust_coefficient(expansion_ratio: float, gamma: float,
                       p_chamber: float, p_ambient: float) -> tuple:
    """Thrust coefficient Cf for a fixed area ratio (Sutton & Biblarz eq. 3-30).

    Momentum term plus pressure term, for a calorically perfect gas expanding isentropically
    at `gamma` from `p_chamber` to the exit area ratio. Sizing and simulation both go through
    here, so a nozzle designed for a thrust reproduces that thrust when simulated.

    Returns:
        ``(Cf [-], p_exit [Pa], exit Mach [-])``.
    """
    if p_chamber <= 0:
        raise ValueError(f"p_chamber must be positive (got {p_chamber})")

    mach = exit_mach(expansion_ratio, gamma)
    p_exit = p_chamber * (1.0 + 0.5 * (gamma - 1.0) * mach ** 2) ** (-gamma / (gamma - 1.0))

    momentum = np.sqrt(
        (2.0 * gamma ** 2 / (gamma - 1.0))
        * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))
        * (1.0 - (p_exit / p_chamber) ** ((gamma - 1.0) / gamma))
    )
    pressure_term = (p_exit - p_ambient) / p_chamber * expansion_ratio
    return momentum + pressure_term, p_exit, mach


class Nozzle:
    def __init__(self,
                 throat_diameter: float,
                 exit_diameter: float = None,
                 expansion_ratio: float = None,
                 units: str = "Imperial"):
        """
        Provide either the exit diameter (`exit_diameter`) or the area ratio
        (`expansion_ratio`) alongside the throat diameter. `units` applies to the diameters.
        """
        if units not in ("Imperial", "SI"):
            raise ValueError("Error: units must equal 'Imperial' or 'SI'")
        imperial = units == "Imperial"

        # UNITS HANDLING -- everything past this block is SI
        self.DThroat = cv.in2m(throat_diameter) if imperial else throat_diameter
        if self.DThroat <= 0:
            raise ValueError("throat_diameter must be positive")
        self.AThroat = cv.d2a(self.DThroat)

        if exit_diameter is not None:
            self.DExit = cv.in2m(exit_diameter) if imperial else exit_diameter
            self.AExit = cv.d2a(self.DExit)
            self.ExpansionRatio = self.AExit / self.AThroat
        elif expansion_ratio is not None:
            self.ExpansionRatio = expansion_ratio
            self.AExit = self.AThroat * expansion_ratio
            self.DExit = cv.a2d(self.AExit)
        else:
            raise ValueError("Error: provide either exit_diameter or expansion_ratio")

        if self.ExpansionRatio <= 1.0:
            raise ValueError(f"Expansion ratio must be greater than 1 for a supersonic nozzle "
                             f"(got {self.ExpansionRatio:.4f})")

        # Per-step performance, populated by simulate()
        self.Mdot = None            # kg/s
        self.MachExit = None        # -
        self.PExit = None           # Pa
        self.Cf = None              # -
        self.Thrust = None          # N
        self.Isp = None             # s
        self.ExhaustVelocity = None # m/s (effective, F/mdot -- not the true exit velocity)

    def simulate(self, p_chamber: float, combustion, p_ambient: float) -> float:
        """One transient step: choked-flow nozzle performance at the current chamber state.

        Args:
            p_chamber: chamber pressure [Pa].
            combustion: combustion result providing `.gamma` and `.cstar` [m/s] at the
                current chamber pressure and mixture ratio.
            p_ambient: ambient back pressure [Pa].

        Returns:
            Thrust [N]. The full per-step breakdown is left on the object (`Mdot`, `Cf`,
            `PExit`, `MachExit`, `Isp`, `ExhaustVelocity`).
        """
        # Choked throat: c* = Pc * At / mdot
        self.Mdot = p_chamber * self.AThroat / combustion.cstar

        # Isentropic expansion to the fixed area ratio
        self.Cf, self.PExit, self.MachExit = thrust_coefficient(
            self.ExpansionRatio, combustion.gamma, p_chamber, p_ambient)

        self.Thrust = self.Cf * p_chamber * self.AThroat
        self.ExhaustVelocity = self.Thrust / self.Mdot if self.Mdot > 0 else 0.0
        self.Isp = self.Thrust / (self.Mdot * G0) if self.Mdot > 0 else 0.0

        return self.Thrust

    # Printing and string representation
    def __str__(self):
        output = f"""
        Nozzle
        -----------------------------
        Throat D:   {self.DThroat:.4f} m  |  {cv.m2in(self.DThroat):.3f} in
        Exit D:     {self.DExit:.4f} m  |  {cv.m2in(self.DExit):.3f} in
        Throat A:   {self.AThroat:.3e} m^2  |  {cv.m2_2_in2(self.AThroat):.4f} in^2
        Exp. Ratio: {self.ExpansionRatio:.2f}
        """
        if self.Thrust is not None:
            output += f"""
        Last Operating Point
        -----------------------------
        Thrust:     {self.Thrust:.1f} N  |  {cv.n2lbf(self.Thrust):.1f} lbf
        Cf:         {self.Cf:.4f}
        Isp:        {self.Isp:.1f} s
        Mass Flow:  {self.Mdot:.4f} kg/s  |  {cv.kg2lbm(self.Mdot):.4f} lbm/s
        Exit Mach:  {self.MachExit:.3f}
        Exit P:     {(self.PExit/1e3):.1f} kPa  |  {cv.pa2psi(self.PExit):.2f} psi
        """
        return output

    def __repr__(self):
        return (f"Nozzle(DThroat={self.DThroat:.4f} m, DExit={self.DExit:.4f} m, "
                f"ExpansionRatio={self.ExpansionRatio:.2f})")


def design_nozzle(mdot_total: float, p_combustion: float, combustion) -> Nozzle:
    """Size the nozzle (SI in) and return a ready-to-simulate Nozzle.

    Inverts the characteristic-velocity definition for the throat that chokes `mdot_total`
    at `p_combustion`:  At = mdot * c* / Pc.

    Args:
        mdot_total: total propellant mass flow [kg/s].
        p_combustion: chamber pressure [Pa].
        combustion: combustion result providing `.cstar` [m/s] and `.expansion_ratio`.

    Returns:
        Nozzle: ready-to-simulate Nozzle
    """
    if mdot_total <= 0:
        raise ValueError("mdot_total must be positive")
    if p_combustion <= 0:
        raise ValueError("p_combustion must be positive")

    throat_area = mdot_total * combustion.cstar / p_combustion
    return Nozzle(throat_diameter=cv.a2d(throat_area),
                  expansion_ratio=combustion.expansion_ratio,
                  units="SI")
