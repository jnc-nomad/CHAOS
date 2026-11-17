"""
Fuel grain: port/length sizing (design) + regressing port geometry (simulation).

Class Summary
    FuelGrain: single circular-port cylindrical grain regressing under r = a * G_ox^n,
        advanced each step by `simulate()`

Functions Summary
    fuel_flow: regression rate and shed fuel flow at a port state, without regressing it
    design_fuel_grain: sizes the port and length from the outer diameter, burn time, and flows

All internal state is SI
"""

# Dependency imports
import numpy as np
from scipy.optimize import brentq

# Package imports
from ..Physics.fuel_handling import Fuel
from .. import conversions as cv


def fuel_flow(grain, mdot_ox: float) -> tuple:
    """Fuel the grain sheds at this oxidizer flow, evaluated on its *current* port.

    Pure: the grain is not advanced. The transient chamber solve calls this at trial chamber
    pressures, where regressing the grain on every root-find iteration would be wrong.

    Returns:
        ``(mdot_fuel [kg/s], regression rate [m/s], G_ox [kg/(m^2 s)])``.
    """
    g_ox = mdot_ox / grain.PortArea
    if grain.Burnout:
        return 0.0, 0.0, g_ox
    rdot = grain.a * (g_ox ** grain.n)
    return grain.rho * grain.BurnArea * rdot, rdot, g_ox


class FuelGrain:
    def __init__(self,
                 fuel: Fuel,
                 inner_diameter: float,
                 outer_diameter: float,
                 length: float,
                 units: str = "Imperial"):
        """
        Args:
            fuel: Physics.fuel_handling.Fuel providing regression coefficients (a, n) and density
            inner_diameter: initial port diameter
            outer_diameter: grain outer diameter (defines the web / burnout)
            length: grain length
            units: "Imperial" (inches) or "SI" (meters) for the geometry inputs
        """
        if units not in ("Imperial", "SI"):
            raise ValueError("Error: units must equal 'Imperial' or 'SI'")
        imperial = units == "Imperial"

        self.Fuel = fuel
        self.a = fuel.a_metric
        self.n = fuel.n
        self.rho = fuel.rho_metric
        self.GOxMax = fuel.g_ox_max

        if self.a is None or self.n is None:
            raise ValueError(
                f"Fuel '{fuel.name}' has no regression law (a, n) and cannot be burned as a "
                f"grain on its own. Use it as a component of a Fuel.mixture() instead."
            )

        # UNITS HANDLING -- everything past this block is SI
        self.PortDiameter = cv.in2m(inner_diameter) if imperial else inner_diameter
        self.OuterDiameter = cv.in2m(outer_diameter) if imperial else outer_diameter
        self.Length = cv.in2m(length) if imperial else length

        if not 0.0 < self.PortDiameter < self.OuterDiameter:
            raise ValueError(
                f"Port diameter must be positive and smaller than the outer diameter "
                f"(got port {self.PortDiameter:.4f} m, outer {self.OuterDiameter:.4f} m)"
            )
        if self.Length <= 0:
            raise ValueError("length must be positive")

        self.InitialPortDiameter = self.PortDiameter
        self.Burnout = False

        # Current port-derived geometry, refreshed by simulate()
        self.PortArea = cv.d2a(self.PortDiameter)                       # m^2
        self.BurnArea = np.pi * self.PortDiameter * self.Length         # m^2
        self.WebRemaining = (self.OuterDiameter - self.PortDiameter) / 2.0
        self.FuelMass = self.rho * np.pi * self.Length * (
            (self.OuterDiameter / 2.0) ** 2 - (self.PortDiameter / 2.0) ** 2)

        # Per-step regression, populated by simulate()
        self.GOx = None             # kg/(m^2 s)
        self.RegressionRate = None  # m/s
        self.MdotFuel = None        # kg/s
        self.DesignGOx = None       # initial G_ox at the design point, set by design_fuel_grain()
        self.DesignMdotFuel = None  # fuel flow actually shed at the design point

    def simulate(self, mdot_ox: float, dt: float) -> float:
        """One transient step: regress the port for `dt` [s] under oxidizer flow `mdot_ox` [kg/s].

        The flux, regression rate, and shed fuel are all evaluated on the port geometry at the
        *start* of the step (explicit Euler), then the port is opened out by 2*rdot*dt.

        Returns:
            Fuel mass flow [kg/s] shed over this step. `Burnout` is set once the port reaches
            the outer diameter, after which the grain stops regressing and sheds no fuel.
        """
        self.MdotFuel, self.RegressionRate, self.GOx = fuel_flow(self, mdot_ox)
        if self.Burnout:
            return 0.0

        # Grow the port: the radius advances at the regression rate
        self.PortDiameter += 2.0 * self.RegressionRate * dt
        if self.PortDiameter >= self.OuterDiameter:
            self.PortDiameter = self.OuterDiameter
            self.Burnout = True

        # Refresh the port-derived geometry
        self.PortArea = cv.d2a(self.PortDiameter)
        self.BurnArea = np.pi * self.PortDiameter * self.Length
        self.WebRemaining = (self.OuterDiameter - self.PortDiameter) / 2.0
        self.FuelMass = self.rho * np.pi * self.Length * (
            (self.OuterDiameter / 2.0) ** 2 - (self.PortDiameter / 2.0) ** 2)

        return self.MdotFuel

    # Print and string representation
    def __str__(self):
        output = f"""
        Fuel Grain ({self.Fuel.name})
        -----------------------------
        Port Diameter:  {self.PortDiameter:.4f} m  |  {cv.m2in(self.PortDiameter):.3f} in
        Outer Diameter: {self.OuterDiameter:.4f} m  |  {cv.m2in(self.OuterDiameter):.3f} in
        Length:         {self.Length:.4f} m  |  {cv.m2in(self.Length):.3f} in
        Web Remaining:  {self.WebRemaining:.4f} m  |  {cv.m2in(self.WebRemaining):.3f} in
        Fuel Remaining: {self.FuelMass:.3f} kg  |  {cv.kg2lbm(self.FuelMass):.3f} lbm
        """
        if self.DesignGOx is not None:
            output += f"""
        Design Point
        -----------------------------
        Initial G_ox:   {self.DesignGOx:.0f} kg/(m^2 s)  (fuel limit: {self.GOxMax if self.GOxMax else 'none'})
        Fuel Flow:      {self.DesignMdotFuel:.4f} kg/s  |  {cv.kg2lbm(self.DesignMdotFuel):.4f} lbm/s
        """
        return output

    def __repr__(self):
        return (f"FuelGrain({self.Fuel.name}, PortDiameter={self.PortDiameter:.4f} m, "
                f"OuterDiameter={self.OuterDiameter:.4f} m, Length={self.Length:.4f} m)")


def design_fuel_grain(fuel,
                      mdot_ox: float,
                      mdot_fuel: float,
                      outer_diameter: float,
                      burn_time: float,
                      length: float = None,
                      g_ox_max: float = None,
                      units: str = "Imperial") -> FuelGrain:
    """Size the grain and return a ready-to-simulate FuelGrain.

    A grain has two independent requirements, and both matter:

    - Flow:  it must SHED fuel at `mdot_fuel` at the design point, so the engine actually
             runs at the design O/F.  mdot_fuel = rho * (pi * D * L) * a * G_ox^n
    - Mass:  it must CONTAIN `mdot_fuel * burn_time` of fuel, so it lasts the burn.
             m = rho * pi * L * (r_outer^2 - r_port^2)

    With the outer diameter fixed, that is two equations in the two unknowns (port diameter,
    length), and `design_fuel_grain` solves both when `length` is free. Sizing on mass alone
    -- the obvious approach -- leaves the burning surface free to come out several times too
    small, and the engine then runs far leaner than designed.

    Dictating `length` over-constrains the problem: the port is then solved from the mass
    requirement and the flow requirement cannot also be met. The grain still reports the flow
    it will actually shed as `DesignMdotFuel`, so the real design point is visible.

    Args:
        fuel: Physics.fuel_handling.Fuel (or a fuel name) for regression coefficients & density
        mdot_ox: design oxidizer mass flow [kg/s]
        mdot_fuel: design fuel mass flow [kg/s]
        outer_diameter: grain outer diameter (in for Imperial, m for SI)
        burn_time: design burn time [s]
        length: grain length (in for Imperial, m for SI); solved for if omitted
        g_ox_max: max initial oxidizer flux [kg/(m^2 s)], overriding the fuel's own limit.
            The solved port is checked against it; pass float("inf") to waive the check.
        units: "Imperial" or "SI" for `outer_diameter` and `length`

    Returns:
        FuelGrain: ready-to-simulate FuelGrain
    """
    if units not in ("Imperial", "SI"):
        raise ValueError("Error: units must equal 'Imperial' or 'SI'")
    imperial = units == "Imperial"

    # UNITS HANDLING -- everything past this block is SI
    outer_diameter = cv.in2m(outer_diameter) if imperial else outer_diameter
    if length is not None:
        length = cv.in2m(length) if imperial else length

    fuel = fuel if isinstance(fuel, Fuel) else Fuel(fuel)
    if fuel.a_metric is None or fuel.n is None:
        raise ValueError(f"Fuel '{fuel.name}' has no regression law (a, n) and cannot be "
                         f"sized as a grain on its own")
    if mdot_ox <= 0 or mdot_fuel <= 0 or burn_time <= 0 or outer_diameter <= 0:
        raise ValueError("mdot_ox, mdot_fuel, burn_time and outer_diameter must all be positive")

    fuel_mass = mdot_fuel * burn_time
    r_outer = outer_diameter / 2.0

    def length_for_flow(port_diameter):
        """Grain length whose burning surface sheds exactly `mdot_fuel` at this port."""
        g_ox = mdot_ox / cv.d2a(port_diameter)
        return mdot_fuel / (fuel.rho_metric * np.pi * port_diameter
                            * fuel.a_metric * g_ox ** fuel.n)

    if length is not None:
        # Length dictated: only the mass requirement can be met
        r_port_sq = r_outer ** 2 - fuel_mass / (fuel.rho_metric * np.pi * length)
        if r_port_sq <= 0:
            raise ValueError(
                f"Fuel grain design failed: a solid grain of OD {outer_diameter:.4f} m "
                f"({cv.m2in(outer_diameter):.2f} in) x length {length:.4f} m "
                f"({cv.m2in(length):.2f} in) cannot hold the {fuel_mass:.3f} kg design "
                f"fuel mass. Increase outer_diameter or length."
            )
        port_diameter = 2.0 * np.sqrt(r_port_sq)

    else:
        # Solve the port that satisfies the flow and mass requirements together: at the
        # length that sheds mdot_fuel, the remaining web must hold exactly fuel_mass
        def residual(port_diameter):
            return (fuel.rho_metric * np.pi * length_for_flow(port_diameter)
                    * (r_outer ** 2 - (port_diameter / 2.0) ** 2)) - fuel_mass

        lo, hi = 1e-4 * outer_diameter, 0.999 * outer_diameter
        if residual(lo) * residual(hi) > 0:
            raise ValueError(
                f"Fuel grain design failed: no port diameter inside a {outer_diameter:.4f} m "
                f"({cv.m2in(outer_diameter):.2f} in) OD both sheds {mdot_fuel:.4f} kg/s and "
                f"holds the {fuel_mass:.3f} kg design fuel mass. A grain that sheds this much "
                f"needs a length of about {cv.m2in(length_for_flow(0.5 * outer_diameter)):.1f} in; "
                f"increase outer_diameter, or shorten burn_time."
            )
        port_diameter = brentq(residual, lo, hi)
        length = length_for_flow(port_diameter)

    grain = FuelGrain(fuel, port_diameter, outer_diameter, length, units="SI")
    grain.DesignGOx = mdot_ox / grain.PortArea
    # The fuel flow this grain actually sheds at the design point -- equal to `mdot_fuel` when
    # the length was solved for, and an honest mismatch when the length was dictated
    grain.DesignMdotFuel = fuel_flow(grain, mdot_ox)[0]

    g_limit = g_ox_max if g_ox_max is not None else fuel.g_ox_max
    if g_limit is not None and np.isfinite(g_limit) and grain.DesignGOx > g_limit:
        raise ValueError(
            f"Fuel grain design failed: the sized port gives an initial oxidizer flux of "
            f"{grain.DesignGOx:.0f} kg/(m^2 s), above the {g_limit:.0f} kg/(m^2 s) limit for "
            f"{fuel.name}. Above this, entrainment outruns combustion and c* efficiency "
            f"collapses. Increase outer_diameter, reduce the oxidizer flow, or pass "
            f"g_ox_max=float('inf') to waive the check."
        )
    return grain
