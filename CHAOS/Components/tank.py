"""
N2O oxidizer tank: volume sizing (design) + self-pressurized blowdown (simulation).

Class Summary
    Tank: holds the fixed geometry and the current thermodynamic state, advanced by `simulate()`

Functions Summary
    design_tank: sizes the internal volume for a liquid oxidizer load and ullage

All internal state is SI
"""

# Dependency imports
import CoolProp.CoolProp as CP
import numpy as np
from scipy.optimize import brentq

# Package imports
from ..Physics.fluid_handling import FluidProperties, N2O
from .. import conversions as cv

# Phase-change limits of the oxidizer, used to bracket the temperature solves
T_TRIPLE = CP.PropsSI("Ttriple", N2O)   # K -- below this CoolProp extrapolates into the solid
T_CRIT = CP.PropsSI("Tcrit", N2O)       # K -- above this there is no saturation dome


def _solve_temperature(residual, T_previous: float) -> float:
    """Solve ``residual(T) = 0`` for the tank temperature after a mass/energy balance.

    Bracketed from the triple point up to the previous temperature: a blowdown always cools.
    The one exception is the step that crosses from saturated to superheated, where the
    reclassified state can land a fraction of a degree above `T_previous`, so the upper end
    is walked up until the root is enclosed.
    """
    lo = T_TRIPLE + 1e-3
    hi = min(T_previous, T_CRIT - 1e-6)
    f_lo, f_hi = residual(lo), residual(hi)

    while f_lo * f_hi > 0.0 and hi < T_CRIT - 1e-6:
        hi = min(hi + 5.0, T_CRIT - 1e-6)
        f_hi = residual(hi)

    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"Tank state solve failed: no temperature in [{lo:.2f}, {hi:.2f}] K satisfies the "
            f"mass/energy balance (tank is likely empty or has gone supercritical)"
        )
    return brentq(residual, lo, hi)


class Tank:
    def __init__(self,
                 liquid_mass: float,
                 starting_temperature: float,
                 internal_volume: float = None,
                 tank_inner_diameter: float = None,
                 tank_internal_length: float = None,
                 units: str = "Imperial"):
        """
        Provide either the internal volume (`internal_volume`) or the inner diameter and
        internal length together. The tank starts saturated: the liquid load sits on its own
        vapor at `starting_temperature`, which fixes the pressure.
        """
        if units not in ("Imperial", "SI"):
            raise ValueError("Error: units must equal 'Imperial' or 'SI'")
        self.units = units
        imperial = units == "Imperial"

        # UNITS HANDLING -- everything past this block is SI
        if internal_volume is not None:
            self.Volume = cv.ft3_2_m3(internal_volume) if imperial else internal_volume
        elif tank_inner_diameter is not None and tank_internal_length is not None:
            d = cv.in2m(tank_inner_diameter) if imperial else tank_inner_diameter
            length = cv.in2m(tank_internal_length) if imperial else tank_internal_length
            self.Volume = cv.d2a(d) * length
        else:
            raise ValueError("Error: Either internal volume or ID & Length must be provided")

        self.LiquidMass = cv.lbm2kg(liquid_mass) if imperial else liquid_mass
        self.Temperature = cv.f2k(starting_temperature) if imperial else starting_temperature

        self.Properties = FluidProperties.update(self.Temperature)
        self.Pressure = self.Properties.pressure

        # Liquid & vapor volumes / masses (overall quality)
        self.LiquidVolume = self.LiquidMass / self.Properties.density_liquid
        self.VaporVolume = self.Volume - self.LiquidVolume

        if self.VaporVolume <= 0:
            raise ValueError(
                f"Tank overfilled: Total volume: {self.Volume}; "
                f"Liquid volume: {self.LiquidVolume}; Vapor volume: {self.VaporVolume}"
            )

        self.VaporMass = self.VaporVolume * self.Properties.density_vapor
        self.TotalMass = self.VaporMass + self.LiquidMass
        self.Quality = self.VaporMass / self.TotalMass
        self.TwoPhase = self.Quality <= 0.999
        self.TotalU = (self.Properties.energy_liquid * self.LiquidMass) + (self.Properties.energy_vapor * self.VaporMass)

    def simulate(self, removed_mass):
        """One transient step: advance the blowdown after `removed_mass` is drawn off.

        `removed_mass` is interpreted in the tank's declared `units` (lbm or kg). The mass and
        energy balances fix the new (V, U, m); the temperature satisfying them is then solved
        for -- on the saturation dome while liquid remains, at fixed bulk density once the
        tank has gone all vapor.
        """
        removed_mass = cv.lbm2kg(removed_mass) if self.units == "Imperial" else removed_mass
        T_old = self.Temperature

        # Energy balance: the departing stream carries the enthalpy of whichever phase is
        # being drawn, evaluated on the start-of-step state -- saturated liquid off the
        # bottom while liquid remains, superheated vapor at the current (T, rho) once it is
        # gone (self.Properties is saturated data, meaningless past the transition)
        if self.TwoPhase:
            h_out = self.Properties.enthalpy_liquid
        else:
            h_out = CP.PropsSI("H", "T", T_old, "D", self.TotalMass / self.Volume, N2O)

        # Mass balance
        self.TotalMass -= removed_mass
        if self.TotalMass <= 0:
            raise ValueError("Tank empty")
        self.TotalU -= removed_mass * h_out

        m = self.TotalMass
        u = self.TotalU / m             # specific internal energy [J/kg]

        if self.TwoPhase:
            # Saturated: find the T whose equilibrium quality exactly fills the tank
            def residual(T):
                props = FluidProperties.update(T)
                x = np.clip((u - props.energy_liquid) /
                            (props.energy_vapor - props.energy_liquid), 0.0, 1.0)
                return ((1.0 - x) * m / props.density_liquid
                        + x * m / props.density_vapor) - self.Volume

            T_new = _solve_temperature(residual, T_old)
            self.Properties = FluidProperties.update(T_new)
            self.Pressure = self.Properties.pressure
            quality = np.clip((u - self.Properties.energy_liquid) /
                              (self.Properties.energy_vapor - self.Properties.energy_liquid), 0.0, 1.0)
        else:
            # Superheated vapor: no dome left to sit on, so solve at fixed bulk density
            density = m / self.Volume
            T_new = _solve_temperature(lambda T: CP.PropsSI("U", "T", T, "D", density, N2O) - u, T_old)
            self.Pressure = CP.PropsSI("P", "T", T_new, "D", density, N2O)
            quality = 1.0

        # Commit the new state
        self.Temperature = T_new
        self.Quality = quality
        self.TwoPhase = quality <= 0.999
        if self.TwoPhase:
            self.VaporMass = quality * m
            self.LiquidMass = (1.0 - quality) * m
            self.LiquidVolume = self.LiquidMass / self.Properties.density_liquid
            self.VaporVolume = self.VaporMass / self.Properties.density_vapor
        else:
            self.Quality = 1.0
            self.VaporMass = m
            self.LiquidMass = 0.0
            self.LiquidVolume = 0.0
            self.VaporVolume = self.Volume

    # Printing and string representation
    def __str__(self):
        string = f"""
        N2O Tank Thermodynamics
        -----------------------
        Temperature: {cv.k2c(self.Temperature):.3f} °C  |  {cv.k2f(self.Temperature):.3f} °F
        Pressure:    {(self.Pressure/1e6):.3f} MPa  |  {cv.pa2psi(self.Pressure):.3f} psi

        Total Mass:    {self.TotalMass:.3f} kg  |  {cv.kg2lbm(self.TotalMass):.3f} lbm
        Liquid Mass:   {self.LiquidMass:.3f} kg  |  {cv.kg2lbm(self.LiquidMass):.3f} lbm
        Vapor Mass:    {self.VaporMass:.3f} kg  |  {cv.kg2lbm(self.VaporMass):.3f} lbm

        Total Volume:  {self.Volume:.3f} m^3  |  {cv.m3_2_ft3(self.Volume):.3f} ft^3
        Volume of Liq: {self.LiquidVolume:.3f} m^3  |  {cv.m3_2_ft3(self.LiquidVolume):.3f} ft^3
        Volume of Vap: {self.VaporVolume:.3f} m^3  |  {cv.m3_2_ft3(self.VaporVolume):.3f} ft^3
        """
        return string

    def __repr__(self):
        return (f"Tank(Volume={self.Volume:.3f} m^3, LiquidMass={self.LiquidMass:.3f} kg, "
                f"Temperature={self.Temperature:.3f} K, Pressure={self.Pressure:.3f} Pa)")


def design_tank(oxidizer_mass: float,
                starting_temperature: float,
                ullage_fraction: float = 0.15,
                units: str = "Imperial") -> Tank:
    """Size the tank volume for a required liquid oxidizer load and return the Tank.

    Args:
        oxidizer_mass: liquid oxidizer to hold (lbm for Imperial, kg for SI).
        starting_temperature: fill temperature (°F for Imperial, K for SI).
        ullage_fraction: vapor headspace as a fraction of total volume (0.15 default).
        units: "Imperial" or "SI" for the two inputs above.

    Returns:
        Tank: ready-to-simulate Tank
    """
    if units == "Imperial":
        mass = cv.lbm2kg(oxidizer_mass)
        temperature = cv.f2k(starting_temperature)
    elif units == "SI":
        mass = oxidizer_mass
        temperature = starting_temperature
    else:
        raise ValueError("Error: units must equal 'Imperial' or 'SI'")

    if not 0.0 < ullage_fraction < 1.0:
        raise ValueError("ullage_fraction must be between 0 and 1 (exclusive)")

    liquid_volume = mass / FluidProperties.update(temperature).density_liquid
    volume = liquid_volume / (1.0 - ullage_fraction)
    return Tank(internal_volume=volume, liquid_mass=mass,
                starting_temperature=temperature, units="SI")


if __name__ == "__main__":
    pass