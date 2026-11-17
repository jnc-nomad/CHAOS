"""
Transient hybrid engine simulation: blowdown, chamber solve, and burn to depletion.

Built from as-built hardware dimensions, not from a design estimate -- measure the engine,
put the numbers in here, and the simulation says what it will do. Sizing a new engine from
requirements is Hybrid.design's job; this module never calls it.

Class Summary
    HybridSimulation: assembles the components from hardware dimensions and runs the burn
    SimulationResults: per-step history as column arrays, plus the run's performance metrics

Functions Summary
    solve_chamber: root-solves the chamber pressure where propellant inflow chokes the throat

All internal state is SI
"""

# Dependency imports
import csv
import numpy as np
from scipy.optimize import brentq

# Package imports
from ..Components.tank import Tank, design_tank
from ..Components.injector import Injector
from ..Components.fuel_grain import FuelGrain, fuel_flow
from ..Components.nozzle import Nozzle, G0
from ..Physics.nasa_cea import NasaCEA
from ..Physics.fuel_handling import Fuel
from .. import conversions as cv
from ..tools.thrust_curve import generate_thrustcurve as _write_thrustcurve

def solve_chamber(injector, grain, nozzle, cea, p_tank, upstream_state, p_ambient,
                  cstar_guess: float = None, tol: float = 1e-6,
                  max_iter: int = 20) -> tuple:
    """Quasi-steady chamber pressure for the current tank and grain state.

    Finds the Pc at which the propellant the injector and grain deliver exactly equals what
    the choked throat passes:  mdot_ox + mdot_fuel = Pc * At / c*.  Both sides move with Pc
    -- raising Pc throttles the injector and opens the throat -- so the balance is a root
    find, redone every timestep.

    Solved on two levels, because a CEA call costs ~1000x what an injector evaluation does
    and c* barely moves with Pc (about 1.4% across a 4x pressure range). The inner root find
    holds c* fixed and touches no combustion at all; the outer loop re-evaluates CEA at the
    solved Pc and repeats. The weak coupling makes that contract in two or three passes,
    against roughly twenty CEA calls for a single-level solve. Convergence is on c* itself,
    to `tol` relative, so the returned state satisfies the mass balance with CEA's own c*.

    The grain is evaluated, never advanced: regressing it inside the root find would march
    the geometry once per iteration.

    Args:
        cstar_guess: c* [m/s] to seed the outer loop, normally the previous timestep's.
            Without one, a mid-bracket CEA evaluation seeds it at the cost of one call.

    Returns:
        ``(p_chamber [Pa], mdot_ox [kg/s], mdot_fuel [kg/s], OF [-], rdot [m/s],
        G_ox [kg/(m^2 s)], CEAResults)``.

    Raises:
        ValueError: the feed can no longer sustain a chamber above ambient.
        RuntimeError: the c* iteration failed to converge.
    """
    # Bracket strictly inside (ambient, tank): the injector needs a positive drop to flow,
    # and the chamber can never reach the feed pressure.
    lower, upper = p_ambient * 1.01, p_tank * 0.999
    if upper <= lower:
        raise ValueError(
            f"Feed pressure {cv.pa2psi(p_tank):.1f} psi is too low to sustain a chamber above "
            f"ambient {cv.pa2psi(p_ambient):.1f} psi"
        )

    def flows(p_chamber):
        """Propellant delivered at this chamber pressure. No combustion call."""
        mdot_ox = injector.simulate(p_tank, p_chamber, upstream_state)
        mdot_fuel, rdot, g_ox = fuel_flow(grain, mdot_ox)
        return mdot_ox, mdot_fuel, rdot, g_ox

    def combustion_at(p_chamber, mdot_ox, mdot_fuel):
        of = mdot_ox / mdot_fuel if mdot_fuel > 0 else np.inf
        return of, cea.solve(p_chamber, of, expansion_ratio=nozzle.ExpansionRatio)

    cstar = cstar_guess
    if cstar is None:
        p_seed = 0.5 * (lower + upper)
        mdot_ox, mdot_fuel, _, _ = flows(p_seed)
        cstar = combustion_at(p_seed, mdot_ox, mdot_fuel)[1].cstar

    for _ in range(max_iter):
        # Inner solve, c* frozen: injector and grain only, no combustion
        def residual(p):
            mdot_ox, mdot_fuel, _, _ = flows(p)
            return (mdot_ox + mdot_fuel) - p * nozzle.AThroat / cstar

        p_chamber = brentq(residual, lower, upper, xtol=1.0, rtol=1e-10)
        mdot_ox, mdot_fuel, rdot, g_ox = flows(p_chamber)
        of, combustion = combustion_at(p_chamber, mdot_ox, mdot_fuel)

        if abs(combustion.cstar - cstar) <= tol * cstar:
            return p_chamber, mdot_ox, mdot_fuel, of, rdot, g_ox, combustion
        cstar = combustion.cstar

    raise RuntimeError(
        f"Chamber solve did not converge in {max_iter} c* iterations at "
        f"Pc={cv.pa2psi(p_chamber):.1f} psi, O/F={of:.3f}"
    )


class HybridSimulation:
    def __init__(self,
                 oxidizer: str,
                 fuel,
                 # Oxidizer tank
                 oxidizer_mass: float,
                 oxidizer_temperature: float,
                 tank_volume: float = None,
                 tank_inner_diameter: float = None,
                 tank_internal_length: float = None,
                 tank_ullage: float = None,
                 # Injector
                 injector_area: float = None,
                 injector_orifice_diameter: float = None,
                 injector_orifice_count: int = 1,
                 injector_cd: float = 0.61,
                 injector_model: str = "NHNE",
                 # Fuel grain
                 fuel_port_diameter: float = None,
                 fuel_outer_diameter: float = None,
                 fuel_length: float = None,
                 # Nozzle
                 throat_diameter: float = None,
                 exit_diameter: float = None,
                 expansion_ratio: float = None,
                 # Run configuration
                 p_ambient: float = 14.7,
                 combustion_efficiency: float = 1.0,
                 time_step: float = 0.01,
                 max_time: float = 60.0,
                 min_liquid_mass: float = 1e-3,
                 units: str = "Imperial"):
        """Assemble an engine from its as-built hardware and prepare to run it.

        Every dimension here is something you can measure on the bench. Nothing is inferred
        from a design point.

        Args:
            oxidizer: oxidizer species name for CEA (e.g. "N2O").
            fuel: a Physics.fuel_handling.Fuel, or a fuel name from fuels.csv.
            oxidizer_mass: liquid oxidizer loaded (lbm for Imperial, kg for SI).
            oxidizer_temperature: fill temperature (F for Imperial, K for SI).
            tank_volume: tank internal volume (ft^3 for Imperial, m^3 for SI); or give
                `tank_inner_diameter` and `tank_internal_length` (in / m); or give
                `tank_ullage` to size the volume from the load and that headspace fraction.
            injector_area: total injector flow area (in^2 / m^2); or give
                `injector_orifice_diameter` (in / m) and `injector_orifice_count`.
            injector_cd: measured discharge coefficient.
            injector_model: "NHNE" (default), "HEM", or "SPI".
            fuel_port_diameter: initial port diameter (in / m).
            fuel_outer_diameter: grain outer diameter (in / m).
            fuel_length: grain length (in / m).
            throat_diameter: nozzle throat diameter (in / m).
            exit_diameter: nozzle exit diameter (in / m); or give `expansion_ratio`.
            p_ambient: ambient pressure (psi for Imperial, bar for SI).
            combustion_efficiency: c* efficiency (eta_c*) for the whole run.
            time_step: integration step [s].
            max_time: hard stop [s].
            min_liquid_mass: the burn ends when tank liquid falls below this [kg]; the
                injector models all assume a liquid feed.
            units: "Imperial" or "SI" for every dimensional input above.
        """
        if units not in ("Imperial", "SI"):
            raise ValueError("Error: units must equal 'Imperial' or 'SI'")
        imperial = units == "Imperial"
        self.units = units

        self.Oxidizer = oxidizer
        self.Fuel = fuel if isinstance(fuel, Fuel) else Fuel(fuel)
        self.PAmbient = cv.psi2pa(p_ambient) if imperial else cv.bar2pa(p_ambient)
        self.CombustionEfficiency = combustion_efficiency
        self.dt = time_step
        self.MaxTime = max_time
        self.MinLiquidMass = min_liquid_mass

        if time_step <= 0 or max_time <= 0:
            raise ValueError("time_step and max_time must be positive")

        # Oxidizer tank. An ullage fraction sizes the volume from the load, the way a tank
        # is actually filled; otherwise the measured volume (or bore x length) is used.
        self.OxidizerTemperature = (cv.f2k(oxidizer_temperature) if imperial
                                    else oxidizer_temperature)
        oxidizer_mass_si = cv.lbm2kg(oxidizer_mass) if imperial else oxidizer_mass
        if tank_ullage is not None:
            self.tank = design_tank(oxidizer_mass_si, self.OxidizerTemperature,
                                    ullage_fraction=tank_ullage, units="SI")
        else:
            self.tank = Tank(liquid_mass=oxidizer_mass, starting_temperature=oxidizer_temperature,
                             internal_volume=tank_volume,
                             tank_inner_diameter=tank_inner_diameter,
                             tank_internal_length=tank_internal_length,
                             units=units)

        # Injector, fuel grain, nozzle -- straight from the measured hardware
        self.injector = Injector(area_inj=injector_area,
                                 orifice_diameter=injector_orifice_diameter,
                                 n_orifices=injector_orifice_count,
                                 Cd=injector_cd, model=injector_model, units=units)
        self.grain = FuelGrain(self.Fuel, fuel_port_diameter, fuel_outer_diameter,
                               fuel_length, units=units)
        self.nozzle = Nozzle(throat_diameter=throat_diameter, exit_diameter=exit_diameter,
                             expansion_ratio=expansion_ratio, units=units)

        # Combustion. The efficiency goes into the backend, so c* and Isp are scaled together
        # and every chamber solve sees the de-rated c* directly.
        self.CEA = NasaCEA(oxidizer, self.Fuel,
                           p_ambient=self.PAmbient,
                           combustion_efficiency=combustion_efficiency,
                           oxidizer_temperature=self.OxidizerTemperature)

    def run(self, verbose: bool = False) -> "SimulationResults":
        """March the burn to depletion and return the history and metrics.

        One step: solve the chamber on the current state, evaluate the nozzle, log, then
        advance the grain and the tank by `dt`. Explicit Euler throughout -- impulse and
        propellant are accumulated with the same rectangular rule the state is stepped with,
        so the reported totals are exactly consistent with the integration, not a separate
        quadrature over the samples.
        """
        history = {name: [] for name in (
            "time", "p_chamber", "thrust", "cstar", "cf", "isp", "OF", "T_chamber",
            "mdot_ox", "mdot_fuel", "mdot_total", "port_diameter", "regression_rate",
            "G_ox", "web_remaining", "p_tank", "dp_injector", "T_tank", "tank_liquid_mass",
            "tank_vapor_mass", "tank_total_mass", "grain_fuel_mass")}

        # Exact mass bookkeeping: what the tank and grain actually lose, rather than an
        # integral of logged flow rates
        tank_mass_start = self.tank.TotalMass
        grain_mass_start = self.grain.FuelMass

        impulse = 0.0
        t = 0.0
        reason = "max_time"
        cstar_guess = None      # warm-starts the next step's chamber solve

        while t < self.MaxTime:
            # The injector models all assume a liquid feed: stop when the liquid is gone
            if not self.tank.TwoPhase or self.tank.LiquidMass <= self.MinLiquidMass:
                reason = "liquid_exhausted"
                break
            if self.grain.Burnout:
                reason = "burnout"
                break

            p_tank = self.tank.Pressure
            try:
                (p_chamber, mdot_ox, mdot_fuel, of, rdot, g_ox,
                 combustion) = solve_chamber(self.injector, self.grain, self.nozzle, self.CEA,
                                             p_tank, self.tank.Properties, self.PAmbient,
                                             cstar_guess=cstar_guess)
            except ValueError:
                reason = "pressure_collapse"
                break
            cstar_guess = combustion.cstar

            thrust = self.nozzle.simulate(p_chamber, combustion, self.PAmbient)

            history["time"].append(t)
            history["p_chamber"].append(p_chamber)
            history["thrust"].append(thrust)
            history["cstar"].append(combustion.cstar)
            history["cf"].append(self.nozzle.Cf)
            history["isp"].append(self.nozzle.Isp)
            history["OF"].append(of)
            history["T_chamber"].append(combustion.T_chamber)
            history["mdot_ox"].append(mdot_ox)
            history["mdot_fuel"].append(mdot_fuel)
            history["mdot_total"].append(mdot_ox + mdot_fuel)
            history["port_diameter"].append(self.grain.PortDiameter)
            history["regression_rate"].append(rdot)
            history["G_ox"].append(g_ox)
            history["web_remaining"].append(self.grain.WebRemaining)
            history["grain_fuel_mass"].append(self.grain.FuelMass)
            history["p_tank"].append(p_tank)
            history["dp_injector"].append(p_tank - p_chamber)
            history["T_tank"].append(self.tank.Temperature)
            history["tank_liquid_mass"].append(self.tank.LiquidMass)
            history["tank_vapor_mass"].append(self.tank.VaporMass)
            history["tank_total_mass"].append(self.tank.TotalMass)

            if verbose:
                print(f"t={t:6.3f}s  Pc={cv.pa2psi(p_chamber):7.1f}psi  F={cv.n2lbf(thrust):8.1f}lbf  "
                      f"O/F={of:5.2f}  Dport={cv.m2in(self.grain.PortDiameter):5.3f}in")

            # Advance the state by one step, accumulating impulse the same way
            impulse += thrust * self.dt
            self.grain.simulate(mdot_ox, self.dt)
            removed = mdot_ox * self.dt
            try:
                self.tank.simulate(cv.kg2lbm(removed) if self.tank.units == "Imperial" else removed)
            except ValueError:
                reason = "tank_empty"
                t += self.dt
                break

            t += self.dt

        return SimulationResults(history, reason, burn_time=t, impulse=impulse,
                                 oxidizer_used=tank_mass_start - self.tank.TotalMass,
                                 fuel_used=grain_mass_start - self.grain.FuelMass,
                                 final_port_diameter=self.grain.PortDiameter,
                                 final_web_remaining=self.grain.WebRemaining,
                                 fuel_length=self.grain.Length,
                                 propellant_loaded=tank_mass_start + grain_mass_start)

    def __str__(self):
        return f"""
        CHAOS Hybrid Simulation (as-built)
        -----------------------------
        Oxidizer:   {self.Oxidizer}  ({self.tank.LiquidMass:.3f} kg | {cv.kg2lbm(self.tank.LiquidMass):.3f} lbm liquid)
        Fuel:       {self.Fuel.name}
        Fill Temp:  {self.OxidizerTemperature:.2f} K  |  {cv.k2f(self.OxidizerTemperature):.1f} F
        Tank P:     {(self.tank.Pressure/1e6):.3f} MPa  |  {cv.pa2psi(self.tank.Pressure):.1f} psi
        Tank Vol:   {self.tank.Volume:.4f} m^3  |  {cv.m3_2_ft3(self.tank.Volume):.3f} ft^3

        Injector:   {cv.m2_2_in2(self.injector.Area):.4f} in^2, Cd {self.injector.Cd}, {self.injector.Model}
        Grain:      port {cv.m2in(self.grain.PortDiameter):.3f} in, OD {cv.m2in(self.grain.OuterDiameter):.3f} in, L {cv.m2in(self.grain.Length):.3f} in
        Nozzle:     throat {cv.m2in(self.nozzle.DThroat):.3f} in, exit {cv.m2in(self.nozzle.DExit):.3f} in, eps {self.nozzle.ExpansionRatio:.3f}

        eta_c*:     {self.CombustionEfficiency:.3f}
        Ambient:    {cv.pa2psi(self.PAmbient):.2f} psi
        Time Step:  {self.dt:.4f} s
        """

    def __repr__(self):
        return (f"HybridSimulation({self.Fuel.name}/{self.Oxidizer}, "
                f"throat={cv.m2in(self.nozzle.DThroat):.3f} in, dt={self.dt:.4f} s)")


class SimulationResults:
    """History and performance of one burn.

    Every logged channel is a numpy array attribute (``results.thrust``, ``results.p_chamber``,
    ...) and every metric is a plain float attribute, both set once when the run ends.
    """

    def __init__(self, history: dict, termination_reason: str, burn_time: float,
                 impulse: float, oxidizer_used: float, fuel_used: float,
                 final_port_diameter: float, final_web_remaining: float,
                 fuel_length: float, propellant_loaded: float):
        self.channels = list(history)
        for name, values in history.items():
            setattr(self, name, np.array(values))

        self.TerminationReason = termination_reason
        self.Steps = len(self.time)
        self.BurnTime = burn_time                   # s, the duration actually integrated

        # Impulse accumulated inside the loop as sum(F*dt) -- the same rectangular rule the
        # state was stepped with. A trapezoid over the logged samples would mix schemes and
        # would also drop the final step, which is the one the burn ends on.
        self.TotalImpulse = impulse                 # N s
        self.AverageThrust = impulse / burn_time if burn_time > 0 else 0.0
        self.PeakThrust = float(self.thrust.max()) if self.Steps else 0.0

        # Propellant from the components' own mass change -- exact, no quadrature error
        self.OxidizerUsed = oxidizer_used           # kg
        self.FuelUsed = fuel_used                   # kg
        self.PropellantUsed = oxidizer_used + fuel_used
        self.PropellantLoaded = propellant_loaded   # kg, tank + grain at ignition
        self.OverallIsp = (impulse / (self.PropellantUsed * G0)
                           if self.PropellantUsed > 0 else 0.0)
        # The run's true mixture ratio: consumed masses, not the mean of the instantaneous
        # O/F trace, which weights every timestep equally regardless of how much flowed
        self.AverageOF = oxidizer_used / fuel_used if fuel_used > 0 else float("inf")
        # Mean oxidizer flow the same way: consumed mass over the integrated duration, not
        # the mean of the logged rates, which weights every timestep equally
        self.AverageOxFlow = oxidizer_used / burn_time if burn_time > 0 else 0.0
        self.AverageFuelFlow = fuel_used / burn_time if burn_time > 0 else 0.0

        if self.Steps:
            self.PeakChamberPressure = float(self.p_chamber.max())
            self.AverageChamberPressure = float(self.p_chamber.mean())
            self.InitialOF = float(self.OF[0])
            self.FinalOF = float(self.OF[-1])
            self.InitialThrust = float(self.thrust[0])
            self.FinalThrust = float(self.thrust[-1])
            self.PeakGOx = float(self.G_ox.max())
            # Injector authority over the blowdown. The drop decays as the tank drains, and
            # Pc follows it down, so the fraction of Pc matters as much as the psi does
            self.AveragePressureDrop = float(self.dp_injector.mean())
            self.InitialPressureDrop = float(self.dp_injector[0])
            self.FinalPressureDrop = float(self.dp_injector[-1])
            self.AverageStiffness = self.AveragePressureDrop / self.AverageChamberPressure
        else:
            for name in ("PeakChamberPressure", "AverageChamberPressure", "InitialOF",
                         "FinalOF", "InitialThrust", "FinalThrust", "PeakGOx",
                         "AveragePressureDrop", "InitialPressureDrop", "FinalPressureDrop",
                         "AverageStiffness"):
                setattr(self, name, 0.0)

        # True end-of-burn geometry, read off the grain rather than the last logged row
        self.FinalPortDiameter = final_port_diameter
        self.WebRemaining = final_web_remaining
        self.FuelLength = fuel_length               # m, grain length (motor length on export)

    def generate_thrustcurve(self, diameter: float = 98, length: float = None,
                             manufacturer: str = "BYUPropulsion", datapoints: int = None,
                             path: str = None) -> str:
        """Write the burn's thrust curve as a RASP .eng file for OpenRocket / RockSim.

        Delegates to CHAOS.tools.generate_thrustcurve, the one writer shared with the
        data-analysis path. Exported as a plugged motor with the loaded propellant (tank +
        grain at ignition) in both mass fields; hardware mass belongs in the rocket design.

        Args:
            diameter: motor diameter [mm].
            length: motor length [mm]; defaults to the fuel grain length.
            datapoints: samples in the curve; defaults to 4 per second of burn.
            path: output file; defaults to "<manufacturer> <designation>.eng".

        Returns:
            The path written.
        """
        if not self.Steps:
            raise ValueError("No burn history to export")
        length_mm = round(self.FuelLength * 1000) if length is None else length
        return _write_thrustcurve(self.time, self.thrust,
                                  propellant_mass=self.PropellantLoaded, length=length_mm,
                                  diameter=diameter, manufacturer=manufacturer,
                                  datapoints=datapoints, path=path)["path"]

    def to_csv(self, path: str):
        """Write the per-step history, one row per timestep, SI throughout."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.channels)
            columns = [getattr(self, name) for name in self.channels]
            for row in zip(*columns):
                writer.writerow(row)

    def plot(self, show: bool = True, save_path: str = None):
        """Overview: thrust, pressures, O/F, port diameter, mass flows, Isp."""
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 3, figsize=(15, 7))
        t = self.time

        axs[0, 0].plot(t, cv.n2lbf(self.thrust))
        axs[0, 0].set(title="Thrust", xlabel="time [s]", ylabel="F [lbf]")

        axs[0, 1].plot(t, cv.pa2psi(self.p_tank), label="tank")
        axs[0, 1].plot(t, cv.pa2psi(self.p_chamber), label="chamber")
        axs[0, 1].set(title="Pressure", xlabel="time [s]", ylabel="P [psi]")
        axs[0, 1].legend()

        axs[0, 2].plot(t, self.OF)
        axs[0, 2].set(title="O/F Ratio", xlabel="time [s]", ylabel="O/F [-]")

        axs[1, 0].plot(t, cv.m2in(self.port_diameter))
        axs[1, 0].set(title="Port Diameter", xlabel="time [s]", ylabel="D_port [in]")

        axs[1, 1].plot(t, self.mdot_ox, label="oxidizer")
        axs[1, 1].plot(t, self.mdot_fuel, label="fuel")
        axs[1, 1].set(title="Mass Flow", xlabel="time [s]", ylabel="mdot [kg/s]")
        axs[1, 1].legend()

        axs[1, 2].plot(t, self.isp)
        axs[1, 2].set(title="Isp", xlabel="time [s]", ylabel="Isp [s]")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120)
        if show:
            plt.show()
        return fig

    def __str__(self):
        return f"""
        CHAOS Hybrid Simulation Results
        -----------------------------
        Termination:  {self.TerminationReason}
        Burn Time:    {self.BurnTime:.3f} s  ({self.Steps} steps)

        Total Impulse:{self.TotalImpulse:.1f} N-s  |  {cv.n2lbf(self.TotalImpulse):.1f} lbf-s
        Avg Thrust:   {self.AverageThrust:.1f} N  |  {cv.n2lbf(self.AverageThrust):.1f} lbf
        Peak Thrust:  {self.PeakThrust:.1f} N  |  {cv.n2lbf(self.PeakThrust):.1f} lbf
        Thrust Decay: {cv.n2lbf(self.InitialThrust):.1f} -> {cv.n2lbf(self.FinalThrust):.1f} lbf
        Overall Isp:  {self.OverallIsp:.1f} s

        Chamber P:    {cv.pa2psi(self.AverageChamberPressure):.1f} psi avg  |  {cv.pa2psi(self.PeakChamberPressure):.1f} psi peak
        Injector dP:  {cv.pa2psi(self.AveragePressureDrop):.1f} psi avg  ({self.AverageStiffness:.1%} of Pc)  (drifted {cv.pa2psi(self.InitialPressureDrop):.1f} -> {cv.pa2psi(self.FinalPressureDrop):.1f} psi)

        Oxidizer Used:{self.OxidizerUsed:.3f} kg  |  {cv.kg2lbm(self.OxidizerUsed):.3f} lbm
        Avg Ox Flow:  {self.AverageOxFlow:.4f} kg/s  |  {cv.kg2lbm(self.AverageOxFlow):.4f} lbm/s
        Fuel Used:    {self.FuelUsed:.3f} kg  |  {cv.kg2lbm(self.FuelUsed):.3f} lbm
        Avg Fuel Flow:{self.AverageFuelFlow:.3f} kg/s | {cv.kg2lbm(self.AverageFuelFlow):.4f} lbm/s
        Propellant:   {self.PropellantUsed:.3f} kg  |  {cv.kg2lbm(self.PropellantUsed):.3f} lbm
        Overall O/F:  {self.AverageOF:.3f}  (drifted {self.InitialOF:.2f} -> {self.FinalOF:.2f})

        Final Port D: {cv.m2in(self.FinalPortDiameter):.3f} in
        Web Left:     {cv.m2in(self.WebRemaining):.3f} in
        Peak G_ox:    {self.PeakGOx:.0f} kg/(m^2 s)
        """

    def __repr__(self):
        return (f"SimulationResults({self.TerminationReason}, BurnTime={self.BurnTime:.3f} s, "
                f"TotalImpulse={self.TotalImpulse:.1f} N-s, Steps={self.Steps})")
