"""
Steady-state point design: requirements in, sized + ready-to-simulate components out.

Class Summary
    EngineDesign: sizes the Nozzle / Injector / FuelGrain / Tank from engine requirements

Functions Summary
    (None)

All internal state is SI
"""

# Package imports
from ..Components.tank import design_tank
from ..Components.injector import design_injector
from ..Components.fuel_grain import design_fuel_grain
from ..Components.nozzle import design_nozzle, thrust_coefficient
from ..Physics.nasa_cea import NasaCEA, G0
from ..Physics.fluid_handling import FluidProperties, liquid_density
from ..Physics.fuel_handling import Fuel
from .. import conversions as cv


class EngineDesign:
    def __init__(self,
                 oxidizer: str,
                 fuel,
                 p_combustion: float,
                 thrust: float,
                 burn_time: float,
                 OF: float,
                 fuel_od: float,
                 fuel_length: float = None,
                 p_ambient: float = 14.7,
                 combustion_efficiency: float = 1.0,
                 oxidizer_temperature: float = None,
                 tank_ullage: float = 0.15,
                 injector_cd: float = 0.61,
                 injector_model: str = "NHNE",
                 injector_pressure_drop: float = None,
                 injector_orifice_diameter: float = None,
                 units: str = "Imperial",
                 print_output: bool = True):
        """Size a hybrid engine from its requirements.

        Sizing chain, all of it done here:
          1. CEA at (Pc, O/F) picks the optimum expansion ratio for Pc/Pambient and returns
             c*, gamma and the chamber state.
          2. Cf comes from the Nozzle model at that area ratio -- the same function the
             simulation calls -- so the sized engine reproduces the requested thrust when
             simulated. CEA's own equilibrium Cf is kept alongside it for comparison.
          3. Total mass flow follows from F = Cf * c* * mdot, split by O/F.
          4. Each component is sized by its own design function.

        Args:
            oxidizer: oxidizer species name for CEA (e.g. "N2O").
            fuel: a Physics.fuel_handling.Fuel, or a fuel name from fuels.csv.
            p_combustion: chamber pressure (psi for Imperial, bar for SI).
            thrust: design thrust (lbf for Imperial, N for SI).
            burn_time: design burn time [s].
            OF: design oxidizer/fuel mass ratio.
            fuel_od: fuel grain outer diameter (in for Imperial, m for SI).
            fuel_length: fuel grain length; free if omitted, in which case the port is sized
                by the fuel's G_ox limit and the length follows from the fuel mass.
            p_ambient: ambient pressure (psi for Imperial, bar for SI).
            combustion_efficiency: c* efficiency (eta_c*), actual over ideal c*.
            oxidizer_temperature: tank fill temperature (F for Imperial, K for SI).
                Defaults to 68 F / 293.15 K.
            tank_ullage: tank vapor headspace as a fraction of total volume.
            injector_cd: injector discharge coefficient.
            injector_model: flashing model to size and simulate the injector with --
                "NHNE" (default), "HEM", or "SPI".
            injector_pressure_drop: forces closed-form SPI sizing at this pressure drop
                (psi for Imperial, bar for SI), for a pressure-fed system where the feed
                pressure is set by a regulator. Leave this off for a self-pressurized tank:
                the drop is then an *output*, fixed by the oxidizer's saturation pressure.
            injector_orifice_diameter: if given, reports how many orifices of this diameter
                make up the sized flow area.
            units: "Imperial" or "SI" for the dimensional inputs above.
            print_output: print the design summary on construction.
        """
        if units not in ("Imperial", "SI"):
            raise ValueError("Error: units must equal 'Imperial' or 'SI'")
        imperial = units == "Imperial"
        self.units = units

        # UNITS HANDLING -- everything past this block is SI
        self.PCombustion = cv.psi2pa(p_combustion) if imperial else cv.bar2pa(p_combustion)
        self.PAmbient = cv.psi2pa(p_ambient) if imperial else cv.bar2pa(p_ambient)
        self.Thrust = cv.lbf2n(thrust) if imperial else thrust
        self.FuelOD = cv.in2m(fuel_od) if imperial else fuel_od
        self.FuelLength = (cv.in2m(fuel_length) if imperial else fuel_length) if fuel_length else None
        if oxidizer_temperature is None:
            self.OxidizerTemperature = cv.f2k(68.0) if imperial else 293.15
        else:
            self.OxidizerTemperature = cv.f2k(oxidizer_temperature) if imperial else oxidizer_temperature

        if self.PCombustion <= self.PAmbient:
            raise ValueError(
                f"Chamber pressure ({cv.pa2psi(self.PCombustion):.1f} psi) must exceed ambient "
                f"({cv.pa2psi(self.PAmbient):.1f} psi)"
            )
        if OF <= 0 or thrust <= 0 or burn_time <= 0:
            raise ValueError("OF, thrust and burn_time must all be positive")

        self.Oxidizer = oxidizer
        self.Fuel = fuel if isinstance(fuel, Fuel) else Fuel(fuel)
        self.OF = OF
        self.BurnTime = burn_time
        self.CombustionEfficiency = combustion_efficiency
        self.TankUllage = tank_ullage

        # 1. Combustion at the design point. No expansion ratio given, so CEA returns the
        #    optimum area ratio for this chamber/ambient pressure ratio.
        self.CEA = NasaCEA(oxidizer, self.Fuel,
                           p_ambient=self.PAmbient,
                           combustion_efficiency=combustion_efficiency,
                           oxidizer_temperature=self.OxidizerTemperature)
        self.Combustion = self.CEA.solve(self.PCombustion, OF)
        self.ExpansionRatio = self.Combustion.expansion_ratio

        # 2. Cf from the Nozzle model at CEA's area ratio. Sizing against the model that
        #    simulates keeps design and simulation self-consistent; CEA's equilibrium Cf is
        #    kept for comparison, and the gap between them is a real modelling uncertainty.
        self.Cf, self.PExit, self.MachExit = thrust_coefficient(
            self.ExpansionRatio, self.Combustion.gamma, self.PCombustion, self.PAmbient)
        self.CfCEA = self.Combustion.cf

        # 3. Mass flows: F = Cf * Pc * At and c* = Pc * At / mdot, so F = Cf * c* * mdot
        self.TotalMassFlow = self.Thrust / (self.Cf * self.Combustion.cstar)
        self.OxidMassFlow = self.TotalMassFlow * (OF / (OF + 1.0))
        self.FuelMassFlow = self.TotalMassFlow / (OF + 1.0)
        self.Isp = self.Thrust / (self.TotalMassFlow * G0)
        self.Impulse = self.Thrust * self.BurnTime
        self.OxidizerLoad = self.OxidMassFlow * self.BurnTime
        self.FuelLoad = self.FuelMassFlow * self.BurnTime

        # 4a. Nozzle
        self.nozzle = design_nozzle(self.TotalMassFlow, self.PCombustion, self.Combustion)

        # 4b. Injector. Default is flashing-flow sizing against the saturated tank state: for
        #     a self-pressurized N2O feed the upstream pressure is Psat(T_ox), not a free
        #     choice, so the pressure drop falls out of the design rather than into it.
        self.OxidizerState = FluidProperties.update(self.OxidizerTemperature)
        if injector_pressure_drop is not None:
            dP = (cv.psi2pa(injector_pressure_drop) if imperial
                  else cv.bar2pa(injector_pressure_drop))
            self.PFeed = self.PCombustion + dP
            self.injector = design_injector(self.OxidMassFlow,
                                            rho_ox=liquid_density(self.OxidizerTemperature),
                                            pressure_drop=dP,
                                            Cd=injector_cd, model=injector_model)
        else:
            self.PFeed = self.OxidizerState.pressure      # tank saturation pressure
            if self.PFeed <= self.PCombustion:
                raise ValueError(
                    f"Self-pressurized feed cannot push the chamber: N2O saturation pressure "
                    f"at {cv.k2f(self.OxidizerTemperature):.1f} F is "
                    f"{cv.pa2psi(self.PFeed):.1f} psi, at or below the "
                    f"{cv.pa2psi(self.PCombustion):.1f} psi chamber. Lower the chamber "
                    f"pressure, warm the oxidizer, or supply injector_pressure_drop for a "
                    f"pressure-fed system."
                )
            self.injector = design_injector(self.OxidMassFlow,
                                            upstream_state=self.OxidizerState,
                                            p_downstream=self.PCombustion,
                                            Cd=injector_cd, model=injector_model)
        self.InjectorPressureDrop = self.injector.PressureDrop

        self.InjectorOrificeCount = None
        if injector_orifice_diameter is not None:
            d = cv.in2m(injector_orifice_diameter) if imperial else injector_orifice_diameter
            self.InjectorOrificeCount = self.injector.Area / cv.d2a(d)

        # 4c. Fuel grain. A dictated length solves the port from the fuel-mass constraint;
        #     otherwise the port comes from the fuel's G_ox limit and the length follows.
        self.grain = design_fuel_grain(self.Fuel, self.OxidMassFlow, self.FuelMassFlow,
                                       self.FuelOD, self.BurnTime,
                                       length=self.FuelLength, units="SI")
        self.FuelID = self.grain.PortDiameter
        self.FuelLength = self.grain.Length
        # What the grain will actually shed at t=0, and the O/F that really results. These
        # match the design values whenever the grain length was solved for; a dictated
        # fuel_length over-constrains the grain and they drift.
        self.ActualFuelMassFlow = self.grain.DesignMdotFuel
        self.ActualOF = self.OxidMassFlow / self.ActualFuelMassFlow

        # 4d. Tank, sized for the design oxidizer load
        self.tank = design_tank(self.OxidizerLoad, self.OxidizerTemperature,
                                ullage_fraction=tank_ullage, units="SI")

        if print_output:
            print(self)

    # Printing and string representation
    def __str__(self):
        cf_gap = 100.0 * (self.Cf / self.CfCEA - 1.0)
        return f"""
        CHAOS Hybrid Engine Design

        Requirements
        --------------------------
        Oxidizer:   {self.Oxidizer}
        Fuel:       {self.Fuel.name}
        Thrust:     {self.Thrust:.1f} N  |  {cv.n2lbf(self.Thrust):.1f} lbf
        Burn Time:  {self.BurnTime:.2f} s
        Chamber P:  {(self.PCombustion/1e6):.3f} MPa  |  {cv.pa2psi(self.PCombustion):.1f} psi
        O/F:        {self.OF:.3f}

        Combustion (NASA CEA, eta_c* = {self.CombustionEfficiency:.3f})
        --------------------------
        Chamber T:  {self.Combustion.T_chamber:.1f} K
        gamma:      {self.Combustion.gamma:.4f}
        c*:         {self.Combustion.cstar:.1f} m/s
        Cf:         {self.Cf:.4f} (nozzle model)  |  {self.CfCEA:.4f} (CEA)  |  {cf_gap:+.2f}%
        Isp:        {self.Isp:.1f} s  |  {self.Combustion.isp:.1f} s (CEA)
        Impulse:    {self.Impulse:.0f} N-s  |  {cv.n2lbf(self.Impulse):.0f} lbf-s

        Propellants
        --------------------------
        Oxidizer Flow: {self.OxidMassFlow:.4f} kg/s  |  {cv.kg2lbm(self.OxidMassFlow):.4f} lbm/s
        Fuel Flow:     {self.FuelMassFlow:.4f} kg/s  |  {cv.kg2lbm(self.FuelMassFlow):.4f} lbm/s
        Total Oxidizer:{self.OxidizerLoad:.3f} kg  |  {cv.kg2lbm(self.OxidizerLoad):.3f} lbm
        Total Fuel:    {self.FuelLoad:.3f} kg  |  {cv.kg2lbm(self.FuelLoad):.3f} lbm

        Nozzle
        --------------------------
        Throat D:   {self.nozzle.DThroat:.4f} m  |  {cv.m2in(self.nozzle.DThroat):.3f} in
        Exit D:     {self.nozzle.DExit:.4f} m  |  {cv.m2in(self.nozzle.DExit):.3f} in
        Exp. Ratio: {self.ExpansionRatio:.3f}
        Exit P:     {(self.PExit/1e3):.1f} kPa  |  {cv.pa2psi(self.PExit):.2f} psi

        Fuel Grain
        --------------------------
        Outer D:    {self.FuelOD:.4f} m  |  {cv.m2in(self.FuelOD):.3f} in
        Port D:     {self.FuelID:.4f} m  |  {cv.m2in(self.FuelID):.3f} in
        Length:     {self.FuelLength:.4f} m  |  {cv.m2in(self.FuelLength):.3f} in
        Initial G_ox:{self.grain.DesignGOx:.0f} kg/(m^2 s)  (fuel limit: {self.grain.GOxMax if self.grain.GOxMax else 'none'})
        Sheds:      {self.ActualFuelMassFlow:.4f} kg/s at t=0  ->  actual O/F {self.ActualOF:.3f} (design {self.OF:.3f})

        Injector ({self.injector.Model})
        --------------------------
        Feed P:     {(self.PFeed/1e6):.3f} MPa  |  {cv.pa2psi(self.PFeed):.1f} psi
        dP:         {(self.InjectorPressureDrop/1e6):.3f} MPa  |  {cv.pa2psi(self.InjectorPressureDrop):.1f} psi  ({100*self.InjectorPressureDrop/self.PCombustion:.1f}% of Pc)
        Flow Area:  {self.injector.Area*1e6:.3f} mm^2  |  {cv.m2_2_in2(self.injector.Area):.4f} in^2
        Equiv. D:   {cv.a2d(self.injector.Area)*1000:.3f} mm  |  {cv.m2in(cv.a2d(self.injector.Area)):.4f} in{f"""
        Orifices:   {self.InjectorOrificeCount:.2f}""" if self.InjectorOrificeCount else ""}

        Tank ({self.TankUllage:.0%} ullage)
        --------------------------
        Volume:     {self.tank.Volume:.4f} m^3  |  {cv.m3_2_ft3(self.tank.Volume):.3f} ft^3
        Liquid:     {self.tank.LiquidMass:.3f} kg  |  {cv.kg2lbm(self.tank.LiquidMass):.3f} lbm
        Pressure:   {(self.tank.Pressure/1e6):.3f} MPa  |  {cv.pa2psi(self.tank.Pressure):.1f} psi
        Fill Temp:  {self.OxidizerTemperature:.2f} K  |  {cv.k2f(self.OxidizerTemperature):.1f} F
        """

    def __repr__(self):
        return (f"EngineDesign({self.Fuel.name}/{self.Oxidizer}, Thrust={self.Thrust:.1f} N, "
                f"BurnTime={self.BurnTime:.2f} s, PCombustion={self.PCombustion/1e6:.3f} MPa)")
