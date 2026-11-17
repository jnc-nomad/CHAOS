"""
NASA CEA combustion: chamber equilibrium at a given (Pc, O/F).

Wraps NASA GRC's official ``cea`` Python package (Chemical Equilibrium with Applications).
The heavy objects -- the reactant/product mixtures and the rocket solver, which carry
hundreds of product species -- are built once in `__init__` and reused for every `solve()`,
so a per-timestep call costs single-digit milliseconds.

Class Summary
    CEAResults: one chamber-equilibrium solution, the contract the Components rely on
    NasaCEA: propellant pair + solver configuration, evaluated by `solve()`

Functions Summary
    (None)

CEA works in bar, K, m/s and g/mol; ``cea.R`` is J/(kmol K), so a specific gas constant in
J/(kg K) is ``cea.R / MW[g/mol]``. This module converts at the boundary: all inputs and
outputs here are SI (pressures in Pa).
"""

# Dependency imports
import numpy as np
from dataclasses import dataclass

# Package imports
from .fuel_handling import Fuel, fuel_registry
from .. import conversions as cv

# Constants
PA_PER_BAR = 1.0e5
G0 = cv.G0      # standard gravity [m/s^2], for reporting Isp in seconds


@dataclass
class CEAResults:
    """Chamber-equilibrium combustion output at a given (Pc, O/F).

    The first block is what the engine models consume. The second block is CEA's own
    ideal-nozzle answer at the same expansion ratio, carried along so the Nozzle model's
    calorically-perfect expansion can be checked against CEA's equilibrium expansion --
    they will not agree exactly, and the gap is a real part of the error budget.
    """
    T_chamber: float        # K   adiabatic flame / chamber temperature
    gamma: float            # -   isentropic exponent at the chamber station
    R: float                # J/(kg K)  specific gas constant (Ru / MW)
    MW: float               # kg/mol  mean molecular weight of products
    cstar: float            # m/s  characteristic velocity (already scaled by efficiency)
    expansion_ratio: float  # -   nozzle area ratio Ae/At the call was evaluated at
    p_chamber: float        # Pa  chamber pressure the result was computed at
    OF: float               # -   oxidizer/fuel ratio the result was computed at

    # CEA's own ideal-nozzle result at this expansion ratio, for cross-checking
    p_exit: float           # Pa  exit static pressure
    mach_exit: float        # -   exit Mach number
    cf: float               # -   thrust coefficient (nozzle only; no efficiency applied)
    isp: float              # s   specific impulse (scaled by efficiency, = cf*cstar/g0)

    def __str__(self):
        return f"""
        CEA Chamber Equilibrium
        -----------------------------
        Chamber P:   {(self.p_chamber/1e6):.3f} MPa  |  {self.p_chamber/PA_PER_BAR:.2f} bar
        O/F:         {self.OF:.3f}
        Chamber T:   {self.T_chamber:.1f} K
        gamma:       {self.gamma:.4f}
        MW:          {self.MW*1000:.3f} g/mol
        R:           {self.R:.2f} J/(kg K)
        c*:          {self.cstar:.1f} m/s

        CEA Ideal Nozzle (eps = {self.expansion_ratio:.3f})
        -----------------------------
        Exit P:      {(self.p_exit/1e3):.1f} kPa
        Exit Mach:   {self.mach_exit:.3f}
        Cf:          {self.cf:.4f}
        Isp:         {self.isp:.1f} s
        """


class NasaCEA:
    def __init__(self,
                 oxidizer: str,
                 fuel,
                 p_ambient: float = 101325.0,
                 combustion_efficiency: float = 1.0,
                 oxidizer_temperature: float = 298.15,
                 fuel_temperature: float = None,
                 cache_size: int = 4096):
        """
        Args:
            oxidizer: oxidizer species name as known to the CEA thermo database (e.g. "N2O").
            fuel: a Physics.fuel_handling.Fuel (single ingredient or a Fuel.mixture()), or a
                fuel name. A name in fuels.csv becomes a custom CEA reactant built from its
                formula and heat of formation; any other name is passed straight through as a
                CEA database species.
            p_ambient: ambient pressure [Pa], used to pick the optimum expansion ratio when
                `solve()` is not given one.
            combustion_efficiency: c* efficiency (eta_c*), actual over ideal characteristic
                velocity -- typically 0.85-0.97 for hybrids. Scales c* and, with it, Isp and
                the equilibrium chamber pressure. Cf is a nozzle quantity and is left alone.
            oxidizer_temperature: assigned oxidizer inlet temperature [K].
            fuel_temperature: assigned fuel inlet temperature [K]; defaults to each
                component's reference temperature from fuels.csv.
            cache_size: how many solved points to retain. Keys are the exact (Pc, O/F, eps)
                floats, so a cache hit returns the solution for precisely those inputs.
        """
        import cea  # imported lazily so the package imports without a CEA install

        if not 0.0 < combustion_efficiency <= 1.0:
            raise ValueError(
                f"combustion_efficiency is a ratio of actual to ideal c* and must lie in "
                f"(0, 1] (got {combustion_efficiency})"
            )
        if p_ambient <= 0:
            raise ValueError("p_ambient must be positive")

        self._cea = cea
        self.Oxidizer = oxidizer
        self.PAmbient = p_ambient
        self.CombustionEfficiency = combustion_efficiency

        # Resolve the fuel to a registry Fuel, else fall back to a CEA database species
        if isinstance(fuel, Fuel):
            fuel_obj = fuel
        elif isinstance(fuel, str):
            fuel_registry.load()
            try:
                fuel_obj = Fuel(fuel)
            except ValueError:
                fuel_obj = None
        else:
            raise TypeError("fuel must be a fuel name (str) or a Physics.fuel_handling.Fuel")

        # One CEA reactant per fuel component (a mixture has several), plus the oxidizer. The
        # weight vectors carry each component's mass fraction, so O/F applies to the fuel as a
        # whole while the internal blend is preserved.
        if fuel_obj is not None:
            self.Fuel = fuel_obj.name
            fuel_reactants, fractions, temperatures = [], [], []
            for c in fuel_obj.components:
                T = fuel_temperature if fuel_temperature is not None else c.t_ref
                fuel_reactants.append(cea.Reactant(c.name, formula=c.formula, enthalpy=c.hf,
                                                   enthalpy_units=c.hf_units, temperature=T))
                fractions.append(c.mass_fraction)
                temperatures.append(T)
        else:
            self.Fuel = fuel
            fuel_reactants = [fuel]
            fractions = [1.0]
            temperatures = [fuel_temperature if fuel_temperature is not None else 298.15]

        # Reactant order is [fuel components..., oxidizer]; the weight vectors match it
        reactants = list(fuel_reactants) + [oxidizer]
        self._reactants = cea.Mixture(reactants)
        self._products = cea.Mixture(reactants, products_from_reactants=True)
        self._solver = cea.RocketSolver(self._products, reactants=self._reactants)

        n_fuel = len(fuel_reactants)
        self._fuel_weights = np.array(fractions + [0.0])
        self._oxid_weights = np.array([0.0] * n_fuel + [1.0])
        self._temperatures = np.array(temperatures + [oxidizer_temperature])

        self._cache = {}
        self._cache_size = cache_size

    def solve(self, p_chamber: float, OF: float, expansion_ratio: float = None) -> CEAResults:
        """Evaluate chamber equilibrium at `p_chamber` [Pa] and mixture ratio `OF`.

        Args:
            p_chamber: chamber pressure [Pa].
            OF: oxidizer/fuel mass ratio.
            expansion_ratio: nozzle area ratio Ae/At to evaluate the exit station at. When
                omitted, CEA picks the ratio that expands the chamber exactly to `p_ambient`
                (the optimum-expansion design point).

        Returns:
            CEAResults for this operating point.
        """
        if p_chamber <= 0:
            raise ValueError(f"p_chamber must be positive (got {p_chamber})")
        if OF <= 0:
            raise ValueError(f"OF must be positive (got {OF})")
        if expansion_ratio is not None and expansion_ratio <= 1.0:
            raise ValueError(f"expansion_ratio must be greater than 1 (got {expansion_ratio})")

        key = (p_chamber, OF, expansion_ratio)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        cea = self._cea
        pc_bar = p_chamber / PA_PER_BAR

        soln = cea.RocketSolution(self._solver)
        weights = self._reactants.of_ratio_to_weights(self._oxid_weights, self._fuel_weights, OF)
        # Assigned chamber enthalpy of the reactants, normalized by Ru as CEA expects
        hc = self._reactants.calc_property(cea.ENTHALPY, weights, self._temperatures) / cea.R

        if expansion_ratio is None:
            # Optimum expansion: run to the chamber/ambient pressure ratio and read off Ae/At
            self._solver.solve(soln, weights, pc_bar,
                               pi_p=[pc_bar / (self.PAmbient / PA_PER_BAR)], iac=True, hc=hc)
        else:
            self._solver.solve(soln, weights, pc_bar,
                               supar=[float(expansion_ratio)], iac=True, hc=hc)

        if not soln.converged:
            raise RuntimeError(
                f"CEA did not converge at Pc={pc_bar:.2f} bar, O/F={OF:.3f}"
                + (f", eps={expansion_ratio:.3f}" if expansion_ratio is not None else "")
                + f" (last error: {soln.last_error})"
            )

        # Stations are [chamber, throat, exit]: index 0 is the chamber, -1 the exit.
        # c* is a chamber property and is constant across stations.
        MW_g = float(soln.MW[0])                    # g/mol
        eta = self.CombustionEfficiency
        cstar = float(soln.c_star[0]) * eta

        result = CEAResults(
            T_chamber=float(soln.T[0]),             # K
            gamma=float(soln.gamma_s[0]),           # -
            R=cea.R / MW_g,                         # J/(kg K)  (cea.R is J/(kmol K))
            MW=MW_g / 1000.0,                       # kg/mol
            cstar=cstar,                            # m/s
            expansion_ratio=float(soln.ae_at[-1]),  # -
            p_chamber=p_chamber,                    # Pa
            OF=OF,                                  # -
            p_exit=float(soln.P[-1]) * PA_PER_BAR,  # Pa  (CEA reports bar)
            mach_exit=float(soln.Mach[-1]),         # -
            cf=float(soln.coefficient_of_thrust[-1]),               # -
            isp=float(soln.Isp[-1]) * eta / G0,     # s   (CEA reports Isp in m/s)
        )

        # Bounded cache: a long sim would otherwise grow it without limit
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[key] = result
        return result

    def __str__(self):
        return f"""
        NASA CEA Combustion
        -----------------------------
        Oxidizer:   {self.Oxidizer}
        Fuel:       {self.Fuel}
        Ambient P:  {(self.PAmbient/1e3):.1f} kPa
        eta_c*:     {self.CombustionEfficiency:.3f}
        Cached:     {len(self._cache)} / {self._cache_size} points
        """

    def __repr__(self):
        return (f"NasaCEA(Oxidizer={self.Oxidizer!r}, Fuel={self.Fuel!r}, "
                f"CombustionEfficiency={self.CombustionEfficiency:.3f})")
