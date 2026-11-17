"""
Oxidizer injector: flow-area sizing (design) + two-phase mass-flow models (simulation).

Class Summary
    Injector: holds the sized geometry and gives oxidizer flow via `simulate()`

Functions Summary
    mass_flux: mass flux [kg/(m^2 s)] for the SPI / HEM / NHNE (Dyer) flashing models
    design_injector: sizes a flow area for a required oxidizer flow and returns an Injector

All internal state is SI
"""

# Dependency imports
import numpy as np

# Package imports
from ..Physics.fluid_handling import two_phase_downstream
from .. import conversions as cv

# Model Types
_MODELS = ("SPI", "HEM", "NHNE")


def mass_flux(upstream_pressure: float,
              downstream_pressure: float,
              upstream_state,
              model: str = "NHNE") -> float:
    """Oxidizer mass flux G [kg/(m^2 s)] through the injector, before Cd and area.

    Args:
        upstream_pressure: feed pressure P_1 [Pa] (tank pressure).
        downstream_pressure: chamber pressure P_2 [Pa].
        upstream_state: Physics.fluid_handling.FluidProperties for the saturated upstream
            liquid, providing density, enthalpy, entropy, and saturation pressure P_v.
        model: "SPI" (incompressible liquid), "HEM" (equilibrium flashing), or
            "NHNE" (Dyer's non-equilibrium blend of the two).

    Returns:
        Mass flux (kg/(m^2 s)); 0.0 if the injector is not flowing forward.
    """
    dP = upstream_pressure - downstream_pressure
    if dP <= 0:
        return 0.0

    # Single-phase incompressible: liquid never flashes
    G_spi = np.sqrt(2.0 * upstream_state.density_liquid * dP)
    if model == "SPI":
        return G_spi

    # Homogeneous equilibrium: isentropic flash to the chamber, G = rho_2 * sqrt(2*(h_1 - h_2))
    h2, rho2 = two_phase_downstream(upstream_state.entropy_liquid, downstream_pressure)
    dh = upstream_state.enthalpy_liquid - h2
    G_hem = rho2 * np.sqrt(2.0 * dh) if dh > 0 else 0.0
    if model == "HEM":
        return G_hem

    # NHNE (Dyer): weight SPI against HEM by how far the chamber sits below vapor pressure
    denom = upstream_state.pressure - downstream_pressure
    if denom <= 0:
        return G_spi          # chamber above P_v: no flashing, pure liquid
    kappa = np.sqrt(dP / denom)
    return (kappa * G_spi + G_hem) / (1.0 + kappa)


class Injector:
    def __init__(self,
                 area_inj: float = None,
                 orifice_diameter: float = None,
                 n_orifices: int = 1,
                 Cd: float = 0.61,          # Sutton & Biblarz Ch. 8, sharp-edged orifice
                 model: str = "NHNE",
                 units: str = "Imperial"):
        """
        Provide either the total injector flow area (`area_inj`) or a single-orifice diameter
        (`orifice_diameter`) together with `n_orifices`. `model` is one of "SPI", "HEM", "NHNE".

        Cd note: calibrate against fired data where possible -- Dyer's NHNE was fit on plain
        orifices, and swirl elements have shown effective two-phase Cd well below the ~0.61
        liquid-service handbook value.
        """
        if area_inj is not None:
            self.Area = cv.in2_2_m2(area_inj) if units == "Imperial" else area_inj
        elif orifice_diameter is not None:
            d = cv.in2m(orifice_diameter) if units == "Imperial" else orifice_diameter
            self.Area = n_orifices * cv.d2a(d)
        else:
            raise ValueError("Error: provide either area_inj or orifice_diameter")

        if model not in _MODELS:
            raise ValueError(f"Unknown injector model '{model}' (expected one of {_MODELS})")

        self.Cd = Cd
        self.NOrifices = n_orifices
        self.Model = model

        # Design-point metadata, populated by design_injector()
        self.PressureDrop = None    # Pa
        self.DesignMassFlow = None  # kg/s

    def simulate(self, upstream_pressure: float, downstream_pressure: float, upstream_state) -> float:
        """Oxidizer mass flow [kg/s] at this operating point: mdot = Cd * A * G."""
        return self.Cd * self.Area * mass_flux(upstream_pressure, downstream_pressure,
                                               upstream_state, self.Model)

    def __str__(self):
        output = f"""
        Injector ({self.Model})
        -----------------------------
        Flow Area:  {self.Area:.3e} m^2  |  {cv.m2_2_in2(self.Area):.4f} in^2
        Equiv. D:   {cv.a2d(self.Area)*1000:.3f} mm  |  {cv.m2in(cv.a2d(self.Area)):.4f} in
        Orifices:   {self.NOrifices}
        Cd:         {self.Cd}
        """
        if self.PressureDrop is not None:
            output += f"""
        Design Point ({self.Model} sizing)
        -----------------------------
        Oxidizer Flow: {self.DesignMassFlow:.4f} kg/s
        Pressure Drop: {(self.PressureDrop/1e6):.3f} MPa  |  {cv.pa2psi(self.PressureDrop):.1f} psi
        """
        return output

    def __repr__(self):
        return f"Injector(Area={self.Area:.3e} m^2, Cd={self.Cd}, Model={self.Model})"


def design_injector(mdot_ox: float,
                    upstream_state=None,
                    p_downstream: float = None,
                    rho_ox: float = None,
                    pressure_drop: float = None,
                    Cd: float = 0.61,
                    model: str = "NHNE") -> Injector:
    """Size the injector flow area (SI in) and return a ready-to-simulate Injector.

    Inverts the mass-flux law for the area that passes `mdot_ox` [kg/s]:
    A = mdot / (Cd * G). The sizing route follows the inputs provided:

    - Flashing flow (preferred): pass the saturated `upstream_state`
          (Physics.fluid_handling.FluidProperties at the design fill temperature) and the
          design chamber pressure `p_downstream` [Pa]. Sized against `model`'s flux, so
          "HEM"/"NHNE" size larger than SPI -- the correct sizing for self-pressurized feed,
          where SPI overpredicts the flux -- by roughly 1.5x at modest pressure drops, and
          by more as the drop grows (over 2x at a 350 psi drop from a 70 F saturated tank).
    - SPI closed form: pass `rho_ox` [kg/m^3] and `pressure_drop` [Pa], for when no
          thermodynamic state is on hand. A = mdot / (Cd * sqrt(2 * rho * dP)), used
          regardless of `model`.

    `model` also sets the flow model the finished injector simulates with.
    """
    if upstream_state is not None and p_downstream is not None:
        G = mass_flux(upstream_state.pressure, p_downstream, upstream_state, model)
        dP = upstream_state.pressure - p_downstream
    elif rho_ox is not None and pressure_drop is not None:
        G = np.sqrt(2.0 * rho_ox * pressure_drop)
        dP = pressure_drop
    else:
        raise ValueError(
            "design_injector needs either (upstream_state, p_downstream) for HEM/NHNE "
            "flashing-flow sizing or (rho_ox, pressure_drop) for SPI sizing"
        )

    if G <= 0:
        raise ValueError("No flow at the requested design point "
                         "(p_downstream >= tank saturation pressure)")

    injector = Injector(area_inj=mdot_ox / (Cd * G), Cd=Cd, model=model, units="SI")
    injector.PressureDrop = dP
    injector.DesignMassFlow = mdot_ox
    return injector
