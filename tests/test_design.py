"""Verification for CHAOS rev2 / Hybrid/design.py."""
import math
import numpy as np

import sys
import pathlib

# Make the CHAOS package importable no matter where this is run from
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from CHAOS.Hybrid.design import EngineDesign
from CHAOS.Components.nozzle import thrust_coefficient, exit_mach, G0
from CHAOS.Components.injector import mass_flux
from CHAOS.Physics.fluid_handling import FluidProperties
from CHAOS.Physics.fuel_handling import Fuel
from CHAOS import conversions as cv

FAILS = []
def check(name, got, want, rtol=1e-9, atol=0.0):
    ok = np.isclose(got, want, rtol=rtol, atol=atol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r} want {want!r}")
    if not ok: FAILS.append(name)

def check_true(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond: FAILS.append(name)


print("=" * 72); print("ENGINE DESIGN: N2O / HTPB, 1000 lbf, 500 psi, O/F 7"); print("=" * 72)
eng = EngineDesign(oxidizer="N2O", fuel="htpb", p_combustion=500.0, thrust=1000.0,
                   burn_time=8.0, OF=7.0, fuel_od=6.0, units="Imperial", print_output=True)

# --- Unit conversion of the inputs ---
check("Pc converted", eng.PCombustion, cv.psi2pa(500.0), rtol=1e-12)
check("thrust converted", eng.Thrust, cv.lbf2n(1000.0), rtol=1e-12)
check("fuel OD converted", eng.FuelOD, cv.in2m(6.0), rtol=1e-12)
check("default ox temp = 68 F", eng.OxidizerTemperature, cv.f2k(68.0), rtol=1e-12)

# --- Mass flow bookkeeping ---
check("mdot splits by O/F", eng.OxidMassFlow / eng.FuelMassFlow, 7.0, rtol=1e-12)
check("flows sum to total", eng.OxidMassFlow + eng.FuelMassFlow, eng.TotalMassFlow, rtol=1e-12)
check("F = Cf * cstar * mdot", eng.Thrust, eng.Cf * eng.Combustion.cstar * eng.TotalMassFlow, rtol=1e-12)
check("Isp = F/(mdot*g0)", eng.Isp, eng.Thrust / (eng.TotalMassFlow * G0), rtol=1e-12)
check("impulse = F*t", eng.Impulse, eng.Thrust * 8.0, rtol=1e-12)
check("oxidizer load = mdot_ox * t", eng.OxidizerLoad, eng.OxidMassFlow * 8.0, rtol=1e-12)

# --- THE KEY SELF-CONSISTENCY TEST -------------------------------------------------
# Simulating the sized engine at its own design point must reproduce the requested thrust
# and the requested mass flows. This is what sizing against the nozzle model's Cf buys.
F_sim = eng.nozzle.simulate(eng.PCombustion, eng.Combustion, eng.PAmbient)
check("SIM REPRODUCES DESIGN THRUST", F_sim, eng.Thrust, rtol=1e-9)
check("SIM REPRODUCES DESIGN MASS FLOW", eng.nozzle.Mdot, eng.TotalMassFlow, rtol=1e-9)
check("sim Cf matches design Cf", eng.nozzle.Cf, eng.Cf, rtol=1e-12)
check("sim exit pressure matches design", eng.nozzle.PExit, eng.PExit, rtol=1e-12)

# Injector at the design point must pass the design oxidizer flow
mdot_ox_sim = eng.injector.simulate(eng.PFeed, eng.PCombustion, eng.OxidizerState)
check("SIM REPRODUCES DESIGN OXIDIZER FLOW", mdot_ox_sim, eng.OxidMassFlow, rtol=1e-9)

# Fuel grain at the design point must shed the design fuel flow
mdot_f_sim = eng.grain.simulate(eng.OxidMassFlow, 1e-9)
check("SIM REPRODUCES DESIGN FUEL FLOW", mdot_f_sim, eng.FuelMassFlow, rtol=1e-6)
check("grain reports the flow it sheds", eng.ActualFuelMassFlow, eng.FuelMassFlow, rtol=1e-9)
check("ACTUAL O/F EQUALS DESIGN O/F", eng.ActualOF, eng.OF, rtol=1e-9)
check("grain holds the design fuel mass", eng.grain.FuelMass, eng.FuelLoad, rtol=1e-6)
check("grain initial G_ox = mdot_ox/A_port", eng.grain.DesignGOx,
      eng.OxidMassFlow / (math.pi/4 * eng.FuelID**2), rtol=1e-9)

# Tank holds the design load
check("tank holds the design oxidizer load", eng.tank.LiquidMass, eng.OxidizerLoad, rtol=1e-12)
check("tank ullage honored", eng.tank.VaporVolume / eng.tank.Volume, 0.15, rtol=1e-9)
check("tank pressure = Psat(T_fill)", eng.tank.Pressure, eng.OxidizerState.pressure, rtol=1e-12)

# --- Nozzle geometry consistency ---
check("At = mdot*cstar/Pc", eng.nozzle.AThroat,
      eng.TotalMassFlow * eng.Combustion.cstar / eng.PCombustion, rtol=1e-12)
check("nozzle eps = CEA optimum eps", eng.nozzle.ExpansionRatio, eng.ExpansionRatio, rtol=1e-12)
# Optimum expansion means exit pressure lands near ambient
print(f"\n  exit P {cv.pa2psi(eng.PExit):.2f} psi vs ambient {cv.pa2psi(eng.PAmbient):.2f} psi "
      f"(eps={eng.ExpansionRatio:.3f})")
check_true("optimum expansion puts exit P near ambient (within 15%)",
           abs(eng.PExit/eng.PAmbient - 1) < 0.15, f"(ratio {eng.PExit/eng.PAmbient:.4f})")

# --- Cf cross-check against CEA ---
gap = 100*(eng.Cf/eng.CfCEA - 1)
print(f"  Cf nozzle model {eng.Cf:.4f} vs CEA equilibrium {eng.CfCEA:.4f}  ({gap:+.2f}%)")
check_true("nozzle-model Cf within 8% of CEA", abs(gap) < 8.0, f"({gap:+.2f}%)")
check_true("Cf physical", 1.2 < eng.Cf < 1.9, f"({eng.Cf:.4f})")
check_true("Isp physical", 180 < eng.Isp < 300, f"({eng.Isp:.1f} s)")

# --- Injector: self-pressurized sizing -------------------------------------------------
print(f"\n  Feed P {cv.pa2psi(eng.PFeed):.1f} psi = N2O Psat at 68 F; "
      f"dP {cv.pa2psi(eng.InjectorPressureDrop):.1f} psi "
      f"({100*eng.InjectorPressureDrop/eng.PCombustion:.1f}% of Pc)")
check("feed pressure IS the tank saturation pressure", eng.PFeed, eng.OxidizerState.pressure, rtol=1e-12)
check("dP = Psat - Pc", eng.InjectorPressureDrop, eng.PFeed - eng.PCombustion, rtol=1e-12)
check_true("self-pressurized dP is NOT the 20% rule of thumb",
           abs(eng.InjectorPressureDrop/eng.PCombustion - 0.20) > 0.10,
           f"(actual {100*eng.InjectorPressureDrop/eng.PCombustion:.1f}% of Pc)")
check_true("injector uses NHNE by default", eng.injector.Model == "NHNE")

# The old SPI-at-20%-of-Pc route undersizes badly: quantify it
spi = EngineDesign(oxidizer="N2O", fuel="htpb", p_combustion=500.0, thrust=1000.0,
                   burn_time=8.0, OF=7.0, fuel_od=6.0, units="Imperial",
                   injector_pressure_drop=0.20*500.0, injector_model="SPI",
                   print_output=False)
ratio = eng.injector.Area / spi.injector.Area
print(f"  NHNE-at-Psat area {eng.injector.Area*1e6:.2f} mm^2 vs "
      f"SPI-at-20%-Pc area {spi.injector.Area*1e6:.2f} mm^2  (x{ratio:.2f})")
check_true("old SPI/20% route sizes a different hole", abs(ratio - 1.0) > 0.10, f"(x{ratio:.3f})")
# And it does NOT deliver the design flow at the real feed pressure
mdot_spi_real = spi.injector.simulate(eng.PFeed, eng.PCombustion, eng.OxidizerState)
print(f"  SPI-sized injector at the REAL feed pressure passes {mdot_spi_real:.4f} kg/s "
      f"vs {eng.OxidMassFlow:.4f} kg/s design ({100*(mdot_spi_real/eng.OxidMassFlow-1):+.1f}%)")
check_true("SPI-sized injector misses the design flow at the real feed pressure",
           abs(mdot_spi_real/eng.OxidMassFlow - 1) > 0.10)

# --- Orifice count ---
eng_or = EngineDesign(oxidizer="N2O", fuel="htpb", p_combustion=500.0, thrust=1000.0,
                      burn_time=8.0, OF=7.0, fuel_od=6.0, units="Imperial",
                      injector_orifice_diameter=0.060, print_output=False)
check("orifice count = area / single-orifice area", eng_or.InjectorOrificeCount,
      eng_or.injector.Area / (math.pi/4 * cv.in2m(0.060)**2), rtol=1e-12)
check_true("orifice count is a sane number", 5 < eng_or.InjectorOrificeCount < 200,
           f"({eng_or.InjectorOrificeCount:.2f} holes of 0.060 in)")

# --- Combustion efficiency propagates into sizing ---
eta = 0.90
eng_eta = EngineDesign(oxidizer="N2O", fuel="htpb", p_combustion=500.0, thrust=1000.0,
                       burn_time=8.0, OF=7.0, fuel_od=6.0, units="Imperial",
                       combustion_efficiency=eta, print_output=False)
check("eta scales c*", eng_eta.Combustion.cstar, eng.Combustion.cstar*eta, rtol=1e-12)
check("eta leaves Cf alone", eng_eta.Cf, eng.Cf, rtol=1e-12)
check("lower eta needs MORE mass flow for the same thrust",
      eng_eta.TotalMassFlow, eng.TotalMassFlow/eta, rtol=1e-12)
# At = mdot*c*/Pc with mdot ~ 1/eta and c* ~ eta, so the throat is exactly eta-invariant
# (equivalently At = F/(Cf*Pc), which has no c* in it at all).
check("throat is exactly invariant to eta", eng_eta.nozzle.AThroat, eng.nozzle.AThroat, rtol=1e-12)
check_true("lower eta gives a bigger injector and tank",
           eng_eta.injector.Area > eng.injector.Area and eng_eta.tank.Volume > eng.tank.Volume)
check_true("lower eta lowers Isp", eng_eta.Isp < eng.Isp, f"({eng.Isp:.1f} -> {eng_eta.Isp:.1f} s)")
F_eta = eng_eta.nozzle.simulate(eng_eta.PCombustion, eng_eta.Combustion, eng_eta.PAmbient)
check("eta engine still reproduces design thrust", F_eta, eng_eta.Thrust, rtol=1e-9)

# --- Length-constrained grain ---
eng_L = EngineDesign(oxidizer="N2O", fuel="htpb", p_combustion=500.0, thrust=1000.0,
                     burn_time=8.0, OF=7.0, fuel_od=6.0, fuel_length=24.0,
                     units="Imperial", print_output=False)
check("dictated length honored", eng_L.FuelLength, cv.in2m(24.0), rtol=1e-12)
check("length-constrained grain holds the design fuel mass", eng_L.grain.FuelMass,
      eng_L.FuelLoad, rtol=1e-9)
# A dictated length over-constrains the grain: flow cannot also be met. What matters is
# that the resulting mismatch is reported rather than hidden.
check("dictated-length grain reports what it really sheds", eng_L.ActualFuelMassFlow,
      eng_L.grain.simulate(eng_L.OxidMassFlow, 1e-9), rtol=1e-9)
check("dictated-length actual O/F is consistent with that flow", eng_L.ActualOF,
      eng_L.OxidMassFlow / eng_L.ActualFuelMassFlow, rtol=1e-12)
print(f"  dictated 24 in length -> sheds {eng_L.ActualFuelMassFlow:.4f} kg/s "
      f"(design {eng_L.FuelMassFlow:.4f}), actual O/F {eng_L.ActualOF:.2f} vs design {eng_L.OF:.2f}")

# --- SI units path must agree with the Imperial path ---
eng_si = EngineDesign(oxidizer="N2O", fuel="htpb",
                      p_combustion=cv.psi2bar(500.0), thrust=cv.lbf2n(1000.0),
                      burn_time=8.0, OF=7.0, fuel_od=cv.in2m(6.0),
                      p_ambient=cv.psi2bar(14.7), oxidizer_temperature=cv.f2k(68.0),
                      units="SI", print_output=False)
check("SI path: same Pc", eng_si.PCombustion, eng.PCombustion, rtol=1e-12)
check("SI path: same thrust", eng_si.Thrust, eng.Thrust, rtol=1e-12)
check("SI path: same throat", eng_si.nozzle.AThroat, eng.nozzle.AThroat, rtol=1e-9)
check("SI path: same injector area", eng_si.injector.Area, eng.injector.Area, rtol=1e-9)
check("SI path: same tank volume", eng_si.tank.Volume, eng.tank.Volume, rtol=1e-9)

# --- Paraffin and a composite fuel ---
par = EngineDesign(oxidizer="N2O", fuel="paraffin", p_combustion=500.0, thrust=1000.0,
                   burn_time=8.0, OF=7.0, fuel_od=6.0, units="Imperial", print_output=False)
check_true("paraffin engine sizes", par.nozzle.AThroat > 0 and par.injector.Area > 0)
check_true("paraffin reproduces design thrust",
           np.isclose(par.nozzle.simulate(par.PCombustion, par.Combustion, par.PAmbient),
                      par.Thrust, rtol=1e-9))
mix = Fuel.mixture({"paraffin": 0.90, "al": 0.10})
alu = EngineDesign(oxidizer="N2O", fuel=mix, p_combustion=500.0, thrust=1000.0,
                   burn_time=8.0, OF=7.0, fuel_od=6.0, units="Imperial", print_output=False)
check_true("composite fuel engine sizes", alu.nozzle.AThroat > 0)
check_true("composite reproduces design thrust",
           np.isclose(alu.nozzle.simulate(alu.PCombustion, alu.Combustion, alu.PAmbient),
                      alu.Thrust, rtol=1e-9))

# --- Guards ---
base = dict(oxidizer="N2O", fuel="htpb", p_combustion=500.0, thrust=1000.0,
            burn_time=8.0, OF=7.0, fuel_od=6.0, units="Imperial", print_output=False)
for over, label in [(dict(p_combustion=10.0), "Pc <= ambient rejected"),
                    (dict(OF=-1.0), "negative OF rejected"),
                    (dict(thrust=-5.0), "negative thrust rejected"),
                    (dict(units="metric"), "bad units rejected")]:
    try:
        EngineDesign(**{**base, **over}); check_true(label, False, "(no error)")
    except ValueError:
        check_true(label, True)
# Self-pressurized feed that cannot push the chamber must be caught with a clear message
try:
    EngineDesign(**{**base, "p_combustion": 900.0}); check_true("feed-below-chamber rejected", False)
except ValueError as e:
    check_true("feed-below-chamber rejected", "saturation pressure" in str(e))

# --- Thrust sweep: sizes must scale sensibly ---
print("\n  Thrust sweep (Pc 500 psi, O/F 7):")
areas, throats = [], []
for F in (250.0, 500.0, 1000.0, 2000.0):
    e = EngineDesign(**{**base, "thrust": F})
    areas.append(e.injector.Area); throats.append(e.nozzle.AThroat)
    print(f"    {F:6.0f} lbf: mdot {e.TotalMassFlow:.4f} kg/s  At {e.nozzle.AThroat*1e6:7.2f} mm^2  "
          f"Ainj {e.injector.Area*1e6:6.2f} mm^2  port {cv.m2in(e.FuelID):.3f} in  L {cv.m2in(e.FuelLength):.2f} in")
check_true("throat area grows with thrust", all(b > a for a, b in zip(throats, throats[1:])))
check_true("injector area grows with thrust", all(b > a for a, b in zip(areas, areas[1:])))
check_true("throat area scales linearly with thrust", np.isclose(throats[-1]/throats[0], 8.0, rtol=1e-6),
           f"(x{throats[-1]/throats[0]:.4f} for 8x thrust)")

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS: print("   -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
