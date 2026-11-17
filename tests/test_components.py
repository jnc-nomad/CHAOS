"""Independent verification of every equation in CHAOS rev2 / Components.

Each check recomputes the physics from first principles (or a textbook closed form) and
compares against what the module produces. Nothing here imports the module's own helpers to
build its reference value.
"""
import math
import numpy as np
import CoolProp.CoolProp as CP

import sys
import pathlib

# Make the CHAOS package importable no matter where this is run from
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from CHAOS.Components.tank import Tank, design_tank
from CHAOS.Components.injector import Injector, design_injector, mass_flux
from CHAOS.Components.nozzle import Nozzle, design_nozzle, exit_mach, G0
from CHAOS.Components.fuel_grain import FuelGrain, design_fuel_grain
from CHAOS.Physics.fluid_handling import FluidProperties, N2O
from CHAOS.Physics.fuel_handling import Fuel
from CHAOS import conversions as cv

FAILS = []
def check(name, got, want, rtol=1e-9, atol=0.0):
    ok = np.isclose(got, want, rtol=rtol, atol=atol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r} want {want!r}")
    if not ok:
        FAILS.append(name)

def check_true(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


print("=" * 70); print("CONVERSIONS"); print("=" * 70)
check("in2m(1)", cv.in2m(1), 0.0254)
check("d2a(2) == pi", cv.d2a(2.0), math.pi)
check("a2d(d2a(0.037))", cv.a2d(cv.d2a(0.037)), 0.037)
check("f2k(32F)", cv.f2k(32.0), 273.15, rtol=1e-12)
check("f2k(-40F)", cv.f2k(-40.0), 233.15, rtol=1e-12)
check("k2f(300K)", cv.k2f(300.0), 80.33, rtol=1e-6)
check("psi2pa(1)", cv.psi2pa(1), 6894.757293168)
check("lbm2kg(1)", cv.lbm2kg(1), 0.45359237)
check("ft3_2_m3(1)", cv.ft3_2_m3(1), 0.3048**3)


print(); print("=" * 70); print("NOZZLE"); print("=" * 70)
gamma, cstar, eps = 1.20, 1550.0, 4.0

# exit_mach: feed the solved Mach back through the area-Mach relation
Me = exit_mach(eps, gamma)
ex = (gamma + 1.0) / (2.0 * (gamma - 1.0))
eps_back = (1.0 / Me) * ((2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * Me**2)) ** ex
check("exit_mach round-trips area ratio", eps_back, eps, rtol=1e-8)
check_true("exit_mach supersonic", Me > 1.0, f"(Me={Me:.4f})")

# gamma=1.4, eps=4 has a well-known answer: Me ~= 2.9402
check("exit_mach(4, 1.4) vs tables", exit_mach(4.0, 1.4), 2.9402, rtol=1e-4)
# gamma=1.4, eps=10 -> Me ~= 3.92255
check("exit_mach(10, 1.4) vs tables", exit_mach(10.0, 1.4), 3.92255, rtol=1e-4)

class CEA:  # duck-typed combustion result
    def __init__(self, gamma, cstar, expansion_ratio):
        self.gamma, self.cstar, self.expansion_ratio = gamma, cstar, expansion_ratio

comb = CEA(gamma, cstar, eps)
mdot_total, pc, pa = 2.5, cv.psi2pa(400.0), 101325.0

noz = design_nozzle(mdot_total, pc, comb)
check("design_nozzle: At = mdot*cstar/Pc", noz.AThroat, mdot_total * cstar / pc, rtol=1e-12)
check("design_nozzle: expansion ratio carried", noz.ExpansionRatio, eps, rtol=1e-12)
check("design_nozzle: Ae = eps*At", noz.AExit, eps * noz.AThroat, rtol=1e-12)

F = noz.simulate(pc, comb, pa)
check("simulate: mdot = Pc*At/cstar reproduces design mdot", noz.Mdot, mdot_total, rtol=1e-10)
# exit pressure, isentropic
pe_ref = pc * (1.0 + 0.5 * (gamma - 1.0) * Me**2) ** (-gamma / (gamma - 1.0))
check("simulate: p_exit isentropic", noz.PExit, pe_ref, rtol=1e-12)
# Cf from Sutton eq 3-30, recomputed independently
cf_mom = math.sqrt((2 * gamma**2 / (gamma - 1)) * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))
                   * (1 - (pe_ref / pc) ** ((gamma - 1) / gamma)))
cf_ref = cf_mom + (pe_ref - pa) / pc * eps
check("simulate: Cf (Sutton 3-30)", noz.Cf, cf_ref, rtol=1e-12)
check("simulate: F = Cf*Pc*At", F, cf_ref * pc * noz.AThroat, rtol=1e-12)
check("simulate: Isp = F/(mdot*g0)", noz.Isp, F / (mdot_total * G0), rtol=1e-12)
check("simulate: c_eff = F/mdot", noz.ExhaustVelocity, F / mdot_total, rtol=1e-12)
# Cross-check: Isp*g0 should equal cstar*Cf
check("cstar*Cf == Isp*g0 (identity)", cstar * noz.Cf, noz.Isp * G0, rtol=1e-10)
check_true("Cf in physical range", 1.0 < noz.Cf < 2.2, f"(Cf={noz.Cf:.4f})")
check_true("Isp in physical range for hybrid", 150 < noz.Isp < 350, f"(Isp={noz.Isp:.1f} s)")

# Momentum-only Cf at perfect expansion must equal the vacuum-free ideal
noz_pe = Nozzle(throat_diameter=cv.a2d(noz.AThroat), expansion_ratio=eps, units="SI")
noz_pe.simulate(pc, comb, pe_ref)   # ambient == exit pressure -> pressure term vanishes
check("Cf at perfect expansion == momentum term", noz_pe.Cf, cf_mom, rtol=1e-12)

# Geometry round trip: exit_diameter and expansion_ratio constructors agree
n1 = Nozzle(throat_diameter=1.5, exit_diameter=3.0, units="SI")
check("Nozzle(De): eps = (De/Dt)^2", n1.ExpansionRatio, 4.0, rtol=1e-12)
n2 = Nozzle(throat_diameter=1.5, expansion_ratio=4.0, units="SI")
check("Nozzle(eps): De matches", n2.DExit, 3.0, rtol=1e-12)
n3 = Nozzle(throat_diameter=2.0, expansion_ratio=4.0, units="Imperial")
check("Nozzle Imperial: Dt = 2 in", n3.DThroat, cv.in2m(2.0), rtol=1e-12)

# Guards
for bad, label in [(dict(throat_diameter=1.0, expansion_ratio=0.5), "eps<1 rejected"),
                   (dict(throat_diameter=1.0), "no exit spec rejected"),
                   (dict(throat_diameter=-1.0, expansion_ratio=4.0), "negative throat rejected")]:
    try:
        Nozzle(units="SI", **bad); check_true(label, False, "(no error raised)")
    except ValueError:
        check_true(label, True)
try:
    exit_mach(4.0, 1.0); check_true("gamma<=1 rejected", False)
except ValueError:
    check_true("gamma<=1 rejected", True)


print(); print("=" * 70); print("INJECTOR"); print("=" * 70)
T_fill = cv.f2k(70.0)
st = FluidProperties.update(T_fill)
p_up, p_dn = st.pressure, cv.psi2pa(400.0)

# SPI flux against a hand-computed Bernoulli value
G_spi = mass_flux(p_up, p_dn, st, "SPI")
check("SPI flux = sqrt(2*rho_l*dP)", G_spi, math.sqrt(2 * st.density_liquid * (p_up - p_dn)), rtol=1e-12)

# HEM flux against an independent CoolProp isentropic flash
h2 = CP.PropsSI("H", "P", p_dn, "S", st.entropy_liquid, N2O)
rho2 = CP.PropsSI("D", "P", p_dn, "S", st.entropy_liquid, N2O)
G_hem_ref = rho2 * math.sqrt(2 * (st.enthalpy_liquid - h2))
G_hem = mass_flux(p_up, p_dn, st, "HEM")
check("HEM flux = rho2*sqrt(2*dh)", G_hem, G_hem_ref, rtol=1e-12)

# NHNE (Dyer) blend
kappa = math.sqrt((p_up - p_dn) / (st.pressure - p_dn))
G_nhne_ref = (kappa * G_spi + G_hem_ref) / (1 + kappa)
G_nhne = mass_flux(p_up, p_dn, st, "NHNE")
check("NHNE = (k*Gspi + Ghem)/(1+k)", G_nhne, G_nhne_ref, rtol=1e-12)
# Upstream saturated => P_up == P_v => kappa == 1 => plain average
check("kappa == 1 when upstream saturated", kappa, 1.0, rtol=1e-12)
check("NHNE == mean(SPI,HEM) when saturated", G_nhne, 0.5 * (G_spi + G_hem), rtol=1e-12)
check_true("SPI overpredicts HEM", G_spi > G_hem, f"(SPI {G_spi:.0f} vs HEM {G_hem:.0f})")
check_true("NHNE between HEM and SPI", G_hem < G_nhne < G_spi)
# The SPI/HEM ratio is not a constant: it grows with pressure drop and collapses to 1 as
# dP -> 0. Check that trend rather than a single magic number.
ratios = []
for pc_test in (cv.psi2pa(700.0), cv.psi2pa(600.0), cv.psi2pa(500.0), cv.psi2pa(400.0), cv.psi2pa(200.0)):
    ratios.append(mass_flux(p_up, pc_test, st, "SPI") / mass_flux(p_up, pc_test, st, "HEM"))
check_true("SPI/HEM ratio grows monotonically with dP",
           all(b > a for a, b in zip(ratios, ratios[1:])),
           f"(ratios {[round(r,3) for r in ratios]})")
check_true("SPI/HEM -> 1 as dP -> 0", ratios[0] < 1.6, f"(ratio at 50 psi drop = {ratios[0]:.3f})")

# No-flow and reverse-flow guards
check("no flow when dP<=0", mass_flux(p_dn, p_up, st, "NHNE"), 0.0)
check("no flow when dP==0", mass_flux(p_up, p_up, st, "NHNE"), 0.0)

# design_injector inverts simulate exactly, for every model
for model in ("SPI", "HEM", "NHNE"):
    mdot_target = 1.75
    inj = design_injector(mdot_target, upstream_state=st, p_downstream=p_dn, Cd=0.61, model=model)
    check(f"design_injector({model}) round-trips mdot",
          inj.simulate(p_up, p_dn, st), mdot_target, rtol=1e-10)
    check(f"design_injector({model}) records dP", inj.PressureDrop, p_up - p_dn, rtol=1e-12)

# Areas must order HEM > NHNE > SPI for the same target flow (lower flux -> bigger hole)
a_spi = design_injector(1.75, upstream_state=st, p_downstream=p_dn, model="SPI").Area
a_hem = design_injector(1.75, upstream_state=st, p_downstream=p_dn, model="HEM").Area
a_nhne = design_injector(1.75, upstream_state=st, p_downstream=p_dn, model="NHNE").Area
check_true("area ordering HEM > NHNE > SPI", a_hem > a_nhne > a_spi,
           f"(SPI {a_spi:.3e}, NHNE {a_nhne:.3e}, HEM {a_hem:.3e})")

# Closed-form SPI sizing route
inj_cf = design_injector(1.75, rho_ox=st.density_liquid, pressure_drop=p_up - p_dn,
                         Cd=0.61, model="SPI")
check("closed-form SPI area == state-based SPI area", inj_cf.Area, a_spi, rtol=1e-12)
check("closed-form area = mdot/(Cd*sqrt(2*rho*dP))", inj_cf.Area,
      1.75 / (0.61 * math.sqrt(2 * st.density_liquid * (p_up - p_dn))), rtol=1e-12)

# Geometry constructors
i1 = Injector(orifice_diameter=0.0015, n_orifices=24, units="SI")
check("n orifices -> total area", i1.Area, 24 * math.pi * 0.0015**2 / 4, rtol=1e-12)
i2 = Injector(area_inj=0.25, units="Imperial")
check("Imperial area input", i2.Area, 0.25 * 0.0254**2, rtol=1e-12)
try:
    Injector(area_inj=1.0, model="BOGUS"); check_true("bad model rejected", False)
except ValueError:
    check_true("bad model rejected", True)
try:
    Injector(); check_true("no geometry rejected", False)
except ValueError:
    check_true("no geometry rejected", True)


print(); print("=" * 70); print("TANK"); print("=" * 70)
tk = design_tank(oxidizer_mass=30.0, starting_temperature=70.0, ullage_fraction=0.15,
                 units="Imperial")
m_want = cv.lbm2kg(30.0)
check("design_tank: liquid mass", tk.LiquidMass, m_want, rtol=1e-12)
check("design_tank: ullage fraction", tk.VaporVolume / tk.Volume, 0.15, rtol=1e-9)
check("design_tank: temperature", tk.Temperature, cv.f2k(70.0), rtol=1e-12)
check("design_tank: P = Psat(T)", tk.Pressure,
      CP.PropsSI("P", "T", cv.f2k(70.0), "Q", 0, N2O), rtol=1e-12)
check("init: liquid volume = m/rho_l", tk.LiquidVolume,
      tk.LiquidMass / CP.PropsSI("D", "T", tk.Temperature, "Q", 0, N2O), rtol=1e-12)
check("init: vapor mass = V_v*rho_v", tk.VaporMass,
      tk.VaporVolume * CP.PropsSI("D", "T", tk.Temperature, "Q", 1, N2O), rtol=1e-12)
check("init: volumes sum to total", tk.LiquidVolume + tk.VaporVolume, tk.Volume, rtol=1e-12)
check("init: masses sum to total", tk.LiquidMass + tk.VaporMass, tk.TotalMass, rtol=1e-12)
check("init: U = m_l*u_l + m_v*u_v", tk.TotalU,
      tk.LiquidMass * CP.PropsSI("U", "T", tk.Temperature, "Q", 0, N2O)
      + tk.VaporMass * CP.PropsSI("U", "T", tk.Temperature, "Q", 1, N2O), rtol=1e-12)

# Geometry constructor: cylinder volume
tcyl = Tank(liquid_mass=5.0, starting_temperature=290.0, tank_inner_diameter=0.15,
            tank_internal_length=1.0, units="SI")
check("cylinder volume = pi/4*d^2*L", tcyl.Volume, math.pi / 4 * 0.15**2 * 1.0, rtol=1e-12)

# Blowdown: conservation and monotonicity
m0, U0, V0 = tk.TotalMass, tk.TotalU, tk.Volume
draw, dt, n = 0.35, 0.05, 60          # kg/s
removed = 0.0
T_prev, P_prev = tk.Temperature, tk.Pressure
mono_T = mono_P = True
for i in range(n):
    step = draw * dt
    tk.simulate(step)
    removed += step
    mono_T &= tk.Temperature <= T_prev + 1e-9
    mono_P &= tk.Pressure <= P_prev + 1e-6
    T_prev, P_prev = tk.Temperature, tk.Pressure
    # volume closure every step
    if tk.TwoPhase:
        vsum = tk.LiquidVolume + tk.VaporVolume
        if not np.isclose(vsum, V0, rtol=1e-6):
            FAILS.append(f"volume closure step {i}")
            print(f"  [FAIL] volume closure at step {i}: {vsum} vs {V0}")

check("blowdown: mass conserved", tk.TotalMass, m0 - removed, rtol=1e-12)
check_true("blowdown: T monotonically falls", mono_T)
check_true("blowdown: P monotonically falls", mono_P)
check_true("blowdown: still on saturation dome", tk.TwoPhase)
check("blowdown: P == Psat(T)", tk.Pressure,
      CP.PropsSI("P", "T", tk.Temperature, "Q", 0, N2O), rtol=1e-9)
check("blowdown: volume closure", tk.LiquidVolume + tk.VaporVolume, V0, rtol=1e-6)
check("blowdown: masses still sum", tk.LiquidMass + tk.VaporMass, tk.TotalMass, rtol=1e-10)
check_true("blowdown: liquid depleting", tk.LiquidMass < m_want, f"({tk.LiquidMass:.3f} kg left)")
# Energy bookkeeping: U should equal m_l*u_l + m_v*u_v at the new T
props = FluidProperties.update(tk.Temperature)
check("blowdown: U consistent with phase split", tk.TotalU,
      tk.LiquidMass * props.energy_liquid + tk.VaporMass * props.energy_vapor, rtol=1e-6)

# Run to gas transition
steps = 0
while tk.TwoPhase and steps < 4000:
    tk.simulate(0.02); steps += 1
check_true("reaches vapor-only phase", not tk.TwoPhase, f"(after {steps} extra steps)")
check("gas phase: liquid mass zeroed", tk.LiquidMass, 0.0)
check("gas phase: vapor volume = tank volume", tk.VaporVolume, V0, rtol=1e-12)
check("gas phase: liquid volume zeroed", tk.LiquidVolume, 0.0)
check("gas phase: quality == 1", tk.Quality, 1.0)
# Continue in gas: pressure from real-gas EOS at bulk density
tk.simulate(0.02)
rho_bulk = tk.TotalMass / tk.Volume
check("gas phase: P = P(T, rho) real gas", tk.Pressure,
      CP.PropsSI("P", "T", tk.Temperature, "D", rho_bulk, N2O), rtol=1e-9)
check("gas phase: U = m*u(T,rho)", tk.TotalU,
      tk.TotalMass * CP.PropsSI("U", "T", tk.Temperature, "D", rho_bulk, N2O), rtol=1e-6)

# Guards
try:
    Tank(liquid_mass=1000.0, starting_temperature=290.0, internal_volume=0.01, units="SI")
    check_true("overfill rejected", False)
except ValueError:
    check_true("overfill rejected", True)
try:
    design_tank(10.0, 290.0, ullage_fraction=1.5, units="SI"); check_true("bad ullage rejected", False)
except ValueError:
    check_true("bad ullage rejected", True)


print(); print("=" * 70); print("FUEL GRAIN"); print("=" * 70)
fuel = Fuel("paraffin")
check("fuels.csv loads", fuel.rho_metric, 900.0)
check("paraffin a", fuel.a_metric, 0.000144)
check("paraffin n", fuel.n, 0.5)
check("paraffin G_ox max", fuel.g_ox_max, 350.0)

mdot_ox, mdot_f, t_burn = 1.8, 0.45, 8.0
OD = cv.in2m(6.0)

# Flux-limited sizing (length free)
g = design_fuel_grain(fuel, mdot_ox, mdot_f, 6.0, t_burn, units="Imperial")
check("sizing: sheds the design fuel flow", g.DesignMdotFuel, mdot_f, rtol=1e-9)
check("sizing: G_ox respects the fuel limit", g.DesignGOx <= 350.0, True)
check_true("sizing: G_ox below the limit", g.DesignGOx <= 350.0, f"({g.DesignGOx:.0f} <= 350)")
# Both requirements hold simultaneously: flow from the burn area, mass from the web
check("sizing: mdot_f = rho*A_burn*a*G^n", g.DesignMdotFuel,
      900.0 * g.BurnArea * 0.000144 * g.DesignGOx**0.5, rtol=1e-12)
fuel_mass_want = mdot_f * t_burn
check("flux sizing: grain holds the design fuel mass", g.FuelMass, fuel_mass_want, rtol=1e-12)
check("fuel mass = rho*pi*L*(ro^2-ri^2)", g.FuelMass,
      900.0 * math.pi * g.Length * ((OD/2)**2 - (g.PortDiameter/2)**2), rtol=1e-12)
check("burn area = pi*D*L", g.BurnArea, math.pi * g.PortDiameter * g.Length, rtol=1e-12)
check("port area = pi/4*D^2", g.PortArea, math.pi / 4 * g.PortDiameter**2, rtol=1e-12)
check("web = (OD-ID)/2", g.WebRemaining, (OD - g.PortDiameter) / 2, rtol=1e-12)

# Length-constrained sizing
gl = design_fuel_grain(fuel, mdot_ox, mdot_f, 6.0, t_burn, length=18.0, units="Imperial")
check("length sizing: length honored", gl.Length, cv.in2m(18.0), rtol=1e-12)
check("length sizing: holds design fuel mass", gl.FuelMass, fuel_mass_want, rtol=1e-10)

# Regression law and mass shed, recomputed by hand
g2 = design_fuel_grain(fuel, mdot_ox, mdot_f, 6.0, t_burn, units="Imperial")
D0, A0, B0 = g2.PortDiameter, g2.PortArea, g2.BurnArea
dt = 0.01
mdot_fuel_step = g2.simulate(mdot_ox, dt)
G_ref = mdot_ox / A0
rdot_ref = 0.000144 * G_ref**0.5
check("G_ox = mdot_ox/A_port", g2.GOx, G_ref, rtol=1e-12)
check("rdot = a*G^n", g2.RegressionRate, rdot_ref, rtol=1e-12)
check("mdot_fuel = rho*A_burn*rdot", mdot_fuel_step, 900.0 * B0 * rdot_ref, rtol=1e-12)
check("port grows by 2*rdot*dt", g2.PortDiameter, D0 + 2 * rdot_ref * dt, rtol=1e-12)
check("geometry refreshed after step", g2.BurnArea, math.pi * g2.PortDiameter * g2.Length, rtol=1e-12)

# Mass-balance check: integrated fuel flow must match the mass the grain actually lost
g3 = design_fuel_grain(fuel, mdot_ox, mdot_f, 6.0, t_burn, units="Imperial")
m_start = g3.FuelMass
dt = 0.001
integrated = 0.0
t = 0.0
while not g3.Burnout and t < 60.0:
    integrated += g3.simulate(mdot_ox, dt) * dt
    t += dt
m_lost = m_start - g3.FuelMass
check("integrated mdot_fuel == mass actually removed", integrated, m_lost, rtol=2e-3)
check_true("burnout reached", g3.Burnout, f"(t={t:.2f} s)")
check("burnout: port == OD", g3.PortDiameter, OD, rtol=1e-12)
check("burnout: web zero", g3.WebRemaining, 0.0, atol=1e-15)
check("burnout: fuel mass zero", g3.FuelMass, 0.0, atol=1e-12)
check("burnout: no more fuel shed", g3.simulate(mdot_ox, dt), 0.0)
check("burnout: port frozen", g3.PortDiameter, OD, rtol=1e-12)

# Design-point burn time sanity: paraffin at this flux should burn near the design time
check_true("burn time matches the design burn time", abs(t - t_burn)/t_burn < 0.02,
           f"(design {t_burn}s, actual {t:.2f}s)")

# For n = 0.5 the shed fuel flow is constant through the burn, so a grain that holds
# mdot_fuel*t_burn and sheds mdot_fuel must burn out at exactly t_burn.
gb = design_fuel_grain(fuel, mdot_ox, mdot_f, 6.0, t_burn, units="Imperial")
tb = 0.0
while not gb.Burnout and tb < 60.0:
    gb.simulate(mdot_ox, 0.001); tb += 0.001
check_true("flow+mass grain burns out at the design time", abs(tb - t_burn) / t_burn < 0.02,
           f"(design {t_burn}s, actual {tb:.2f}s)")

# Guards
try:
    FuelGrain(Fuel("al"), 0.02, 0.15, 0.5, units="SI"); check_true("fuel with no a/n rejected", False)
except ValueError:
    check_true("fuel with no a/n rejected", True)
try:
    FuelGrain(fuel, 0.20, 0.15, 0.5, units="SI"); check_true("port >= OD rejected", False)
except ValueError:
    check_true("port >= OD rejected", True)
try:
    design_fuel_grain(fuel, mdot_ox, mdot_f, 1.0, t_burn, length=1.0, units="Imperial")
    check_true("undersized grain rejected", False)
except ValueError:
    check_true("undersized grain rejected", True)


print(); print("=" * 70); print("CROSS-COMPONENT: coupled blowdown")
print("=" * 70)
tank = design_tank(30.0, 70.0, units="Imperial")
st0 = FluidProperties.update(tank.Temperature)
pc_design = cv.psi2pa(400.0)
inj = design_injector(1.8, upstream_state=st0, p_downstream=pc_design, model="NHNE")
grain = design_fuel_grain(fuel, 1.8, 0.45, 6.0, 8.0, units="Imperial")
noz = design_nozzle(1.8 + 0.45, pc_design, CEA(1.20, 1550.0, 4.0))

dt, t = 0.01, 0.0
pc = pc_design
hist = []
while t < 6.0 and tank.TwoPhase and not grain.Burnout:
    mdot_ox = inj.simulate(tank.Pressure, pc, tank.Properties)
    if mdot_ox <= 0:
        break
    mdot_f = grain.simulate(mdot_ox, dt)
    F = noz.simulate(pc, CEA(1.20, 1550.0, 4.0), 101325.0)
    tank.simulate(mdot_ox * dt)
    pc = (mdot_ox + mdot_f) * 1550.0 / noz.AThroat     # chamber pressure from choked throat
    hist.append((t, tank.Pressure, mdot_ox, mdot_f, pc, F))
    t += dt

check_true("coupled loop ran", len(hist) > 100, f"({len(hist)} steps, t={t:.2f} s)")
check_true("tank pressure fell", hist[-1][1] < hist[0][1],
           f"({cv.pa2psi(hist[0][1]):.0f} -> {cv.pa2psi(hist[-1][1]):.0f} psi)")
check_true("ox flow decayed with tank", hist[-1][2] < hist[0][2],
           f"({hist[0][2]:.3f} -> {hist[-1][2]:.3f} kg/s)")
check_true("thrust decayed", hist[-1][5] < hist[0][5],
           f"({cv.n2lbf(hist[0][5]):.0f} -> {cv.n2lbf(hist[-1][5]):.0f} lbf)")
of = hist[0][2] / hist[0][3]
check_true("initial O/F physical for paraffin/N2O", 2.0 < of < 12.0, f"(O/F={of:.2f})")
check_true("all mass flows positive", all(h[2] > 0 and h[3] > 0 for h in hist))
check_true("all pressures positive", all(h[1] > 0 and h[4] > 0 for h in hist))
check_true("injector dP stayed positive", all(h[1] > h[4] for h in hist))

print()
print("=" * 70)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
