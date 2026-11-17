"""Verification for CHAOS rev2 / Hybrid/simulate.py."""
import math, os
import numpy as np

import sys
import pathlib

# Make the CHAOS package importable no matter where this is run from
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from CHAOS.Hybrid.simulate import HybridSimulation, SimulationResults, solve_chamber
from CHAOS.Hybrid.design import EngineDesign
from CHAOS.Components.fuel_grain import fuel_flow
from CHAOS.Components.nozzle import G0
from CHAOS import conversions as cv

FAILS = []
def check(name, got, want, rtol=1e-9, atol=0.0):
    ok = np.isclose(got, want, rtol=rtol, atol=atol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r} want {want!r}")
    if not ok: FAILS.append(name)

def check_true(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond: FAILS.append(name)

HW = dict(oxidizer="N2O", fuel="htpb",
          oxidizer_mass=28.0, oxidizer_temperature=68.0, tank_ullage=0.15,
          injector_orifice_diameter=0.060, injector_orifice_count=36, injector_cd=0.61,
          fuel_port_diameter=2.507, fuel_outer_diameter=6.0, fuel_length=21.5,
          throat_diameter=1.292, expansion_ratio=5.50,
          p_ambient=14.7, time_step=0.02, max_time=60.0, units="Imperial")

print("=" * 72); print("SIMULATION: as-built hardware in, performance out"); print("=" * 72)
sim = HybridSimulation(**HW)
print(sim)
res = sim.run()
print(res)

# --- Hardware went in unchanged (no design estimate anywhere) ---
check("throat from hardware", sim.nozzle.DThroat, cv.in2m(1.292), rtol=1e-12)
check("port from hardware", sim.grain.InitialPortDiameter, cv.in2m(2.507), rtol=1e-12)
check("grain length from hardware", sim.grain.Length, cv.in2m(21.5), rtol=1e-12)
check("injector area from orifices", sim.injector.Area,
      36 * math.pi/4 * cv.in2m(0.060)**2, rtol=1e-12)
check_true("simulate imports nothing from design", "EngineDesign" not in
           (pathlib.Path(__file__).resolve().parent.parent / "CHAOS" / "Hybrid" / "simulate.py").read_text())

# --- The chamber solve actually balances ---
print()
sim2 = HybridSimulation(**HW)
(pc, mdot_ox, mdot_f, of, rdot, g_ox, comb) = solve_chamber(
    sim2.injector, sim2.grain, sim2.nozzle, sim2.CEA,
    sim2.tank.Pressure, sim2.tank.Properties, sim2.PAmbient)
mdot_throat = pc * sim2.nozzle.AThroat / comb.cstar
check("CHAMBER BALANCES: inflow == choked throat flow", mdot_ox + mdot_f, mdot_throat, rtol=1e-6)
check("solve returns consistent O/F", of, mdot_ox / mdot_f, rtol=1e-12)
check("injector flow matches the solved Pc", mdot_ox,
      sim2.injector.simulate(sim2.tank.Pressure, pc, sim2.tank.Properties), rtol=1e-12)
check("fuel flow matches the solved state", mdot_f, fuel_flow(sim2.grain, mdot_ox)[0], rtol=1e-12)
check_true("chamber sits between ambient and tank", sim2.PAmbient < pc < sim2.tank.Pressure,
           f"({cv.pa2psi(pc):.1f} psi vs tank {cv.pa2psi(sim2.tank.Pressure):.1f} psi)")
# The root find must not have regressed the grain
check("root find did NOT regress the grain", sim2.grain.PortDiameter,
      sim2.grain.InitialPortDiameter, rtol=0.0)

# --- Metric accuracy: impulse, propellant, Isp, O/F --------------------------------------
print()
# Impulse must equal sum(F*dt) over the logged steps, exactly
check("TotalImpulse == sum(F*dt)", res.TotalImpulse, float(res.thrust.sum() * sim.dt), rtol=1e-12)
check("BurnTime == steps * dt", res.BurnTime, res.Steps * sim.dt, rtol=1e-9)
check("AverageThrust == impulse/burn_time", res.AverageThrust, res.TotalImpulse/res.BurnTime, rtol=1e-12)
check("PeakThrust == max of trace", res.PeakThrust, float(res.thrust.max()), rtol=1e-12)

# Burn time must cover the LAST step, not stop at the last sample's start time
check("BurnTime covers the final step", res.BurnTime, float(res.time[-1]) + sim.dt, rtol=1e-9)
check_true("BurnTime > last logged sample time", res.BurnTime > res.time[-1])

# Propellant from component mass change must match the integrated flow to O(dt)
ox_integrated = float(res.mdot_ox.sum() * sim.dt)
fuel_integrated = float(res.mdot_fuel.sum() * sim.dt)
print(f"  oxidizer: {res.OxidizerUsed:.4f} kg from tank mass change vs "
      f"{ox_integrated:.4f} kg integrated ({100*(ox_integrated/res.OxidizerUsed-1):+.3f}%)")
print(f"  fuel:     {res.FuelUsed:.4f} kg from grain mass change vs "
      f"{fuel_integrated:.4f} kg integrated ({100*(fuel_integrated/res.FuelUsed-1):+.3f}%)")
check("oxidizer used == tank mass change", res.OxidizerUsed, ox_integrated, rtol=2e-3)
check("fuel used == grain mass change", res.FuelUsed, fuel_integrated, rtol=5e-3)
check("propellant = ox + fuel", res.PropellantUsed, res.OxidizerUsed + res.FuelUsed, rtol=1e-12)
check("OverallIsp = impulse/(m_prop*g0)", res.OverallIsp,
      res.TotalImpulse/(res.PropellantUsed*G0), rtol=1e-12)

# Overall O/F must be the mass ratio, NOT the mean of the instantaneous trace
of_mass = res.OxidizerUsed / res.FuelUsed
of_naive = float(res.OF.mean())
print(f"  overall O/F {res.AverageOF:.4f} (mass ratio) vs {of_naive:.4f} (mean of trace) "
      f"-- {100*(of_naive/of_mass-1):+.2f}% apart")
check("AverageOF is the consumed-mass ratio", res.AverageOF, of_mass, rtol=1e-12)
check_true("mass-ratio O/F differs from the naive mean", abs(of_naive/of_mass - 1) > 1e-4)

# --- Conservation across the whole run ---
check("tank lost exactly what the injector passed", res.OxidizerUsed,
      float(res.tank_total_mass[0] - sim.tank.TotalMass), rtol=1e-12)
check_true("grain fuel mass decreases monotonically",
           bool(np.all(np.diff(res.grain_fuel_mass) <= 1e-12)))
check_true("port diameter increases monotonically",
           bool(np.all(np.diff(res.port_diameter) >= -1e-15)))
check_true("tank pressure falls monotonically",
           bool(np.all(np.diff(res.p_tank) <= 1e-6)))
check_true("thrust decays over the burn", res.FinalThrust < res.InitialThrust,
           f"({cv.n2lbf(res.InitialThrust):.0f} -> {cv.n2lbf(res.FinalThrust):.0f} lbf)")
check_true("all chamber pressures above ambient", bool(np.all(res.p_chamber > sim.PAmbient)))
check_true("chamber always below tank", bool(np.all(res.p_chamber < res.p_tank)))
check_true("all mass flows positive", bool(np.all(res.mdot_ox > 0) and np.all(res.mdot_fuel > 0)))
check_true("O/F drifts as the port opens", abs(res.FinalOF/res.InitialOF - 1) > 0.01,
           f"({res.InitialOF:.2f} -> {res.FinalOF:.2f})")
check_true("termination reason is meaningful",
           res.TerminationReason in ("burnout", "liquid_exhausted", "tank_empty",
                                     "pressure_collapse", "max_time"),
           f"('{res.TerminationReason}')")

# --- Per-step Isp consistency ---
isp_from_trace = res.thrust / (res.mdot_total * G0)
check_true("logged Isp == F/(mdot*g0) every step",
           bool(np.allclose(res.isp, isp_from_trace, rtol=1e-6)),
           f"(max dev {np.abs(res.isp/isp_from_trace - 1).max():.2e})")
check_true("logged Cf == F/(Pc*At) every step",
           bool(np.allclose(res.cf, res.thrust/(res.p_chamber*sim.nozzle.AThroat), rtol=1e-9)))

# --- Timestep convergence: metrics must converge as dt shrinks --------------------------
print("\n  Timestep convergence:")
prev = None
rows = []
for dt in (0.08, 0.04, 0.02, 0.01):
    r = HybridSimulation(**{**HW, "time_step": dt}).run()
    rows.append((dt, r.TotalImpulse, r.BurnTime, r.OverallIsp, r.AverageOF))
    print(f"    dt={dt:5.3f}: impulse {r.TotalImpulse:9.1f} N-s  burn {r.BurnTime:6.3f} s  "
          f"Isp {r.OverallIsp:6.2f} s  O/F {r.AverageOF:.4f}  ({r.TerminationReason})")
imp = [r[1] for r in rows]
rel = [abs(imp[i+1]/imp[i] - 1) for i in range(len(imp)-1)]
check_true("impulse converges as dt shrinks", abs(imp[-1]/imp[0] - 1) < 0.01,
           f"(coarse->fine drift {100*abs(imp[-1]/imp[0]-1):.3f}%, "
           f"successive {[f'{x:.2e}' for x in rel]})")
check_true("finest-step impulse change under 1%", rel[-1] < 0.01, f"({100*rel[-1]:.3f}%)")
check_true("all timesteps terminate the same way",
           len({HybridSimulation(**{**HW, "time_step": dt}).run().TerminationReason
                for dt in (0.04, 0.02)}) == 1)

# --- eta_c* propagates correctly through the coupled solve ---
print()
r_full = HybridSimulation(**HW).run()
r_eta = HybridSimulation(**{**HW, "combustion_efficiency": 0.90}).run()
print(f"  eta=1.00: Pc {cv.pa2psi(r_full.AverageChamberPressure):.1f} psi, "
      f"impulse {r_full.TotalImpulse:.0f} N-s, Isp {r_full.OverallIsp:.1f} s")
print(f"  eta=0.90: Pc {cv.pa2psi(r_eta.AverageChamberPressure):.1f} psi, "
      f"impulse {r_eta.TotalImpulse:.0f} N-s, Isp {r_eta.OverallIsp:.1f} s")
check_true("lower eta lowers chamber pressure",
           r_eta.AverageChamberPressure < r_full.AverageChamberPressure)
check_true("lower eta lowers total impulse", r_eta.TotalImpulse < r_full.TotalImpulse)
check_true("lower eta lowers Isp", r_eta.OverallIsp < r_full.OverallIsp)
check_true("eta effect is not absurd", 0.80 < r_eta.OverallIsp/r_full.OverallIsp < 1.0,
           f"(Isp ratio {r_eta.OverallIsp/r_full.OverallIsp:.4f})")

# --- Termination paths ---
print()
r_short = HybridSimulation(**{**HW, "oxidizer_mass": 4.0}).run()
check_true("small oxidizer load ends on liquid exhaustion",
           r_short.TerminationReason in ("liquid_exhausted", "tank_empty"),
           f"('{r_short.TerminationReason}', {r_short.BurnTime:.2f} s)")
check_true("short run still reports sane metrics",
           r_short.TotalImpulse > 0 and r_short.OverallIsp > 100 and r_short.BurnTime > 0)
r_cap = HybridSimulation(**{**HW, "max_time": 1.0}).run()
check("max_time cap respected", r_cap.BurnTime, 1.0, rtol=1e-9)
check_true("max_time run reports that reason", r_cap.TerminationReason == "max_time")
r_thin = HybridSimulation(**{**HW, "fuel_outer_diameter": 3.2,
                             "fuel_port_diameter": 2.507, "fuel_length": 21.5}).run()
check_true("thin web burns out", r_thin.TerminationReason == "burnout",
           f"('{r_thin.TerminationReason}', web left {cv.m2in(r_thin.WebRemaining):.4f} in)")
check("burnout leaves zero web", r_thin.WebRemaining, 0.0, atol=1e-12)
check_true("burnout end-state is the grain's, not the last logged row",
           r_thin.WebRemaining < float(r_thin.web_remaining[-1]),
           f"(reported {r_thin.WebRemaining:.2e} m vs last logged {r_thin.web_remaining[-1]:.2e} m)")

# --- CSV export round trip ---
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_out.csv")
res.to_csv(path)
with open(path) as f:
    lines = f.read().strip().split("\n")
check("CSV has one row per step plus a header", len(lines), res.Steps + 1)
check_true("CSV header matches the logged channels", lines[0].split(",") == res.channels)
first = [float(x) for x in lines[1].split(",")]
check("CSV first row matches the first logged thrust",
      first[res.channels.index("thrust")], float(res.thrust[0]), rtol=1e-12)
os.remove(path)

# --- Design -> hardware -> simulate: the two halves must agree at t=0 -------------------
print()
print("=" * 72); print("DESIGN HANDOFF: t=0 of the sim vs the design point"); print("=" * 72)
d = EngineDesign(oxidizer="N2O", fuel="htpb", p_combustion=500.0, thrust=1000.0,
                 burn_time=8.0, OF=7.0, fuel_od=6.0, units="Imperial", print_output=False)
sim_d = HybridSimulation(
    oxidizer="N2O", fuel="htpb",
    oxidizer_mass=cv.kg2lbm(d.OxidizerLoad), oxidizer_temperature=68.0, tank_ullage=0.15,
    injector_area=cv.m2_2_in2(d.injector.Area), injector_cd=0.61,
    fuel_port_diameter=cv.m2in(d.FuelID), fuel_outer_diameter=6.0,
    fuel_length=cv.m2in(d.FuelLength),
    throat_diameter=cv.m2in(d.nozzle.DThroat), expansion_ratio=d.ExpansionRatio,
    p_ambient=14.7, time_step=0.01, units="Imperial")
rd = sim_d.run()
print(f"  design point : Pc {cv.pa2psi(d.PCombustion):6.1f} psi  F {cv.n2lbf(d.Thrust):6.1f} lbf  "
      f"O/F {d.OF:.3f}  mdot_ox {d.OxidMassFlow:.4f}  burn {d.BurnTime:.2f} s")
print(f"  sim at t=0   : Pc {cv.pa2psi(rd.p_chamber[0]):6.1f} psi  F {cv.n2lbf(rd.thrust[0]):6.1f} lbf  "
      f"O/F {rd.OF[0]:.3f}  mdot_ox {rd.mdot_ox[0]:.4f}  burn {rd.BurnTime:.2f} s ({rd.TerminationReason})")
for label, got, want in (("chamber pressure", rd.p_chamber[0], d.PCombustion),
                         ("thrust", rd.thrust[0], d.Thrust),
                         ("O/F", rd.OF[0], d.OF),
                         ("oxidizer flow", rd.mdot_ox[0], d.OxidMassFlow),
                         ("fuel flow", rd.mdot_fuel[0], d.FuelMassFlow)):
    err = 100*(got/want - 1)
    print(f"    {label:18} {err:+7.3f} %")
    check_true(f"sim t=0 {label} within 2% of design", abs(err) < 2.0, f"({err:+.3f}%)")
check_true("burn lasts near the design burn time", abs(rd.BurnTime/d.BurnTime - 1) < 0.35,
           f"(design {d.BurnTime:.2f} s, actual {rd.BurnTime:.2f} s)")

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS: print("   -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
