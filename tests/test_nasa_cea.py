"""Accuracy verification for CHAOS rev2 / Physics/nasa_cea.py."""
import math
import numpy as np

import sys
import pathlib

# Make the CHAOS package importable no matter where this is run from
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from CHAOS.Physics.nasa_cea import NasaCEA, CEAResults, PA_PER_BAR, G0
from CHAOS.Physics.fuel_handling import Fuel
from CHAOS.Components.nozzle import Nozzle, design_nozzle, exit_mach
from CHAOS import conversions as cv

FAILS = []
def check(name, got, want, rtol=1e-9, atol=0.0):
    ok = np.isclose(got, want, rtol=rtol, atol=atol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r} want {want!r}")
    if not ok: FAILS.append(name)

def check_true(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond: FAILS.append(name)


print("=" * 72); print("CEA: N2O / HTPB"); print("=" * 72)
solver = NasaCEA("N2O", "htpb")
pc, of, eps = cv.psi2pa(500.0), 7.0, 4.0
r = solver.solve(pc, of, eps)
print(r)

# --- Internal consistency, recomputed from the returned numbers alone ---
check("R = Ru/MW", r.R, 8314.51 / (r.MW * 1000.0), rtol=1e-12)
check("Isp = Cf*cstar/g0", r.isp, r.cf * r.cstar / G0, rtol=1e-9)
check("expansion ratio honored", r.expansion_ratio, eps, rtol=1e-4)
check("p_chamber echoed exactly", r.p_chamber, pc, rtol=0.0)
check("OF echoed exactly", r.OF, of, rtol=0.0)

# --- Independent physics cross-checks ---
# Ideal c* from chamber properties: c* = sqrt(R*Tc/g) * ((g+1)/2)^((g+1)/(2(g-1)))
g, R, Tc = r.gamma, r.R, r.T_chamber
cstar_ideal = math.sqrt(R * Tc / g) * ((g + 1) / 2) ** ((g + 1) / (2 * (g - 1)))
print(f"\n  CEA c* = {r.cstar:.1f} m/s, ideal-gas c* from (g,R,Tc) = {cstar_ideal:.1f} m/s, "
      f"ratio = {r.cstar/cstar_ideal:.4f}")
check_true("CEA c* within 3% of ideal-gas c* from its own chamber state",
           abs(r.cstar / cstar_ideal - 1.0) < 0.03, f"(ratio {r.cstar/cstar_ideal:.4f})")

# Exit pressure from the isentropic relation at CEA's own exit Mach
pe_isen = pc * (1 + 0.5 * (g - 1) * r.mach_exit**2) ** (-g / (g - 1))
print(f"  CEA p_exit = {r.p_exit/1e3:.2f} kPa, isentropic (chamber gamma) = {pe_isen/1e3:.2f} kPa, "
      f"ratio = {r.p_exit/pe_isen:.4f}")
check_true("CEA exit pressure within 12% of frozen-gamma isentropic",
           abs(r.p_exit / pe_isen - 1.0) < 0.12, f"(ratio {r.p_exit/pe_isen:.4f})")

# --- Published-range sanity for N2O/HTPB ---
check_true("T_chamber in published range (3000-3600 K)", 3000 < r.T_chamber < 3600, f"({r.T_chamber:.0f} K)")
check_true("c* in published range (1400-1700 m/s)", 1400 < r.cstar < 1700, f"({r.cstar:.0f} m/s)")
check_true("gamma in range (1.10-1.30)", 1.10 < r.gamma < 1.30, f"({r.gamma:.4f})")
check_true("MW in range (20-32 g/mol)", 20 < r.MW*1000 < 32, f"({r.MW*1000:.2f} g/mol)")
check_true("Isp in range (200-280 s)", 200 < r.isp < 280, f"({r.isp:.1f} s)")
check_true("Cf in range (1.2-1.8)", 1.2 < r.cf < 1.8, f"({r.cf:.4f})")
check_true("exit Mach supersonic", r.mach_exit > 1.0, f"({r.mach_exit:.3f})")

# --- O/F sweep: c* must peak at an interior optimum ---
print("\n  O/F sweep (Pc = 500 psi, eps = 4):")
ofs = np.arange(2.0, 14.1, 1.0)
cstars, temps, isps = [], [], []
for o in ofs:
    ri = solver.solve(pc, float(o), eps)
    cstars.append(ri.cstar); temps.append(ri.T_chamber); isps.append(ri.isp)
    print(f"    O/F {o:5.1f}:  Tc {ri.T_chamber:7.1f} K   c* {ri.cstar:7.1f} m/s   Isp {ri.isp:6.1f} s")
i_best = int(np.argmax(cstars))
check_true("c* has an interior peak (not at a sweep end)", 0 < i_best < len(ofs) - 1,
           f"(peak at O/F={ofs[i_best]:.1f})")
check_true("peak c* O/F physical for N2O/HTPB (4-9)", 4.0 <= ofs[i_best] <= 9.0,
           f"(O/F={ofs[i_best]:.1f})")
check_true("Tc peaks near the c* peak", abs(int(np.argmax(temps)) - i_best) <= 2)
check_true("all sweep points converged", all(c > 0 for c in cstars))

# --- Pressure sweep: c* rises weakly with Pc, monotonic ---
print("\n  Pc sweep (O/F = 7, eps = 4):")
pcs = [cv.psi2pa(p) for p in (150, 250, 350, 450, 550, 650)]
cs = []
for p in pcs:
    rp = solver.solve(p, 7.0, eps)
    cs.append(rp.cstar)
    print(f"    Pc {cv.pa2psi(p):6.0f} psi:  c* {rp.cstar:7.1f} m/s   Tc {rp.T_chamber:7.1f} K")
check_true("c* increases monotonically with Pc", all(b > a for a, b in zip(cs, cs[1:])))
check_true("c* rise over 150->650 psi is weak (<6%)", (cs[-1]/cs[0] - 1) < 0.06,
           f"({100*(cs[-1]/cs[0]-1):.2f}%)")

# --- Optimum expansion: exit pressure must land on ambient ---
print()
r_opt = solver.solve(pc, 7.0)          # no expansion_ratio -> optimum
check("optimum expansion puts p_exit at p_ambient", r_opt.p_exit, 101325.0, rtol=2e-3)
check_true("optimum eps is physical", 2.0 < r_opt.expansion_ratio < 12.0,
           f"(eps={r_opt.expansion_ratio:.3f})")
# Re-solving at that eps must reproduce the same point
r_back = solver.solve(pc, 7.0, r_opt.expansion_ratio)
check("re-solving at the optimum eps reproduces p_exit", r_back.p_exit, r_opt.p_exit, rtol=2e-3)
check("re-solving at the optimum eps reproduces c*", r_back.cstar, r_opt.cstar, rtol=1e-9)

# --- Combustion efficiency scaling ---
eta = 0.92
solver_eta = NasaCEA("N2O", "htpb", combustion_efficiency=eta)
re = solver_eta.solve(pc, 7.0, eps)
check("eta scales c* exactly", re.cstar, r.cstar * eta, rtol=1e-12)
check("eta scales Isp exactly", re.isp, r.isp * eta, rtol=1e-9)
check("eta leaves Cf alone", re.cf, r.cf, rtol=1e-12)
check("eta leaves Tc alone", re.T_chamber, r.T_chamber, rtol=1e-12)
check("Isp = Cf*cstar/g0 still holds with eta", re.isp, re.cf * re.cstar / G0, rtol=1e-9)

# --- Cache is exact, not fuzzy ---
a = solver.solve(cv.psi2pa(500.0), 7.0, eps)
b = solver.solve(cv.psi2pa(500.0) + 500.0, 7.0, eps)   # 500 Pa apart
check_true("cache does not collapse nearby Pc", a.p_chamber != b.p_chamber and a.cstar != b.cstar,
           f"(c* {a.cstar:.4f} vs {b.cstar:.4f})")
c = solver.solve(cv.psi2pa(500.0), 7.0, eps)
check_true("exact repeat is a cache hit (same object)", c is a)

# --- Guards ---
for args, label in [((0.0, 7.0, 4.0), "Pc<=0 rejected"), ((pc, 0.0, 4.0), "OF<=0 rejected"),
                    ((pc, 7.0, 0.5), "eps<=1 rejected")]:
    try:
        solver.solve(*args); check_true(label, False, "(no error)")
    except ValueError:
        check_true(label, True)
try:
    NasaCEA("N2O", "htpb", combustion_efficiency=1.4); check_true("eta>1 rejected", False)
except ValueError:
    check_true("eta>1 rejected", True)
try:
    NasaCEA("N2O", 42); check_true("bad fuel type rejected", False)
except TypeError:
    check_true("bad fuel type rejected", True)


print(); print("=" * 72); print("CEA: composite fuel + other propellants"); print("=" * 72)
# Paraffin
par = NasaCEA("N2O", "paraffin").solve(pc, 7.0, eps)
print(f"  N2O/paraffin  O/F 7: Tc {par.T_chamber:.0f} K  c* {par.cstar:.0f} m/s  Isp {par.isp:.1f} s")
check_true("paraffin Tc in range", 3000 < par.T_chamber < 3700, f"({par.T_chamber:.0f} K)")
check_true("paraffin c* in range", 1450 < par.cstar < 1750, f"({par.cstar:.0f} m/s)")

# Composite: paraffin + aluminum. Al raises flame temperature and lowers c* (heavy condensed products)
mix = Fuel.mixture({"paraffin": 0.90, "al": 0.10})
alu = NasaCEA("N2O", mix).solve(pc, 7.0, eps)
print(f"  N2O/{mix.name}  O/F 7: Tc {alu.T_chamber:.0f} K  c* {alu.cstar:.0f} m/s  Isp {alu.isp:.1f} s")
check_true("composite fuel solves", alu.cstar > 0)
check_true("aluminized grain runs hotter", alu.T_chamber > par.T_chamber,
           f"({par.T_chamber:.0f} -> {alu.T_chamber:.0f} K)")
check_true("aluminized MW is heavier", alu.MW > par.MW,
           f"({par.MW*1000:.2f} -> {alu.MW*1000:.2f} g/mol)")

# CEA database species passthrough (not in fuels.csv)
ch4 = NasaCEA("O2", "CH4").solve(cv.psi2pa(500.0), 3.0, 4.0)
print(f"  O2/CH4  O/F 3: Tc {ch4.T_chamber:.0f} K  c* {ch4.cstar:.0f} m/s  Isp {ch4.isp:.1f} s")
check_true("CEA-database fuel passthrough works", ch4.cstar > 0)
check_true("O2/CH4 Tc in published range (3400-3800 K)", 3400 < ch4.T_chamber < 3800,
           f"({ch4.T_chamber:.0f} K)")
check_true("O2/CH4 c* in published range (1700-1900 m/s)", 1700 < ch4.cstar < 1900,
           f"({ch4.cstar:.0f} m/s)")


print(); print("=" * 72); print("INTEGRATION: CEA -> Nozzle"); print("=" * 72)
comb = solver.solve(pc, 7.0, eps)
mdot = 2.5
noz = design_nozzle(mdot, pc, comb)
check("design_nozzle: At = mdot*cstar/Pc", noz.AThroat, mdot * comb.cstar / pc, rtol=1e-12)
check("design_nozzle takes eps from CEA", noz.ExpansionRatio, comb.expansion_ratio, rtol=1e-12)
F = noz.simulate(pc, comb, 101325.0)
check("nozzle mdot round-trips", noz.Mdot, mdot, rtol=1e-10)
print(f"\n  Nozzle (perfect-gas, chamber gamma): Me {noz.MachExit:.3f}  pe {noz.PExit/1e3:.2f} kPa  "
      f"Cf {noz.Cf:.4f}  Isp {noz.Isp:.1f} s")
print(f"  CEA    (equilibrium expansion):      Me {comb.mach_exit:.3f}  pe {comb.p_exit/1e3:.2f} kPa  "
      f"Cf {comb.cf:.4f}  Isp {comb.isp:.1f} s")
print(f"  -> Isp gap: {100*(noz.Isp/comb.isp - 1):+.2f} %   Cf gap: {100*(noz.Cf/comb.cf - 1):+.2f} %")
check_true("nozzle and CEA agree on exit Mach within 10%",
           abs(noz.MachExit/comb.mach_exit - 1) < 0.10, f"({noz.MachExit:.3f} vs {comb.mach_exit:.3f})")
check_true("nozzle and CEA agree on Isp within 8%",
           abs(noz.Isp/comb.isp - 1) < 0.08, f"({noz.Isp:.1f} vs {comb.isp:.1f} s)")
check_true("nozzle thrust positive and physical", 1000 < F < 20000, f"({cv.n2lbf(F):.0f} lbf)")

print()
print("=" * 72)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S):")
    for f in FAILS: print("   -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
