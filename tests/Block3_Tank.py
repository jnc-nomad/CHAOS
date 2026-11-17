# Dependency Imports
import sys
import pathlib

# Package Imports
# Make the CHAOS package importable no matter where this is run from
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import CHAOS
from CHAOS.Components.tank import Tank
from CHAOS.Physics.fluid_handling import FluidProperties
from CHAOS import conversions as cv

# -- tank geometry + fill condition ----------------------------------------------------------
inner_diameter = 5.625      # in
internal_length = 19.0      # in
fill_temperature = 85.0     # F

# Internal volume from the geometry
volume = cv.d2a(cv.in2m(inner_diameter)) * cv.in2m(internal_length)      # m^3

# N2O saturated liquid density at the fill temperature sets how much liquid fits
rho_liquid = FluidProperties.update(cv.f2k(fill_temperature)).density_liquid   # kg/m^3

# "Mass that fits" = a brim-full liquid volume; real fills leave vapor ullage on top
mass_full = rho_liquid * volume                     # kg, 0% ullage (theoretical max)
mass_15 = rho_liquid * volume * (1.0 - 0.15)        # kg, 15% vapor ullage (typical)

print(f"Tank: {inner_diameter} in ID x {internal_length} in  ->  "
      f"{volume*1e6:.1f} cm^3 | {cv.m3_2_ft3(volume):.4f} ft^3")
print(f"N2O saturated liquid density at {fill_temperature} F: {rho_liquid:.1f} kg/m^3")
print(f"Liquid N2O that fits (0% ullage):  {mass_full:.3f} kg | {cv.kg2lbm(mass_full):.3f} lbm")
print(f"Liquid N2O that fits (15% ullage): {mass_15:.3f} kg | {cv.kg2lbm(mass_15):.3f} lbm")
