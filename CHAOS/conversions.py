"""
    Custom conversions file
"""
# Dependency Imports
import numpy as np

# Package Imports

# Exact definitions
LBF = 4.4482216152605   # N per lbf
LBM = 0.45359237        # kg per lbm
INCH = 0.0254           # m per in
FOOT = 0.3048           # m per ft
PSI = 6894.757293168    # Pa per psi
G0 = 9.80665            # m/s^2 (standard gravity, for Isp in seconds)

# Forces
def lbf2n(lbf):
    return lbf * LBF
def n2lbf(n):
    return n / LBF

# Masses
def kg2lbm(kg):
    return kg / LBM
def lbm2kg(lbm):
    return lbm * LBM

# Pressures
def psi2pa(psi):
    return psi * PSI
def psi2bar(psi):
    return psi * (PSI / 100000)
def bar2psi(bar):
    return bar / (PSI / 100000)
def bar2pa(bar):
    return bar * 100000
def pa2bar(pa):
    return pa / 100000
def pa2psi(pa):
    return pa / PSI

# Diameter to Area
def d2a(d):
    return np.pi * (d ** 2) / 4
def a2d(a):
    return np.sqrt((4 * a) / np.pi)

# Lengths
def in2m(inches):
    return inches * INCH
def m2in(m):
    return m / INCH

# Areas
def m2_2_in2(m2):
    return m2 / (INCH ** 2)
def in2_2_m2(in2):
    return in2 * (INCH ** 2)
def m2_2_ft2(m2):
    return m2 / (FOOT ** 2)
def ft2_2_m2(ft2):
    return ft2 * (FOOT ** 2)

# Volumes
def m3_2_in3(m3):
    return m3 / (INCH ** 3)
def in3_2_m3(in3):
    return in3 * (INCH ** 3)
def m3_2_ft3(m3):
    return m3 / (FOOT ** 3)
def ft3_2_m3(ft3):
    return ft3 * (FOOT ** 3)

# Temperatures
def k2r(k):
    return k * (9 / 5)
def r2k(r):
    return r / (9 / 5)
def k2c(k):
    return k - 273.15
def c2k(c):
    return c + 273.15
def r2f(r):
    return r - 459.67
def f2r(f):
    return f + 459.67
def c2f(c):
    return r2f(k2r(c2k(c)))
def f2c(f):
    return k2c(r2k(f2r(f)))
def f2k(f):
    return r2k(f2r(f))
def k2f(k):
    return r2f(k2r(k))
