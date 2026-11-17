import csv
import sys
import pathlib
import numpy as np

# Package Imports
# Make the CHAOS package importable no matter where this is run from
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import CHAOS
from CHAOS.Components import Nozzle

n2 = Nozzle(throat_diameter=1.5, expansion_ratio=4.0, units="SI")

print(n2)