# CHAOS

Hybrid rocket engine design and simulation.

- **`EngineDesign`**: requirements in, nominal geometry out. A steady-state sizing of the nozzle, injector, fuel grain and tank.
- **`HybridSimulation`**: as-built hardware in, transient performance out. Marches the coupled tank / injector / chamber / grain / nozzle blowdown forward in time.

## Install

```bash
pip install chaos-hybrid            # core
pip install "chaos-hybrid[plot]"    # with matplotlib for plotting
```

For development, from a clone:

```bash
pip install -e ".[dev]"
```

## Usage

```python
import CHAOS

# Size a new engine from requirements
engine = CHAOS.EngineDesign(oxidizer="N2O", fuel="htpb", p_combustion=500.0,
                            thrust=1000.0, burn_time=8.0, OF=7.0, fuel_od=6.0)

# Simulate hardware you can measure
sim = CHAOS.HybridSimulation(oxidizer="N2O", fuel="htpb",
                             oxidizer_mass=28.0, oxidizer_temperature=68.0,
                             tank_ullage=0.15,
                             injector_orifice_diameter=0.060, injector_orifice_count=36,
                             fuel_port_diameter=2.507, fuel_outer_diameter=6.0,
                             fuel_length=21.5,
                             throat_diameter=1.292, expansion_ratio=5.50)
results = sim.run()
results.to_csv("burn.csv")
```

Inputs are Imperial by default (in, lbm, lbf, psi, °F); pass `units="SI"` for m, kg, N, bar and K.

## Tests

The scripts in `tests/` are standalone verification scripts:

```bash
python tests/test_design.py
```
