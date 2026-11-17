"""
Propellant registry.

Loads fuel/ingredient definitions from ``fuels.csv`` (next to this module) exactly once and caches them.
Each row is an *ingredient*: a chemical species usable on its own (a fuel, with regression
coefficients a, n) or as a component of a composite grain (e.g. an aluminum additive, which
has no standalone a/n). Data is CEA-native: chemical formula (with CEA element symbols, which
are UPPERCASE — e.g. ``AL`` for aluminum), heat of formation, and reference temperature, plus
grain density.

This module has no combustion/CEA dependency — turning a Fuel into solver reactants is the
combustion backend's job (see ``CHAOS.Physics.Combustion``).
"""

# Dependency imports
from pathlib import Path
from dataclasses import dataclass
import csv

# Package imports
# (none)


def _parse_formula(formula: str) -> dict[str, float]:
    """Parse a CEA-style element/count string, e.g. "C 75 H 128" -> {"C":75.0,"H":128.0}.

    Element symbols must match CEA (uppercase, e.g. "AL" for aluminum)."""
    tokens = formula.split()
    if len(tokens) % 2 != 0:
        raise ValueError(f"Malformed formula string: {formula!r}")
    return {tokens[i]: float(tokens[i + 1]) for i in range(0, len(tokens), 2)}


def _opt_float(value) -> float | None:
    """Parse a possibly-empty CSV cell into a float or None (for optional a/n on additives)."""
    value = (value or "").strip()
    return float(value) if value else None


class FuelRegistry:
    """
    Handles ingredient data. Acts as a one-time import from fuels.csv and caches the parsed
    rows so repeated engine designs / simulations do not reload.
    """
    def __init__(self):
        self._fuels = {}
        self._loaded = False

    def load(self):
        """Parse fuels.csv into memory"""
        if self._loaded:
            return

        data_path = Path(__file__).with_name("fuels.csv")

        with data_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._fuels[row["fuel_name"]] = row

        self._loaded = True

    def get(self, name: str):
        if not self._loaded:
            self.load()

        key = name.lower()
        for fuel_name in self._fuels:
            if fuel_name.lower() == key:
                return self._fuels[fuel_name]

        raise ValueError(f'Error: Fuel/ingredient name "{name}" not found')


@dataclass
class FuelComponent:
    """One ingredient of a (possibly composite) fuel, with its mass fraction in the fuel."""
    name: str
    formula: dict            # CEA element symbols (uppercase) -> atom count
    hf: float                # heat of formation in `hf_units`
    hf_units: str            # CEA molar enthalpy unit string (e.g. "cal/mol", "J/mol")
    t_ref: float             # reference temperature [K]
    density: float           # kg/m^3
    mass_fraction: float     # fraction of the fuel's mass (components sum to 1)


def _component_from_row(row: dict, mass_fraction: float) -> FuelComponent:
    return FuelComponent(
        name=row["fuel_name"],
        formula=_parse_formula(row["formula"]),
        hf=float(row["hf_value"]),
        hf_units=row["hf_units"],
        t_ref=float(row["T_ref_K"]),
        density=float(row["density_kg_m3"]),
        mass_fraction=mass_fraction,
    )


class Fuel:
    """
    A grain fuel, either a single ingredient or a composite (mixture) of several.

    Attributes:
        name (str): fuel name.
        components (list[FuelComponent]): ingredient(s) and their mass fractions.
        a_metric (float | None): regression law coefficient a in r = a * G^n (SI).
        n (float | None): regression law exponent n.
        rho_metric (float): grain density [kg/m^3].
        g_ox_max (float | None): recommended upper limit on initial oxidizer mass flux
            G_ox [kg/(m^2 s)]. Above this, entrainment/flooding outruns combustion and
            c* efficiency collapses; the fuel-grain design opens the port to respect it.

    Construct a single-ingredient fuel with ``Fuel("paraffin")`` or a composite with
    ``Fuel.mixture({"paraffin": 0.95, "Al": 0.05})``.
    """

    def __init__(self, name: str, *, components=None, a_metric=None, n=None, rho_metric=None,
                 g_ox_max=None):
        if components is not None:
            # Built via Fuel.mixture(); fields already resolved.
            self.name = name
            self.components = components
            self.a_metric = a_metric
            self.n = n
            self.rho_metric = rho_metric
            self.g_ox_max = g_ox_max
            return

        # Single ingredient looked up from the registry.
        row = fuel_registry.get(name)
        self.name = row["fuel_name"]
        self.components = [_component_from_row(row, 1.0)]
        self.a_metric = _opt_float(row["a_metric"])
        self.n = _opt_float(row["n"])
        self.rho_metric = float(row["density_kg_m3"])
        self.g_ox_max = _opt_float(row.get("g_ox_max_kg_m2s"))

    @property
    def mass_fractions(self) -> list:
        return [c.mass_fraction for c in self.components]

    @classmethod
    def mixture(cls, composition: dict, *, a=None, n=None, density=None, name=None,
                g_ox_max=None) -> "Fuel":
        """Build a composite fuel from registry ingredients.

        Args:
            composition: {ingredient_name: mass_fraction}. Fractions need not sum to 1; they
                are normalized.
            a, n: regression coefficients for the composite grain. Default to the dominant
                (largest mass fraction) ingredient's values.
            density: grain density [kg/m^3]. Defaults to the volume-additive mixture density
                computed from the component densities (1/rho = sum w_i/rho_i).
            name: display name; defaults to a "comp1XX/comp2YY" summary.
            g_ox_max: recommended max initial G_ox [kg/(m^2 s)]. Defaults to the dominant
                ingredient's value.
        """
        fuel_registry.load()
        items = list(composition.items())
        if not items:
            raise ValueError("composition must contain at least one ingredient")

        total = float(sum(frac for _, frac in items))
        if total <= 0:
            raise ValueError("composition mass fractions must sum to a positive value")

        components = [_component_from_row(fuel_registry.get(nm), frac / total) for nm, frac in items]

        # Regression / flux-limit defaults from the dominant ingredient.
        dominant_row = fuel_registry.get(max(components, key=lambda c: c.mass_fraction).name)
        a_val = a if a is not None else _opt_float(dominant_row["a_metric"])
        n_val = n if n is not None else _opt_float(dominant_row["n"])
        g_val = g_ox_max if g_ox_max is not None else _opt_float(dominant_row.get("g_ox_max_kg_m2s"))

        # Composite density: volumes are additive (1/rho = sum of mass-fraction/density).
        rho = density if density is not None else 1.0 / sum(c.mass_fraction / c.density for c in components)

        display = name or "/".join(f"{c.name}{c.mass_fraction * 100:.0f}" for c in components)
        return cls(display, components=components, a_metric=a_val, n=n_val, rho_metric=rho,
                   g_ox_max=g_val)

    def __str__(self):
        comp_lines = "\n".join(
            f"          {c.name}: {c.mass_fraction * 100:.1f}%  (rho={c.density:.0f} kg/m^3)"
            for c in self.components
        )
        return f"""
        Fuel: {self.name}
        Components:
{comp_lines}
        Grain Density: {self.rho_metric:.0f} kg/m^3

        Regression Characteristics
        --------------------------
        a: {self.a_metric} (metric)
        n: {self.n}
        Max initial G_ox: {self.g_ox_max} kg/(m^2 s)
        """


# Module-level singleton shared by every Fuel instance.
fuel_registry = FuelRegistry()
