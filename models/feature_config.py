"""
models/feature_config.py
========================

Single source of truth for the ML feature contract.

CRITICAL: lap_time_model.py, train_lap_time_model.py, and simulation_engine.py
ALL import from this file. Never define feature lists locally in those files.
"""

# ---------------------------------------------------------------------------
# Feature columns — order matters for consistency
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "driver_id",        # str:   driver code (VER, HAM, NOR, ...)
    "team_id",          # str:   constructor name (Red Bull Racing, Mercedes, ...)
    "circuit_key",      # str:   stable track identifier (silverstone, monza, ...)
    "lap_number",       # int:   lap number in race (1 to total_laps)
    "race_progress",    # float: lap_number / total_laps (0.0 to 1.0)
    "tire_compound",    # str:   SOFT / MEDIUM / HARD / INTERMEDIATE / WET
    "tire_age",         # int:   laps on current set of tires (0 = fresh)
    "fuel_load",        # float: laps remaining (proxy for fuel weight)
    "track_temp",       # float: track temperature (Celsius)
    "air_temp",         # float: air temperature (Celsius)
    "regulation_era",   # str:   ground_effect / v6_hybrid / 2026_pu
    "stint_number",     # int:   which stint (1, 2, 3, ...)
]

CATEGORICAL_FEATURES = [
    "driver_id",
    "team_id",
    "circuit_key",
    "tire_compound",
    "regulation_era",
]

NUMERIC_FEATURES = [
    "lap_number",
    "race_progress",
    "tire_age",
    "fuel_load",
    "track_temp",
    "air_temp",
    "stint_number",
]

# ---------------------------------------------------------------------------
# Monotone constraints — physics-backed inductive bias for LightGBM
# ---------------------------------------------------------------------------
# Without these, the model can learn a NET-NEGATIVE slope on tire_age within
# a stint because fuel_load and lap_number carry a strong "later in race =
# faster" signal that masks tire wear. We force the partial effect of
# tire_age and fuel_load to monotonically slow the lap.
#
# Values: -1 (decreasing), 0 (unconstrained), +1 (increasing).
# Order MUST match FEATURE_COLUMNS exactly.

MONOTONE_CONSTRAINTS = {
    "driver_id":      0,
    "team_id":        0,
    "circuit_key":    0,
    "lap_number":     0,   # track evolution can rubber-in or get hotter — keep free
    "race_progress":  0,
    "tire_compound":  0,
    "tire_age":      +1,   # older tires = slower (degradation)
    "fuel_load":     +1,   # heavier car = slower (laps_remaining proxy)
    "track_temp":     0,   # compound-dependent, complex
    "air_temp":       0,
    "regulation_era": 0,
    "stint_number":   0,
}

# ---------------------------------------------------------------------------
# Defaults — used by _coerce_features() when a value is not provided
# ---------------------------------------------------------------------------

FEATURE_DEFAULTS = {
    "driver_id": "unknown",
    "team_id": "unknown",
    "circuit_key": "unknown",
    "lap_number": 1,
    "race_progress": 0.5,
    "tire_compound": "MEDIUM",
    "tire_age": 0,
    "fuel_load": 0.0,
    "track_temp": 25.0,
    "air_temp": 20.0,
    "regulation_era": "ground_effect",
    "stint_number": 1,
}


# ---------------------------------------------------------------------------
# Regulation era mapping
# ---------------------------------------------------------------------------

def get_regulation_era(year: int) -> str:
    """
    Map a season year to its F1 regulation era.
    Used as a categorical feature so the model learns era-specific patterns.
    """
    if year <= 2013:
        return "v8_era"
    if year <= 2021:
        return "v6_hybrid"
    if year <= 2025:
        return "ground_effect"
    return "2026_pu"
