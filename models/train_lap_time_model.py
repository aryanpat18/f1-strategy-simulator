"""
models/train_lap_time_model.py
==============================

Builds the training DataFrame for LapTimeModel and provides a CLI entry
point for running training directly.

Training data filter: 2022 onwards (ground effect era)
-------------------------------------------------------
The 2022 regulations introduced ground effect aerodynamics, which reset
tire degradation behavior, lap time distributions, and car characteristics
compared to 2020-2021. Training across both eras causes distribution shift.

By filtering to 2022+, we train on consistent car behavior.
The regulation_era feature future-proofs the system for 2026+ regs.

Phase 1 data hygiene (2026-04-26)
---------------------------------
The dry-pace model is trained ONLY on representative dry race-pace laps.
Filters applied (in order):

  1. Drop wet-compound laps (INTER/WET) — separate physics entirely.
  2. Drop laps in races flagged wet (rainfall observed in session).
  3. Drop SC/VSC-affected laps (field median >120% of race-fastest at that lap).
  4. Drop lap 1 (standing start — not degradation physics).
  5. Drop stint warmup laps (first lap of any non-first stint, tire_age==0).
  6. Drop in-laps (last lap of any non-final stint — driver backs off / pushes
     for pit window, depending on strategy, but unrepresentative either way).
  7. 107% rule on remaining laps (catches data errors / heavy traffic).
  8. Per-lap-number outlier removal (>115% of field median for that lap).
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from models.lap_time_model import LapTimeModel
from models.feature_config import FEATURE_COLUMNS, get_regulation_era


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_TRAINING_YEAR = 2022

DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
SC_FIELD_MEDIAN_THRESHOLD = 1.20   # field median >120% of race fastest = SC/VSC
LAP_OUTLIER_THRESHOLD = 1.15       # lap >115% of field median for that lap = traffic/spin
RULE_107_PCT = 1.07
LAP_TIME_FLOOR_PCT = 0.80          # below 80% of fastest = data error


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_WEATHER_CTE = """
WITH race_weather AS (
    SELECT
        race_id,
        AVG(track_temp)   AS avg_track_temp,
        AVG(air_temp)     AS avg_air_temp,
        BOOL_OR(rainfall) AS any_rainfall
    FROM session_weather
    WHERE session_type = 'R'
    GROUP BY race_id
)
"""

_LAP_QUERY = (
    _WEATHER_CTE
    + """
SELECT
    r.id                                         AS race_id,
    r.year,
    r.round,
    r.total_laps,
    r.circuit_key,
    l.driver_code,
    l.team,
    l.lap_number,
    l.lap_time_seconds,
    l.compound,
    l.tyre_life,
    COALESCE(w.avg_track_temp, 25.0)             AS track_temp,
    COALESCE(w.avg_air_temp,   20.0)             AS air_temp,
    COALESCE(w.any_rainfall,   FALSE)            AS is_wet_race
FROM laps l
JOIN  races         r ON r.id       = l.race_id
LEFT JOIN race_weather w ON w.race_id = l.race_id
WHERE l.lap_time_seconds IS NOT NULL
  AND l.lap_time_seconds > 0
  AND r.year >= %(min_year)s
"""
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_training_df(
    engine: Engine,
    min_year: int = MIN_TRAINING_YEAR,
) -> pd.DataFrame:
    """
    Build a flat training DataFrame ready for LapTimeModel.train().

    Returns DataFrame with FEATURE_COLUMNS + lap_time_seconds.
    """
    df = pd.read_sql(_LAP_QUERY, engine, params={"min_year": min_year})

    if df.empty:
        raise ValueError(
            f"No lap records found for year >= {min_year}. "
            "Run the ingestion pipeline or backfill script before training."
        )

    print(
        f"  Loaded {len(df):,} raw lap rows from {df['race_id'].nunique()} races "
        f"(year >= {min_year})."
    )

    # ------------------------------------------------------------------
    # Feature derivation
    # ------------------------------------------------------------------

    df["driver_id"]     = df["driver_code"].astype(str)
    df["team_id"]       = df["team"].fillna("unknown").astype(str)
    df["circuit_key"]   = df["circuit_key"].fillna("unknown").astype(str)
    df["tire_compound"] = df["compound"].astype(str)
    df["tire_age"]      = df["tyre_life"].fillna(0).astype(int)

    # Fuel load proxy: laps remaining
    total = df["total_laps"].fillna(60).astype(int)
    df["fuel_load"] = (total - df["lap_number"].astype(int)).clip(lower=0)

    # Race progress: normalized 0-1
    df["race_progress"] = (df["lap_number"].astype(float) / total.astype(float)).clip(0, 1)

    df["track_temp"] = df["track_temp"].astype(float)
    df["air_temp"]   = df["air_temp"].astype(float)

    # Regulation era from year
    df["regulation_era"] = df["year"].apply(get_regulation_era)

    # Stint number: inferred from tyre_life resets
    df = df.sort_values(["race_id", "driver_id", "lap_number"])
    prev_age = df.groupby(["race_id", "driver_id"])["tire_age"].shift(1)
    pit_reset = prev_age.notna() & (df["tire_age"] < prev_age)
    df["stint_number"] = (
        pit_reset.groupby([df["race_id"], df["driver_id"]]).cumsum().astype(int) + 1
    )

    df["lap_number"]       = df["lap_number"].astype(int)
    df["lap_time_seconds"] = df["lap_time_seconds"].astype(float)
    df["stint_number"]     = df["stint_number"].astype(int)

    # ------------------------------------------------------------------
    # Phase 1 multi-stage data hygiene
    # ------------------------------------------------------------------
    df = _clean_training_data(df)

    # ------------------------------------------------------------------
    # Weather coverage report
    # ------------------------------------------------------------------
    _log_weather_coverage(df)

    # ------------------------------------------------------------------
    # Return only the columns LapTimeModel expects + target
    # ------------------------------------------------------------------
    keep_cols = FEATURE_COLUMNS + ["lap_time_seconds"]
    return df[keep_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-stage data hygiene for the dry-pace model.

    Each stage logs how many laps it removes so we can see what dominates
    the cleanup. Order matters: cheaper / coarser filters run first so
    the expensive per-lap-median computation runs on a smaller frame.
    """
    initial = len(df)
    print(f"\n  Data hygiene starting from {initial:,} raw laps:")

    # Stage 1: drop wet compounds entirely (separate model territory)
    before = len(df)
    df = df[df["tire_compound"].isin(DRY_COMPOUNDS)].copy()
    _log_stage("wet-compound laps (INTER/WET)", before - len(df))

    # Stage 2: drop races flagged with rainfall
    before = len(df)
    if "is_wet_race" in df.columns:
        df = df[~df["is_wet_race"].fillna(False)].copy()
    _log_stage("laps from wet races", before - len(df))

    # Stage 3: SC/VSC laps (field-median >120% of race fastest at that lap)
    before = len(df)
    race_fastest = df.groupby("race_id")["lap_time_seconds"].transform("min")
    lap_median   = df.groupby(["race_id", "lap_number"])["lap_time_seconds"].transform("median")
    sc_mask = lap_median > race_fastest * SC_FIELD_MEDIAN_THRESHOLD
    df = df[~sc_mask].copy()
    _log_stage("SC/VSC-affected laps", before - len(df))

    # Stage 4: drop lap 1 (standing start)
    before = len(df)
    df = df[df["lap_number"] > 1].copy()
    _log_stage("lap 1 (standing start)", before - len(df))

    # Stage 5: drop stint warmup laps
    # First lap of a stint where tire_age == 0 is the out-lap. We already
    # exclude race-start (lap_number > 1 above), so this catches post-pit
    # warmups only.
    before = len(df)
    df = df.sort_values(["race_id", "driver_id", "lap_number"]).reset_index(drop=True)
    first_lap_of_stint = df.groupby(["race_id", "driver_id", "stint_number"])["lap_number"].transform("min")
    warmup_mask = (df["lap_number"] == first_lap_of_stint) & (df["tire_age"] == 0)
    df = df[~warmup_mask].copy()
    _log_stage("stint warmup laps (out-laps)", before - len(df))

    # Stage 6: drop in-laps (last lap of any non-final stint)
    before = len(df)
    next_stint = df.groupby(["race_id", "driver_id"])["stint_number"].shift(-1)
    last_in_stint = df.groupby(["race_id", "driver_id", "stint_number"])["lap_number"].transform("max")
    in_lap_mask = (df["lap_number"] == last_in_stint) & next_stint.notna() & (next_stint > df["stint_number"])
    df = df[~in_lap_mask].copy()
    _log_stage("in-laps (pre-pit)", before - len(df))

    # Stage 7: 107% rule + lap-time floor (catches data errors, severe traffic)
    before = len(df)
    fastest = df.groupby("race_id")["lap_time_seconds"].transform("min")
    df = df[
        (df["lap_time_seconds"] <= fastest * RULE_107_PCT) &
        (df["lap_time_seconds"] >= fastest * LAP_TIME_FLOOR_PCT)
    ].copy()
    _log_stage("107% rule + low-time floor", before - len(df))

    # Stage 8: per-lap field-median outlier (catches dirty-air / minor incidents)
    before = len(df)
    lap_median = df.groupby(["race_id", "lap_number"])["lap_time_seconds"].transform("median")
    df = df[df["lap_time_seconds"] <= lap_median * LAP_OUTLIER_THRESHOLD].copy()
    _log_stage("per-lap median outliers", before - len(df))

    final = len(df)
    pct = (initial - final) / initial if initial else 0.0
    print(f"  Hygiene complete: {final:,} laps remain ({pct:.1%} removed).\n")
    return df


def _log_stage(label: str, removed: int) -> None:
    if removed > 0:
        print(f"    - {label:<40} {removed:>7,} laps removed")


def _log_weather_coverage(df: pd.DataFrame) -> None:
    """Print weather data coverage to stdout."""
    race_temps = df.groupby("circuit_key")["track_temp"].mean()
    fallback = (race_temps == 25.0).sum()
    real = (race_temps != 25.0).sum()

    print(
        f"  Weather coverage: {real} circuits with real data, "
        f"{fallback} circuits using fallback (25.0C / 20.0C)."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise EnvironmentError("DATABASE_URL is not set.")

    model_dir = os.getenv("MODEL_DIR", "/app/models/artifacts")
    os.makedirs(model_dir, exist_ok=True)

    min_year = int(os.getenv("MIN_TRAINING_YEAR", MIN_TRAINING_YEAR))

    engine = create_engine(db_url)
    df = build_training_df(engine, min_year=min_year)

    print(f"\n  Training LapTimeModel on {len(df):,} laps...")
    model = LapTimeModel(model_dir=model_dir)
    model.train(df)

    print(f"  Saved lap time models to {model_dir}")


if __name__ == "__main__":
    main()
