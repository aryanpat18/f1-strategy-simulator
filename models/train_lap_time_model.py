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
"""

import os
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from models.lap_time_model import LapTimeModel
from models.feature_config import FEATURE_COLUMNS, get_regulation_era


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_TRAINING_YEAR = 2022


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
    # Quality filter
    # ------------------------------------------------------------------
    df = _filter_outliers(df)

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


def _filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove lap-time outliers using the 107% rule (same as F1 regulations).
    Also drops laps faster than 80% of the best lap (data errors).
    """
    before = len(df)

    fastest = df.groupby("race_id")["lap_time_seconds"].min().rename("fastest_lap")
    df = df.join(fastest, on="race_id")

    df = df[
        (df["lap_time_seconds"] <= df["fastest_lap"] * 1.07) &
        (df["lap_time_seconds"] >= df["fastest_lap"] * 0.80)
    ].drop(columns=["fastest_lap"])

    removed = before - len(df)
    if removed > 0:
        print(f"  Filtered {removed:,} outlier laps ({removed / before:.1%} of total).")

    return df


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
