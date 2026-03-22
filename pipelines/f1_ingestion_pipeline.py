"""
F1 Strategy Production Pipeline — Airflow DAG

Tasks:
  1. validate_and_ingest_f1_data  — Fetches R + Q (+ SQ for sprint weekends),
                                    persists laps, qualifying laps, and weather.
  2. run_automation_suite         — Trains LapTimeModel, then runs strategy
                                    optimization for the ingested race and
                                    persists results to simulation_results.

Idempotency guarantee:
  Every (year, round) ingestion deletes the existing Race row first. Because all
  child tables (laps, qualifying_laps, session_weather, track_models,
  simulation_results) are declared with ON DELETE CASCADE, a single
  db.delete(race) + db.commit() clears everything atomically.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param
from datetime import datetime, timedelta
import sys
import os
from typing import Optional

import pandas as pd
import fastf1
from fastf1.core import Session as FF1Session

from sqlalchemy import create_engine
from sqlalchemy.orm import Session as DBSession

sys.path.append('/opt/airflow/')

from db.database import (
    SessionLocal,
    Race,
    Lap,
    QualifyingLap,
    SessionWeather,
    TrackModel,
    SimulationResult,
    init_db,
)
from db.calculate_metrics import calculate_pit_loss

from models.modeling_engine import ModelingEngine
from models.model_config import SimulationConfig, ModelConfig
from models.optimization.optimizer_config import OptimizerConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_PATH = os.getenv("FASTF1_CACHE_PATH", "/opt/airflow/cache")
MODEL_DIR = os.getenv("MODEL_DIR", "/opt/airflow/models/artifacts")

# Drivers to auto-simulate after each ingestion.
# Covers the current grid's top performers across teams.
AUTO_SIMULATE_DRIVERS = ["VER", "NOR", "LEC", "HAM", "RUS"]
AUTO_SIMULATE_NUM_SIMULATIONS = 200


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ingest_race_session(
    db: DBSession,
    race_id: int,
    ff1_session: FF1Session,
) -> None:
    """
    Parse race lap data from a pre-loaded FastF1 session and bulk-insert
    into the `laps` table.
    """
    laps_df = ff1_session.laps.copy()

    if laps_df.empty:
        raise ValueError("Race session returned zero laps — cannot continue.")

    laps_df = laps_df.dropna(
        subset=["LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Driver"]
    )
    laps_df["TyreLife"] = laps_df["TyreLife"].fillna(0).astype(int)
    laps_df["lap_seconds"] = laps_df["LapTime"].dt.total_seconds()

    laps_to_insert = [
        Lap(
            race_id=race_id,
            driver_code=row["Driver"],
            lap_number=int(row["LapNumber"]),
            lap_time_seconds=row["lap_seconds"],
            s1=row["Sector1Time"].total_seconds(),
            s2=row["Sector2Time"].total_seconds(),
            s3=row["Sector3Time"].total_seconds(),
            compound=row["Compound"],
            tyre_life=row["TyreLife"],
            is_pit_out_lap=pd.notnull(row["PitOutTime"]),
            team=row.get("Team"),
        )
        for _, row in laps_df.iterrows()
    ]

    db.bulk_save_objects(laps_to_insert)
    print(f"  ↳ Race: inserted {len(laps_to_insert)} lap rows.")


def _ingest_qualifying_session(
    db: DBSession,
    race_id: int,
    ff1_session: FF1Session,
    session_type: str,
) -> None:
    """
    Parse qualifying lap data and bulk-insert into `qualifying_laps`.
    """
    laps_df = ff1_session.laps.copy()

    if laps_df.empty:
        print(f"  ↳ {session_type}: no laps found — skipping.")
        return

    laps_df = laps_df.dropna(subset=["Driver"])
    laps_df["TyreLife"] = laps_df["TyreLife"].fillna(0).astype(int)

    def _safe_seconds(td) -> Optional[float]:
        try:
            return td.total_seconds() if pd.notnull(td) else None
        except Exception:
            return None

    def _is_deleted(row) -> bool:
        val = row.get("Deleted", False)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() == "true"
        return bool(val) if pd.notnull(val) else False

    rows_to_insert = [
        QualifyingLap(
            race_id=race_id,
            session_type=session_type,
            driver_code=row["Driver"],
            lap_number=int(row["LapNumber"]),
            lap_time_seconds=_safe_seconds(row.get("LapTime")),
            s1=_safe_seconds(row.get("Sector1Time")),
            s2=_safe_seconds(row.get("Sector2Time")),
            s3=_safe_seconds(row.get("Sector3Time")),
            compound=row.get("Compound"),
            tyre_life=row["TyreLife"],
            is_deleted=_is_deleted(row),
            team=row.get("Team"),
        )
        for _, row in laps_df.iterrows()
    ]

    db.bulk_save_objects(rows_to_insert)
    print(f"  ↳ {session_type}: inserted {len(rows_to_insert)} qualifying lap rows.")


def _ingest_weather(
    db: DBSession,
    race_id: int,
    ff1_session: FF1Session,
    session_type: str,
) -> None:
    """
    Persist the full FastF1 weather time-series into `session_weather`.
    """
    weather_df = ff1_session.weather_data

    if weather_df is None or weather_df.empty:
        print(f"  ↳ {session_type}: no weather data available — skipping.")
        return

    weather_rows = []
    for _, row in weather_df.iterrows():
        try:
            time_offset = row["Time"].total_seconds()
        except Exception:
            continue

        weather_rows.append(
            SessionWeather(
                race_id=race_id,
                session_type=session_type,
                time_offset_seconds=time_offset,
                air_temp=_safe_float(row.get("AirTemp")),
                track_temp=_safe_float(row.get("TrackTemp")),
                humidity=_safe_float(row.get("Humidity")),
                pressure=_safe_float(row.get("Pressure")),
                wind_speed=_safe_float(row.get("WindSpeed")),
                wind_direction=_safe_float(row.get("WindDirection")),
                rainfall=bool(row.get("Rainfall", False)),
            )
        )

    db.bulk_save_objects(weather_rows)
    print(f"  ↳ {session_type} weather: inserted {len(weather_rows)} observations.")


def _safe_float(val) -> Optional[float]:
    """Convert a value to float, returning None if not convertible."""
    try:
        return float(val) if pd.notnull(val) else None
    except (TypeError, ValueError):
        return None


def _load_session(
    year: int,
    round_num: int,
    session_type: str,
) -> Optional[FF1Session]:
    """
    Attempt to load a FastF1 session.
    Returns None for non-existent session types (e.g. SQ on non-sprint weekends).
    """
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load(laps=True, telemetry=False, weather=True, messages=False)
        return session
    except ValueError as e:
        print(f"  ↳ {session_type}: session does not exist for {year} R{round_num} — skipping. ({e})")
        return None
    except Exception:
        raise


# ---------------------------------------------------------------------------
# Airflow task: ingest
# ---------------------------------------------------------------------------


def validate_and_ingest_f1_data(**kwargs) -> None:
    """
    Airflow task — ingest all sessions for a given (year, round).

    Idempotency: if a Race row already exists for (year, round), it is
    deleted first. All child rows are removed via CASCADE.
    """
    year = kwargs["params"]["year"]
    round_num = kwargs["params"]["round"]

    print(f"🚀 Starting ingestion for {year} Round {round_num}")
    init_db()
    fastf1.Cache.enable_cache(CACHE_PATH)

    print(f"📡 Loading Race session...")
    race_session = fastf1.get_session(year, round_num, "R")
    race_session.load(laps=True, telemetry=False, weather=True, messages=False)
    total_laps = int(race_session.total_laps) if race_session.total_laps else 0

    print(f"📡 Loading Qualifying session...")
    quali_session = _load_session(year, round_num, "Q")

    print(f"📡 Checking for Sprint Qualifying session...")
    sprint_quali_session = _load_session(year, round_num, "SQ")

    db = SessionLocal()
    try:
        existing_race = (
            db.query(Race).filter(Race.year == year, Race.round == round_num).first()
        )
        if existing_race:
            print(f"🧹 Cleaning existing data for Race ID: {existing_race.id}")
            db.delete(existing_race)
            db.commit()

        # Extract stable circuit identifier from FastF1 event location
        circuit_key = (
            race_session.event.get("Location", "unknown")
            .strip().lower().replace(" ", "_").replace("-", "_")
        )

        new_race = Race(
            year=year,
            round=round_num,
            event_name=race_session.event["EventName"],
            total_laps=total_laps,
            circuit_key=circuit_key,
        )
        db.add(new_race)
        db.commit()
        db.refresh(new_race)
        race_id = new_race.id
        print(f"✅ Race row created — ID: {race_id}, Event: {new_race.event_name}")

        print(f"💾 Persisting Race ('R') data...")
        _ingest_race_session(db, race_id, race_session)
        _ingest_weather(db, race_id, race_session, "R")
        db.commit()

        if quali_session is not None:
            print(f"💾 Persisting Qualifying ('Q') data...")
            _ingest_qualifying_session(db, race_id, quali_session, "Q")
            _ingest_weather(db, race_id, quali_session, "Q")
            db.commit()
        else:
            print("⚠️  Qualifying session unavailable — skipping Q data.")

        if sprint_quali_session is not None:
            print(f"💾 Persisting Sprint Qualifying ('SQ') data...")
            _ingest_qualifying_session(db, race_id, sprint_quali_session, "SQ")
            _ingest_weather(db, race_id, sprint_quali_session, "SQ")
            db.commit()

        calculate_pit_loss()
        print("✅ Ingestion complete.")

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Airflow task: train + auto-simulate
# ---------------------------------------------------------------------------


def run_automation_suite(**kwargs) -> None:
    """
    Airflow task — after ingestion:
      1. Train and save the LapTimeModel on all available data.
      2. Run strategy simulation for AUTO_SIMULATE_DRIVERS against the
         just-ingested race and persist results to simulation_results.

    Step 2 populates the DB with pre-computed strategy rankings so the
    dashboard can load them instantly without waiting for a simulation.
    """
    year = kwargs["params"]["year"]
    round_num = kwargs["params"]["round"]
    print(f"🧠 Running automation suite for {year} Round {round_num}")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set.")

    db_engine = create_engine(database_url)

    modeling_engine = ModelingEngine(
        model_config=ModelConfig(model_dir=MODEL_DIR),
        simulation_config=SimulationConfig(
            pit_loss_seconds=float(os.getenv("PIT_LOSS_SECONDS", 22.0)),
            lap_variance=float(os.getenv("LAP_VARIANCE", 0.1)),
            default_fuel_load=float(os.getenv("DEFAULT_FUEL_LOAD", 100.0)),
        ),
        optimizer_config=OptimizerConfig(
            num_simulations=int(os.getenv("OPT_NUM_SIMULATIONS", 300)),
            risk_penalty=float(os.getenv("OPT_RISK_PENALTY", 1.0)),
            n_trials=int(os.getenv("OPT_N_TRIALS", 60)),
            max_strategies=int(os.getenv("OPT_MAX_STRATEGIES", 100)),
            max_stops=int(os.getenv("OPT_MAX_STOPS", 2)),
        ),
        db_engine=db_engine,
    )

    # ------------------------------------------------------------------
    # Step 1: Train model
    # ------------------------------------------------------------------
    print("🛠️  Training LapTimeModel...")
    modeling_engine.train_and_save_all_models()
    print("✅ Training complete.")

    # ------------------------------------------------------------------
    # Step 2: Auto-simulate top drivers for this race
    # ------------------------------------------------------------------
    db = SessionLocal()
    try:
        race = (
            db.query(Race)
            .filter(Race.year == year, Race.round == round_num)
            .first()
        )
        if race is None:
            print(f"⚠️  Race {year} R{round_num} not found in DB — skipping simulation.")
            return

        race_id = race.id
        race_laps = race.total_laps or 52  # fallback if total_laps not set
        event_name = race.event_name

        # Delete existing simulation results for this race
        db.query(SimulationResult).filter(
            SimulationResult.race_id == race_id
        ).delete()
        db.commit()

        print(f"🏎️  Auto-simulating {len(AUTO_SIMULATE_DRIVERS)} drivers for {event_name} ({race_laps} laps)...")

        for driver_id in AUTO_SIMULATE_DRIVERS:
            try:
                result = modeling_engine.simulate_driver_gp(
                    year=year,
                    round_number=round_num,
                    driver_id=driver_id,
                    race_laps=race_laps,
                    num_simulations=AUTO_SIMULATE_NUM_SIMULATIONS,
                    is_wet_race=False,
                    circuit_key=race.circuit_key or "unknown",
                )

                # Persist top 5 strategies per driver
                rows = []
                for r in result["results"][:5]:
                    rows.append(
                        SimulationResult(
                            race_id=race_id,
                            driver_code=driver_id,
                            strategy_compounds="-".join(r["strategy"]["compounds"]),
                            strategy_stints="-".join(str(s) for s in r["strategy"]["stints"]),
                            mean_time=r["mean_time"],
                            std_time=r["std_time"],
                            p10=r["p10"],
                            p50=r["p50"],
                            p90=r["p90"],
                        )
                    )
                db.bulk_save_objects(rows)
                db.commit()
                print(f"  ↳ {driver_id}: best strategy {result['results'][0]['strategy']['compounds']} — {result['results'][0]['p50']:.1f}s median")

            except Exception as e:
                print(f"  ↳ {driver_id}: simulation failed — {e}")
                continue

        print("✅ Auto-simulation complete.")

    except Exception as e:
        db.rollback()
        print(f"❌ Auto-simulation error: {e}")
        # Don't re-raise — training succeeded, simulation failure is non-critical
    finally:
        db.close()


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    "owner": "aryan",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "f1_strategy_production_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    params={
        "year": Param(2024, type="integer"),
        "round": Param(1, type="integer"),
    },
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_and_clean_race_data",
        python_callable=validate_and_ingest_f1_data,
    )

    automate_task = PythonOperator(
        task_id="run_modeling_and_simulation",
        python_callable=run_automation_suite,
    )

    ingest_task >> automate_task