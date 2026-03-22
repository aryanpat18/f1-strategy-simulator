"""
api/routes/data.py
==================

Read-only endpoints for dashboard data: races, track metrics,
and pre-computed simulation results.
"""

import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

router = APIRouter(prefix="/data", tags=["Data"])


def _get_db_engine():
    """Lazily create a DB engine from DATABASE_URL."""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise HTTPException(
            status_code=503,
            detail="DATABASE_URL not configured — data endpoints unavailable.",
        )
    return create_engine(url)


# -------------------------------------------------------------------
# Response models
# -------------------------------------------------------------------

class RaceInfo(BaseModel):
    id: int
    year: int
    round: int
    event_name: str
    total_laps: Optional[int] = None
    circuit_key: Optional[str] = None


class TrackMetricInfo(BaseModel):
    event_name: str
    avg_pit_loss: Optional[float] = None
    fuel_penalty_per_lap: Optional[float] = None


class StoredSimulationResult(BaseModel):
    driver_code: str
    strategy_compounds: Optional[str] = None
    strategy_stints: Optional[str] = None
    mean_time: Optional[float] = None
    std_time: Optional[float] = None
    p10: Optional[float] = None
    p50: Optional[float] = None
    p90: Optional[float] = None


class RaceDetail(BaseModel):
    race: RaceInfo
    track_metric: Optional[TrackMetricInfo] = None
    simulations: List[StoredSimulationResult]


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@router.get("/races", response_model=List[RaceInfo])
def list_races(year: Optional[int] = None):
    """Return all ingested races, optionally filtered by year."""
    engine = _get_db_engine()
    query = "SELECT id, year, round, event_name, total_laps, circuit_key FROM races"
    params = {}
    if year is not None:
        query += " WHERE year = :year"
        params["year"] = year
    query += " ORDER BY year DESC, round ASC"

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    return [RaceInfo(**dict(r)) for r in rows]


@router.get("/tracks", response_model=List[TrackMetricInfo])
def list_track_metrics():
    """Return pit-loss constants for all tracks."""
    engine = _get_db_engine()
    query = "SELECT event_name, avg_pit_loss, fuel_penalty_per_lap FROM track_metrics ORDER BY event_name"

    with engine.connect() as conn:
        rows = conn.execute(text(query)).mappings().all()

    return [TrackMetricInfo(**dict(r)) for r in rows]


@router.get("/race/{year}/{round_number}", response_model=RaceDetail)
def get_race_detail(year: int, round_number: int):
    """
    Full detail for one race: metadata, track metrics, and
    any pre-computed simulation results.
    """
    engine = _get_db_engine()

    with engine.connect() as conn:
        # Race info
        race_row = conn.execute(
            text("SELECT id, year, round, event_name, total_laps, circuit_key FROM races WHERE year = :y AND round = :r"),
            {"y": year, "r": round_number},
        ).mappings().first()

        if not race_row:
            raise HTTPException(status_code=404, detail=f"Race {year} round {round_number} not found.")

        race = RaceInfo(**dict(race_row))

        # Track metric
        tm_row = conn.execute(
            text("SELECT event_name, avg_pit_loss, fuel_penalty_per_lap FROM track_metrics WHERE event_name = :en"),
            {"en": race.event_name},
        ).mappings().first()

        track_metric = TrackMetricInfo(**dict(tm_row)) if tm_row else None

        # Pre-computed simulations
        sim_rows = conn.execute(
            text(
                "SELECT driver_code, strategy_compounds, strategy_stints, "
                "mean_time, std_time, p10, p50, p90 "
                "FROM simulation_results WHERE race_id = :rid "
                "ORDER BY p50 ASC"
            ),
            {"rid": race.id},
        ).mappings().all()

        simulations = [StoredSimulationResult(**dict(s)) for s in sim_rows]

    return RaceDetail(race=race, track_metric=track_metric, simulations=simulations)


@router.get("/drivers", response_model=List[str])
def list_drivers(year: Optional[int] = None):
    """Return distinct driver codes from the laps table."""
    engine = _get_db_engine()
    if year is not None:
        query = (
            "SELECT DISTINCT l.driver_code FROM laps l "
            "JOIN races r ON l.race_id = r.id "
            "WHERE r.year = :year ORDER BY l.driver_code"
        )
        params = {"year": year}
    else:
        query = "SELECT DISTINCT driver_code FROM laps ORDER BY driver_code"
        params = {}

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).all()

    return [r[0] for r in rows]


@router.get("/teams", response_model=List[str])
def list_teams(year: Optional[int] = None):
    """Return distinct team names from the laps table."""
    engine = _get_db_engine()
    if year is not None:
        query = (
            "SELECT DISTINCT l.team FROM laps l "
            "JOIN races r ON l.race_id = r.id "
            "WHERE r.year = :year AND l.team IS NOT NULL "
            "ORDER BY l.team"
        )
        params = {"year": year}
    else:
        query = "SELECT DISTINCT team FROM laps WHERE team IS NOT NULL ORDER BY team"
        params = {}

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).all()

    return [r[0] for r in rows]


# -------------------------------------------------------------------
# Race Analysis — Actual Lap Data
# -------------------------------------------------------------------

class LapData(BaseModel):
    driver_code: str
    lap_number: int
    lap_time_seconds: Optional[float] = None
    compound: Optional[str] = None
    tyre_life: Optional[int] = None
    is_pit_out_lap: Optional[bool] = None
    s1: Optional[float] = None
    s2: Optional[float] = None
    s3: Optional[float] = None


class DriverRaceSummary(BaseModel):
    driver_code: str
    total_laps: int
    best_lap: Optional[float] = None
    avg_lap: Optional[float] = None
    stints: List[dict]  # [{compound, start_lap, end_lap, avg_time, laps}]
    laps: List[LapData]


class RaceAnalysis(BaseModel):
    race: RaceInfo
    track_metric: Optional[TrackMetricInfo] = None
    drivers: List[DriverRaceSummary]


@router.get("/race-analysis/{year}/{round_number}", response_model=RaceAnalysis)
def get_race_analysis(year: int, round_number: int):
    """
    Full race analysis with actual lap data, stint breakdowns,
    and per-driver performance for a completed race.
    """
    engine = _get_db_engine()

    with engine.connect() as conn:
        race_row = conn.execute(
            text("SELECT id, year, round, event_name, total_laps, circuit_key FROM races WHERE year = :y AND round = :r"),
            {"y": year, "r": round_number},
        ).mappings().first()

        if not race_row:
            raise HTTPException(status_code=404, detail=f"Race {year} round {round_number} not found.")

        race = RaceInfo(**dict(race_row))

        # Track metric
        tm_row = conn.execute(
            text("SELECT event_name, avg_pit_loss, fuel_penalty_per_lap FROM track_metrics WHERE event_name = :en"),
            {"en": race.event_name},
        ).mappings().first()
        track_metric = TrackMetricInfo(**dict(tm_row)) if tm_row else None

        # All laps for this race
        lap_rows = conn.execute(
            text(
                "SELECT driver_code, lap_number, lap_time_seconds, compound, "
                "tyre_life, is_pit_out_lap, s1, s2, s3 "
                "FROM laps WHERE race_id = :rid "
                "ORDER BY driver_code, lap_number"
            ),
            {"rid": race.id},
        ).mappings().all()

    # Group by driver and compute summaries
    from collections import defaultdict
    driver_laps = defaultdict(list)
    for row in lap_rows:
        d = dict(row)
        driver_laps[d["driver_code"]].append(d)

    drivers = []
    for driver_code in sorted(driver_laps.keys()):
        laps = driver_laps[driver_code]
        valid_times = [l["lap_time_seconds"] for l in laps
                       if l["lap_time_seconds"] is not None and l["lap_time_seconds"] > 0]

        # Compute stint breakdown (detect compound changes / pit stops)
        stints = []
        current_stint = None
        for lap in laps:
            compound = lap.get("compound")
            if current_stint is None or compound != current_stint["compound"] or lap.get("is_pit_out_lap"):
                if current_stint is not None:
                    stints.append(current_stint)
                current_stint = {
                    "compound": compound,
                    "start_lap": lap["lap_number"],
                    "end_lap": lap["lap_number"],
                    "laps": 0,
                    "times": [],
                }
            current_stint["end_lap"] = lap["lap_number"]
            current_stint["laps"] += 1
            if lap["lap_time_seconds"] and lap["lap_time_seconds"] > 0 and not lap.get("is_pit_out_lap"):
                current_stint["times"].append(lap["lap_time_seconds"])

        if current_stint is not None:
            stints.append(current_stint)

        # Compute avg time per stint and clean up
        stint_summaries = []
        for s in stints:
            stint_summaries.append({
                "compound": s["compound"],
                "start_lap": s["start_lap"],
                "end_lap": s["end_lap"],
                "laps": s["laps"],
                "avg_time": round(sum(s["times"]) / len(s["times"]), 3) if s["times"] else None,
            })

        drivers.append(DriverRaceSummary(
            driver_code=driver_code,
            total_laps=len(laps),
            best_lap=round(min(valid_times), 3) if valid_times else None,
            avg_lap=round(sum(valid_times) / len(valid_times), 3) if valid_times else None,
            stints=stint_summaries,
            laps=[LapData(**l) for l in laps],
        ))

    return RaceAnalysis(race=race, track_metric=track_metric, drivers=drivers)


# -------------------------------------------------------------------
# Pre-Race Intelligence — Current Season Form + Track History
# -------------------------------------------------------------------

class DriverForm(BaseModel):
    driver_code: str
    races_completed: int
    avg_delta_to_race_best: Optional[float] = None   # normalized: avg gap to race winner pace
    best_delta_to_race_best: Optional[float] = None   # best single-race gap to winner pace
    avg_position_proxy: Optional[float] = None         # rank by pace delta (1 = fastest)


class TrackHistory(BaseModel):
    year: int
    round: int
    event_name: str
    total_laps: Optional[int] = None
    avg_pit_loss: Optional[float] = None


class PreRaceIntelligence(BaseModel):
    target_event: str
    season: int
    driver_form: List[DriverForm]
    track_history: List[TrackHistory]


@router.get("/pre-race/{year}/{event_name}", response_model=PreRaceIntelligence)
def get_pre_race_intelligence(year: int, event_name: str):
    """
    Pre-race intelligence: current season form for all drivers
    + historical data for this track from previous years.

    Pace is NORMALIZED per-race: for each GP, we compute the
    race-best average lap time, then each driver's delta from that.
    This prevents track length bias (Monza 80s vs Spa 105s).
    """
    engine = _get_db_engine()

    with engine.connect() as conn:
        # ---------------------------------------------------------------
        # 1. Normalized season form
        #
        # Step A: Per race, per driver — compute clean average lap time.
        #         Exclude pit-out laps and apply 107% rule per race.
        # Step B: Per race — find the fastest driver's avg (race reference).
        # Step C: Per driver — compute delta to race reference, then average
        #         across all races they participated in.
        # ---------------------------------------------------------------
        form_rows = conn.execute(
            text("""
                WITH driver_race_avg AS (
                    -- Step A: avg lap time per driver per race (clean laps only)
                    SELECT
                        l.driver_code,
                        r.id AS race_id,
                        AVG(l.lap_time_seconds) AS avg_time
                    FROM laps l
                    JOIN races r ON l.race_id = r.id
                    WHERE r.year = :year
                      AND l.lap_time_seconds IS NOT NULL
                      AND l.lap_time_seconds > 0
                      AND l.is_pit_out_lap = false
                    GROUP BY l.driver_code, r.id
                ),
                race_best AS (
                    -- Step B: fastest driver avg per race (reference pace)
                    SELECT race_id, MIN(avg_time) AS best_avg
                    FROM driver_race_avg
                    GROUP BY race_id
                ),
                driver_deltas AS (
                    -- Step C: driver delta to race best
                    SELECT
                        dra.driver_code,
                        dra.race_id,
                        dra.avg_time - rb.best_avg AS delta
                    FROM driver_race_avg dra
                    JOIN race_best rb ON dra.race_id = rb.race_id
                )
                SELECT
                    driver_code,
                    COUNT(DISTINCT race_id) AS races_completed,
                    AVG(delta) AS avg_delta,
                    MIN(delta) AS best_delta
                FROM driver_deltas
                GROUP BY driver_code
                ORDER BY avg_delta ASC
            """),
            {"year": year},
        ).mappings().all()

        driver_form = []
        for i, row in enumerate(form_rows):
            driver_form.append(DriverForm(
                driver_code=row["driver_code"],
                races_completed=row["races_completed"],
                avg_delta_to_race_best=round(float(row["avg_delta"]), 3) if row["avg_delta"] is not None else None,
                best_delta_to_race_best=round(float(row["best_delta"]), 3) if row["best_delta"] is not None else None,
                avg_position_proxy=i + 1,
            ))

        # ---------------------------------------------------------------
        # 2. Track history: same event from previous years
        # ---------------------------------------------------------------
        track_rows = conn.execute(
            text(
                "SELECT r.year, r.round, r.event_name, r.total_laps, tm.avg_pit_loss "
                "FROM races r "
                "LEFT JOIN track_metrics tm ON r.event_name = tm.event_name "
                "WHERE r.event_name = :event_name "
                "ORDER BY r.year DESC"
            ),
            {"event_name": event_name},
        ).mappings().all()

        track_history = [
            TrackHistory(
                year=r["year"],
                round=r["round"],
                event_name=r["event_name"],
                total_laps=r["total_laps"],
                avg_pit_loss=float(r["avg_pit_loss"]) if r["avg_pit_loss"] else None,
            )
            for r in track_rows
        ]

    return PreRaceIntelligence(
        target_event=event_name,
        season=year,
        driver_form=driver_form,
        track_history=track_history,
    )
