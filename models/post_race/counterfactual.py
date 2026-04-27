"""
models/post_race/counterfactual.py
==================================

Post-race counterfactual engine.

Answers questions of the form: "what if Driver X had pitted under the SC
on lap 25 instead of staying out?" — for one driver, or a set of drivers
simultaneously (so we can compare 4 scenarios for two drivers, or 2^N
permutations for N drivers).

How it works
------------
1. We reconstruct each driver's *actual* strategy from the laps table by
   walking tyre_life resets — that gives us stint lengths + compounds.
2. The caller supplies an `overrides` dict mapping driver_id -> alternative
   strategy. Anyone without an override uses their actual strategy.
3. Each driver is simulated independently using the existing LapTimeModel
   (already conditioned on their driver_id, team_id, circuit_key, era,
   weather). For each Monte Carlo draw we accumulate lap-by-lap time so
   we can compute virtual gaps between drivers, not just totals.
4. We compare the simulated total time to the *actual* total time for
   each driver and return the delta — this is the counterfactual gap.

What we deliberately do NOT model (yet)
---------------------------------------
- Dirty-air penalty when one car catches another.
- Overtake difficulty (Monaco vs Monza).
- Pit stop ordering (when two cars pit on the same lap).

These are layered on top of this engine in a follow-up; tracking the
lap-by-lap virtual gap matrix here is the foundation that makes them
possible without rewriting the simulation core.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from models.feature_config import get_regulation_era


# ---------------------------------------------------------------------------
# Strategy data model
# ---------------------------------------------------------------------------


@dataclass
class Strategy:
    stints: List[int]            # laps per stint, e.g. [20, 30]
    compounds: List[str]         # one per stint, e.g. ["MEDIUM", "HARD"]

    def total_laps(self) -> int:
        return int(sum(self.stints))

    def num_stops(self) -> int:
        return max(0, len(self.stints) - 1)

    def to_dict(self) -> Dict:
        return {"stints": list(self.stints), "compounds": list(self.compounds)}

    @classmethod
    def from_dict(cls, d: Dict) -> "Strategy":
        return cls(stints=list(d["stints"]), compounds=list(d["compounds"]))


@dataclass
class DriverContext:
    driver_id: str
    team_id: str
    circuit_key: str
    regulation_era: str
    track_temp: float
    air_temp: float
    actual_strategy: Strategy
    actual_total_time: float     # observed sum of valid lap times


@dataclass
class DriverScenarioResult:
    driver_id: str
    strategy: Strategy
    is_override: bool
    actual_total_time: float
    sim_p10: float
    sim_p50: float
    sim_p90: float
    delta_p50: float             # sim_p50 - actual (positive = slower than reality)
    cumulative_time_p50: List[float]   # length = total_laps


@dataclass
class ScenarioResult:
    name: str
    overrides_applied: List[str]                  # driver_ids who had alternative strategies
    drivers: List[DriverScenarioResult]
    finishing_order_p50: List[str]                # sorted by sim_p50 ascending
    gap_matrix_p50: Dict[str, Dict[str, float]] = field(default_factory=dict)
    """gap_matrix_p50[driver_a][driver_b] = sim_p50[a] - sim_p50[b]
       (positive => a finished BEHIND b in this scenario)."""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _race_metadata(engine: Engine, year: int, round_number: int) -> Dict:
    """Resolve race_id, total_laps, circuit_key, and avg track/air temps."""
    sql = text("""
        SELECT
            r.id, r.total_laps, COALESCE(r.circuit_key, 'unknown') AS circuit_key,
            COALESCE((
                SELECT AVG(track_temp) FROM session_weather
                WHERE race_id = r.id AND session_type = 'R'
            ), 25.0) AS track_temp,
            COALESCE((
                SELECT AVG(air_temp) FROM session_weather
                WHERE race_id = r.id AND session_type = 'R'
            ), 20.0) AS air_temp,
            r.year
        FROM races r
        WHERE r.year = :y AND r.round = :r
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"y": year, "r": round_number}).fetchone()
    if row is None:
        raise ValueError(f"Race not found: year={year} round={round_number}")
    return {
        "race_id": int(row[0]),
        "total_laps": int(row[1] or 0),
        "circuit_key": str(row[2]),
        "track_temp": float(row[3]),
        "air_temp": float(row[4]),
        "year": int(row[5]),
    }


def _driver_context(engine: Engine, race_meta: Dict, driver_id: str) -> DriverContext:
    """Pull team + reconstruct actual strategy + observed total time."""
    race_id = race_meta["race_id"]

    # Find team for this driver in this race
    with engine.connect() as conn:
        team_row = conn.execute(
            text("""
                SELECT team FROM laps
                WHERE race_id = :rid AND driver_code = :d AND team IS NOT NULL
                LIMIT 1
            """),
            {"rid": race_id, "d": driver_id},
        ).fetchone()
    team_id = str(team_row[0]) if team_row and team_row[0] else "unknown"

    actual = _reconstruct_actual_strategy(engine, race_id, driver_id)
    actual_total = _actual_total_time(engine, race_id, driver_id)

    return DriverContext(
        driver_id=driver_id,
        team_id=team_id,
        circuit_key=race_meta["circuit_key"],
        regulation_era=get_regulation_era(race_meta["year"]),
        track_temp=race_meta["track_temp"],
        air_temp=race_meta["air_temp"],
        actual_strategy=actual,
        actual_total_time=actual_total,
    )


def _reconstruct_actual_strategy(
    engine: Engine, race_id: int, driver_id: str
) -> Strategy:
    """
    Walk through the driver's laps, detect stint boundaries via tyre_life
    resets, and return Strategy(stints, compounds).
    """
    sql = text("""
        SELECT lap_number, compound, tyre_life
        FROM laps
        WHERE race_id = :rid AND driver_code = :d
        ORDER BY lap_number
    """)
    with engine.connect() as conn:
        rows = list(conn.execute(sql, {"rid": race_id, "d": driver_id}))
    if not rows:
        raise ValueError(f"No laps found for driver {driver_id} in race_id={race_id}")

    stints: List[int] = []
    compounds: List[str] = []
    current_stint_laps = 0
    current_compound: Optional[str] = None
    prev_age: Optional[int] = None

    for lap_number, compound, tyre_life in rows:
        compound = str(compound) if compound else "MEDIUM"
        age = int(tyre_life) if tyre_life is not None else 0

        # Stint boundary: tyre_life reset (new tire age < previous)
        # OR compound change (rare without a pit, but be defensive)
        is_new_stint = (
            current_compound is None or
            (prev_age is not None and age < prev_age) or
            (compound != current_compound)
        )
        if is_new_stint and current_compound is not None:
            stints.append(current_stint_laps)
            compounds.append(current_compound)
            current_stint_laps = 0

        current_compound = compound
        current_stint_laps += 1
        prev_age = age

    # Close the final stint
    if current_compound is not None and current_stint_laps > 0:
        stints.append(current_stint_laps)
        compounds.append(current_compound)

    return Strategy(stints=stints, compounds=compounds)


def _actual_total_time(engine: Engine, race_id: int, driver_id: str) -> float:
    sql = text("""
        SELECT COALESCE(SUM(lap_time_seconds), 0.0)
        FROM laps
        WHERE race_id = :rid AND driver_code = :d
              AND lap_time_seconds IS NOT NULL
    """)
    with engine.connect() as conn:
        return float(conn.execute(sql, {"rid": race_id, "d": driver_id}).scalar() or 0.0)


# ---------------------------------------------------------------------------
# Per-driver simulation with cumulative time tracking
# ---------------------------------------------------------------------------


def simulate_driver_cumtime(
    lap_time_model,
    strategy: Strategy,
    ctx: DriverContext,
    pit_loss_seconds: float,
    num_simulations: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate this driver's race lap-by-lap. Returns a (num_simulations, total_laps)
    matrix of *cumulative* race time at the end of each lap. Pit loss is added on
    the lap of the pit stop (the in-lap), matching how observed lap times include
    the pit-stop delta.
    """
    rng = rng or np.random.default_rng()
    total_laps = strategy.total_laps()

    cumtime = np.zeros((num_simulations, total_laps), dtype=float)
    running = np.zeros(num_simulations, dtype=float)
    race_lap = 0  # 0-indexed; race_lap-th column = lap number race_lap+1

    for stint_idx, (stint_laps, compound) in enumerate(
        zip(strategy.stints, strategy.compounds)
    ):
        if stint_laps <= 0:
            continue

        rows: List[Dict] = []
        for age in range(stint_laps):
            current_lap = race_lap + age + 1
            rows.append({
                "driver_id": ctx.driver_id,
                "team_id": ctx.team_id,
                "circuit_key": ctx.circuit_key,
                "regulation_era": ctx.regulation_era,
                "tire_compound": compound,
                "tire_age": age,
                "lap_number": current_lap,
                "race_progress": float(current_lap) / max(total_laps, 1),
                "fuel_load": float(max(total_laps - current_lap, 0)),
                "stint_number": stint_idx + 1,
                "track_temp": ctx.track_temp,
                "air_temp": ctx.air_temp,
            })

        batch_df = pd.DataFrame(rows)
        p10, p50, p90 = lap_time_model.predict_quantiles_batch(batch_df)

        # Triangular distribution per lap (same family as simulation_engine)
        lo = np.minimum(np.minimum(p10, p50), p90)
        hi = np.maximum(np.maximum(p10, p50), p90)
        mode = np.clip(p50, lo, hi)

        stint_samples = rng.triangular(
            left=lo, mode=mode, right=hi,
            size=(num_simulations, stint_laps),
        )

        # Stitch into cumulative trace
        for k in range(stint_laps):
            running = running + stint_samples[:, k]
            # If this is the last lap of a non-final stint, this is the in-lap:
            # add pit loss to it (mirrors how FastF1's lap_time_seconds includes
            # the pit-stop delta on the in-lap).
            if k == stint_laps - 1 and stint_idx < len(strategy.stints) - 1:
                running = running + pit_loss_seconds
            cumtime[:, race_lap + k] = running

        race_lap += stint_laps

    return cumtime


# ---------------------------------------------------------------------------
# Scenario orchestration
# ---------------------------------------------------------------------------


def run_scenario(
    *,
    engine: Engine,
    lap_time_model,
    year: int,
    round_number: int,
    drivers: List[str],
    overrides: Dict[str, Strategy],
    pit_loss_seconds: float,
    num_simulations: int = 300,
    name: str = "scenario",
    rng: Optional[np.random.Generator] = None,
) -> ScenarioResult:
    """
    Simulate `drivers` independently in a single hypothetical world.
    Each driver uses `overrides[driver_id]` if present, else their actual strategy.
    """
    rng = rng or np.random.default_rng()
    race_meta = _race_metadata(engine, year, round_number)

    per_driver: List[DriverScenarioResult] = []
    cumtime_p50_by_driver: Dict[str, np.ndarray] = {}

    for driver_id in drivers:
        ctx = _driver_context(engine, race_meta, driver_id)
        strategy = overrides.get(driver_id, ctx.actual_strategy)

        cumtime = simulate_driver_cumtime(
            lap_time_model=lap_time_model,
            strategy=strategy,
            ctx=ctx,
            pit_loss_seconds=pit_loss_seconds,
            num_simulations=num_simulations,
            rng=rng,
        )

        totals = cumtime[:, -1]  # total race time per simulation
        sim_p10 = float(np.percentile(totals, 10))
        sim_p50 = float(np.percentile(totals, 50))
        sim_p90 = float(np.percentile(totals, 90))
        cumtime_p50 = np.percentile(cumtime, 50, axis=0)

        per_driver.append(DriverScenarioResult(
            driver_id=driver_id,
            strategy=strategy,
            is_override=driver_id in overrides,
            actual_total_time=ctx.actual_total_time,
            sim_p10=sim_p10,
            sim_p50=sim_p50,
            sim_p90=sim_p90,
            delta_p50=sim_p50 - ctx.actual_total_time,
            cumulative_time_p50=cumtime_p50.tolist(),
        ))
        cumtime_p50_by_driver[driver_id] = cumtime_p50

    # Pairwise final-time gap matrix
    gap_matrix: Dict[str, Dict[str, float]] = {}
    for a in drivers:
        gap_matrix[a] = {}
        for b in drivers:
            if a == b:
                gap_matrix[a][b] = 0.0
            else:
                gap_matrix[a][b] = float(
                    cumtime_p50_by_driver[a][-1] - cumtime_p50_by_driver[b][-1]
                )

    finishing_order = sorted(per_driver, key=lambda r: r.sim_p50)
    return ScenarioResult(
        name=name,
        overrides_applied=sorted(overrides.keys()),
        drivers=per_driver,
        finishing_order_p50=[r.driver_id for r in finishing_order],
        gap_matrix_p50=gap_matrix,
    )


def compare_scenarios(
    *,
    engine: Engine,
    lap_time_model,
    year: int,
    round_number: int,
    drivers: List[str],
    scenarios: List[Tuple[str, Dict[str, Strategy]]],
    pit_loss_seconds: float,
    num_simulations: int = 300,
    seed: Optional[int] = 42,
) -> List[ScenarioResult]:
    """
    Run multiple scenarios over the same set of drivers and return their
    results in input order. A common shape for two drivers X, Y:

        scenarios = [
            ("baseline_actual",     {}),
            ("X_alt_only",          {X: alt_x}),
            ("Y_alt_only",          {Y: alt_y}),
            ("both_alt",            {X: alt_x, Y: alt_y}),
        ]

    Caller can then diff the gap_matrix_p50 across scenarios to answer
    "if Y had pitted, where would they have finished relative to X?"
    """
    rng = np.random.default_rng(seed) if seed is not None else None
    return [
        run_scenario(
            engine=engine,
            lap_time_model=lap_time_model,
            year=year,
            round_number=round_number,
            drivers=drivers,
            overrides=overrides,
            pit_loss_seconds=pit_loss_seconds,
            num_simulations=num_simulations,
            name=name,
            rng=rng,
        )
        for name, overrides in scenarios
    ]
