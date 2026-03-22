"""
F1 Strategy Platform — Season Backfill Script
==============================================

Ingests one or more full F1 seasons into the database, then trains the
LapTimeModel on all ingested data.

Usage
-----
# Ingest default range (2020–2024):
    python pipelines/backfill_seasons.py

# Ingest specific years (covers Phase 1e — 2025 re-fetch + 2026 AUS):
    python pipelines/backfill_seasons.py --years 2025 2026

# Re-ingest years even if already present in the DB:
    python pipelines/backfill_seasons.py --years 2024 --force

# Preview the work list without touching the DB or FastF1:
    python pipelines/backfill_seasons.py --dry-run

# Skip model training at the end (data only):
    python pipelines/backfill_seasons.py --no-train

Run inside the Airflow container:
    docker exec -it <airflow-worker> python /opt/airflow/pipelines/backfill_seasons.py

Environment variables (same as the main pipeline):
    DATABASE_URL          — required
    FASTF1_CACHE_PATH     — default: /opt/airflow/cache
    MODEL_DIR             — default: /opt/airflow/models/artifacts
    PIT_LOSS_SECONDS      — default: 22.0
    LAP_VARIANCE          — default: 0.1
    DEFAULT_FUEL_LOAD     — default: 100.0
    OPT_NUM_SIMULATIONS   — default: 300
    OPT_RISK_PENALTY      — default: 1.0
    OPT_N_TRIALS          — default: 40
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Set, Tuple

import fastf1
from sqlalchemy import create_engine

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from db.database import SessionLocal, Race, init_db
from pipelines.f1_ingestion_pipeline import validate_and_ingest_f1_data
from models.modeling_engine import ModelingEngine
from models.model_config import SimulationConfig, ModelConfig
from models.optimization.optimizer_config import OptimizerConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_YEARS = [2020, 2021, 2022, 2023, 2024]
CACHE_PATH = os.getenv("FASTF1_CACHE_PATH", "/opt/airflow/cache")
MODEL_DIR = os.getenv("MODEL_DIR", "/opt/airflow/models/artifacts")

# Seconds to wait between rounds to avoid hammering the FastF1 API.
# FastF1 uses a local cache, so most requests hit disk — but the first
# fetch for uncached sessions goes to the Ergast/F1 API.
INTER_ROUND_SLEEP_SECONDS = 45


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


@dataclass
class RoundResult:
    year: int
    round_num: int
    event_name: str = ""
    status: str = "pending"   # pending | skipped | success | failed
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class BackfillSummary:
    total: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    results: List[RoundResult] = field(default_factory=list)

    def print_report(self) -> None:
        print("\n" + "=" * 60)
        print("BACKFILL SUMMARY")
        print("=" * 60)
        print(f"  Total rounds attempted : {self.total}")
        print(f"  ✅ Succeeded           : {self.succeeded}")
        print(f"  ⏭️  Skipped (already in DB): {self.skipped}")
        print(f"  ❌ Failed              : {self.failed}")

        if self.failed > 0:
            print("\nFailed rounds:")
            for r in self.results:
                if r.status == "failed":
                    print(f"  {r.year} R{r.round_num:02d} ({r.event_name}): {r.error}")

        if self.succeeded > 0:
            total_time = sum(r.duration_seconds for r in self.results if r.status == "success")
            print(f"\nTotal ingestion time: {total_time:.0f}s")

        print("=" * 60)


# ---------------------------------------------------------------------------
# Work list construction
# ---------------------------------------------------------------------------


def _get_already_ingested(db) -> Set[Tuple[int, int]]:
    """
    Return the set of (year, round) pairs already present in the DB.
    Used to skip rounds on non-forced runs.
    """
    rows = db.query(Race.year, Race.round).all()
    return {(r.year, r.round) for r in rows}


def _build_work_list(
    years: List[int],
    already_ingested: Set[Tuple[int, int]],
    force: bool,
) -> List[RoundResult]:
    """
    For each year, fetch the FastF1 schedule and build a list of
    (year, round) pairs to process.

    Rules:
    - Only include rounds where EventDate is in the past (completed races).
    - Exclude pre-season testing sessions.
    - Skip rounds already in the DB unless --force is set.
    - For the current year (or future years), only include completed rounds.
    """
    fastf1.Cache.enable_cache(CACHE_PATH)
    work_list: List[RoundResult] = []
    today = datetime.now()

    for year in sorted(years):
        print(f"\n📅 Fetching schedule for {year}...")
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"  ⚠️  Could not fetch schedule for {year}: {e} — skipping year.")
            continue

        # Filter: completed rounds only, no testing
        completed = schedule[
            (schedule["EventDate"] < today) &
            (schedule["EventFormat"] != "testing")
        ]

        year_rounds = completed["RoundNumber"].tolist()
        year_names = dict(zip(
            completed["RoundNumber"].tolist(),
            completed["EventName"].tolist(),
        ))

        print(f"  Found {len(year_rounds)} completed rounds: {year_rounds}")

        for round_num in sorted(year_rounds):
            result = RoundResult(
                year=year,
                round_num=round_num,
                event_name=year_names.get(round_num, ""),
            )

            if (year, round_num) in already_ingested and not force:
                result.status = "skipped"

            work_list.append(result)

    return work_list


# ---------------------------------------------------------------------------
# Core backfill loop
# ---------------------------------------------------------------------------


def run_backfill(
    years: List[int],
    force: bool = False,
    dry_run: bool = False,
    no_train: bool = False,
) -> BackfillSummary:
    """
    Main backfill entry point.

    Parameters
    ----------
    years:    List of season years to ingest.
    force:    Re-ingest rounds already present in the DB.
    dry_run:  Print the work list and exit without any DB or FastF1 calls.
    no_train: Skip model training at the end.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise EnvironmentError(
            "DATABASE_URL is not set. "
            "Export it before running: export DATABASE_URL=postgresql://..."
        )

    init_db()
    db = SessionLocal()

    try:
        already_ingested = _get_already_ingested(db)
    finally:
        db.close()

    print(f"\n🗃️  Already ingested: {len(already_ingested)} rounds in DB")

    work_list = _build_work_list(years, already_ingested, force)

    pending = [r for r in work_list if r.status == "pending"]
    skipped = [r for r in work_list if r.status == "skipped"]

    print(f"\n📋 Work list: {len(pending)} to ingest, {len(skipped)} to skip")

    if dry_run:
        print("\n[DRY RUN] Would ingest:")
        for r in pending:
            print(f"  {r.year} R{r.round_num:02d} — {r.event_name}")
        print("\n[DRY RUN] Would skip:")
        for r in skipped:
            print(f"  {r.year} R{r.round_num:02d} — {r.event_name}")
        summary = BackfillSummary(
            total=len(work_list),
            succeeded=0,
            skipped=len(skipped),
            failed=0,
            results=work_list,
        )
        summary.print_report()
        return summary

    # ------------------------------------------------------------------
    # Main ingestion loop
    # ------------------------------------------------------------------
    for idx, result in enumerate(pending, start=1):
        print(f"\n[{idx}/{len(pending)}] {result.year} R{result.round_num:02d} — {result.event_name}")

        start = time.time()
        try:
            # Call the same function Airflow calls — zero duplication.
            validate_and_ingest_f1_data(
                params={"year": result.year, "round": result.round_num}
            )
            result.status = "success"
            result.duration_seconds = time.time() - start
            print(f"  ✅ Done in {result.duration_seconds:.1f}s")

        except Exception as e:
            result.status = "failed"
            result.duration_seconds = time.time() - start
            result.error = str(e)
            print(f"  ❌ FAILED: {e}")
            # Continue — one bad round should not abort the backfill.

        if idx < len(pending):
            time.sleep(INTER_ROUND_SLEEP_SECONDS)

    # ------------------------------------------------------------------
    # Train once after all data is loaded
    # ------------------------------------------------------------------
    succeeded = [r for r in pending if r.status == "success"]

    if not no_train and succeeded:
        print(f"\n🧠 Training LapTimeModel on {len(succeeded)} newly ingested rounds...")
        try:
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
                    n_trials=int(os.getenv("OPT_N_TRIALS", 40)),
                ),
                db_engine=db_engine,
            )
            modeling_engine.train_and_save_all_models()
            print("✅ Model training complete.")
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            print("   Data is intact — re-run training separately if needed:")
            print("   python -c \"from models.train_lap_time_model import main; main()\"")
    elif no_train:
        print("\n⏭️  Skipping model training (--no-train flag set).")
    else:
        print("\n⚠️  No rounds were successfully ingested — skipping model training.")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    all_results = skipped + pending  # skipped rounds included for completeness
    summary = BackfillSummary(
        total=len(work_list),
        succeeded=len(succeeded),
        skipped=len(skipped),
        failed=len([r for r in pending if r.status == "failed"]),
        results=all_results,
    )
    summary.print_report()
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill F1 race data into the strategy platform database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
        metavar="YEAR",
        help=f"Season years to ingest. Defaults to {DEFAULT_YEARS}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest rounds already present in the database.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the work list without touching the DB or fetching data.",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip model training after ingestion.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print("🏎️  F1 Strategy Platform — Season Backfill")
    print(f"   Years    : {args.years}")
    print(f"   Force    : {args.force}")
    print(f"   Dry run  : {args.dry_run}")
    print(f"   No train : {args.no_train}")

    summary = run_backfill(
        years=args.years,
        force=args.force,
        dry_run=args.dry_run,
        no_train=args.no_train,
    )

    # Exit with non-zero code if any rounds failed — useful for CI/alerting.
    sys.exit(1 if summary.failed > 0 else 0)