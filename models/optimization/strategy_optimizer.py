"""
models/optimization/strategy_optimizer.py
==========================================

Prescriptive optimization layer.

Finds the optimal race strategy under uncertainty using Bayesian
optimization (Optuna) over a candidate strategy pool.

Key improvements over v1:
  - Uses vectorized _simulate_strategy_monte_carlo() — 10-50x faster
  - Accepts inference_context (track_id, driver_id) so predictions
    are conditioned on the specific race, not a generic unknown track
  - Track-specific pit loss via get_pit_loss_for_event()
  - Candidate pool size configurable (default 100 vs old 25)
  - Evaluation cache — same strategy never simulated twice
  - Returns full ranked list of evaluated strategies
  - Optuna logging suppressed for clean Airflow logs
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import optuna

from models.simulation_engine import SimulationEngine
from models.strategy_generator import StrategyGenerator
from models.tire_rules import validate_strategy

optuna.logging.set_verbosity(optuna.logging.WARNING)


class StrategyOptimizer:
    """
    Finds the risk-aware optimal race strategy using Bayesian
    optimization over a Monte Carlo simulation engine.

    Parameters
    ----------
    simulation_engine : SimulationEngine
    race_laps : int
    is_wet_race : bool
    risk_penalty : float
        Objective = mean_time + risk_penalty * std_time
    num_simulations : int
        Monte Carlo runs per trial
    max_strategies : int
        Candidate pool size
    max_stops : int
        Max pit stops to consider
    inference_context : dict, optional
        Extra features merged into every predict_quantiles call.
        Use to condition predictions on a specific track and driver:
        {"track_id": "2024_1", "driver_id": "HAM"}
    """

    def __init__(
        self,
        simulation_engine: SimulationEngine,
        race_laps: int,
        is_wet_race: bool,
        risk_penalty: float,
        num_simulations: int,
        max_strategies: int = 100,
        max_stops: int = 2,
        inference_context: Optional[Dict] = None,
    ):
        self.simulation_engine = simulation_engine
        self.race_laps = race_laps
        self.is_wet_race = is_wet_race
        self.risk_penalty = risk_penalty
        self.num_simulations = num_simulations
        self.inference_context = inference_context or {}

        generator = StrategyGenerator(
            race_laps=race_laps,
            is_wet_race=is_wet_race,
        )
        self.candidate_strategies = generator.generate_strategies(
            max_stops=max_stops,
            max_strategies=max_strategies,
        )

        if not self.candidate_strategies:
            raise ValueError(
                f"No valid strategies generated for {race_laps} laps "
                f"(is_wet_race={is_wet_race})."
            )

        context_str = (
            f"circuit={self.inference_context.get('circuit_key', 'unknown')} "
            f"team={self.inference_context.get('team_id', 'unknown')} "
            f"driver={self.inference_context.get('driver_id', 'unknown')}"
        )
        print(
            f"🎯 Optimizer: {len(self.candidate_strategies)} candidates, "
            f"{num_simulations} sims/trial, risk_penalty={risk_penalty}, "
            f"{context_str}"
        )

        self._evaluation_cache: Dict[int, Dict] = {}

    # --------------------------------------------------
    # OPTUNA OBJECTIVE
    # --------------------------------------------------

    def _objective(self, trial: optuna.Trial) -> float:
        idx = trial.suggest_int("strategy_idx", 0, len(self.candidate_strategies) - 1)
        strategy = self.candidate_strategies[idx]

        if not validate_strategy(strategy):
            return float("inf")

        if idx in self._evaluation_cache:
            cached = self._evaluation_cache[idx]
            return cached["mean_time"] + self.risk_penalty * cached["std_time"]

        # Vectorized Monte Carlo with inference context
        sim_times = self.simulation_engine._simulate_strategy_monte_carlo(
            strategy=strategy,
            race_laps=self.race_laps,
            num_simulations=self.num_simulations,
            inference_context=self.inference_context,
        )

        mean_time = float(np.mean(sim_times))
        std_time = float(np.std(sim_times))
        p10 = float(np.percentile(sim_times, 10))
        p50 = float(np.percentile(sim_times, 50))
        p90 = float(np.percentile(sim_times, 90))

        self._evaluation_cache[idx] = {
            "strategy": strategy,
            "mean_time": mean_time,
            "std_time": std_time,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "objective": mean_time + self.risk_penalty * std_time,
        }

        return mean_time + self.risk_penalty * std_time

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def optimize(self, n_trials: int = 60) -> Dict:
        """Run Bayesian optimization and return ranked results."""
        print(f"🔍 Running {n_trials} optimization trials...")

        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=n_trials)

        best_idx = study.best_params["strategy_idx"]
        best_strategy = self.candidate_strategies[best_idx]
        best_cache = self._evaluation_cache.get(best_idx, {})

        all_evaluated = sorted(
            self._evaluation_cache.values(),
            key=lambda x: x["objective"],
        )

        print(
            f"✅ Optimization complete. "
            f"Best: {'-'.join(best_strategy['compounds'])} "
            f"({best_strategy['num_stops']} stop{'s' if best_strategy['num_stops'] > 1 else ''}) "
            f"— {best_cache.get('mean_time', 0):.1f}s ± {best_cache.get('std_time', 0):.1f}s"
        )

        return {
            "best_strategy": best_strategy,
            "expected_time": study.best_value,
            "mean_time": best_cache.get("mean_time", study.best_value),
            "std_time": best_cache.get("std_time", 0.0),
            "p10": best_cache.get("p10", 0.0),
            "p50": best_cache.get("p50", 0.0),
            "p90": best_cache.get("p90", 0.0),
            "risk_penalty": self.risk_penalty,
            "trials": n_trials,
            "strategies_evaluated": len(self._evaluation_cache),
            "all_evaluated": all_evaluated,
        }


# --------------------------------------------------
# Pit loss DB lookup
# --------------------------------------------------


def get_pit_loss_for_event(
    event_name: str,
    db_engine,
    default: float = 22.0,
) -> float:
    """
    Look up track-specific pit loss from track_metrics table.
    Falls back to default if not found or DB unavailable.
    """
    if not event_name or db_engine is None:
        return default

    try:
        import pandas as pd
        result = pd.read_sql(
            "SELECT avg_pit_loss FROM track_metrics WHERE event_name = %(name)s",
            db_engine,
            params={"name": event_name},
        )
        if not result.empty and result["avg_pit_loss"].notna().any():
            pit_loss = float(result["avg_pit_loss"].iloc[0])
            print(f"  ↳ Pit loss for '{event_name}': {pit_loss:.2f}s (from DB)")
            return pit_loss
    except Exception as e:
        print(f"  ↳ Could not read pit loss from DB ({e}) — using default {default}s")

    return default