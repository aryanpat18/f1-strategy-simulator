"""
models/simulation_engine.py
============================

Core race simulation engine.

Performance note
----------------
The key bottleneck in v1 was _sample_stint_laps_matrix calling
predict_quantiles() once per tire_age in a Python loop. For a 57-lap
race with 3 quantile models, this meant 171 individual sklearn pipeline
predictions per strategy evaluation — ~85 seconds per Optuna trial.

The fix: LapTimeModel now exposes predict_quantiles_batch() which
accepts a DataFrame of N rows and calls model.predict() once,
returning N predictions in one sklearn forward pass. A 57-lap stint
drops from ~85s to ~0.1s — a ~800x speedup for the optimizer.

Feature contract
----------------
All inference contexts now use the feature_config.py contract:
  circuit_key, team_id, regulation_era (instead of track_id).
Race-level features (lap_number, race_progress, fuel_load, stint_number)
are computed per-lap inside _sample_stint_laps_matrix().
"""

import random
from typing import Dict, Optional

import numpy as np
import pandas as pd

from models.strategy_generator import StrategyGenerator
from models.tire_rules import validate_strategy
from models.feature_config import get_regulation_era


class SimulationEngine:
    """Core race simulation engine."""

    def __init__(
        self,
        lap_time_model=None,
        pit_loss_seconds: float = 22.0,
        lap_variance: float = 0.1,
    ):
        self.lap_time_model = lap_time_model
        self.pit_loss_seconds = pit_loss_seconds
        self.lap_variance = lap_variance

    # ---------------------------------------------------
    # PUBLIC API — MANUAL MODE
    # ---------------------------------------------------

    def simulate_manual_strategy(
        self,
        base_lap_time: float,
        strategy: Dict,
    ) -> float:
        """Simulate a single race for a manually provided strategy."""
        if not validate_strategy(strategy):
            raise ValueError("Invalid strategy provided.")

        total_time = 0.0
        for stint_laps in strategy["stints"]:
            for _ in range(stint_laps):
                total_time += self._sample_lap_time(base_lap_time)
            total_time += self.pit_loss_seconds

        total_time -= self.pit_loss_seconds
        return total_time

    # ---------------------------------------------------
    # PUBLIC API — AUTOMATED MONTE CARLO MODE
    # ---------------------------------------------------

    def simulate_gp_driver(
        self,
        race_laps: int,
        driver_id: str,
        circuit_key: str = "unknown",
        team_id: str = "unknown",
        year: int = 2024,
        num_simulations: int = 500,
        max_strategies: int = 20,
        is_wet_race: bool = False,
    ) -> Dict:
        """Generate valid strategies, run Monte Carlo, return ranked results."""
        generator = StrategyGenerator(race_laps=race_laps, is_wet_race=is_wet_race)
        strategies = generator.generate_strategies(max_stops=2, max_strategies=max_strategies)

        inference_context = {
            "driver_id": driver_id,
            "circuit_key": circuit_key,
            "team_id": team_id,
            "regulation_era": get_regulation_era(year),
        }

        results = []
        for strategy in strategies:
            sim_times = self._simulate_strategy_monte_carlo(
                strategy=strategy,
                race_laps=race_laps,
                num_simulations=num_simulations,
                inference_context=inference_context,
            )
            results.append({
                "strategy": strategy,
                "mean_time": float(np.mean(sim_times)),
                "std_time": float(np.std(sim_times)),
                "p10": float(np.percentile(sim_times, 10)),
                "p50": float(np.percentile(sim_times, 50)),
                "p90": float(np.percentile(sim_times, 90)),
            })

        results.sort(key=lambda x: x["p50"])

        return {
            "driver": driver_id,
            "circuit_key": circuit_key,
            "team_id": team_id,
            "num_simulations": num_simulations,
            "results": results,
            "best_strategy": results[0] if results else None,
        }

    # ---------------------------------------------------
    # INTERNAL MONTE CARLO
    # ---------------------------------------------------

    def _simulate_strategy_monte_carlo(
        self,
        strategy: Dict,
        num_simulations: int,
        race_laps: int = 0,
        inference_context: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Vectorized Monte Carlo simulation for a single strategy.
        Returns np.ndarray of shape (num_simulations,).
        """
        if not validate_strategy(strategy):
            raise ValueError("Invalid strategy provided.")

        context = inference_context or {}
        total_times = np.zeros(num_simulations, dtype=float)
        race_total_laps = race_laps or sum(strategy["stints"])
        current_race_lap = 0

        for stint_idx, stint_laps in enumerate(strategy["stints"]):
            compound = strategy["compounds"][stint_idx]

            laps_matrix = self._sample_stint_laps_matrix(
                compound=compound,
                stint_laps=stint_laps,
                num_simulations=num_simulations,
                stint_number=stint_idx + 1,
                race_total_laps=race_total_laps,
                start_race_lap=current_race_lap,
                inference_context=context,
            )

            current_race_lap += stint_laps
            total_times += laps_matrix.sum(axis=1)
            total_times += self.pit_loss_seconds

        total_times -= self.pit_loss_seconds
        return total_times

    def _sample_stint_laps_matrix(
        self,
        compound: str,
        stint_laps: int,
        num_simulations: int,
        stint_number: int = 1,
        race_total_laps: int = 57,
        start_race_lap: int = 0,
        inference_context: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Build a (num_simulations, stint_laps) matrix of lap times.

        Computes per-lap features: lap_number, race_progress, fuel_load,
        tire_age, stint_number — matching the feature_config.py contract.
        """
        context = inference_context or {}

        # Fallback — no ML model
        if self.lap_time_model is None:
            base = 90.0
            degradation = np.arange(stint_laps, dtype=float) * 0.05
            mean = base + degradation
            noise = np.random.normal(0.0, self.lap_variance, size=(num_simulations, stint_laps))
            return mean + noise

        # Build a DataFrame with one row per lap in the stint
        rows = []
        for age in range(stint_laps):
            race_lap = start_race_lap + age + 1  # 1-indexed
            row = {
                **context,
                "tire_compound": compound,
                "tire_age": age,
                "lap_number": race_lap,
                "race_progress": float(race_lap) / max(race_total_laps, 1),
                "fuel_load": float(max(race_total_laps - race_lap, 0)),
                "stint_number": stint_number,
            }
            rows.append(row)

        batch_df = pd.DataFrame(rows)

        # Use batched prediction if model supports it (preferred)
        if hasattr(self.lap_time_model, "predict_quantiles_batch"):
            p10_arr, p50_arr, p90_arr = self.lap_time_model.predict_quantiles_batch(batch_df)
        else:
            p10_arr = np.empty(stint_laps)
            p50_arr = np.empty(stint_laps)
            p90_arr = np.empty(stint_laps)
            for age in range(stint_laps):
                feat = {k: batch_df.iloc[age][k] for k in batch_df.columns}
                q10, q50, q90 = self.lap_time_model.predict_quantiles(feat)
                p10_arr[age] = q10
                p50_arr[age] = q50
                p90_arr[age] = q90

        # Ensure valid triangular distribution (left <= mode <= right)
        lo = np.minimum(np.minimum(p10_arr, p50_arr), p90_arr)
        hi = np.maximum(np.maximum(p10_arr, p50_arr), p90_arr)
        mode = np.clip(p50_arr, lo, hi)

        return np.random.triangular(
            left=lo,
            mode=mode,
            right=hi,
            size=(num_simulations, stint_laps),
        )

    # ---------------------------------------------------
    # LEGACY HELPERS
    # ---------------------------------------------------

    def _sample_lap_time(self, base_lap_time: float) -> float:
        return random.gauss(base_lap_time, self.lap_variance)

    # ---------------------------------------------------
    # PUBLIC API — DEGRADATION CURVES (lap-by-lap)
    # ---------------------------------------------------

    def predict_degradation_curve(
        self,
        strategy: Dict,
        inference_context: Optional[Dict] = None,
    ) -> Dict:
        """
        Return lap-by-lap p10/p50/p90 predictions for a strategy.

        Returns
        -------
        {
            "laps": [1, 2, ...],
            "p10": [89.1, 89.2, ...],
            "p50": [89.5, 89.6, ...],
            "p90": [90.0, 90.1, ...],
            "compound": ["SOFT", "SOFT", ..., "HARD", ...],
            "stint_number": [1, 1, ..., 2, ...],
            "pit_laps": [20],
        }
        """
        context = inference_context or {}
        race_total_laps = sum(strategy["stints"])
        all_laps = []
        all_p10, all_p50, all_p90 = [], [], []
        all_compounds = []
        all_stint_nums = []
        pit_laps = []

        current_lap = 0
        for stint_idx, stint_laps in enumerate(strategy["stints"]):
            compound = strategy["compounds"][stint_idx]

            if self.lap_time_model is None:
                for age in range(stint_laps):
                    current_lap += 1
                    base = 90.0 + age * 0.05
                    all_laps.append(current_lap)
                    all_p10.append(base - 0.3)
                    all_p50.append(base)
                    all_p90.append(base + 0.3)
                    all_compounds.append(compound)
                    all_stint_nums.append(stint_idx + 1)
            else:
                rows = []
                for age in range(stint_laps):
                    current_lap += 1
                    row = {
                        **context,
                        "tire_compound": compound,
                        "tire_age": age,
                        "lap_number": current_lap,
                        "race_progress": float(current_lap) / max(race_total_laps, 1),
                        "fuel_load": float(max(race_total_laps - current_lap, 0)),
                        "stint_number": stint_idx + 1,
                    }
                    rows.append(row)
                    all_laps.append(current_lap)
                    all_compounds.append(compound)
                    all_stint_nums.append(stint_idx + 1)

                batch_df = pd.DataFrame(rows)
                if hasattr(self.lap_time_model, "predict_quantiles_batch"):
                    p10_arr, p50_arr, p90_arr = self.lap_time_model.predict_quantiles_batch(batch_df)
                else:
                    p10_arr = np.empty(stint_laps)
                    p50_arr = np.empty(stint_laps)
                    p90_arr = np.empty(stint_laps)
                    for i, row in enumerate(rows):
                        q10, q50, q90 = self.lap_time_model.predict_quantiles(row)
                        p10_arr[i], p50_arr[i], p90_arr[i] = q10, q50, q90

                all_p10.extend(p10_arr.tolist())
                all_p50.extend(p50_arr.tolist())
                all_p90.extend(p90_arr.tolist())

            if stint_idx < len(strategy["stints"]) - 1:
                pit_laps.append(current_lap)

        return {
            "laps": all_laps,
            "p10": all_p10,
            "p50": all_p50,
            "p90": all_p90,
            "compound": all_compounds,
            "stint_number": all_stint_nums,
            "pit_laps": pit_laps,
        }

    # ---------------------------------------------------
    # PUBLIC API — SAFETY CAR SCENARIO
    # ---------------------------------------------------

    def simulate_safety_car_scenario(
        self,
        strategy: Dict,
        sc_lap: int,
        num_simulations: int = 300,
        inference_context: Optional[Dict] = None,
        sc_pit_loss_seconds: float = 12.0,
        sc_neutralised_time: float = 95.0,
        sc_laps: int = 3,
    ) -> Dict:
        """
        Compare 'pit under SC' vs 'stay out' for a given strategy and SC lap.

        The SC bunches the field, so pit loss during SC is reduced (~12s vs ~22s)
        because the field is going slowly. Staying out means you keep position
        but your tyres are older; pitting means fresh rubber but reduced pit cost.

        Parameters
        ----------
        strategy : Strategy currently being followed
        sc_lap : Lap on which SC is deployed
        sc_pit_loss_seconds : Effective pit loss under SC (~12s)
        sc_neutralised_time : Lap time during SC laps (slow pace)
        sc_laps : Number of SC laps before restart

        Returns
        -------
        {
            "sc_lap": int,
            "stay_out": {"mean_time": ..., "std_time": ..., "p50": ..., "strategy": ...},
            "pit_under_sc": {"mean_time": ..., "std_time": ..., "p50": ..., "strategy": ...},
            "recommendation": "PIT" or "STAY OUT",
            "time_delta": float  (positive = pit is faster)
        }
        """
        context = inference_context or {}
        race_laps = sum(strategy["stints"])

        # Determine which stint the SC falls in
        lap_counter = 0
        sc_stint_idx = -1
        sc_lap_in_stint = -1
        for idx, stint_len in enumerate(strategy["stints"]):
            if lap_counter + stint_len >= sc_lap:
                sc_stint_idx = idx
                sc_lap_in_stint = sc_lap - lap_counter
                break
            lap_counter += stint_len

        if sc_stint_idx == -1:
            return {"error": f"SC lap {sc_lap} is beyond race distance {race_laps}"}

        # --- OPTION A: STAY OUT (keep current strategy, add SC neutralised laps) ---
        stay_out_times = self._simulate_strategy_monte_carlo(
            strategy=strategy,
            race_laps=race_laps,
            num_simulations=num_simulations,
            inference_context=context,
        )

        # --- OPTION B: PIT UNDER SC ---
        # Build a new strategy: pit at sc_lap, put on fresh compound
        current_compound = strategy["compounds"][sc_stint_idx]
        # Choose best compound for remaining stint
        remaining_laps = race_laps - sc_lap
        fresh_compound = "HARD" if remaining_laps > 20 else "MEDIUM"

        # New strategy: everything before sc_lap stays, then fresh stint
        pit_stints = []
        pit_compounds = []
        laps_before_sc = 0
        for idx in range(sc_stint_idx):
            pit_stints.append(strategy["stints"][idx])
            pit_compounds.append(strategy["compounds"][idx])
            laps_before_sc += strategy["stints"][idx]

        # Partial current stint up to SC
        partial_stint = sc_lap - laps_before_sc
        if partial_stint > 0:
            pit_stints.append(partial_stint)
            pit_compounds.append(current_compound)

        # Fresh stint for remaining laps
        if remaining_laps > 0:
            pit_stints.append(remaining_laps)
            pit_compounds.append(fresh_compound)

        pit_strategy = {
            "stints": pit_stints,
            "compounds": pit_compounds,
            "num_stops": len(pit_stints) - 1,
        }

        # Simulate with reduced pit loss for the SC stop
        original_pit_loss = self.pit_loss_seconds
        self.pit_loss_seconds = sc_pit_loss_seconds
        pit_times = self._simulate_strategy_monte_carlo(
            strategy=pit_strategy,
            race_laps=race_laps,
            num_simulations=num_simulations,
            inference_context=context,
        )
        self.pit_loss_seconds = original_pit_loss

        stay_out_mean = float(np.mean(stay_out_times))
        pit_mean = float(np.mean(pit_times))
        delta = stay_out_mean - pit_mean  # positive = pit is faster

        return {
            "sc_lap": sc_lap,
            "stay_out": {
                "mean_time": stay_out_mean,
                "std_time": float(np.std(stay_out_times)),
                "p50": float(np.percentile(stay_out_times, 50)),
                "strategy": strategy,
            },
            "pit_under_sc": {
                "mean_time": pit_mean,
                "std_time": float(np.std(pit_times)),
                "p50": float(np.percentile(pit_times, 50)),
                "strategy": pit_strategy,
            },
            "recommendation": "PIT" if delta > 0 else "STAY OUT",
            "time_delta": round(delta, 2),
        }