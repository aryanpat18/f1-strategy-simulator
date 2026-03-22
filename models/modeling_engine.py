"""
models/modeling_engine.py
==========================

End-to-end intelligence orchestrator.
"""

from typing import Dict, Optional

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import text

from models.lap_time_model import LapTimeModel
from models.simulation_engine import SimulationEngine
from models.model_config import SimulationConfig, ModelConfig
from models.post_race.residual_logger import ResidualLogger
from models.optimization.strategy_optimizer import StrategyOptimizer, get_pit_loss_for_event
from models.optimization.optimizer_config import OptimizerConfig
from models.feature_config import get_regulation_era


class ModelingEngine:
    """End-to-end intelligence orchestrator."""

    def __init__(
        self,
        model_config: ModelConfig,
        simulation_config: SimulationConfig,
        optimizer_config: Optional[OptimizerConfig] = None,
        db_engine: Optional[Engine] = None,
    ):
        self.lap_time_model = LapTimeModel(
            model_dir=model_config.model_dir
        )

        self.simulation_config = simulation_config

        self.simulation_engine = SimulationEngine(
            lap_time_model=self.lap_time_model,
            pit_loss_seconds=simulation_config.pit_loss_seconds,
            lap_variance=simulation_config.lap_variance,
        )

        self.optimizer_config = optimizer_config
        self.db_engine = db_engine

        self.residual_logger = (
            ResidualLogger(db_engine) if db_engine is not None else None
        )

    # --------------------------------------------------
    # SIMULATION
    # --------------------------------------------------

    def simulate_driver_gp(
        self,
        year: int,
        round_number: int,
        driver_id: str,
        race_laps: int,
        num_simulations: int,
        is_wet_race: bool,
        circuit_key: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Dict:
        # Look up circuit_key from DB if not provided
        if circuit_key is None:
            circuit_key = self._lookup_circuit_key(year, round_number)
        if team_id is None:
            team_id = self._lookup_team_for_driver(year, round_number, driver_id)

        return self.simulation_engine.simulate_gp_driver(
            race_laps=race_laps,
            driver_id=driver_id,
            circuit_key=circuit_key,
            team_id=team_id,
            year=year,
            num_simulations=num_simulations,
            is_wet_race=is_wet_race,
        )

    def simulate_manual(
        self,
        base_lap_time: float,
        strategy: Dict,
    ) -> float:
        return self.simulation_engine.simulate_manual_strategy(
            base_lap_time=base_lap_time,
            strategy=strategy,
        )

    # --------------------------------------------------
    # STRATEGY OPTIMIZATION
    # --------------------------------------------------

    def optimize_strategy(
        self,
        race_laps: int,
        is_wet_race: bool,
        risk_penalty: Optional[float] = None,
        num_simulations: Optional[int] = None,
        circuit_key: Optional[str] = None,
        team_id: Optional[str] = None,
        driver_id: Optional[str] = None,
        year: int = 2024,
        event_name: Optional[str] = None,
    ) -> Dict:
        """
        Run Bayesian strategy optimization.

        When circuit_key / team_id / driver_id are provided, the optimizer
        conditions lap time predictions on that specific context.

        When event_name is provided, the track-specific pit loss constant
        is looked up from the DB rather than using the global default.
        """
        if self.optimizer_config is None:
            raise RuntimeError("OptimizerConfig not provided.")

        effective_risk_penalty = (
            risk_penalty if risk_penalty is not None
            else self.optimizer_config.risk_penalty
        )
        effective_num_simulations = (
            num_simulations if num_simulations is not None
            else self.optimizer_config.num_simulations
        )

        # Look up track-specific pit loss if event_name provided
        pit_loss = get_pit_loss_for_event(
            event_name=event_name or "",
            db_engine=self.db_engine,
            default=self.simulation_config.pit_loss_seconds,
        )

        if pit_loss != self.simulation_config.pit_loss_seconds:
            sim_engine = SimulationEngine(
                lap_time_model=self.lap_time_model,
                pit_loss_seconds=pit_loss,
                lap_variance=self.simulation_config.lap_variance,
            )
        else:
            sim_engine = self.simulation_engine

        # Build inference context using new feature contract
        inference_context: Dict = {}
        if circuit_key is not None:
            inference_context["circuit_key"] = circuit_key
        if team_id is not None:
            inference_context["team_id"] = team_id
        if driver_id is not None:
            inference_context["driver_id"] = driver_id
        inference_context["regulation_era"] = get_regulation_era(year)

        optimizer = StrategyOptimizer(
            simulation_engine=sim_engine,
            race_laps=race_laps,
            is_wet_race=is_wet_race,
            risk_penalty=effective_risk_penalty,
            num_simulations=effective_num_simulations,
            max_strategies=self.optimizer_config.max_strategies,
            max_stops=self.optimizer_config.max_stops,
            inference_context=inference_context,
        )

        result = optimizer.optimize(n_trials=self.optimizer_config.n_trials)
        result["pit_loss_used"] = pit_loss
        return result

    # --------------------------------------------------
    # DB LOOKUPS
    # --------------------------------------------------

    def _lookup_circuit_key(self, year: int, round_number: int) -> str:
        """Look up circuit_key from races table, fallback to 'unknown'."""
        if self.db_engine is None:
            return "unknown"
        try:
            with self.db_engine.connect() as conn:
                row = conn.execute(
                    text("SELECT circuit_key FROM races WHERE year = :y AND round = :r"),
                    {"y": year, "r": round_number},
                ).fetchone()
            return row[0] if row and row[0] else "unknown"
        except Exception:
            return "unknown"

    def _lookup_team_for_driver(
        self, year: int, round_number: int, driver_id: str
    ) -> str:
        """Look up the most recent team for a driver, fallback to 'unknown'."""
        if self.db_engine is None:
            return "unknown"
        try:
            with self.db_engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT l.team FROM laps l "
                        "JOIN races r ON r.id = l.race_id "
                        "WHERE r.year = :y AND r.round = :r "
                        "AND l.driver_code = :d AND l.team IS NOT NULL "
                        "LIMIT 1"
                    ),
                    {"y": year, "r": round_number, "d": driver_id},
                ).fetchone()
            return row[0] if row and row[0] else "unknown"
        except Exception:
            return "unknown"

    # --------------------------------------------------
    # POST-RACE LEARNING
    # --------------------------------------------------

    def log_post_race_lap(
        self,
        race_id: str,
        driver_id: str,
        lap_number: int,
        predicted_time: float,
        actual_time: float,
    ) -> None:
        if self.residual_logger is None:
            return

        self.residual_logger.log_residual(
            race_id=race_id,
            driver_id=driver_id,
            lap_number=lap_number,
            predicted_time=predicted_time,
            actual_time=actual_time,
        )

    # --------------------------------------------------
    # MODEL TRAINING
    # --------------------------------------------------

    def train_and_save_all_models(self) -> None:
        from models.train_lap_time_model import build_training_df

        if self.residual_logger is None:
            raise ValueError(
                "ModelingEngine initialized without db_engine. "
                "A database connection is required for training."
            )

        df = build_training_df(self.residual_logger.engine)
        self.lap_time_model.train(df)