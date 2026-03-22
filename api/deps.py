import os
from functools import lru_cache
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from models.modeling_engine import ModelingEngine
from models.model_config import SimulationConfig, ModelConfig
from models.optimization.optimizer_config import OptimizerConfig


def _maybe_create_db_engine(database_url: Optional[str]) -> Optional[Engine]:
    """
    Create a SQLAlchemy engine only if DATABASE_URL is provided.
    Keeps the API runnable in simulation-only mode without a DB.
    """
    if not database_url:
        return None
    return create_engine(database_url)


@lru_cache(maxsize=1)
def get_engine() -> ModelingEngine:
    """
    Dependency provider for a shared ModelingEngine instance.
    Cached so it is created once per process.
    """
    database_url = os.getenv("DATABASE_URL")
    model_dir = os.getenv("MODEL_DIR", "/app/models/artifacts")

    os.makedirs(model_dir, exist_ok=True)

    db_engine = _maybe_create_db_engine(database_url)

    modeling_engine = ModelingEngine(
        model_config=ModelConfig(model_dir=model_dir),
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

    return modeling_engine