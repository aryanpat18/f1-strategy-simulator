from dataclasses import dataclass


@dataclass
class SimulationConfig:
    pit_loss_seconds: float
    lap_variance: float
    default_fuel_load: float


@dataclass
class ModelConfig:
    model_dir: str