from typing import List, Optional
from pydantic import BaseModel


# ----------------------------
# Simulation Requests
# ----------------------------

class AutoSimulationRequest(BaseModel):
    year: int
    round: int
    driver_id: str
    race_laps: int
    num_simulations: int = 500
    is_wet_race: bool = False
    circuit_key: Optional[str] = None   # auto-looked up from DB if omitted
    team_id: Optional[str] = None       # auto-looked up from DB if omitted


class ManualSimulationRequest(BaseModel):
    base_lap_time: float
    stints: List[int]
    compounds: List[str]


# ----------------------------
# Optimization Requests
# ----------------------------

class StrategyOptimizationRequest(BaseModel):
    race_laps: int
    is_wet_race: bool = False
    risk_penalty: float = 1.0
    num_simulations: int = 300

    # Track + driver context — when provided, the optimizer
    # conditions lap time predictions on this specific track
    # and driver rather than using "unknown" defaults.
    year: Optional[int] = None
    round: Optional[int] = None
    driver_id: Optional[str] = None
    circuit_key: Optional[str] = None   # auto-looked up from DB if omitted
    team_id: Optional[str] = None       # auto-looked up from DB if omitted
    event_name: Optional[str] = None    # used for pit loss DB lookup


# ----------------------------
# Shared sub-models
# ----------------------------

class StrategyResult(BaseModel):
    stints: List[int]
    compounds: List[str]
    num_stops: int


class SimulationResult(BaseModel):
    strategy: StrategyResult
    mean_time: float
    std_time: float
    p10: float
    p50: float
    p90: float


# ----------------------------
# Simulation Responses
# ----------------------------

class AutoSimulationResponse(BaseModel):
    driver: str
    circuit_key: str
    team_id: str
    num_simulations: int
    best_strategy: Optional[SimulationResult]
    results: List[SimulationResult]


# ----------------------------
# Optimization Response
# ----------------------------

class OptimizationStrategyResult(BaseModel):
    strategy: StrategyResult
    mean_time: float
    std_time: float
    p10: float
    p50: float
    p90: float
    objective_score: float


class OptimizationResponse(BaseModel):
    best_strategy: StrategyResult
    expected_time: float
    mean_time: float
    std_time: float
    p10: float
    p50: float
    p90: float
    risk_penalty: float
    trials: int
    strategies_evaluated: int
    circuit_key: Optional[str] = None      # echoed back so caller knows what was used
    team_id: Optional[str] = None
    driver_id: Optional[str] = None
    pit_loss_used: Optional[float] = None
    all_strategies: List[OptimizationStrategyResult]


# ----------------------------
# Degradation Curve
# ----------------------------

class DegradationCurveRequest(BaseModel):
    year: int
    round: int
    driver_id: str
    stints: List[int]
    compounds: List[str]
    circuit_key: Optional[str] = None
    team_id: Optional[str] = None


class DegradationCurveResponse(BaseModel):
    laps: List[int]
    p10: List[float]
    p50: List[float]
    p90: List[float]
    compound: List[str]
    stint_number: List[int]
    pit_laps: List[int]


# ----------------------------
# Safety Car Scenario
# ----------------------------

class SafetyCarRequest(BaseModel):
    year: int
    round: int
    driver_id: str
    race_laps: int
    stints: List[int]
    compounds: List[str]
    sc_lap: int
    num_simulations: int = 300
    event_name: Optional[str] = None
    circuit_key: Optional[str] = None
    team_id: Optional[str] = None


class SafetyCarOptionResult(BaseModel):
    mean_time: float
    std_time: float
    p50: float
    strategy: StrategyResult


class SafetyCarResponse(BaseModel):
    sc_lap: int
    stay_out: SafetyCarOptionResult
    pit_under_sc: SafetyCarOptionResult
    recommendation: str
    time_delta: float


# ----------------------------
# Post-race counterfactual
# ----------------------------

class StrategyOverride(BaseModel):
    """Alternative strategy for one driver in a counterfactual scenario."""
    driver_id: str
    stints: List[int]
    compounds: List[str]


class CounterfactualScenario(BaseModel):
    """One hypothetical world: a name + the driver overrides applied in it.
    Drivers without an override use their *actual* race strategy."""
    name: str
    overrides: List[StrategyOverride] = []


class CounterfactualRequest(BaseModel):
    year: int
    round: int
    drivers: List[str]
    scenarios: List[CounterfactualScenario]
    num_simulations: int = 300
    pit_loss_seconds: Optional[float] = None   # falls back to engine default
    seed: Optional[int] = 42


class DriverScenarioOutcome(BaseModel):
    driver_id: str
    strategy: StrategyResult
    is_override: bool
    actual_total_time: float
    sim_p10: float
    sim_p50: float
    sim_p90: float
    delta_p50: float
    cumulative_time_p50: List[float]


class ScenarioOutcome(BaseModel):
    name: str
    overrides_applied: List[str]
    drivers: List[DriverScenarioOutcome]
    finishing_order_p50: List[str]
    gap_matrix_p50: dict   # {driver_a: {driver_b: gap_seconds}}


class CounterfactualResponse(BaseModel):
    year: int
    round: int
    drivers: List[str]
    pit_loss_used: float
    num_simulations: int
    scenarios: List[ScenarioOutcome]