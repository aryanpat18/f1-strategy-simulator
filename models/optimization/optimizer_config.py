"""
models/optimization/optimizer_config.py
========================================

Configuration dataclass for the StrategyOptimizer.

Fields
------
num_simulations : int
    Monte Carlo simulations run per Optuna trial to estimate the
    distribution of race outcomes for a given strategy.
    Higher = more accurate but slower. 200-500 is a good range.

risk_penalty : float
    Weight applied to the standard deviation of race times in the
    objective function:  objective = mean_time + risk_penalty * std_time
    0.0 = pure time optimisation (aggressive)
    1.0 = balanced time + consistency (default)
    2.0+ = heavily penalise variance (conservative)

n_trials : int
    Number of Optuna trials (strategy evaluations) to run.
    Each trial samples a strategy from the candidate pool and
    evaluates it with num_simulations Monte Carlo runs.
    40-100 is a good range for the candidate pool sizes we use.

max_strategies : int
    Maximum number of candidate strategies to generate for the
    optimizer to search over. Larger = richer search space but
    slightly more memory. 50-150 is a good range.

max_stops : int
    Maximum number of pit stops to consider when generating
    candidate strategies. 2 covers the vast majority of F1 races.
    Set to 3 for street circuits with high degradation.
"""

from dataclasses import dataclass, field


@dataclass
class OptimizerConfig:
    num_simulations: int = 300
    risk_penalty: float = 1.0
    n_trials: int = 60
    max_strategies: int = 100
    max_stops: int = 2