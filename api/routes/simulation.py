from fastapi import APIRouter, Depends

from api.schemas import (
    AutoSimulationRequest,
    ManualSimulationRequest,
    StrategyOptimizationRequest,
    AutoSimulationResponse,
    OptimizationResponse,
    OptimizationStrategyResult,
    SimulationResult,
    StrategyResult,
    DegradationCurveRequest,
    DegradationCurveResponse,
    SafetyCarRequest,
    SafetyCarResponse,
    SafetyCarOptionResult,
)
from models.modeling_engine import ModelingEngine


router = APIRouter(prefix="/simulate", tags=["Simulation"])


def get_engine() -> ModelingEngine:
    from api.deps import get_engine as deps_get_engine
    return deps_get_engine()


@router.post("/auto", response_model=AutoSimulationResponse)
def run_auto_simulation(
    request: AutoSimulationRequest,
    engine: ModelingEngine = Depends(get_engine),
):
    result = engine.simulate_driver_gp(
        year=request.year,
        round_number=request.round,
        driver_id=request.driver_id,
        race_laps=request.race_laps,
        num_simulations=request.num_simulations,
        is_wet_race=request.is_wet_race,
        circuit_key=request.circuit_key,
        team_id=request.team_id,
    )

    formatted_results = [
        SimulationResult(
            strategy=StrategyResult(
                stints=r["strategy"]["stints"],
                compounds=r["strategy"]["compounds"],
                num_stops=r["strategy"]["num_stops"],
            ),
            mean_time=r["mean_time"],
            std_time=r["std_time"],
            p10=r["p10"],
            p50=r["p50"],
            p90=r["p90"],
        )
        for r in result["results"]
    ]

    best = formatted_results[0] if formatted_results else None

    return AutoSimulationResponse(
        driver=result["driver"],
        circuit_key=result["circuit_key"],
        team_id=result["team_id"],
        num_simulations=result["num_simulations"],
        best_strategy=best,
        results=formatted_results,
    )


@router.post("/manual")
def run_manual_simulation(
    request: ManualSimulationRequest,
    engine: ModelingEngine = Depends(get_engine),
):
    strategy = {
        "stints": request.stints,
        "compounds": request.compounds,
    }

    total_time = engine.simulate_manual(
        base_lap_time=request.base_lap_time,
        strategy=strategy,
    )

    return {"total_time": total_time}


@router.post("/optimize", response_model=OptimizationResponse)
def optimize_strategy(
    request: StrategyOptimizationRequest,
    engine: ModelingEngine = Depends(get_engine),
):
    # Look up circuit_key from DB if year/round provided but circuit_key not
    circuit_key = request.circuit_key
    if circuit_key is None and request.year is not None and request.round is not None:
        circuit_key = engine._lookup_circuit_key(request.year, request.round)

    team_id = request.team_id
    if team_id is None and request.driver_id and request.year and request.round:
        team_id = engine._lookup_team_for_driver(request.year, request.round, request.driver_id)

    result = engine.optimize_strategy(
        race_laps=request.race_laps,
        is_wet_race=request.is_wet_race,
        risk_penalty=request.risk_penalty,
        num_simulations=request.num_simulations,
        circuit_key=circuit_key,
        team_id=team_id,
        driver_id=request.driver_id,
        year=request.year or 2024,
        event_name=request.event_name,
    )

    best = result["best_strategy"]

    all_strategies = [
        OptimizationStrategyResult(
            strategy=StrategyResult(
                stints=s["strategy"]["stints"],
                compounds=s["strategy"]["compounds"],
                num_stops=s["strategy"]["num_stops"],
            ),
            mean_time=s["mean_time"],
            std_time=s["std_time"],
            p10=s["p10"],
            p50=s["p50"],
            p90=s["p90"],
            objective_score=s["objective"],
        )
        for s in result.get("all_evaluated", [])
    ]

    return OptimizationResponse(
        best_strategy=StrategyResult(
            stints=best["stints"],
            compounds=best["compounds"],
            num_stops=best["num_stops"],
        ),
        expected_time=result["expected_time"],
        mean_time=result["mean_time"],
        std_time=result["std_time"],
        p10=result["p10"],
        p50=result["p50"],
        p90=result["p90"],
        risk_penalty=result["risk_penalty"],
        trials=result["trials"],
        strategies_evaluated=result["strategies_evaluated"],
        circuit_key=circuit_key,
        team_id=team_id,
        driver_id=request.driver_id,
        pit_loss_used=result.get("pit_loss_used"),
        all_strategies=all_strategies,
    )


@router.post("/degradation", response_model=DegradationCurveResponse)
def get_degradation_curve(
    request: DegradationCurveRequest,
    engine: ModelingEngine = Depends(get_engine),
):
    """
    Return lap-by-lap predicted lap times (p10/p50/p90) for a given strategy.
    Shows tire degradation curves and identifies pit laps.
    """
    from models.feature_config import get_regulation_era

    circuit_key = request.circuit_key or engine._lookup_circuit_key(request.year, request.round)
    team_id = request.team_id or engine._lookup_team_for_driver(request.year, request.round, request.driver_id)

    inference_context = {
        "driver_id": request.driver_id,
        "circuit_key": circuit_key,
        "team_id": team_id,
        "regulation_era": get_regulation_era(request.year),
    }

    strategy = {
        "stints": request.stints,
        "compounds": request.compounds,
        "num_stops": len(request.stints) - 1,
    }

    result = engine.simulation_engine.predict_degradation_curve(
        strategy=strategy,
        inference_context=inference_context,
    )

    return DegradationCurveResponse(**result)


@router.post("/safety-car", response_model=SafetyCarResponse)
def simulate_safety_car(
    request: SafetyCarRequest,
    engine: ModelingEngine = Depends(get_engine),
):
    """
    Compare 'pit under safety car' vs 'stay out' for a given strategy
    and safety car deployment lap.
    """
    from models.feature_config import get_regulation_era

    circuit_key = request.circuit_key or engine._lookup_circuit_key(request.year, request.round)
    team_id = request.team_id or engine._lookup_team_for_driver(request.year, request.round, request.driver_id)

    inference_context = {
        "driver_id": request.driver_id,
        "circuit_key": circuit_key,
        "team_id": team_id,
        "regulation_era": get_regulation_era(request.year),
    }

    strategy = {
        "stints": request.stints,
        "compounds": request.compounds,
        "num_stops": len(request.stints) - 1,
    }

    result = engine.simulation_engine.simulate_safety_car_scenario(
        strategy=strategy,
        sc_lap=request.sc_lap,
        num_simulations=request.num_simulations,
        inference_context=inference_context,
    )

    # Format strategies for response
    stay_out_strat = result["stay_out"]["strategy"]
    pit_strat = result["pit_under_sc"]["strategy"]

    return SafetyCarResponse(
        sc_lap=result["sc_lap"],
        stay_out=SafetyCarOptionResult(
            mean_time=result["stay_out"]["mean_time"],
            std_time=result["stay_out"]["std_time"],
            p50=result["stay_out"]["p50"],
            strategy=StrategyResult(
                stints=stay_out_strat["stints"],
                compounds=stay_out_strat["compounds"],
                num_stops=stay_out_strat["num_stops"],
            ),
        ),
        pit_under_sc=SafetyCarOptionResult(
            mean_time=result["pit_under_sc"]["mean_time"],
            std_time=result["pit_under_sc"]["std_time"],
            p50=result["pit_under_sc"]["p50"],
            strategy=StrategyResult(
                stints=pit_strat["stints"],
                compounds=pit_strat["compounds"],
                num_stops=pit_strat["num_stops"],
            ),
        ),
        recommendation=result["recommendation"],
        time_delta=result["time_delta"],
    )