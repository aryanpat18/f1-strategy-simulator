"""
api/routes/post_race.py
=======================

Post-race endpoints. Today: counterfactual scenarios.
"""

from fastapi import APIRouter, Depends, HTTPException

from api.schemas import (
    CounterfactualRequest,
    CounterfactualResponse,
    ScenarioOutcome,
    DriverScenarioOutcome,
    StrategyResult,
)
from models.modeling_engine import ModelingEngine
from models.post_race.counterfactual import (
    Strategy,
    compare_scenarios,
)


router = APIRouter(prefix="/post-race", tags=["Post-Race"])


def get_engine() -> ModelingEngine:
    from api.deps import get_engine as deps_get_engine
    return deps_get_engine()


@router.post("/counterfactual", response_model=CounterfactualResponse)
def run_counterfactual(
    request: CounterfactualRequest,
    engine: ModelingEngine = Depends(get_engine),
):
    if engine.db_engine is None:
        raise HTTPException(
            status_code=503,
            detail="DATABASE_URL not configured — counterfactuals require the laps DB.",
        )

    pit_loss = (
        request.pit_loss_seconds
        if request.pit_loss_seconds is not None
        else engine.simulation_config.pit_loss_seconds
    )

    scenarios_in = []
    for sc in request.scenarios:
        overrides = {
            ov.driver_id: Strategy(stints=ov.stints, compounds=ov.compounds)
            for ov in sc.overrides
        }
        scenarios_in.append((sc.name, overrides))

    try:
        results = compare_scenarios(
            engine=engine.db_engine,
            lap_time_model=engine.lap_time_model,
            year=request.year,
            round_number=request.round,
            drivers=request.drivers,
            scenarios=scenarios_in,
            pit_loss_seconds=pit_loss,
            num_simulations=request.num_simulations,
            seed=request.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    out_scenarios = []
    for r in results:
        out_scenarios.append(ScenarioOutcome(
            name=r.name,
            overrides_applied=r.overrides_applied,
            drivers=[
                DriverScenarioOutcome(
                    driver_id=d.driver_id,
                    strategy=StrategyResult(
                        stints=d.strategy.stints,
                        compounds=d.strategy.compounds,
                        num_stops=d.strategy.num_stops(),
                    ),
                    is_override=d.is_override,
                    actual_total_time=d.actual_total_time,
                    sim_p10=d.sim_p10,
                    sim_p50=d.sim_p50,
                    sim_p90=d.sim_p90,
                    delta_p50=d.delta_p50,
                    cumulative_time_p50=d.cumulative_time_p50,
                )
                for d in r.drivers
            ],
            finishing_order_p50=r.finishing_order_p50,
            gap_matrix_p50=r.gap_matrix_p50,
        ))

    return CounterfactualResponse(
        year=request.year,
        round=request.round,
        drivers=request.drivers,
        pit_loss_used=pit_loss,
        num_simulations=request.num_simulations,
        scenarios=out_scenarios,
    )
