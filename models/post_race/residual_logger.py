from typing import Dict
from sqlalchemy import text
from sqlalchemy.engine import Engine


class ResidualLogger:
    """
    Persists model prediction errors to the database
    for post-race learning and bias analysis.
    """

    def __init__(self, engine: Engine):
        self.engine = engine

    def log_residual(
        self,
        race_id: str,
        driver_id: str,
        lap_number: int,
        predicted_time: float,
        actual_time: float,
    ) -> None:
        error = actual_time - predicted_time

        query = text(
            """
            INSERT INTO model_residuals (
                race_id,
                driver_id,
                lap_number,
                predicted_time,
                actual_time,
                error
            )
            VALUES (
                :race_id,
                :driver_id,
                :lap_number,
                :predicted_time,
                :actual_time,
                :error
            )
            """
        )

        with self.engine.begin() as conn:
            conn.execute(
                query,
                {
                    "race_id": race_id,
                    "driver_id": driver_id,
                    "lap_number": lap_number,
                    "predicted_time": predicted_time,
                    "actual_time": actual_time,
                    "error": error,
                },
            )
