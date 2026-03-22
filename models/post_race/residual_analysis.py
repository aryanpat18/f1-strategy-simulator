import pandas as pd
from sqlalchemy.engine import Engine


class ResidualAnalysis:
    """
    Analyze post-race model errors to detect bias and drift.
    """

    def __init__(self, engine: Engine):
        self.engine = engine

    def load_residuals(self, race_id: str) -> pd.DataFrame:
        query = """
        SELECT *
        FROM model_residuals
        WHERE race_id = %(race_id)s
        """

        return pd.read_sql(query, self.engine, params={"race_id": race_id})

    def summarize_errors(self, race_id: str) -> dict:
        df = self.load_residuals(race_id)

        if df.empty:
            return {}

        return {
            "mean_error": float(df["error"].mean()),
            "median_error": float(df["error"].median()),
            "std_error": float(df["error"].std()),
            "p90_error": float(df["error"].quantile(0.9)),
            "p10_error": float(df["error"].quantile(0.1)),
        }

    def error_by_lap_bucket(self, race_id: str) -> pd.DataFrame:
        df = self.load_residuals(race_id)

        if df.empty:
            return df

        df["lap_bucket"] = pd.cut(
            df["lap_number"],
            bins=[0, 10, 20, 30, 40, 60],
        )

        return (
            df.groupby("lap_bucket")["error"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
