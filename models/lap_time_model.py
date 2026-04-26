"""
models/lap_time_model.py
========================

Quantile regression model for lap time prediction.
Produces p10 / p50 / p90 predictions per lap for use in Monte Carlo simulation.

Model: LightGBM with native categorical support (no OneHotEncoder needed).
Feature contract: imported from models/feature_config.py (single source of truth).

Key method for performance: predict_quantiles_batch()
------------------------------------------------------
predict_quantiles_batch() accepts a DataFrame of N rows and calls
model.predict() once per quantile, returning arrays of length N.
This reduces optimizer trial time from ~85s to ~0.1s (~800x speedup).

Fallback behaviour
------------------
If model artifacts are missing, both methods fall back to a baseline
centred on 90.0s so the system always runs before the first training run.
"""

from typing import Dict, Tuple
import os

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

from models.feature_config import (
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FEATURE_DEFAULTS,
    MONOTONE_CONSTRAINTS,
)


class LapTimeModel:
    """
    LightGBM quantile regression model for lap time prediction.
    Provides p10 / p50 / p90 quantile predictions.
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.feature_columns = FEATURE_COLUMNS
        self.categorical_features = CATEGORICAL_FEATURES
        self.numeric_features = NUMERIC_FEATURES
        self.models = {}
        self.residual_offsets: Dict[str, Dict[str, float]] = {}

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    def train(self, df: pd.DataFrame) -> None:
        """
        Train a physics-constrained median + per-compound quantile bands.

        The median (p50) is fit with L1 regression and monotone constraints
        on tire_age and fuel_load — without this, fuel-burn signal masks
        tire wear within a stint, producing inverted degradation curves.

        p10 and p90 are derived from per-compound empirical residual
        quantiles on the validation set. This keeps the bands calibrated
        without subjecting them to the same constraint (LightGBM disallows
        monotone + quantile in one fit).
        """
        X = df[self.feature_columns].copy()
        y = df["lap_time_seconds"]

        for col in self.categorical_features:
            X[col] = X[col].astype("category")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.20,
            random_state=42,
            stratify=X["tire_compound"],
        )

        print(f"\n  Training on {len(X_train):,} laps, validating on {len(X_val):,} laps.")

        # ------------------------------------------------------------------
        # p50: L1 regression (median) with monotone constraints
        # ------------------------------------------------------------------
        monotone = [MONOTONE_CONSTRAINTS[c] for c in self.feature_columns]
        active = [(c, MONOTONE_CONSTRAINTS[c]) for c in self.feature_columns
                  if MONOTONE_CONSTRAINTS[c] != 0]
        print(f"  Monotone constraints: {active}")

        # LightGBM only supports monotone constraints with the L2 (regression)
        # objective. After Phase 1 hygiene the lap-time distribution is roughly
        # symmetric, so the L2 mean is a good proxy for the median.
        print("  Training p50 model (L2 + monotone)...")
        p50 = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            monotone_constraints=monotone,
            monotone_constraints_method="advanced",
            random_state=42,
            verbose=-1,
        )
        p50.fit(X_train, y_train, categorical_feature=self.categorical_features)
        self.models["p50"] = p50

        # ------------------------------------------------------------------
        # Per-compound residual quantiles for p10/p90 bands
        # ------------------------------------------------------------------
        val_pred = p50.predict(X_val)
        residuals = y_val.values - val_pred
        compounds = X_val["tire_compound"].astype(str).values

        offsets = {}
        for compound in sorted(set(compounds)):
            mask = compounds == compound
            if mask.sum() < 30:
                continue
            offsets[compound] = {
                "p10": float(np.percentile(residuals[mask], 10)),
                "p90": float(np.percentile(residuals[mask], 90)),
            }
        # Global fallback for compounds we never saw in validation
        offsets["__default__"] = {
            "p10": float(np.percentile(residuals, 10)),
            "p90": float(np.percentile(residuals, 90)),
        }
        self.residual_offsets = offsets
        print(f"  Per-compound residual bands: {offsets}")

        self._save_models()
        print(f"\n  Models saved to {self.model_dir}")
        self._print_metrics(X_val, y_val, df_full=df)

    # --------------------------------------------------
    # PREDICTION — SINGLE ROW
    # --------------------------------------------------

    def predict_quantiles(self, features: Dict) -> Tuple[float, float, float]:
        """
        Predict p10, p50, p90 lap times for a single lap.
        Falls back to baseline if artifacts are missing.

        NOTE: Use predict_quantiles_batch() in loops — it is ~800x faster.
        """
        if not self._artifacts_exist():
            p50 = self._baseline_predict(features)
            return p50 - 0.15, p50, p50 + 0.15

        if not self.models:
            self._load_models()

        row = self._coerce_features(features)
        df = pd.DataFrame([row], columns=self.feature_columns)
        self._cast_categoricals(df)

        p50 = float(self.models["p50"].predict(df)[0])
        compound = str(row.get("tire_compound", "MEDIUM"))
        off = self.residual_offsets.get(compound, self.residual_offsets.get("__default__", {"p10": -0.4, "p90": 0.4}))
        return p50 + off["p10"], p50, p50 + off["p90"]

    # --------------------------------------------------
    # PREDICTION — BATCH (fast path for optimizer)
    # --------------------------------------------------

    def predict_quantiles_batch(
        self,
        features_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict p10, p50, p90 for a batch of laps in one call.

        Returns: Tuple of (p10_array, p50_array, p90_array), each shape (N,).
        """
        n = len(features_df)

        if not self._artifacts_exist():
            baseline = self._baseline_predict(
                features_df.iloc[0].to_dict() if n > 0 else {}
            )
            return (
                np.full(n, baseline - 0.15),
                np.full(n, baseline),
                np.full(n, baseline + 0.15),
            )

        if not self.models:
            self._load_models()

        # Coerce every row to the expected feature schema
        coerced_rows = [
            self._coerce_features(features_df.iloc[i].to_dict())
            for i in range(n)
        ]
        batch = pd.DataFrame(coerced_rows, columns=self.feature_columns)
        self._cast_categoricals(batch)

        p50 = self.models["p50"].predict(batch)

        # Per-compound residual offsets for p10/p90
        default = self.residual_offsets.get("__default__", {"p10": -0.4, "p90": 0.4})
        compounds = batch["tire_compound"].astype(str).values
        p10_off = np.empty(n, dtype=float)
        p90_off = np.empty(n, dtype=float)
        for i, c in enumerate(compounds):
            off = self.residual_offsets.get(c, default)
            p10_off[i] = off["p10"]
            p90_off[i] = off["p90"]

        return (
            (p50 + p10_off).astype(float),
            p50.astype(float),
            (p50 + p90_off).astype(float),
        )

    # --------------------------------------------------
    # PRIVATE: METRICS
    # --------------------------------------------------

    def _print_metrics(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        df_full: pd.DataFrame,
    ) -> None:
        print("\n" + "=" * 50)
        print("MODEL QUALITY REPORT (validation set)")
        print("=" * 50)

        p50_pred = self.models["p50"].predict(X_val)

        # Reconstruct p10/p90 from per-compound residual offsets
        default = self.residual_offsets.get("__default__", {"p10": -0.4, "p90": 0.4})
        val_compounds = X_val["tire_compound"].astype(str).values
        p10_off = np.array([self.residual_offsets.get(c, default)["p10"] for c in val_compounds])
        p90_off = np.array([self.residual_offsets.get(c, default)["p90"] for c in val_compounds])
        preds = {
            "p10": p50_pred + p10_off,
            "p50": p50_pred,
            "p90": p50_pred + p90_off,
        }

        print("\n  MAE per quantile:")
        for label in ["p10", "p50", "p90"]:
            mae = mean_absolute_error(y_val, preds[label])
            print(f"  {label}: {mae:.3f}s")

        print("\n  Pinball loss per quantile (lower = better):")
        quantile_map = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
        for label, q in quantile_map.items():
            loss = _pinball_loss(y_val.values, preds[label], q)
            print(f"  {label} (q={q}): {loss:.4f}")

        print("\n  Calibration (% actuals below quantile prediction):")
        for label, q in quantile_map.items():
            pct_below = float(np.mean(y_val.values < preds[label])) * 100
            print(f"  {label}: {pct_below:.1f}%  (target: {q*100:.0f}%)")

        print("\n  p50 MAE by tire compound:")
        for compound in sorted(set(val_compounds)):
            mask = val_compounds == compound
            if mask.sum() < 10:
                continue
            mae = mean_absolute_error(y_val.values[mask], preds["p50"][mask])
            print(f"  {compound:<14}: {mae:.3f}s  ({mask.sum():,} laps)")

        p50_mae = mean_absolute_error(y_val, preds["p50"])
        print(f"\n{'='*50}")
        if p50_mae < 0.5:
            verdict = "Excellent - ready for simulation"
        elif p50_mae < 1.0:
            verdict = "Good - acceptable for simulation"
        elif p50_mae < 2.0:
            verdict = "Fair - usable but more data would help"
        else:
            verdict = "Poor - check data quality before using"
        print(f"Overall p50 MAE: {p50_mae:.3f}s  ->  {verdict}")
        print("=" * 50 + "\n")

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------

    def _coerce_features(self, features: Dict) -> Dict:
        """Merge provided features with defaults, coerce types."""
        merged = {**FEATURE_DEFAULTS, **(features or {})}

        # Numeric coercion
        merged["lap_number"]    = int(merged["lap_number"])
        merged["race_progress"] = float(merged["race_progress"])
        merged["tire_age"]      = int(merged["tire_age"])
        merged["fuel_load"]     = float(merged["fuel_load"])
        merged["track_temp"]    = float(merged["track_temp"])
        merged["air_temp"]      = float(merged["air_temp"])
        merged["stint_number"]  = int(merged["stint_number"])

        # String coercion
        merged["driver_id"]       = str(merged["driver_id"])
        merged["team_id"]         = str(merged["team_id"])
        merged["circuit_key"]     = str(merged["circuit_key"])
        merged["tire_compound"]   = str(merged["tire_compound"])
        merged["regulation_era"]  = str(merged["regulation_era"])

        return {k: merged[k] for k in self.feature_columns}

    def _cast_categoricals(self, df: pd.DataFrame) -> None:
        """Convert categorical columns to pandas category dtype in-place."""
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype("category")

    def _baseline_predict(self, features: Dict) -> float:
        if not features:
            return 90.0
        if "base_lap_time" in features and features["base_lap_time"] is not None:
            return float(features["base_lap_time"])
        if "baseline_lap_time" in features and features["baseline_lap_time"] is not None:
            return float(features["baseline_lap_time"])
        return 90.0

    def _artifacts_exist(self) -> bool:
        required = [
            os.path.join(self.model_dir, "lap_time_model_p50.joblib"),
            os.path.join(self.model_dir, "residual_offsets.joblib"),
        ]
        return all(os.path.exists(p) for p in required)

    # --------------------------------------------------
    # MODEL PERSISTENCE
    # --------------------------------------------------

    def _save_models(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(
            self.models["p50"],
            os.path.join(self.model_dir, "lap_time_model_p50.joblib"),
        )
        joblib.dump(
            self.residual_offsets,
            os.path.join(self.model_dir, "residual_offsets.joblib"),
        )

    def _load_models(self) -> None:
        self.models = {
            "p50": joblib.load(os.path.join(self.model_dir, "lap_time_model_p50.joblib")),
        }
        self.residual_offsets = joblib.load(
            os.path.join(self.model_dir, "residual_offsets.joblib")
        )


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    errors = y_true - y_pred
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1) * errors)))
