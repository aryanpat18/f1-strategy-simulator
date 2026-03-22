# F1 Strategy Intelligence Platform — Developer Guide

## Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Streamlit   │────▶│  FastAPI      │────▶│  PostgreSQL     │
│  Dashboard   │     │  API Server   │     │  (f1_strategy_db)│
│  :8501       │     │  :8000        │     │  :5432          │
└─────────────┘     └──────┬───────┘     └─────────────────┘
                           │                      ▲
                    ┌──────▼───────┐               │
                    │  ML Models    │               │
                    │  (LightGBM)   │               │
                    │  + Simulation  │               │
                    │  + Optimizer   │               │
                    └──────────────┘               │
                                                   │
                    ┌──────────────┐               │
                    │  Airflow      │───────────────┘
                    │  (ingestion)  │
                    │  :8080        │
                    └──────────────┘
```

## Quick Start

```bash
docker compose build
docker compose up -d
```

- Dashboard: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Airflow: http://localhost:8080 (airflow/airflow)
- Postgres: localhost:5432 (f1_user/f1_password/f1_strategy_db)

## Current State (March 2026)

### What Works
- Database: 131 races (2020-2025), ~136k laps, all tables created
- API: All endpoints return 200
- Dashboard: All 4 tabs render
- Safety Car simulation: Works with baseline predictions
- Pre-Race Strategy optimization: Works with baseline predictions
- Race Analysis: Shows actual historical lap data
- Season Form: Shows normalized driver performance

### What's Broken — MUST FIX BEFORE REAL USE
**The ML model artifacts were deleted** because the old sklearn models used
an incompatible feature contract (`track_id`, `dirty_air`). The new LightGBM
code is deployed but NO trained model exists. Everything falls back to a
flat 90.0s baseline — predictions are meaningless.

**To fix — run these steps in order:**

1. **Backfill team data** (existing laps have `team = NULL`):
   ```bash
   # Option A: Re-run backfill pipeline through Airflow
   # This re-ingests all data from FastF1, populating team + circuit_key
   docker compose exec airflow airflow dags trigger f1_season_master

   # Option B: Run backfill script directly
   docker compose exec api python pipelines/backfill_seasons.py \
     --years 2022 2023 2024 2025 --force
   ```

2. **Train new LightGBM model:**
   ```bash
   docker compose exec api python -m models.train_lap_time_model
   ```
   This creates 3 files in `/app/models/artifacts/`:
   - `lap_time_model_p10.joblib`
   - `lap_time_model_p50.joblib`
   - `lap_time_model_p90.joblib`

3. **Restart API** to pick up new models:
   ```bash
   docker compose restart api
   ```

4. **Verify** — predictions should now show real degradation curves, not flat lines.

### Database Schema

```sql
-- Core tables
races (id, year, round, event_name, total_laps, circuit_key)
laps (id, race_id FK, driver_code, lap_number, lap_time_seconds, s1, s2, s3, compound, tyre_life, is_pit_out_lap, team)
qualifying_laps (id, race_id FK, session_type, driver_code, lap_number, lap_time_seconds, s1, s2, s3, compound, tyre_life, is_deleted, team)
session_weather (id, race_id FK, session_type, time_offset_seconds, air_temp, track_temp, humidity, pressure, wind_speed, wind_direction, rainfall)

-- Derived/computed tables
track_metrics (id, event_name UNIQUE, avg_pit_loss, fuel_penalty_per_lap)
track_models (id, race_id FK, compound, deg_coefficient, base_pace, mae_error, r2_score)
simulation_results (id, race_id FK, driver_code, strategy_compounds, strategy_stints, mean_time, std_time, p10, p50, p90)
model_residuals (id, race_id, driver_id, lap_number, predicted_time, actual_time, error)
```

All child tables use `ON DELETE CASCADE` from `races.id`.

### Column Status
| Column | Status | Notes |
|--------|--------|-------|
| `races.circuit_key` | Populated | Backfilled from event_name (e.g., "bahrain", "japanese") |
| `laps.team` | **EMPTY** | Needs re-ingestion via backfill pipeline |
| `qualifying_laps.team` | **EMPTY** | Needs re-ingestion via backfill pipeline |

---

## ML Feature Contract

**CRITICAL**: The feature contract is defined ONCE in `models/feature_config.py`.
All 3 consumers import from there. Never define feature lists locally.

```
models/feature_config.py  ←── SINGLE SOURCE OF TRUTH
    ├── models/lap_time_model.py      (imports FEATURE_COLUMNS, CATEGORICAL_FEATURES, etc.)
    ├── models/train_lap_time_model.py (imports FEATURE_COLUMNS, get_regulation_era)
    └── models/simulation_engine.py    (imports get_regulation_era)
```

### 12 Features
| Feature | Type | Source |
|---------|------|--------|
| driver_id | categorical | driver_code from laps |
| team_id | categorical | team from laps |
| circuit_key | categorical | circuit_key from races |
| lap_number | numeric | lap_number from laps |
| race_progress | numeric | lap_number / total_laps (0-1) |
| tire_compound | categorical | compound from laps |
| tire_age | numeric | tyre_life from laps |
| fuel_load | numeric | total_laps - lap_number (proxy) |
| track_temp | numeric | from session_weather |
| air_temp | numeric | from session_weather |
| regulation_era | categorical | derived from year (v8_era/v6_hybrid/ground_effect/2026_pu) |
| stint_number | numeric | inferred from tyre_life resets |

### Model Architecture
- **LightGBM quantile regression** (3 models: p10, p50, p90)
- `LGBMRegressor(objective="quantile", alpha=q, n_estimators=500, max_depth=6, learning_rate=0.05)`
- Native categorical support — no OneHotEncoder needed
- Training data: 2022+ only (ground effect era, avoids distribution shift)
- Fallback: 90.0s baseline if no artifacts exist

### Key Performance Rule
**NEVER call `predict_quantiles()` in loops.** Use `predict_quantiles_batch()` instead.
Batch = 1 LightGBM forward pass for N rows. Loop = N forward passes. ~800x speedup.

---

## File Map

### Models Layer
```
models/
├── feature_config.py          # Feature contract (SINGLE SOURCE OF TRUTH)
├── lap_time_model.py          # LightGBM quantile regression model
├── train_lap_time_model.py    # Training pipeline (SQL + feature engineering)
├── simulation_engine.py       # Monte Carlo race simulation
├── modeling_engine.py         # Orchestrator (train + simulate + optimize)
├── model_config.py            # SimulationConfig, ModelConfig dataclasses
├── strategy_generator.py      # Generates valid race strategies
├── tire_rules.py              # Strategy validation rules
├── optimization/
│   ├── strategy_optimizer.py  # Bayesian optimization (Optuna)
│   └── optimizer_config.py    # OptimizerConfig dataclass
└── post_race/
    └── residual_logger.py     # Prediction error logging
```

### API Layer
```
api/
├── main.py                    # FastAPI app with routers
├── deps.py                    # Dependency injection (ModelingEngine)
├── schemas.py                 # Pydantic request/response models
└── routes/
    ├── simulation.py          # /simulate/auto, /optimize, /degradation, /safety-car
    └── data.py                # /data/races, /drivers, /teams, /race-analysis, /pre-race
```

### Dashboard Layer
```
app/
└── streamlit_app.py           # 4-tab Streamlit dashboard

dashboard/
└── api_client.py              # HTTP client for FastAPI backend
```

### Data Layer
```
db/
├── database.py                # SQLAlchemy ORM models + engine
├── init.sql                   # Schema DDL + migration ALTER TABLEs
└── calculate_metrics.py       # Pit loss calculation

pipelines/
├── f1_ingestion_pipeline.py   # Airflow DAG: ingest + train + auto-simulate
└── backfill_seasons.py        # Bulk historical data ingestion
```

### Docker
```
docker-compose.yaml            # 4 services: db, api, dashboard, airflow
Dockerfile.api                 # Python 3.9 + FastAPI + LightGBM
Dockerfile.dashboard           # Python 3.9 + Streamlit
```

---

## API Endpoints

### Data (GET)
| Endpoint | Description |
|----------|-------------|
| `/data/races?year=2024` | List ingested races |
| `/data/drivers?year=2024` | List driver codes |
| `/data/teams?year=2024` | List team names |
| `/data/tracks` | Pit loss constants per track |
| `/data/race/{year}/{round}` | Race detail + pre-computed simulations |
| `/data/race-analysis/{year}/{round}` | Full lap data + stint breakdown per driver |
| `/data/pre-race/{year}/{event_name}` | Season form (normalized) + track history |

### Simulation (POST)
| Endpoint | Description |
|----------|-------------|
| `/simulate/auto` | Run Monte Carlo for a driver/race |
| `/simulate/optimize` | Bayesian strategy optimization (Optuna) |
| `/simulate/degradation` | Lap-by-lap p10/p50/p90 degradation curve |
| `/simulate/safety-car` | Pit vs stay out under safety car |

All simulation endpoints accept optional `circuit_key` and `team_id` fields.
If omitted, they are auto-looked up from the database.

---

## Airflow Pipelines

### `f1_strategy_production_pipeline` (manual trigger)
Params: `year`, `round`
1. **ingest_and_clean_race_data**: Fetches R + Q + SQ sessions from FastF1, persists laps/weather
2. **run_modeling_and_simulation**: Trains LightGBM, auto-simulates top 5 drivers

### `f1_season_master` (weekly)
Iterates all rounds in a season, triggers production pipeline for each.

### Idempotency
Every `(year, round)` ingestion deletes the existing Race row first.
`ON DELETE CASCADE` clears all child tables atomically.

---

## Environment Variables
| Variable | Default | Used By |
|----------|---------|---------|
| `DATABASE_URL` | `postgresql://f1_user:f1_password@db:5432/f1_strategy_db` | API, Airflow |
| `API_URL` | `http://api:8000` | Dashboard |
| `MODEL_DIR` | `/app/models/artifacts` | API, Airflow |
| `FASTF1_CACHE_PATH` | `/opt/airflow/cache` | Airflow |
| `MIN_TRAINING_YEAR` | `2022` | Training pipeline |
| `PIT_LOSS_SECONDS` | `22.0` | Simulation |
| `LAP_VARIANCE` | `0.1` | Simulation |

---

## Strategy Generator Rules
- Compounds required: Must use at least 2 different dry compounds per race
- Compound max stint limits: SOFT ≤ 20, MEDIUM ≤ 30, HARD ≤ 45, INTER ≤ 30, WET ≤ 25
- Min stint: 5 laps (except last stint)
- Max stops: configurable (default 2)
- Wet races: only INTERMEDIATE and WET compounds

---

## What Changed (ML Upgrade — March 2026)

### Before
- sklearn GradientBoostingRegressor with OneHotEncoder pipeline
- `track_id = "{year}_{round}"` — Silverstone 2024 ≠ Silverstone 2023
- `dirty_air = "clean"` always — wasted feature
- No team/constructor awareness
- 10 features

### After
- LightGBM with native categorical support
- `circuit_key` — stable track identifier across seasons
- `race_progress` — normalized 0-1 (replaces dirty_air)
- `team_id` — constructor awareness
- `regulation_era` — era-aware (future-proofs 2026 regs)
- 12 features defined in single source of truth

### Migration Checklist
- [x] ALTER TABLE races ADD circuit_key — DONE
- [x] ALTER TABLE laps ADD team — DONE
- [x] ALTER TABLE qualifying_laps ADD team — DONE
- [x] Backfill circuit_key from event_name — DONE (131 races)
- [ ] **Backfill team via re-ingestion — NOT DONE (136k laps with team=NULL)**
- [ ] **Train new LightGBM model — NOT DONE (no artifacts exist)**
- [ ] Delete old sklearn artifacts — DONE (already deleted)
