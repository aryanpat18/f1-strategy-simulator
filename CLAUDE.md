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

## Deployment URLs

| Service | Local (Docker) | Production |
|---------|---------------|------------|
| Dashboard | http://localhost:8501 | https://f1-strategy-simulator-vzczoddjj9lux7t4lstege.streamlit.app |
| API | http://localhost:8000 | https://f1-strategy-simulator-xlev.onrender.com |
| API Docs | http://localhost:8000/docs | https://f1-strategy-simulator-xlev.onrender.com/docs |
| Airflow | http://localhost:8080 | Local only (airflow/airflow) |
| Postgres | localhost:5432 | Render managed (internal) |

## Quick Start (Local Docker)

```bash
docker compose build
docker compose up -d
```

Credentials: f1_user / f1_password / f1_strategy_db

---

## Current State (March 22, 2026)

### What's DONE and WORKING

| Component | Status | Details |
|-----------|--------|---------|
| Database schema | DONE | 8 tables, all columns including `circuit_key`, `team` |
| Data ingestion pipeline | DONE | FastF1 → Postgres, populates team + circuit_key |
| Backfill script | DONE | `pipelines/backfill_seasons.py` with --years, --force, --dry-run |
| Feature config (single source of truth) | DONE | `models/feature_config.py` — 12 features, all consumers import from here |
| LightGBM model code | DONE | `models/lap_time_model.py` — quantile regression p10/p50/p90 |
| Training pipeline | DONE | `models/train_lap_time_model.py` — SQL → feature engineering → train |
| Simulation engine | DONE | `models/simulation_engine.py` — Monte Carlo, degradation curves, safety car |
| Strategy generator | DONE | Compound-specific stint limits (SOFT≤20, MED≤30, HARD≤45) |
| Strategy optimizer | DONE | Bayesian optimization via Optuna |
| Modeling engine orchestrator | DONE | `models/modeling_engine.py` — wires everything together |
| API endpoints (all 9) | DONE | GET: races, drivers, teams, tracks, race-detail, race-analysis, pre-race. POST: simulate, optimize, degradation, safety-car |
| API schemas | DONE | All Pydantic models with optional circuit_key/team_id |
| Dashboard (4 tabs) | DONE | Pre-Race Strategy, Safety Car What-If, Race Analysis, Season Form |
| API client | DONE | `dashboard/api_client.py` — 1:1 with all API endpoints |
| Docker setup | DONE | docker-compose.yaml + Dockerfile.api + Dockerfile.dashboard |
| GitHub repo | DONE | https://github.com/aryanpat18/f1-strategy-simulator |
| Render API deployment | DONE | https://f1-strategy-simulator-xlev.onrender.com |
| Streamlit Cloud deployment | IN PROGRESS | Deploying at streamlit.app |
| **ML model trained (local Docker)** | **DONE** | **Trained on 77k laps, p50 MAE: 0.437s, calibration near-perfect** |

### What's BROKEN / INCOMPLETE

| Issue | Severity | Impact | How to Fix |
|-------|----------|--------|------------|
| **Render DB is empty** | CRITICAL | Production API returns no data, dashboard shows nothing | Need to populate Render Postgres (see "Production Data Setup" below) |
| **No model artifacts on Render** | CRITICAL | Production API falls back to 90.0s flat baseline | Train model on Render after data is loaded |
| **`laps.team` is NULL (local DB)** | HIGH | team_id feature is "unknown" in local model — model works but team-unaware | Re-run backfill with `--force` to re-ingest from FastF1 |
| **Render free tier spins down** | MEDIUM | API takes ~30s to cold-start after inactivity | Expected behavior on free tier |
| **Dashboard "No races found"** | MEDIUM | Shows on Streamlit Cloud because Render DB is empty | Populate Render DB |

### What Was COMPLETED in This Session

1. **Trained LightGBM model** on local Docker (77,027 laps from 90 races, 2022+)
   - p50 MAE: 0.437s (excellent — old sklearn was ~1.0s)
   - Calibration: p10=11.5%, p50=50.6%, p90=88.6% (near-perfect)
   - Real degradation curves now visible (1.87s range across race)
2. **Deployed API to Render** (Dockerfile.api, PostgreSQL, env vars)
3. **Set up Streamlit Cloud** (secrets, requirements-dashboard.txt)
4. **All code uses NEW feature contract** — zero references to old track_id/dirty_air

---

## Production Data Setup (Render)

The Render PostgreSQL database is empty. To populate it:

### Option A: Direct SQL dump from local → Render
```bash
# 1. Dump local Docker Postgres
docker compose exec db pg_dump -U f1_user f1_strategy_db > f1_dump.sql

# 2. Load into Render Postgres (use External Database URL from Render dashboard)
psql "YOUR_RENDER_EXTERNAL_DATABASE_URL" < f1_dump.sql

# 3. Train model on Render (SSH or one-off job — requires paid tier)
# On free tier, model artifacts must be baked into the Docker image
```

### Option B: Bake model artifacts into Docker image
```bash
# 1. Copy trained model from local Docker container
docker compose exec api ls /app/models/artifacts/
# Should show: lap_time_model_p10.joblib, lap_time_model_p50.joblib, lap_time_model_p90.joblib

# 2. Copy artifacts to local repo
docker cp f1_strategy_api:/app/models/artifacts/lap_time_model_p10.joblib models/artifacts/
docker cp f1_strategy_api:/app/models/artifacts/lap_time_model_p50.joblib models/artifacts/
docker cp f1_strategy_api:/app/models/artifacts/lap_time_model_p90.joblib models/artifacts/

# 3. Update .gitignore to NOT ignore model artifacts (or use Git LFS)
# 4. Commit and push — Render auto-deploys with models baked in
```

### Option C: Run backfill on Render (requires paid tier for one-off jobs)
```bash
# Render one-off job (paid tier only)
render jobs create --service f1-strategy-api \
  --command "python pipelines/backfill_seasons.py --years 2022 2023 2024 2025"
```

**Recommended approach**: Option A (dump + load) for data, Option B for model artifacts.

---

## Local Development — Getting Predictions Working

If you're starting fresh with the local Docker setup:

### Step 1: Start services
```bash
docker compose build
docker compose up -d
```

### Step 2: Train model on existing data (team will be "unknown" but works)
```bash
docker compose exec api python -m models.train_lap_time_model
```
Expected output: "p50 MAE: ~0.44s → Excellent"

### Step 3: Restart API to load model
```bash
docker compose restart api
```

### Step 4: Verify predictions are real (not flat 90.0s)
```bash
curl -s -X POST "http://localhost:8000/simulate/degradation" \
  -H "Content-Type: application/json" \
  -d '{"year":2024,"round":1,"driver_id":"VER","stints":[20,37],"compounds":["MEDIUM","HARD"]}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Range: {max(d[\"p50\"])-min(d[\"p50\"]):.2f}s')"
# Should show "Range: 1.87s" (not 0.0s)
```

### Step 5 (Optional): Backfill team data for better predictions
```bash
# This re-ingests all races from FastF1, populating the team column
# Takes several hours (45s sleep between rounds × ~90 races)
docker compose exec airflow python /opt/airflow/pipelines/backfill_seasons.py \
  --years 2022 2023 2024 2025 --force
```

---

## ML Feature Contract

**CRITICAL**: The feature contract is defined ONCE in `models/feature_config.py`.
All consumers import from there. Never define feature lists locally.

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
| team_id | categorical | team from laps (**currently NULL — needs backfill**) |
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

### Model Quality (Last Training — March 22, 2026)
```
Training set: 61,621 laps | Validation set: 15,406 laps

MAE:  p10=0.779s  p50=0.437s  p90=0.867s
Calibration:  p10=11.5% (target 10%)  p50=50.6% (target 50%)  p90=88.6% (target 90%)

Verdict: Excellent — ready for simulation
```

### Key Performance Rule
**NEVER call `predict_quantiles()` in loops.** Use `predict_quantiles_batch()` instead.
Batch = 1 LightGBM forward pass for N rows. Loop = N forward passes. ~800x speedup.

---

## Database Schema

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
| `laps.team` | **NULL** | Needs re-ingestion via `backfill_seasons.py --force` |
| `qualifying_laps.team` | **NULL** | Same fix as above |

---

## File Map

### Models Layer
```
models/
├── feature_config.py          # Feature contract (SINGLE SOURCE OF TRUTH) — 12 features
├── lap_time_model.py          # LightGBM quantile regression model (p10/p50/p90)
├── train_lap_time_model.py    # Training pipeline (SQL → features → train → save)
├── simulation_engine.py       # Monte Carlo race simulation + degradation + safety car
├── modeling_engine.py         # Orchestrator (train + simulate + optimize + DB lookups)
├── model_config.py            # SimulationConfig, ModelConfig dataclasses
├── strategy_generator.py      # Generates valid race strategies (compound-specific limits)
├── tire_rules.py              # FIA tire rules validation
├── optimization/
│   ├── strategy_optimizer.py  # Bayesian optimization (Optuna)
│   └── optimizer_config.py    # OptimizerConfig dataclass
└── post_race/
    ├── residual_logger.py     # Prediction error logging to DB
    └── residual_analysis.py   # Post-race analysis utilities
```

### API Layer
```
api/
├── main.py                    # FastAPI app, mounts routers, init_db on startup
├── deps.py                    # Dependency injection (cached ModelingEngine singleton)
├── schemas.py                 # Pydantic request/response models (all include circuit_key/team_id)
└── routes/
    ├── simulation.py          # POST: /simulate/auto, /optimize, /degradation, /safety-car
    └── data.py                # GET: /races, /drivers, /teams, /tracks, /race-analysis, /pre-race
```

### Dashboard Layer
```
app/
└── streamlit_app.py           # 4-tab Streamlit dashboard (Pre-Race, Safety Car, Race Analysis, Season Form)

dashboard/
└── api_client.py              # HTTP client wrapping all API endpoints
```

### Data Layer
```
db/
├── database.py                # SQLAlchemy ORM models + engine + init_db()
├── init.sql                   # Schema DDL + migration ALTER TABLEs (safe to re-run)
└── calculate_metrics.py       # Pit loss calculation per track

pipelines/
├── f1_ingestion_pipeline.py   # Airflow DAG: ingest + train + auto-simulate
├── f1_season_master.py        # Airflow DAG: iterate all rounds in a season
└── backfill_seasons.py        # CLI bulk ingestion (--years, --force, --dry-run, --no-train)
```

### Docker / Build
```
docker-compose.yaml            # 4 services: db, api, dashboard, airflow
Dockerfile.api                 # Python 3.9 + FastAPI + LightGBM (self-contained pip install)
Dockerfile.dashboard           # Python 3.9 + Streamlit
requirements.txt               # Dashboard-only deps (used by Streamlit Cloud)
requirements-full.txt          # All deps (reference only — API uses Dockerfile.api)
requirements-dashboard.txt     # Same as requirements.txt (kept for clarity)
```

---

## API Endpoints

### Data (GET)
| Endpoint | Description | Status |
|----------|-------------|--------|
| `/data/races?year=2024` | List ingested races | Works |
| `/data/drivers?year=2024` | List driver codes | Works |
| `/data/teams?year=2024` | List team names | Returns empty (team column NULL) |
| `/data/tracks` | Pit loss constants per track | Works |
| `/data/race/{year}/{round}` | Race detail + pre-computed simulations | Works |
| `/data/race-analysis/{year}/{round}` | Full lap data + stint breakdown per driver | Works |
| `/data/pre-race/{year}/{event_name}` | Season form (normalized) + track history | Works |

### Simulation (POST)
| Endpoint | Description | Status |
|----------|-------------|--------|
| `/simulate/auto` | Run Monte Carlo for a driver/race | Works (real preds with trained model) |
| `/simulate/optimize` | Bayesian strategy optimization (Optuna) | Works |
| `/simulate/degradation` | Lap-by-lap p10/p50/p90 degradation curve | Works |
| `/simulate/safety-car` | Pit vs stay out under safety car | Works |

All simulation endpoints accept optional `circuit_key` and `team_id` fields.
If omitted, they are auto-looked up from the database.

---

## Airflow Pipelines

### `f1_strategy_production_pipeline` (manual trigger)
Params: `year`, `round`
1. **ingest_and_clean_race_data**: Fetches R + Q + SQ sessions from FastF1, persists laps/weather/team/circuit_key
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
| `API_URL` | `http://api:8000` (Docker) or via `st.secrets` (Cloud) | Dashboard |
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

### Before (sklearn — deleted)
- sklearn GradientBoostingRegressor with OneHotEncoder pipeline
- `track_id = "{year}_{round}"` — Silverstone 2024 ≠ Silverstone 2023
- `dirty_air = "clean"` always — wasted feature
- No team/constructor awareness
- 10 features, ~1.0s MAE

### After (LightGBM — current)
- LightGBM with native categorical support
- `circuit_key` — stable track identifier across seasons
- `race_progress` — normalized 0-1 (replaces dirty_air)
- `team_id` — constructor awareness (pending data backfill)
- `regulation_era` — era-aware (future-proofs 2026 regs)
- 12 features defined in single source of truth
- **0.437s MAE** — 2x improvement

### Migration Status
- [x] ALTER TABLE races ADD circuit_key
- [x] ALTER TABLE laps ADD team
- [x] ALTER TABLE qualifying_laps ADD team
- [x] Backfill circuit_key from event_name (131 races)
- [x] New feature_config.py (single source of truth)
- [x] Rewrite lap_time_model.py (LightGBM)
- [x] Rewrite train_lap_time_model.py (new SQL + features)
- [x] Update simulation_engine.py (new feature contract)
- [x] Update modeling_engine.py (circuit_key/team_id lookups)
- [x] Update API schemas (optional circuit_key/team_id fields)
- [x] Update API routes (new inference context)
- [x] Train model on local Docker (p50 MAE: 0.437s)
- [x] Deploy API to Render
- [x] Deploy Dashboard to Streamlit Cloud
- [ ] **Backfill team column via re-ingestion (136k laps with team=NULL)**
- [ ] **Populate Render production database with data**
- [ ] **Bake model artifacts into Render deploy (or train on Render)**

---

## Known Issues / Tech Debt

| Issue | Severity | Notes |
|-------|----------|-------|
| `laps.team` is NULL for all rows | HIGH | Run backfill with `--force`; takes hours |
| Render DB is empty | HIGH | Need pg_dump → Render or run backfill there |
| No model artifacts on Render | HIGH | Bake into Docker image or train on Render |
| Render free tier cold starts (~30s) | MEDIUM | Expected; upgrade to paid for always-on |
| `database.py` default DATABASE_URL mismatches docker-compose | LOW | Only affects non-Docker local dev |
| Stale docstring in `strategy_optimizer.py` references `track_id` | TRIVIAL | Cosmetic only |
| `great-expectations` in requirements-full.txt unused | TRIVIAL | Remove |
| Airflow pip installs on every container start | LOW | Could bake into custom Airflow image |
| `style.applymap` deprecated in dashboard | LOW | Should be `style.map` in newer pandas |

---

## Future Improvements (Not Started)

1. **Real-time race tracking** — WebSocket feed from FastF1 live timing
2. **Driver-specific tire degradation models** — separate deg curves per driver
3. **Weather forecast integration** — predict rain probability for strategy
4. **Gap analysis** — simulate undercut/overcut vs specific rivals
5. **Historical strategy accuracy** — compare pre-race predictions vs actual race outcomes
6. **Multi-driver optimization** — optimize team strategy for both cars simultaneously
7. **2026 regulation era** — new PU rules will change car performance characteristics
