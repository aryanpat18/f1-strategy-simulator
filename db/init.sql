-- =============================================================================
-- F1 Strategy Intelligence Platform — Database Schema
-- =============================================================================
-- Safe to re-run: all statements use CREATE TABLE IF NOT EXISTS.
-- Tables are ordered so FK dependencies are satisfied top-to-bottom.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- Core Race Table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS races (
    id          SERIAL PRIMARY KEY,
    year        INT         NOT NULL,
    round       INT         NOT NULL,
    event_name  TEXT        NOT NULL,
    total_laps  INT,
    circuit_key TEXT,
    UNIQUE (year, round)
);


-- ---------------------------------------------------------------------------
-- Race Lap Data
-- One row per driver per lap in the race session.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS laps (
    id                  SERIAL PRIMARY KEY,
    race_id             INT     NOT NULL REFERENCES races(id) ON DELETE CASCADE,
    driver_code         TEXT    NOT NULL,
    lap_number          INT     NOT NULL,
    lap_time_seconds    FLOAT,
    s1                  FLOAT,
    s2                  FLOAT,
    s3                  FLOAT,
    compound            TEXT,
    tyre_life           INT,
    is_pit_out_lap      BOOLEAN DEFAULT FALSE,
    team                TEXT
);

CREATE INDEX IF NOT EXISTS idx_laps_race_driver
    ON laps (race_id, driver_code);


-- ---------------------------------------------------------------------------
-- Qualifying Lap Data  (NEW — Phase 1a)
-- One row per driver per lap in Q or SQ sessions.
--
-- session_type: 'Q'  = standard qualifying
--               'SQ' = sprint qualifying
-- is_deleted:   TRUE when the lap was deleted by the stewards (track limits).
--               Retain for auditability but exclude from pace analysis.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS qualifying_laps (
    id                  SERIAL PRIMARY KEY,
    race_id             INT     NOT NULL REFERENCES races(id) ON DELETE CASCADE,
    session_type        TEXT    NOT NULL,
    driver_code         TEXT    NOT NULL,
    lap_number          INT     NOT NULL,
    lap_time_seconds    FLOAT,
    s1                  FLOAT,
    s2                  FLOAT,
    s3                  FLOAT,
    compound            TEXT,
    tyre_life           INT,
    is_deleted          BOOLEAN DEFAULT FALSE,
    team                TEXT
);

CREATE INDEX IF NOT EXISTS idx_qualifying_laps_race_driver
    ON qualifying_laps (race_id, driver_code);


-- ---------------------------------------------------------------------------
-- Session Weather  (NEW — Phase 1a)
-- Time-series weather observations per session.
--
-- FastF1 returns ~10-second-interval rows for each session.
-- Storing the full series allows downstream code (train_lap_time_model.py)
-- to do a nearest-time join against lap timestamps rather than using
-- hardcoded temperature defaults.
--
-- session_type: 'Q' | 'SQ' | 'R' | 'S'
-- time_offset_seconds: seconds elapsed since session start.
-- rainfall: TRUE if any rain was detected at this observation.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS session_weather (
    id                      SERIAL PRIMARY KEY,
    race_id                 INT     NOT NULL REFERENCES races(id) ON DELETE CASCADE,
    session_type            TEXT    NOT NULL,
    time_offset_seconds     FLOAT   NOT NULL,
    air_temp                FLOAT,
    track_temp              FLOAT,
    humidity                FLOAT,
    pressure                FLOAT,
    wind_speed              FLOAT,
    wind_direction          FLOAT,
    rainfall                BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_session_weather_race_session
    ON session_weather (race_id, session_type);


-- ---------------------------------------------------------------------------
-- Track Metrics  (Pit-Loss Constants)
-- One row per track (event_name). Updated after every ingestion.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS track_metrics (
    id                      SERIAL PRIMARY KEY,
    event_name              TEXT    UNIQUE NOT NULL,
    avg_pit_loss            FLOAT,
    fuel_penalty_per_lap    FLOAT   DEFAULT 0.035
);


-- ---------------------------------------------------------------------------
-- Track Models  (ML Degradation Coefficients)
-- One row per (race_id, compound).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS track_models (
    id                  SERIAL PRIMARY KEY,
    race_id             INT     NOT NULL REFERENCES races(id) ON DELETE CASCADE,
    compound            TEXT    NOT NULL,
    deg_coefficient     FLOAT,
    base_pace           FLOAT,
    mae_error           FLOAT,
    r2_score            FLOAT,
    UNIQUE (race_id, compound)
);


-- ---------------------------------------------------------------------------
-- Simulation Results
-- Persisted strategy simulation outputs per race + driver.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS simulation_results (
    id                  SERIAL PRIMARY KEY,
    race_id             INT     NOT NULL REFERENCES races(id) ON DELETE CASCADE,
    driver_code         TEXT    NOT NULL,
    strategy_compounds  TEXT,       -- e.g. 'MEDIUM-HARD'
    strategy_stints     TEXT,       -- e.g. '28-24'
    mean_time           FLOAT,
    std_time            FLOAT,
    p10                 FLOAT,
    p50                 FLOAT,
    p90                 FLOAT
);


-- ---------------------------------------------------------------------------
-- Model Residuals  (Post-Race Feedback Loop)
-- One row per driver per lap where prediction vs actuals are compared.
-- race_id kept as TEXT for flexibility (can reference by string key).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_residuals (
    id              SERIAL PRIMARY KEY,
    race_id         TEXT    NOT NULL,
    driver_id       TEXT    NOT NULL,
    lap_number      INT,
    predicted_time  FLOAT,
    actual_time     FLOAT,
    error           FLOAT
);


-- ---------------------------------------------------------------------------
-- Migration: Add columns for ML model upgrade (team + circuit_key)
-- Safe to re-run: ALTER TABLE ... ADD COLUMN IF NOT EXISTS
-- ---------------------------------------------------------------------------
ALTER TABLE races ADD COLUMN IF NOT EXISTS circuit_key TEXT;
ALTER TABLE laps ADD COLUMN IF NOT EXISTS team TEXT;
ALTER TABLE qualifying_laps ADD COLUMN IF NOT EXISTS team TEXT;