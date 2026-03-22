"""
Database connection, ORM model definitions, and init helper.

Tables:
  - races              Core race metadata
  - laps               Race lap data (one row per driver per lap)
  - qualifying_laps    Qualifying + sprint-qualifying lap data
  - session_weather    Time-series weather data per race session
  - track_metrics      Derived pit-loss constants per track
  - track_models       Persisted ML degradation coefficients per race/compound
  - simulation_results Persisted simulation outputs
  - model_residuals    Post-race prediction error log (feedback loop)
"""

import os
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    Boolean,
    Interval,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ---------------------------------------------------------------------------
# Engine & Session
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://airflow:airflow@localhost:5432/f1db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------


class Race(Base):
    """One row per race session ingested (year + round combination)."""

    __tablename__ = "races"

    id = Column(Integer, primary_key=True)
    year = Column(Integer, nullable=False)
    round = Column(Integer, nullable=False)
    event_name = Column(String, nullable=False)
    total_laps = Column(Integer)
    circuit_key = Column(String)

    laps = relationship("Lap", back_populates="race", cascade="all, delete-orphan")
    qualifying_laps = relationship(
        "QualifyingLap", back_populates="race", cascade="all, delete-orphan"
    )
    session_weather = relationship(
        "SessionWeather", back_populates="race", cascade="all, delete-orphan"
    )
    track_models = relationship(
        "TrackModel", back_populates="race", cascade="all, delete-orphan"
    )
    simulation_results = relationship(
        "SimulationResult", back_populates="race", cascade="all, delete-orphan"
    )


class Lap(Base):
    """One row per driver per lap in a race session."""

    __tablename__ = "laps"

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id", ondelete="CASCADE"), nullable=False)
    driver_code = Column(String, nullable=False)
    lap_number = Column(Integer, nullable=False)
    lap_time_seconds = Column(Float)
    s1 = Column(Float)
    s2 = Column(Float)
    s3 = Column(Float)
    compound = Column(String)
    tyre_life = Column(Integer)
    is_pit_out_lap = Column(Boolean, default=False)
    team = Column(String)

    race = relationship("Race", back_populates="laps")


class QualifyingLap(Base):
    """
    One row per driver per lap in a qualifying or sprint-qualifying session.

    session_type distinguishes Q (standard qualifying) from SQ (sprint qualifying).
    is_deleted captures FIA lap-deletion flags (exceeded track limits) — these laps
    should be excluded from any pace analysis but are retained for auditability.
    """

    __tablename__ = "qualifying_laps"

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id", ondelete="CASCADE"), nullable=False)
    session_type = Column(String, nullable=False)  # 'Q' or 'SQ'
    driver_code = Column(String, nullable=False)
    lap_number = Column(Integer, nullable=False)
    lap_time_seconds = Column(Float)
    s1 = Column(Float)
    s2 = Column(Float)
    s3 = Column(Float)
    compound = Column(String)
    tyre_life = Column(Integer)
    is_deleted = Column(Boolean, default=False)  # FIA lap deletion flag
    team = Column(String)

    race = relationship("Race", back_populates="qualifying_laps")


class SessionWeather(Base):
    """
    Time-series weather observations for a race or qualifying session.

    FastF1 returns weather data as a DataFrame with one row roughly every
    ~10 seconds of session time. We store every observation so downstream
    code can do a nearest-time join against lap timestamps.

    time_offset_seconds: seconds elapsed since session start (from FastF1 Timedelta).
    rainfall: boolean — True if rain was recorded at that observation.
    """

    __tablename__ = "session_weather"

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id", ondelete="CASCADE"), nullable=False)
    session_type = Column(String, nullable=False)  # 'Q', 'SQ', 'R', 'S'
    time_offset_seconds = Column(Float, nullable=False)
    air_temp = Column(Float)
    track_temp = Column(Float)
    humidity = Column(Float)
    pressure = Column(Float)
    wind_speed = Column(Float)
    wind_direction = Column(Float)
    rainfall = Column(Boolean, default=False)

    race = relationship("Race", back_populates="session_weather")


class TrackMetric(Base):
    """
    Derived pit-loss constant per track (event name).
    Updated by db/calculate_metrics.py after every ingestion.
    """

    __tablename__ = "track_metrics"

    id = Column(Integer, primary_key=True)
    event_name = Column(String, unique=True, nullable=False)
    avg_pit_loss = Column(Float)
    fuel_penalty_per_lap = Column(Float, default=0.035)


class TrackModel(Base):
    """
    Persisted ML tire-degradation model coefficients per race and compound.
    One row per (race_id, compound) combination.
    """

    __tablename__ = "track_models"

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id", ondelete="CASCADE"), nullable=False)
    compound = Column(String, nullable=False)
    deg_coefficient = Column(Float)
    base_pace = Column(Float)
    mae_error = Column(Float)
    r2_score = Column(Float)

    race = relationship("Race", back_populates="track_models")


class SimulationResult(Base):
    """
    Persisted strategy simulation output for a given race and driver.
    Written after each automated simulation run.
    """

    __tablename__ = "simulation_results"

    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id", ondelete="CASCADE"), nullable=False)
    driver_code = Column(String, nullable=False)
    strategy_compounds = Column(String)   # e.g. "MEDIUM-HARD"
    strategy_stints = Column(String)      # e.g. "28-24"
    mean_time = Column(Float)
    std_time = Column(Float)
    p10 = Column(Float)
    p50 = Column(Float)
    p90 = Column(Float)

    race = relationship("Race", back_populates="simulation_results")


class ModelResidual(Base):
    """
    Post-race model error log. One row per driver per lap where we have
    both a model prediction and the actual observed lap time.
    Used by models/post_race/ for bias detection and model iteration.
    """

    __tablename__ = "model_residuals"

    id = Column(Integer, primary_key=True)
    race_id = Column(String, nullable=False)   # kept as String for flexibility
    driver_id = Column(String, nullable=False)
    lap_number = Column(Integer)
    predicted_time = Column(Float)
    actual_time = Column(Float)
    error = Column(Float)


# ---------------------------------------------------------------------------
# Init helper
# ---------------------------------------------------------------------------


def init_db() -> None:
    """
    Create all tables if they do not already exist.
    Safe to call on every Airflow run — CREATE TABLE IF NOT EXISTS semantics
    are handled by SQLAlchemy's checkfirst=True default in create_all().
    """
    Base.metadata.create_all(bind=engine)
    print("✅ Database schema initialised (all tables created or already exist).")