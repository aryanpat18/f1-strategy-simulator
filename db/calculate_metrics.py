import pandas as pd
import numpy as np
from sqlalchemy import func
import sys

# Ensure module access
sys.path.append('/opt/airflow/')
from db.database import SessionLocal, Lap, Race, TrackMetric

def calculate_pit_loss():
    """
    Day 14 Robust Logic: Uses Median and 107% threshold to filter 
    Safety Cars and slow stops.
    """
    db = SessionLocal()
    races = db.query(Race).all()

    for race in races:
        # Fetch laps into DataFrame for statistical cleaning
        query = db.query(Lap.lap_time_seconds, Lap.is_pit_out_lap).filter(Lap.race_id == race.id)
        df = pd.read_sql(query.statement, db.bind)

        if df.empty: continue

        # 1. CLEAN AIR FILTER (107% of best lap)
        best_lap = df[df['is_pit_out_lap'] == False]['lap_time_seconds'].min()
        if not best_lap: continue
        threshold = best_lap * 1.07
        
        # 2. STATISTICAL CLEANING
        # Use median for pit laps to ignore 60s disaster stops
        pit_laps = df[df['is_pit_out_lap'] == True]['lap_time_seconds']
        normal_laps = df[(df['is_pit_out_lap'] == False) & (df['lap_time_seconds'] <= threshold)]['lap_time_seconds']

        if not normal_laps.empty and not pit_laps.empty:
            avg_normal = normal_laps.median() 
            avg_pit = pit_laps.median()
            pit_loss_constant = float(avg_pit - avg_normal)

            # 3. GLOBAL SANITY CAP
            if pit_loss_constant > 32.0 and "Singapore" not in race.event_name:
                pit_loss_constant = 24.0 # Fallback to global average if data is still noisy
            
            if pit_loss_constant < 5.0 and "Monaco" not in race.event_name:
                continue

            # --- ROBUST UPSERT ---
            existing_metric = db.query(TrackMetric).filter(TrackMetric.event_name == race.event_name).first()
            if existing_metric:
                existing_metric.avg_pit_loss = pit_loss_constant
            else:
                db.add(TrackMetric(event_name=race.event_name, avg_pit_loss=pit_loss_constant))
            
            db.flush() 
            print(f"📈 {race.event_name}: Pit Loss = {pit_loss_constant:.2f}s (Median Cleaned)")

    try:
        db.commit()
        print("✅ All track metrics committed successfully.")
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    calculate_pit_loss()