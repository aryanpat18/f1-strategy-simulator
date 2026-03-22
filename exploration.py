import fastf1
import pandas as pd

fastf1.Cache.enable_cache('cache')

# 1. See the Season Schedule
# Great for automating your pipeline later!
schedule = fastf1.get_event_schedule(2024)
print("--- 2024 Schedule (First 5 Rounds) ---")
print(schedule[['RoundNumber', 'EventName', 'EventDate']].head(5))

# 2. Load Silverstone with ALL data (Weather + Telemetry)
session = fastf1.get_session(2024, 'Silverstone', 'R')
session.load(telemetry=True, weather=True)

# 3. Extract Weather Data
# Strategy simulation needs TrackTemp (tire deg) and Rainfall (pit logic)
weather = session.weather_data
print("\n--- Weather Sample (First 5 minutes) ---")
print(weather[['Time', 'AirTemp', 'TrackTemp', 'Rainfall']].head(5))

# 4. Get Detailed Lap Data (Sectors + Speeds)
# .copy() prevents the SettingWithCopyWarning
all_laps = session.laps.copy()
ham_laps = all_laps.pick_drivers('HAM').copy()

# Breaking down the lap into Sectors
# These are Timedeltas, so we convert them to seconds
ham_laps['S1'] = ham_laps['Sector1Time'].dt.total_seconds()
ham_laps['S2'] = ham_laps['Sector2Time'].dt.total_seconds()
ham_laps['S3'] = ham_laps['Sector3Time'].dt.total_seconds()

print("\n--- Sector Breakdown (First 5 Laps) ---")
print(ham_laps[['LapNumber', 'S1', 'S2', 'S3', 'Compound']].head(5))

# 5. Speed Trap (How fast was he going at the end of the straight?)
# This helps us identify if a car has "straight-line speed" (overtaking ability)
print(f"\nHamilton's Max Speed Trap: {ham_laps['SpeedST'].max()} km/h")
