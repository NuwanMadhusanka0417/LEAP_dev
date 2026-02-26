"""
XGBoost-based simulated power for Bundoora meters
==================================================
- Loads solarsitesimulation.csv (simulated data per meter) and Solcast weather (BUNDOORA).
- Trains one XGBoost model per Bundoora meter using: simulated grid_power as target,
  Solcast weather + time features as inputs.
- Predicts simulated power for the full period 2020-07-02 → 2025-08-13 (or Solcast end).
- Output: CSV with timestamp, meter, simulated_power (one row per meter per hour).
  Output period is the intersection of [2020-07-02, 2025-08-13] with Solcast availability.

Only Bundoora meters (meter name contains "bun_") are processed.
Requires: xgboost, pandas, numpy.
"""

import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data","simulated")
OUT_DIR = os.path.join(BASE, "data","simulated")

# Input files
SIMULATION_CSV = "solarsitesimulation.csv"
SOLCAST_CSV = "solcast_df.csv"
SOLCAST_CLEANED_CSV = "solcast_df_cleaned.csv"

# Target period (hourly)
START_DATE = "2020-07-02"
END_DATE = "2025-08-13"

# Bundoora = use weather from campus BUNDOORA; meter names containing this prefix
BUNDOORA_METER_PREFIX = "solar.bun_"
SOLCAST_CAMPUS = "BUNDOORA"

# XGBoost defaults
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

# Weather + time features used for training and prediction
WEATHER_FEATURES = [
    "ghi",
    "dni",
    "dhi",
    "zenith",
    "air_temp",
    "wind_speed_10m",
    "relative_humidity",
    "cloud_opacity",
]
TIME_FEATURES = ["hour", "day_of_year", "month", "weekday"]


def load_solcast_bundoora(path=None):
    """Load Solcast, filter BUNDOORA, resample to 1h."""
    for name in [SOLCAST_CLEANED_CSV, SOLCAST_CSV]:
        p = path or os.path.join(DATA_DIR, name)
        if os.path.isfile(p):
            break
    else:
        p = os.path.join(DATA_DIR, SOLCAST_CSV)
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if "campus" in df.columns:
        df = df[df["campus"].astype(str).str.strip().str.upper() == SOLCAST_CAMPUS].copy()
    # Solcast may use wind_speed_10m
    if "wind_speed_10m" not in df.columns and "wind_speed_10m" in df.columns:
        pass
    # Resample to hourly if 30-min
    if len(df) > 1:
        delta = df["timestamp"].diff().dropna()
        if delta.min() < pd.Timedelta("45min"):
            df = df.set_index("timestamp").resample("1h").mean().reset_index()
    return df


def load_simulation_bundoora(path=None):
    """Load simulation CSV and keep only Bundoora (bun_) meters."""
    p = path or os.path.join(DATA_DIR, SIMULATION_CSV)
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    meter_col = "meter" if "meter" in df.columns else [c for c in df.columns if "meter" in c.lower()][0]
    df = df[df[meter_col].astype(str).str.strip().str.lower().str.contains("bun_", na=False)].copy()
    return df


def build_weather_features(solcast_df):
    """Ensure required weather columns exist; normalize names."""
    df = solcast_df.copy()
    # Map possible column names (Solcast uses wind_speed_10m)
    renames = {}
    if "wind_speed_10m" not in df.columns:
        cand = [x for x in df.columns if "wind" in x.lower() and "10" in x]
        if cand:
            renames[cand[0]] = "wind_speed_10m"
    df = df.rename(columns=renames)
    for f in WEATHER_FEATURES:
        if f not in df.columns:
            df[f] = 0.0
    return df


def add_time_features(df, ts_col="timestamp"):
    """Add hour, day_of_year, month, weekday."""
    t = pd.to_datetime(df[ts_col])
    df = df.copy()
    df["hour"] = t.dt.hour
    df["day_of_year"] = t.dt.dayofyear
    df["month"] = t.dt.month
    df["weekday"] = t.dt.weekday
    return df


def get_feature_columns():
    """List of feature names for model."""
    return [f for f in WEATHER_FEATURES + TIME_FEATURES if f in WEATHER_FEATURES or f in TIME_FEATURES]


def train_and_predict_one_meter(meter_df, solcast_hourly, target_col="grid_power", feature_cols=None):
    """
    Train XGBoost on (timestamp, weather, time) -> target_col; return model and use_cols.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()
    merged = meter_df.merge(solcast_hourly, on="timestamp", how="inner")
    merged = add_time_features(merged)
    use_cols = [c for c in feature_cols if c in merged.columns]
    if not use_cols:
        use_cols = [c for c in WEATHER_FEATURES + TIME_FEATURES if c in merged.columns]
    if target_col not in merged.columns or not use_cols:
        return None, []
    train_df = merged.dropna(subset=[target_col] + use_cols)
    if len(train_df) < 100:
        return None, use_cols
    X = train_df[use_cols]
    y = train_df[target_col]
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)
    return model, use_cols


def predict_full_period(model, use_cols, full_range, solcast_hourly):
    """
    Predict for every timestamp in full_range using Solcast weather.
    Builds a DataFrame with timestamp = full_range, left-merge weather, ffill/bfill, then predict.
    """
    pred_df = pd.DataFrame({"timestamp": full_range})
    pred_df = pred_df.merge(solcast_hourly, on="timestamp", how="left")
    pred_df = pred_df.sort_values("timestamp").set_index("timestamp")
    pred_df = pred_df.ffill().bfill()
    pred_df = add_time_features(pred_df.reset_index())
    for c in use_cols:
        if c not in pred_df.columns:
            pred_df[c] = 0.0
    use_cols_avail = [c for c in use_cols if c in pred_df.columns]
    if not use_cols_avail:
        return np.zeros(len(full_range))
    X_pred = pred_df[use_cols_avail]
    return model.predict(X_pred)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading Solcast (BUNDOORA)...")
    solcast = load_solcast_bundoora()
    solcast = build_weather_features(solcast)
    solcast = add_time_features(solcast, "timestamp")
    # Normalize timestamp to hour
    solcast["timestamp"] = pd.to_datetime(solcast["timestamp"]).dt.floor("h")
    solcast = solcast.groupby("timestamp").first().reset_index()
    print(f"  Solcast hourly: {len(solcast)} rows, {solcast['timestamp'].min()} to {solcast['timestamp'].max()}")

    print("Loading simulation (Bundoora meters only)...")
    sim = load_simulation_bundoora()
    meters = sim["meter"].unique().tolist()
    print(f"  Bundoora meters: {len(meters)}")

    # Full hourly range (capped by Solcast availability)
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    solcast_end = solcast["timestamp"].max()
    solcast_start = solcast["timestamp"].min()
    end_actual = min(end, solcast_end)
    start_actual = max(start, solcast_start)
    full_range = pd.date_range(start=start_actual, end=end_actual, freq="1h")
    print(f"  Prediction period: {start_actual} to {end_actual} ({len(full_range)} hours)")

    # Feature columns available in Solcast
    feature_cols = [c for c in WEATHER_FEATURES + TIME_FEATURES if c in solcast.columns]
    if not feature_cols:
        feature_cols = [c for c in solcast.columns if c not in ["timestamp", "campus"] and np.issubdtype(solcast[c].dtype, np.number)]
    print(f"  Features: {feature_cols}")

    all_results = []

    for i, meter_id in enumerate(meters):
        print(f"  Meter {i+1}/{len(meters)}: {meter_id}")
        meter_sim = sim[sim["meter"] == meter_id].copy()
        meter_sim["timestamp"] = pd.to_datetime(meter_sim["timestamp"]).dt.floor("h")
        meter_sim = meter_sim.drop_duplicates(subset=["timestamp"], keep="first")

        model, use_cols = train_and_predict_one_meter(
            meter_sim,
            solcast,
            target_col="grid_power",
            feature_cols=feature_cols,
        )
        if model is None or not use_cols:
            pred_power = np.zeros(len(full_range))
        else:
            pred_power = predict_full_period(model, use_cols, full_range, solcast)
            pred_power = np.maximum(0.0, pred_power)

        for j, ts in enumerate(full_range):
            all_results.append({"timestamp": ts, "meter": meter_id, "simulated_power": float(pred_power[j])})

    out_df = pd.DataFrame(all_results)
    out_path = os.path.join(OUT_DIR, "simulated_power_2020_2025.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(out_df)} rows)")
    return out_df


if __name__ == "__main__":
    run()
