import os, glob
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = "Results"               # where your *_series_new.csv live
OUT_DIR = Path("Maintenance"); OUT_DIR.mkdir(exist_ok=True)

# ----- helpers -----
def load_series_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    # normalize expected column names if needed
    rename_map = {
        "Meter_reading (actual)": "Meter_reading",
        "Real model prediction": "Real_model_prediction",
        "Simulated model": "Simulated_model",
        "Simulated (degraded)": "Simulated_degraded",
        "Meter reading": "Meter_reading",
    }
    df = df.rename(columns=rename_map)
    needed = ["timestamp","meter","campus","Meter_reading","Real_model_prediction","Simulated_model"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")
    # sort & dedupe time
    return (df
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .set_index("timestamp"))

def daylight_mask(df: pd.DataFrame) -> pd.Series:
    # Preferred: use weather if present
    if {"global_horizontal_irradiance","zenith"} <= set(df.columns):
        ghi = pd.to_numeric(df["global_horizontal_irradiance"], errors="coerce")
        zen = pd.to_numeric(df["zenith"], errors="coerce")
        return (ghi > 5) & (zen < 90)
    # Fallback: use simulated model as proxy for daylight
    sim = pd.to_numeric(df["Simulated_model"], errors="coerce").fillna(0.0)
    # tiny threshold to ignore noise
    return sim > 0.05

def compute_daily_kpis(df: pd.DataFrame) -> pd.DataFrame:
    # daylight mask
    day = daylight_mask(df).reindex(df.index, fill_value=False)

    act  = pd.to_numeric(df["Meter_reading"], errors="coerce").fillna(0.0)
    real = pd.to_numeric(df["Real_model_prediction"], errors="coerce").fillna(0.0)
    sim  = pd.to_numeric(df["Simulated_model"], errors="coerce").fillna(0.0)

    # pointwise KPIs (0 when denominator == 0 or not daylight)
    HS = np.divide(act, sim,  out=np.zeros_like(act, dtype=float),
                   where=day.values & (sim.values > 0))
    HR = np.divide(act, real, out=np.zeros_like(act, dtype=float),
                   where=day.values & (real.values > 0))
    GAP = sim - real

    dfi = pd.DataFrame({
        "HS": HS, "HR": HR, "GAP": GAP,
        "ActE": act, "SimE": sim
    }, index=df.index)

    # aggregate per calendar day (daylight rows only)
    d = dfi[day]
    byday = d.groupby(d.index.date)
    daily = pd.DataFrame({
        "HS": byday["HS"].median(),
        "HR": byday["HR"].median(),
        "GAP": byday["GAP"].median(),
        "ActEnergy": byday["ActE"].sum(),
        "SimEnergy": byday["SimE"].sum(),
        "N_pts": byday["HS"].count()
    })
    daily.index = pd.to_datetime(daily.index)
    # rolling baselines (robust) for alerts
    daily["HS_ref"] = daily["HS"].rolling(30, min_periods=10).median()
    daily["HR_ref"] = daily["HR"].rolling(30, min_periods=10).median()
    daily["ΔHS"] = daily["HS"] - daily["HS_ref"]
    daily["ΔHR"] = daily["HR"] - daily["HR_ref"]
    daily["Lost_kWh"] = (1 - daily["HS"].clip(lower=0, upper=2)) * daily["SimEnergy"]
    return daily

def compute_alerts(daily: pd.DataFrame) -> pd.DataFrame:
    # rules you can tune
    soiling = (daily["HS"] < 0.90).rolling(3).sum() >= 3
    rapid   = (daily["HS"] < 0.75) | (daily["ActEnergy"] < 0.60 * daily["SimEnergy"])
    drift   = (daily["HR"] < 0.95).rolling(5).sum() >= 5
    # summarize
    alerts = pd.DataFrame({
        "Soiling": soiling.astype(bool),
        "RapidFault": rapid.astype(bool),
        "ModelDrift": drift.astype(bool),
        "HS": daily["HS"], "HR": daily["HR"], "GAP": daily["GAP"],
        "Lost_kWh": daily["Lost_kWh"],
        "ActEnergy": daily["ActEnergy"], "SimEnergy": daily["SimEnergy"]
    }, index=daily.index)
    return alerts[alerts[["Soiling","RapidFault","ModelDrift"]].any(axis=1)]

# ----- main loop over files -----
pattern = os.path.join(RESULTS_DIR, "*_series_new.csv")
for path in sorted(glob.glob(pattern)):
    try:
        df = load_series_csv(path)
        meter = df["meter"].iloc[0]
        campus = df["campus"].iloc[0]

        daily = compute_daily_kpis(df)
        alerts = compute_alerts(daily)

        daily.to_csv(OUT_DIR / f"daily_kpis_{meter}.csv", index_label="date")
        alerts.to_csv(OUT_DIR / f"alerts_{meter}.csv", index_label="date")

        print(f"{meter} ({campus}): {len(daily)} daily rows, {len(alerts)} alerts")
    except Exception as e:
        print(f"ERROR on {os.path.basename(path)}: {e}")


