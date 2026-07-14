"""
HSU soiling for the Bundoora library site using FREE air-quality data.

Pipeline:
  1. Load Solcast CSV  -> rainfall (from precipitation_rate) + surface inputs, in UTC
  2. Fetch PM2.5 / PM10 from the Open-Meteo Air Quality API (free, no key, CAMS-sourced)
  3. Merge to a common hourly UTC index
  4. Run pvlib.soiling.hsu -> hourly soiling ratio (1.0 = clean)

Requires: pandas, numpy, requests, pvlib   (pip install pvlib requests)
"""

import pandas as pd
import numpy as np
import requests
import pvlib

# ----------------------------------------------------------------------
# CONFIG  -- edit these for your system
# ----------------------------------------------------------------------
SOLCAST_CSV   = "solcast_df_2020_2025.csv"
LAT, LON      = -37.72, 145.05      # La Trobe University, Bundoora
SURFACE_TILT  = 25.0                # <-- set your actual array tilt (degrees)
CLEAN_THRESH  = 0.5                 # mm of rain in rain_accum_period that resets soiling
# depo_veloc left as None -> pvlib defaults {'2_5':0.0009, '10':0.004}


# ----------------------------------------------------------------------
# 1. Load Solcast -> hourly accumulated rainfall (mm), UTC index
# ----------------------------------------------------------------------
def load_solcast_rain(path):
    df = pd.read_csv(path)
    # Solcast timestamps are UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # precipitation_rate is a RATE in mm/hour. Convert to accumulated mm per step.
    step_hours = df.index.to_series().diff().dropna().median().total_seconds() / 3600.0
    df["rain_mm"] = df["precipitation_rate"].fillna(0.0) * step_hours

    # Resample to hourly: rain = sum of accumulated mm in the hour
    rain_h = df["rain_mm"].resample("1h").sum()
    return rain_h


# ----------------------------------------------------------------------
# 2. Fetch PM2.5 / PM10 from Open-Meteo Air Quality API (free, no key)
#    Historical CAMS data; fetched year-by-year to keep requests small.
# ----------------------------------------------------------------------
def fetch_pm(lat, lon, start_date, end_date):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    frames = []
    for yr in range(start_date.year, end_date.year + 1):
        s = max(pd.Timestamp(f"{yr}-01-01"), start_date.tz_localize(None))
        e = min(pd.Timestamp(f"{yr}-12-31"), end_date.tz_localize(None))
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "pm2_5,pm10",
            "start_date": s.strftime("%Y-%m-%d"),
            "end_date":   e.strftime("%Y-%m-%d"),
            "timezone": "GMT",        # return in UTC to match Solcast
            "domains": "cams_global",
        }
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        h = r.json().get("hourly", {})
        if not h.get("time"):
            continue
        f = pd.DataFrame(h)
        f["time"] = pd.to_datetime(f["time"], utc=True)
        frames.append(f.set_index("time"))
        print(f"  {yr}: {len(f)} hourly PM records")

    if not frames:
        raise RuntimeError("No PM data returned. Check the date range / coordinates.")
    pm = pd.concat(frames).sort_index()
    pm = pm[~pm.index.duplicated(keep="first")]
    return pm  # columns: pm2_5, pm10  (ug/m3)


# ----------------------------------------------------------------------
# 3 + 4. Merge and run HSU
# ----------------------------------------------------------------------
def run_hsu():
    rain_h = load_solcast_rain(SOLCAST_CSV)
    print(f"Solcast rainfall: {rain_h.index.min()} -> {rain_h.index.max()} "
          f"({len(rain_h)} hours)")

    pm = fetch_pm(LAT, LON, rain_h.index.min(), rain_h.index.max())
    print(f"PM coverage:      {pm.index.min()} -> {pm.index.max()}")

    # Align everything on the same hourly UTC index (inner = only where PM exists)
    df = pd.DataFrame({"rainfall": rain_h}).join(pm, how="inner")
    df["pm2_5"] = df["pm2_5"].interpolate(limit=6)   # short PM gaps only
    df["pm10"]  = df["pm10"].interpolate(limit=6)
    df = df.dropna(subset=["pm2_5", "pm10"])
    df["rainfall"] = df["rainfall"].fillna(0.0)

    # IMPORTANT: pvlib.soiling.hsu expects PM in g/m^3, but Open-Meteo returns
    # ug/m^3. Without this 1e-6 conversion the model saturates at its 34.4%
    # max-loss floor (soiling_ratio pinned at 0.6563).
    PM_UGM3_TO_GM3 = 1e-6
    soiling_ratio = pvlib.soiling.hsu(
        rainfall=df["rainfall"],
        cleaning_threshold=CLEAN_THRESH,
        surface_tilt=SURFACE_TILT,
        pm2_5=df["pm2_5"] * PM_UGM3_TO_GM3,
        pm10=df["pm10"] * PM_UGM3_TO_GM3,
        depo_veloc=None,                       # pvlib defaults
        rain_accum_period=pd.Timedelta("1h"),
    )

    out = df.copy()
    out["soiling_ratio"] = soiling_ratio
    out["soiling_loss_pct"] = (1.0 - out["soiling_ratio"]) * 100.0

    daily = out["soiling_ratio"].resample("1D").mean()
    print(f"\nAnalysed period: {out.index.min().date()} -> {out.index.max().date()}")
    print(f"Mean soiling ratio: {out['soiling_ratio'].mean():.4f}  "
          f"(mean loss {out['soiling_loss_pct'].mean():.2f}%)")
    print(f"Worst day SR:       {daily.min():.4f} on {daily.idxmin().date()}")

    out.to_csv("hsu_soiling_output.csv")
    print("Saved hourly results -> hsu_soiling_output.csv")
    return out


if __name__ == "__main__":
    run_hsu()
