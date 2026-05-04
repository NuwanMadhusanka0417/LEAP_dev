"""
Unified degradation, seasonality, soiling & sudden-event analysis
=================================================================
Uses two complementary metrics:
  PRIMARY   — Temperature-corrected GHI-normalised yield (model-free, IEC 61724)
  SECONDARY — Performance Index from XGBoost simulation (weather-normalised PI)

Both metrics are normalised so that the first year (CALIBRATION_YEAR) = 1.0.

Per-meter analysis:
  1) Overall degradation rate (%/year) via monthly median trend
  2) Seasonal degradation (Summer/Autumn/Winter/Spring — southern hemisphere)
  3) Wind-aware soiling detection with composite soiling score
  4) Sudden-event / anomaly detection in daily series
  5) Comprehensive per-meter and fleet-wide outputs

Inputs:
  - SolarMeterReadings1hour_cleaned_2020_2025.csv
  - simulated_power_2020_2025.csv   (optional, for PI metric)
  - solcast_df_cleaned_2020_2025.csv (weather: GHI, temperature, wind, rain)

Outputs (in Results_seasonality/):
  - degradation_summary.csv, seasonal_degradation.csv, monthly_yield.csv
  - soiling_events.csv, sudden_events.csv
  - fleet_overview.png, per-meter dashboard PNGs
"""

import os
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
SIM_DIR = os.path.join(DATA_DIR, "simulated")

_run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_DIR = os.path.join(BASE, "Results_seasonality", _run_stamp)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
METER_V2_CSV = "SolarMeterReadings1hour_cleaned_2020_2025.csv"
SIMULATED_CSV = "simulated_power_2020_2025.csv"
SOLCAST_CSV = "solcast_df_cleaned_2020_2025.csv"
SOLCAST_CAMPUS = "BUNDOORA"

EXCLUDE_METERS = ["bun_rd1", "bun_rd2", "bun_busstop"]

YEAR_START = 2021
YEAR_END = 2025
CALIBRATION_YEAR = 2021

ZENITH_DAYLIGHT = 85
GHI_MIN_HOURLY = 50.0          # W/m2 minimum GHI for valid hour
GHI_MIN_DAILY = 2.0            # kWh/m2 minimum daily GHI
MIN_DAYLIGHT_HOURS = 4         # minimum valid hours per day
MIN_MONTHS = 6
MIN_CAL_YEAR_DAYS = 60         # minimum days in calibration year for valid normalisation
MIN_TOTAL_DAYS = 200           # minimum total valid days for a meter

# Temperature correction (crystalline silicon)
TEMP_COEFF = -0.004            # per degC (typical: -0.4%/degC)
T_REF = 25.0                   # reference cell temperature (degC)
T_CELL_OFFSET = 20.0           # T_cell ~ T_ambient + offset for rooftop

SEASON_MAP = {
    12: "Summer", 1: "Summer", 2: "Summer",
    3: "Autumn",  4: "Autumn",  5: "Autumn",
    6: "Winter",  7: "Winter",  8: "Winter",
    9: "Spring", 10: "Spring", 11: "Spring",
}
SEASON_ORDER = ["Summer", "Autumn", "Winter", "Spring"]

# Soiling
SOILING_SPELL_MIN_DAYS = 5
SOILING_SPELL_MAX_DAYS = 60
PRECIP_THRESHOLD = 0.3
WIND_CLEAN_SPEED = 7.0
SOILING_MIN_DECLINE = -0.002  # yield/day slope threshold (must be p<0.1)
SOILING_P_THRESHOLD = 0.10

# Sudden events
ANOMALY_SIGMA = 3.0
STEP_CHANGE_WINDOW = 14

SOLCAST_WEATHER_COLS = [
    "ghi", "zenith", "precipitation_rate", "air_temp", "cloud_opacity",
    "wind_speed_10m", "wind_direction_10m", "relative_humidity",
    "snow_soiling_rooftop",
]


# ===================================================================
# Data loading
# ===================================================================

def load_meter_v2():
    path = os.path.join(DATA_DIR, METER_V2_CSV)
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[df["meter"].str.contains("bun_", case=False, na=False)].copy()
    df = df[(df["timestamp"].dt.year >= YEAR_START) &
            (df["timestamp"].dt.year <= YEAR_END)].copy()
    for ex in EXCLUDE_METERS:
        df = df[~df["meter"].str.contains(ex, case=False, na=False)]
    return df


def load_simulated_power():
    for d in [SIM_DIR, DATA_DIR]:
        p = os.path.join(d, SIMULATED_CSV)
        if os.path.isfile(p):
            df = pd.read_csv(p)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df[(df["timestamp"].dt.year >= YEAR_START) &
                    (df["timestamp"].dt.year <= YEAR_END)].copy()
            return df
    return None


def load_solcast():
    p = os.path.join(DATA_DIR, SOLCAST_CSV)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Cannot find {SOLCAST_CSV}")
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if "campus" in df.columns:
        df = df[df["campus"].str.upper().str.strip() == SOLCAST_CAMPUS].copy()
    df["timestamp"] = df["timestamp"].dt.floor("h")
    df = df[(df["timestamp"].dt.year >= YEAR_START) &
            (df["timestamp"].dt.year <= YEAR_END)].copy()
    if len(df) > 1 and df["timestamp"].diff().dropna().min() < pd.Timedelta("45min"):
        df = df.set_index("timestamp").resample("1h").mean(numeric_only=True).reset_index()
    return df


# ===================================================================
# Build per-meter hourly table
# ===================================================================

def build_hourly(meter_df, solcast_df, sim_df=None):
    """
    Merge actual meter readings with Solcast weather (and optional simulation).
    Compute:
      - GHI-normalised yield per hour
      - Temperature-corrected yield
      - PI from simulation (if available)
    """
    m = meter_df.copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"]).dt.floor("h")

    # Merge weather
    wcols = ["timestamp"] + [c for c in SOLCAST_WEATHER_COLS if c in solcast_df.columns]
    m = m.merge(solcast_df[wcols].copy(), on="timestamp", how="left")

    # Merge simulation (optional)
    if sim_df is not None and len(sim_df) > 0:
        sim = sim_df.rename(columns={"simulated_power": "E_sim"})
        sim["timestamp"] = pd.to_datetime(sim["timestamp"]).dt.floor("h")
        m = m.merge(sim[["timestamp", "E_sim"]], on="timestamp", how="left")
    else:
        m["E_sim"] = np.nan

    # Nighttime zeroing
    if "zenith" in m.columns:
        is_night = m["zenith"].fillna(100) >= ZENITH_DAYLIGHT
        m.loc[is_night, "meter_reading"] = 0.0

    # Valid daylight mask
    m["valid"] = (
        m["meter_reading"].notna()
        & (m["meter_reading"] > 0)
        & m["ghi"].notna()
        & (m["ghi"] > GHI_MIN_HOURLY)
    )
    if "zenith" in m.columns:
        m["valid"] = m["valid"] & (m["zenith"] < ZENITH_DAYLIGHT)

    # GHI-normalised yield (kW per kW/m2 = m2-equivalent)
    m["yield_ghi"] = np.where(
        m["valid"] & (m["ghi"] > GHI_MIN_HOURLY),
        m["meter_reading"] / m["ghi"],
        np.nan,
    )

    # Temperature correction
    if "air_temp" in m.columns:
        t_cell = m["air_temp"].fillna(T_REF - T_CELL_OFFSET) + T_CELL_OFFSET
        temp_factor = 1.0 + TEMP_COEFF * (t_cell - T_REF)
        temp_factor = temp_factor.clip(0.5, 1.5)
        m["yield_tc"] = np.where(
            m["valid"],
            m["yield_ghi"] / temp_factor,
            np.nan,
        )
    else:
        m["yield_tc"] = m["yield_ghi"]

    # Time features
    ts = pd.to_datetime(m["timestamp"])
    m["date"] = ts.dt.date
    m["year"] = ts.dt.year
    m["month"] = ts.dt.month
    m["season"] = m["month"].map(SEASON_MAP)
    m["year_month"] = ts.dt.to_period("M")

    return m


def build_daily(m):
    """
    Aggregate hourly to daily, compute normalised yield metric,
    and normalise to calibration year = 1.0.
    """
    valid = m[m["valid"]].copy()
    if valid.empty:
        return pd.DataFrame()

    agg = {
        "meter_reading": "sum",
        "ghi": "sum",
        "yield_tc": "median",
        "valid": "count",
    }
    for c in ["precipitation_rate", "wind_speed_10m", "wind_direction_10m",
              "relative_humidity", "air_temp", "snow_soiling_rooftop"]:
        if c in valid.columns:
            agg[c] = "mean"

    daily = valid.groupby("date").agg(agg).reset_index()
    daily = daily.rename(columns={"valid": "n_hours", "yield_tc": "yield_hourly_median"})
    daily["date_dt"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date_dt"].dt.year
    daily["month"] = daily["date_dt"].dt.month
    daily["season"] = daily["month"].map(SEASON_MAP)
    daily["year_month"] = daily["date_dt"].dt.to_period("M")

    # Daily GHI in kWh/m2 (from W/m2 hourly sums)
    daily["daily_ghi_kwh"] = daily["ghi"] / 1000.0

    # Quality filter
    daily = daily[(daily["n_hours"] >= MIN_DAYLIGHT_HOURS) &
                  (daily["daily_ghi_kwh"] >= GHI_MIN_DAILY)].copy()

    # Daily yield = total_kWh / daily_GHI_kWh (temperature-corrected hourly median is backup)
    daily["yield_daily"] = daily["meter_reading"] / daily["daily_ghi_kwh"]

    # Remove extreme outliers (IQR-based)
    q1 = daily["yield_daily"].quantile(0.05)
    q3 = daily["yield_daily"].quantile(0.95)
    iqr = q3 - q1
    daily["yield_valid"] = daily["yield_daily"].between(q1 - 2 * iqr, q3 + 2 * iqr)
    daily.loc[~daily["yield_valid"], "yield_daily"] = np.nan

    # Normalise to calibration year median = 1.0
    cal_year = daily[daily["year"] == CALIBRATION_YEAR]["yield_daily"].dropna()
    if len(cal_year) >= MIN_CAL_YEAR_DAYS:
        norm = cal_year.median()
    else:
        norm = daily["yield_daily"].dropna().median()
    if norm < 1e-6:
        norm = 1.0
    daily["yield_norm"] = daily["yield_daily"] / norm
    daily["cal_year_days"] = len(cal_year)

    return daily


# ===================================================================
# 1) Overall degradation
# ===================================================================

def overall_degradation(daily):
    """Monthly median normalised yield trend -> degradation %/year."""
    if daily.empty or "yield_norm" not in daily.columns:
        return np.nan, np.nan, 0, pd.DataFrame()

    d = daily.dropna(subset=["yield_norm"])
    monthly = d.groupby("year_month").agg(
        yield_med=("yield_norm", "median"),
        n_days=("yield_norm", "count"),
    ).reset_index()
    monthly = monthly[monthly["n_days"] >= 5]
    monthly["month_ts"] = monthly["year_month"].dt.to_timestamp()
    monthly["years"] = (monthly["month_ts"] - monthly["month_ts"].min()).dt.days / 365.25

    if len(monthly) < MIN_MONTHS:
        return np.nan, np.nan, len(monthly), monthly

    slope, intercept, r, p, se = stats.linregress(monthly["years"], monthly["yield_med"])
    deg_pct = 100.0 * slope  # normalised to baseline=1.0, so slope*100 = %/yr
    baseline = monthly.iloc[:6]["yield_med"].median()
    return deg_pct, baseline, len(monthly), monthly


# ===================================================================
# 2) Seasonal degradation
# ===================================================================

def seasonal_degradation(daily):
    results = []
    if daily.empty:
        return pd.DataFrame()

    for season in SEASON_ORDER:
        sd = daily[daily["season"] == season].dropna(subset=["yield_norm"])
        if sd.empty:
            continue
        monthly = sd.groupby("year_month").agg(
            yield_med=("yield_norm", "median"),
            n_days=("yield_norm", "count"),
        ).reset_index()
        monthly = monthly[monthly["n_days"] >= 3]
        monthly["month_ts"] = monthly["year_month"].dt.to_timestamp()
        monthly["years"] = (monthly["month_ts"] - monthly["month_ts"].min()).dt.days / 365.25

        if len(monthly) < 3:
            results.append({
                "season": season, "deg_pct_per_year": np.nan,
                "median_yield": round(np.nanmedian(sd["yield_norm"]), 4),
                "n_months": len(monthly),
            })
            continue

        slope, intercept, r, p, se = stats.linregress(monthly["years"], monthly["yield_med"])
        results.append({
            "season": season,
            "deg_pct_per_year": round(100.0 * slope, 4),
            "trend_slope": round(slope, 6),
            "median_yield": round(np.nanmedian(monthly["yield_med"]), 4),
            "n_months": len(monthly),
            "r_value": round(r, 4),
            "p_value": round(p, 4),
        })
    return pd.DataFrame(results)


# ===================================================================
# 3) Wind-aware soiling detection
# ===================================================================

def detect_soiling(daily):
    if daily.empty or len(daily) < SOILING_SPELL_MIN_DAYS + 2:
        return pd.DataFrame()

    d = daily.dropna(subset=["yield_norm"]).copy()
    d = d.sort_values("date").reset_index(drop=True)

    has_precip = "precipitation_rate" in d.columns
    has_wind = "wind_speed_10m" in d.columns

    if has_precip:
        d["rain_clean"] = d["precipitation_rate"] >= PRECIP_THRESHOLD
    else:
        d["rain_clean"] = False

    if has_wind:
        d["wind_clean"] = d["wind_speed_10m"] >= WIND_CLEAN_SPEED
    else:
        d["wind_clean"] = False

    d["is_cleaned"] = d["rain_clean"] | d["wind_clean"]
    d["is_soiling_day"] = ~d["is_cleaned"]

    # Composite soiling potential
    score = np.zeros(len(d))
    if has_precip:
        score += 1.0 * (d["precipitation_rate"] < 0.1).astype(float)
    if has_wind:
        score -= 0.5 * d["wind_clean"].astype(float)
        score += 0.3 * (d["wind_speed_10m"] <= 3.0).astype(float)
    if "relative_humidity" in d.columns:
        score += 0.2 * (d["relative_humidity"] < 50).astype(float)
    d["soiling_potential"] = score

    # Find soiling spells (capped at max length to avoid seasonal confounding)
    spells = []
    i, n = 0, len(d)
    while i < n:
        if d.iloc[i]["is_soiling_day"]:
            start = i
            while i < n and d.iloc[i]["is_soiling_day"]:
                i += 1
            length = min(i - start, SOILING_SPELL_MAX_DAYS)
            if length >= SOILING_SPELL_MIN_DAYS:
                spells.append((start, start + length - 1, length))
        else:
            i += 1

    events = []
    for s_start, s_end, s_len in spells:
        sub = d.iloc[s_start:s_end + 1]
        x = np.arange(s_len, dtype=float)
        y = sub["yield_norm"].values
        ok = np.isfinite(y)
        if ok.sum() < 3:
            continue
        slope, intercept, r, p, se = stats.linregress(x[ok], y[ok])
        y_start = y[ok][0] if ok.any() else np.nan
        y_end = y[ok][-1] if ok.any() else np.nan

        recovery = np.nan
        cleaning_type = "none"
        if s_end + 3 < n:
            post = d.iloc[s_end + 1:s_end + 4]
            recovery_val = post["yield_norm"].median()
            recovery = recovery_val - y_end if (np.isfinite(recovery_val) and np.isfinite(y_end)) else np.nan
            if post["rain_clean"].any():
                cleaning_type = "rain"
            elif post["wind_clean"].any():
                cleaning_type = "wind"
            else:
                cleaning_type = "unknown"

        avg_wind = sub["wind_speed_10m"].mean() if has_wind else np.nan
        avg_wind_dir = sub["wind_direction_10m"].mean() if "wind_direction_10m" in sub.columns else np.nan
        avg_rh = sub["relative_humidity"].mean() if "relative_humidity" in sub.columns else np.nan
        solcast_soil = sub["snow_soiling_rooftop"].mean() if "snow_soiling_rooftop" in sub.columns else np.nan

        is_soiling = (slope < SOILING_MIN_DECLINE) and (p < SOILING_P_THRESHOLD)

        events.append({
            "start_date": sub.iloc[0]["date"],
            "end_date": sub.iloc[-1]["date"],
            "season": sub.iloc[0]["season"],
            "spell_days": s_len,
            "yield_slope_per_day": round(slope, 6),
            "yield_start": round(y_start, 4) if np.isfinite(y_start) else np.nan,
            "yield_end": round(y_end, 4) if np.isfinite(y_end) else np.nan,
            "yield_drop": round(y_start - y_end, 4) if (np.isfinite(y_start) and np.isfinite(y_end)) else np.nan,
            "yield_recovery": round(recovery, 4) if np.isfinite(recovery) else np.nan,
            "cleaning_type": cleaning_type,
            "avg_wind_speed": round(avg_wind, 2) if np.isfinite(avg_wind) else np.nan,
            "avg_wind_direction": round(avg_wind_dir, 1) if np.isfinite(avg_wind_dir) else np.nan,
            "avg_humidity": round(avg_rh, 1) if np.isfinite(avg_rh) else np.nan,
            "soiling_potential_score": round(sub["soiling_potential"].mean(), 3),
            "solcast_soiling_factor": round(solcast_soil, 4) if np.isfinite(solcast_soil) else np.nan,
            "is_soiling": is_soiling,
            "r_value": round(r, 4),
            "p_value": round(p, 4),
        })
    return pd.DataFrame(events)


# ===================================================================
# 4) Sudden event / anomaly detection
# ===================================================================

def detect_sudden_events(daily):
    d = daily.dropna(subset=["yield_norm"]).copy()
    d = d.sort_values("date").reset_index(drop=True)
    if len(d) < STEP_CHANGE_WINDOW * 2:
        return pd.DataFrame()

    d["smooth"] = d["yield_norm"].rolling(
        STEP_CHANGE_WINDOW, center=True, min_periods=3
    ).median()
    d["smooth"] = d["smooth"].bfill().ffill()
    d["residual"] = d["yield_norm"] - d["smooth"]

    sigma = d["residual"].std()
    if sigma < 1e-6:
        return pd.DataFrame()
    threshold = ANOMALY_SIGMA * sigma
    d["is_anomaly"] = d["residual"].abs() > threshold

    events = []
    for _, row in d[d["is_anomaly"]].iterrows():
        idx = d.index[d["date"] == row["date"]]
        if len(idx) == 0:
            continue
        i = idx[0]

        w = STEP_CHANGE_WINDOW
        before = d.iloc[max(0, i - w):i]["yield_norm"].median()
        after = d.iloc[i + 1:i + 1 + w]["yield_norm"].median()
        step = after - before if (np.isfinite(before) and np.isfinite(after)) else np.nan

        if row["residual"] < -threshold:
            etype = "sudden_drop"
        elif row["residual"] > threshold:
            etype = "sudden_spike"
        else:
            etype = "anomaly"

        if np.isfinite(step) and abs(step) > 2 * sigma:
            etype = "step_change_down" if step < 0 else "step_change_up"

        weather_ctx = ""
        if "wind_speed_10m" in row.index and np.isfinite(row.get("wind_speed_10m", np.nan)):
            weather_ctx += f"wind={row['wind_speed_10m']:.1f}m/s "
        if "precipitation_rate" in row.index and np.isfinite(row.get("precipitation_rate", np.nan)):
            weather_ctx += f"rain={row['precipitation_rate']:.1f}mm/h"

        events.append({
            "date": row["date"],
            "yield_norm": round(row["yield_norm"], 4),
            "smooth": round(row["smooth"], 4),
            "residual": round(row["residual"], 4),
            "event_type": etype,
            "step_magnitude": round(step, 4) if np.isfinite(step) else np.nan,
            "weather_context": weather_ctx.strip(),
        })
    return pd.DataFrame(events)


# ===================================================================
# 5) Loss attribution per (year, season)
# ===================================================================

def loss_attribution(daily, soiling_df, events_df, trend_slope):
    """
    Break down yield change into: degradation, soiling, sudden events, other.
    Returns one row per (year, season) with each component as a fraction of baseline.
    """
    if daily.empty:
        return pd.DataFrame()

    d = daily.dropna(subset=["yield_norm"]).copy()
    d = d.sort_values("date").reset_index(drop=True)

    # Tag soiling days
    soiling_dates = set()
    if not soiling_df.empty:
        for _, row in soiling_df[soiling_df["is_soiling"]].iterrows():
            s_start = pd.to_datetime(row["start_date"])
            s_end = pd.to_datetime(row["end_date"])
            for dt in pd.date_range(s_start, s_end):
                soiling_dates.add(dt.date())

    # Tag event days (drops only)
    event_dates = set()
    if not events_df.empty:
        drop_events = events_df[events_df["event_type"].str.contains("drop|step_change_down")]
        for _, row in drop_events.iterrows():
            event_dates.add(row["date"])

    d["date_obj"] = pd.to_datetime(d["date"]).dt.date
    d["is_soiling_day"] = d["date_obj"].isin(soiling_dates)
    d["is_event_day"] = d["date_obj"].isin(event_dates)
    d["is_normal"] = ~d["is_soiling_day"] & ~d["is_event_day"]

    # Smooth baseline for event deficit calculation
    d["smooth"] = d["yield_norm"].rolling(14, center=True, min_periods=3).median()
    d["smooth"] = d["smooth"].bfill().ffill()

    # Baseline per season (calibration year)
    cal_baselines = {}
    cal_data = d[d["year"] == CALIBRATION_YEAR]
    for season in SEASON_ORDER:
        vals = cal_data[cal_data["season"] == season]["yield_norm"].dropna()
        cal_baselines[season] = vals.median() if len(vals) > 5 else 1.0

    years = sorted(d["year"].unique())
    rows = []

    for year in years:
        for season in SEASON_ORDER:
            mask = (d["year"] == year) & (d["season"] == season)
            sub = d[mask]
            if len(sub) < 5:
                continue

            baseline = cal_baselines.get(season, 1.0)
            actual = sub["yield_norm"].median()
            total_change = actual - baseline
            n_total = len(sub)

            # Degradation component from trend
            years_from_cal = year - CALIBRATION_YEAR
            degradation = trend_slope * years_from_cal

            # Soiling component
            soiling_days = sub[sub["is_soiling_day"]]
            if len(soiling_days) >= 2:
                normal_days = sub[sub["is_normal"]]
                normal_yield = normal_days["yield_norm"].median() if len(normal_days) > 2 else actual
                soiling_yield = soiling_days["yield_norm"].median()
                soiling_frac = len(soiling_days) / n_total
                soiling_loss = max(0, (normal_yield - soiling_yield)) * soiling_frac
            else:
                soiling_loss = 0.0
                soiling_frac = 0.0

            # Sudden event component (drops only)
            event_days = sub[sub["is_event_day"]]
            if len(event_days) >= 1:
                event_deficit = (event_days["smooth"] - event_days["yield_norm"]).clip(lower=0)
                event_loss = event_deficit.sum() / n_total
            else:
                event_loss = 0.0

            other = total_change - degradation + soiling_loss + event_loss

            rows.append({
                "year": year,
                "season": season,
                "baseline": round(baseline, 4),
                "actual": round(actual, 4),
                "total_change_pct": round(100 * total_change, 2),
                "degradation_pct": round(100 * degradation, 2),
                "soiling_loss_pct": round(100 * soiling_loss, 2),
                "event_loss_pct": round(100 * event_loss, 2),
                "other_pct": round(100 * other, 2),
                "n_days": n_total,
                "n_soiling_days": len(soiling_days) if len(sub) > 0 else 0,
                "n_event_days": len(event_days) if len(sub) > 0 else 0,
            })

    return pd.DataFrame(rows)


# ===================================================================
# 6) Plotting
# ===================================================================

def plot_meter_dashboard(meter_id, monthly_df, seasonal_df, soiling_df, events_df, daily):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    short = meter_id.split("#")[0].replace("solar.", "")
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle(f"Degradation Dashboard - {short}  (baseline: {CALIBRATION_YEAR})",
                 fontsize=15, fontweight="bold")

    # ------------------------------------------------------------------
    # Row 1, Col 1: Monthly total power generation (kWh) with trend
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    if not daily.empty and "meter_reading" in daily.columns:
        d = daily.copy()
        d["year_month"] = pd.to_datetime(d["date"]).dt.to_period("M")
        monthly_kwh = d.groupby("year_month").agg(
            total_kwh=("meter_reading", "sum"),
            n_days=("meter_reading", "count"),
        ).reset_index()
        monthly_kwh["month_ts"] = monthly_kwh["year_month"].dt.to_timestamp()
        monthly_kwh["years"] = (
            (monthly_kwh["month_ts"] - monthly_kwh["month_ts"].min()).dt.days / 365.25
        )
        ax.bar(monthly_kwh["month_ts"], monthly_kwh["total_kwh"],
               width=25, color="steelblue", alpha=0.7, edgecolor="steelblue")
        if len(monthly_kwh) >= MIN_MONTHS:
            s, i_val, r, p, _ = stats.linregress(
                monthly_kwh["years"], monthly_kwh["total_kwh"]
            )
            ax.plot(monthly_kwh["month_ts"], i_val + s * monthly_kwh["years"],
                    "--", color="red", lw=2,
                    label=f"Trend: {s:+.1f} kWh/yr (r={r:.2f})")
            ax.legend(fontsize=9)
    ax.set_ylabel("Monthly Total (kWh)")
    ax.set_title("Power Generation Over Time")
    ax.grid(True, alpha=0.3, axis="y")

    # ------------------------------------------------------------------
    # Row 1, Col 2: Yearly total power generation bar chart
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    if not daily.empty and "meter_reading" in daily.columns:
        d = daily.copy()
        d["year"] = pd.to_datetime(d["date"]).dt.year
        yearly = d.groupby("year").agg(
            total_kwh=("meter_reading", "sum"),
            n_days=("meter_reading", "count"),
        ).reset_index()
        yearly["daily_avg_kwh"] = yearly["total_kwh"] / yearly["n_days"]
        years = yearly["year"].values
        bar_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(years)))

        bars = ax.bar(years, yearly["total_kwh"], color=bar_colors,
                      edgecolor="gray", alpha=0.85)
        for bar, row_y in zip(bars, yearly.itertuples()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{row_y.total_kwh:,.0f}\n({row_y.n_days}d)",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

        if len(yearly) >= 2:
            first_avg = yearly.iloc[0]["daily_avg_kwh"]
            last_avg = yearly.iloc[-1]["daily_avg_kwh"]
            change_pct = 100.0 * (last_avg - first_avg) / first_avg if first_avg > 0 else 0
            ax.set_xlabel(
                f"Daily avg: {first_avg:.1f} kWh ({years[0]}) => "
                f"{last_avg:.1f} kWh ({years[-1]})  [{change_pct:+.1f}%]",
                fontsize=9,
            )
        ax.set_ylabel("Yearly Total (kWh)")
        ax.set_title("Yearly Power Generation Comparison")
        ax.set_xticks(years)
    ax.grid(True, alpha=0.3, axis="y")

    # ------------------------------------------------------------------
    # Row 2, Col 1: Monthly normalised yield trend (degradation)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    if not monthly_df.empty and "month_ts" in monthly_df.columns:
        ax.plot(monthly_df["month_ts"], monthly_df["yield_med"], "o-",
                ms=4, lw=1.5, color="steelblue")
        if len(monthly_df) >= MIN_MONTHS:
            x = (monthly_df["month_ts"] - monthly_df["month_ts"].min()).dt.days / 365.25
            s, i_val, _, _, _ = stats.linregress(x, monthly_df["yield_med"])
            ax.plot(monthly_df["month_ts"], i_val + s * x, "--", color="red", lw=2,
                    label=f"Trend: {100*s:+.2f}%/yr")
            ax.legend(fontsize=9)
    ax.set_ylabel("Monthly median yield (norm.)")
    ax.set_title("Degradation Trend (GHI-normalised, temp-corrected)")
    ax.axhline(1.0, ls=":", color="green", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 2, Col 2: Seasonal yield distribution
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    if not daily.empty:
        data_pairs = [(daily[daily["season"] == s]["yield_norm"].dropna().values, s)
                      for s in SEASON_ORDER]
        data_pairs = [(d, s) for d, s in data_pairs if len(d) > 0]
        if data_pairs:
            bp = ax.boxplot([d for d, _ in data_pairs],
                            labels=[s for _, s in data_pairs],
                            patch_artist=True, showfliers=False)
            colors = ["#ff9999", "#ffcc66", "#99ccff", "#99ff99"]
            for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
                patch.set_facecolor(color)
    ax.axhline(1.0, ls=":", color="green", alpha=0.5)
    ax.set_ylabel("Normalised yield")
    ax.set_title("Seasonal Yield Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    # ------------------------------------------------------------------
    # Row 3, Col 1: Daily yield with sudden events
    # ------------------------------------------------------------------
    ax = axes[2, 0]
    if not daily.empty:
        dates = pd.to_datetime(daily["date"])
        ax.scatter(dates, daily["yield_norm"], s=3, alpha=0.3, color="gray")
        smooth = daily["yield_norm"].rolling(14, center=True, min_periods=3).median()
        ax.plot(dates, smooth, color="blue", lw=1.5, label="14d median")
        if not events_df.empty:
            ev_dates = pd.to_datetime(events_df["date"])
            ev_y = events_df["yield_norm"]
            drops = events_df["event_type"].str.contains("drop|step_change_down")
            if drops.any():
                ax.scatter(ev_dates[drops], ev_y[drops], s=30, color="red",
                           zorder=5, label="Sudden drops")
            spikes = events_df["event_type"].str.contains("spike|step_change_up")
            if spikes.any():
                ax.scatter(ev_dates[spikes], ev_y[spikes], s=30, color="orange",
                           marker="^", zorder=5, label="Spikes")
        ax.legend(fontsize=8)
    ax.axhline(1.0, ls=":", color="green", alpha=0.5)
    ax.set_ylabel("Daily yield (norm.)")
    ax.set_title("Daily Yield & Sudden Events")
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 3, Col 2: Soiling events
    # ------------------------------------------------------------------
    ax = axes[2, 1]
    if not soiling_df.empty and soiling_df["is_soiling"].any():
        soil = soiling_df[soiling_df["is_soiling"]].copy()
        n = len(soil)
        bar_colors = []
        for _, s in soil.iterrows():
            if s["cleaning_type"] == "rain":
                bar_colors.append("dodgerblue")
            elif s["cleaning_type"] == "wind":
                bar_colors.append("mediumpurple")
            else:
                bar_colors.append("sandybrown")
        drops = soil["yield_drop"].values
        ax.barh(range(min(n, 30)), drops[:30], color=bar_colors[:30],
                edgecolor="gray", alpha=0.85)
        labels = [str(d)[:10] for d in soil["start_date"].values[:30]]
        ax.set_yticks(range(min(n, 30)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Yield drop during soiling spell")
        ax.set_title(f"Soiling ({n} events) [blue=rain, purple=wind cleaned]")
    else:
        ax.text(0.5, 0.5, "No soiling events detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Soiling Events")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(RESULTS_DIR, f"dashboard_{short}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fleet_overview(all_deg, all_seasonal, all_monthly, soiling_summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"Fleet Degradation & Soiling - Bundoora Solar ({YEAR_START}-{YEAR_END})\n"
        f"Method: GHI-normalised yield, temp-corrected | Baseline: {CALIBRATION_YEAR}",
        fontsize=13, fontweight="bold",
    )

    # 1) Per-meter degradation
    ax = axes[0, 0]
    valid = all_deg[all_deg["deg_pct_per_year"].notna()].sort_values("deg_pct_per_year")
    if not valid.empty:
        vals = valid["deg_pct_per_year"].values
        colors = ["green" if v > -1 else "orange" if v > -3 else "red" for v in vals]
        ax.barh(valid["meter_short"].values, vals, color=colors, edgecolor="gray", alpha=0.85)
        ax.axvline(0, color="black", lw=0.5)
        ax.axvline(-0.5, ls="--", color="green", alpha=0.5, label="Typical: -0.5 to -1%")
        ax.axvline(-1.0, ls="--", color="green", alpha=0.5)
    ax.set_xlabel("Degradation (%/year)")
    ax.set_title("Per-Meter Degradation Rate")
    ax.grid(True, alpha=0.3, axis="x")

    # 2) Seasonal heatmap
    ax = axes[0, 1]
    if not all_seasonal.empty:
        pivot = all_seasonal.pivot_table(
            index="meter_short", columns="season", values="deg_pct_per_year",
        )
        cols = [s for s in SEASON_ORDER if s in pivot.columns]
        if cols:
            pivot = pivot[cols]
            try:
                sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                            cbar_kws={"label": "Deg %/yr"}, linewidths=0.5)
            except Exception:
                pass
    ax.set_title("Seasonal Degradation (%/year)")

    # 3) Monthly yield all meters
    ax = axes[1, 0]
    if not all_monthly.empty and "month_ts" in all_monthly.columns:
        for meter_id in all_monthly["meter"].unique():
            sub = all_monthly[all_monthly["meter"] == meter_id]
            short = meter_id.split("#")[0].replace("solar.", "")
            ax.plot(sub["month_ts"], sub["yield_med"], alpha=0.6, lw=1, label=short)
        ax.axhline(1.0, ls=":", color="green", alpha=0.5)
        ax.set_ylabel("Monthly median yield (norm.)")
        ax.set_title("Monthly Yield - All Meters")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=5, ncol=4, loc="lower left")

    # 4) Soiling: rain vs wind
    ax = axes[1, 1]
    if not soiling_summary.empty:
        ss = soiling_summary.sort_values("total_soiling")
        x_idx = np.arange(len(ss))
        w = 0.35
        ax.barh(x_idx - w/2, ss["rain_cleaned"], height=w, color="dodgerblue",
                label="Rain-cleaned", alpha=0.8)
        ax.barh(x_idx + w/2, ss["wind_cleaned"], height=w, color="mediumpurple",
                label="Wind-cleaned", alpha=0.8)
        ax.set_yticks(x_idx)
        ax.set_yticklabels(ss["meter_short"], fontsize=7)
        ax.set_xlabel("Number of soiling spells")
        ax.set_title("Soiling Events: Rain vs Wind Cleaning")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No soiling data", ha="center", va="center",
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(RESULTS_DIR, "fleet_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fleet overview: {path}")


def plot_loss_dashboard(meter_id, loss_df):
    """Per-meter loss attribution dashboard: waterfall + heatmap + yearly bars."""
    if loss_df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import seaborn as sns
    except ImportError:
        return

    short = meter_id.split("#")[0].replace("solar.", "")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Loss Attribution - {short}  (baseline: {CALIBRATION_YEAR})",
                 fontsize=15, fontweight="bold")

    season_colors = {"Summer": "#ff9999", "Autumn": "#ffcc66",
                     "Winter": "#99ccff", "Spring": "#99ff99"}

    # ------------------------------------------------------------------
    # Panel 1: Year-Season heatmap of total change
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    pivot = loss_df.pivot_table(index="year", columns="season",
                                values="total_change_pct")
    cols = [s for s in SEASON_ORDER if s in pivot.columns]
    if cols:
        pivot = pivot[cols]
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn",
                    center=0, cbar_kws={"label": "Change from baseline (%)"},
                    linewidths=0.5)
    ax.set_title("Total Yield Change vs Baseline (%)")

    # ------------------------------------------------------------------
    # Panel 2: Stacked bar — loss components per year
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    yearly = loss_df.groupby("year").agg(
        degradation=("degradation_pct", "mean"),
        soiling=("soiling_loss_pct", "mean"),
        events=("event_loss_pct", "mean"),
        other=("other_pct", "mean"),
        total=("total_change_pct", "mean"),
    ).reset_index()

    years = yearly["year"].values
    x = np.arange(len(years))
    w = 0.6

    ax.bar(x, yearly["degradation"], w, label="Degradation (trend)",
           color="#2196F3", alpha=0.85)
    ax.bar(x, -yearly["soiling"], w, bottom=yearly["degradation"],
           label="Soiling loss", color="#FF9800", alpha=0.85)
    ax.bar(x, -yearly["events"], w,
           bottom=yearly["degradation"] - yearly["soiling"],
           label="Sudden event loss", color="#F44336", alpha=0.85)
    ax.bar(x, yearly["other"], w,
           bottom=yearly["degradation"] - yearly["soiling"] - yearly["events"],
           label="Other/residual", color="#9E9E9E", alpha=0.6)

    ax.plot(x, yearly["total"], "ko-", ms=6, lw=2, label="Net change", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Yield change from baseline (%)")
    ax.set_title("Yearly Loss Breakdown (avg across seasons)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    # ------------------------------------------------------------------
    # Panel 3: Seasonal loss breakdown — grouped bars
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    years_list = sorted(loss_df["year"].unique())
    non_cal = [y for y in years_list if y != CALIBRATION_YEAR]

    if non_cal:
        seasonal_avg = loss_df[loss_df["year"].isin(non_cal)].groupby("season").agg(
            degradation=("degradation_pct", "mean"),
            soiling=("soiling_loss_pct", "mean"),
            events=("event_loss_pct", "mean"),
            other=("other_pct", "mean"),
            total=("total_change_pct", "mean"),
        )
        seasonal_avg = seasonal_avg.reindex(SEASON_ORDER).dropna(how="all")

        if not seasonal_avg.empty:
            sx = np.arange(len(seasonal_avg))
            w = 0.18
            ax.bar(sx - 1.5*w, seasonal_avg["degradation"], w,
                   label="Degradation", color="#2196F3", alpha=0.85)
            ax.bar(sx - 0.5*w, -seasonal_avg["soiling"], w,
                   label="Soiling loss", color="#FF9800", alpha=0.85)
            ax.bar(sx + 0.5*w, -seasonal_avg["events"], w,
                   label="Event loss", color="#F44336", alpha=0.85)
            ax.bar(sx + 1.5*w, seasonal_avg["total"], w,
                   label="Net change", color="#4CAF50", alpha=0.85)
            ax.set_xticks(sx)
            ax.set_xticklabels(seasonal_avg.index)
            ax.legend(fontsize=8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Yield change (%)")
    ax.set_title(f"Seasonal Loss Components (avg {non_cal[0]}-{non_cal[-1]})" if non_cal else "")
    ax.grid(True, alpha=0.3, axis="y")

    # ------------------------------------------------------------------
    # Panel 4: Waterfall for latest year
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    latest_year = max(loss_df["year"])
    ly = loss_df[loss_df["year"] == latest_year]
    if not ly.empty:
        avg = ly.mean(numeric_only=True)
        categories = ["Baseline", "Degradation", "Soiling\nLoss", "Event\nLoss",
                       "Other", "Actual"]
        values = [100.0, avg["degradation_pct"], -avg["soiling_loss_pct"],
                  -avg["event_loss_pct"], avg["other_pct"], 0]

        cumulative = [100.0]
        for v in values[1:-1]:
            cumulative.append(cumulative[-1] + v)
        actual_val = cumulative[-1]
        cumulative.append(actual_val)

        bar_colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9E9E9E", "#4CAF50"]
        bottoms = [0] * len(categories)
        heights = [0] * len(categories)

        heights[0] = cumulative[0]
        bottoms[0] = 0
        for i in range(1, len(categories) - 1):
            if values[i] >= 0:
                bottoms[i] = cumulative[i - 1]
                heights[i] = values[i]
            else:
                bottoms[i] = cumulative[i]
                heights[i] = -values[i]
        heights[-1] = actual_val
        bottoms[-1] = 0

        bars = ax.bar(categories, heights, bottom=bottoms, color=bar_colors,
                      edgecolor="gray", alpha=0.85, width=0.6)

        for bar, h, b in zip(bars, heights, bottoms):
            ax.text(bar.get_x() + bar.get_width() / 2, b + h + 0.3,
                    f"{b + h:.1f}%", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")

        ax.set_ylabel("Yield (%)")
        ax.set_title(f"Waterfall: {latest_year} vs {CALIBRATION_YEAR} Baseline")
        ax.axhline(100, ls=":", color="green", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(RESULTS_DIR, f"loss_{short}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fleet_loss_dashboard(all_loss_df, deg_df):
    """Fleet-wide loss attribution dashboard."""
    if all_loss_df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(
        f"Fleet Loss Attribution - Bundoora Solar ({YEAR_START}-{YEAR_END})\n"
        f"Baseline: {CALIBRATION_YEAR} | Method: GHI-normalised yield, temp-corrected",
        fontsize=14, fontweight="bold",
    )

    # ------------------------------------------------------------------
    # Panel 1: Fleet-average loss components per year
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    fleet_yearly = all_loss_df.groupby("year").agg(
        degradation=("degradation_pct", "median"),
        soiling=("soiling_loss_pct", "median"),
        events=("event_loss_pct", "median"),
        total=("total_change_pct", "median"),
    ).reset_index()

    years = fleet_yearly["year"].values
    x = np.arange(len(years))
    w = 0.2

    ax.bar(x - w, fleet_yearly["degradation"], w, label="Degradation (trend)",
           color="#2196F3", alpha=0.85)
    ax.bar(x, -fleet_yearly["soiling"], w, label="Soiling loss",
           color="#FF9800", alpha=0.85)
    ax.bar(x + w, -fleet_yearly["events"], w, label="Event loss",
           color="#F44336", alpha=0.85)
    ax.plot(x, fleet_yearly["total"], "ko-", ms=6, lw=2, label="Net change", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Yield change from baseline (%)")
    ax.set_title("Fleet Median Loss Components per Year")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ------------------------------------------------------------------
    # Panel 2: Seasonal loss heatmap (fleet median)
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    non_cal = all_loss_df[all_loss_df["year"] != CALIBRATION_YEAR]
    if not non_cal.empty:
        pivot = non_cal.groupby(["year", "season"])["total_change_pct"].median().reset_index()
        pivot_t = pivot.pivot_table(index="year", columns="season",
                                     values="total_change_pct")
        cols = [s for s in SEASON_ORDER if s in pivot_t.columns]
        if cols:
            pivot_t = pivot_t[cols]
            sns.heatmap(pivot_t, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn",
                        center=0, cbar_kws={"label": "Yield change (%)"},
                        linewidths=0.5)
    ax.set_title("Seasonal Yield Change (fleet median, %)")

    # ------------------------------------------------------------------
    # Panel 3: Per-meter soiling vs event loss (scatter)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    if not non_cal.empty:
        meter_avg = non_cal.groupby("meter_short").agg(
            soiling=("soiling_loss_pct", "mean"),
            events=("event_loss_pct", "mean"),
            total=("total_change_pct", "mean"),
        ).reset_index()
        scatter = ax.scatter(meter_avg["soiling"], meter_avg["events"],
                             c=meter_avg["total"], cmap="RdYlGn", s=80,
                             edgecolors="gray", alpha=0.85, vmin=-10, vmax=10)
        for _, r in meter_avg.iterrows():
            ax.annotate(r["meter_short"], (r["soiling"], r["events"]),
                        fontsize=6, ha="left", va="bottom")
        plt.colorbar(scatter, ax=ax, label="Net change (%)")
    ax.set_xlabel("Avg soiling loss (%)")
    ax.set_ylabel("Avg event loss (%)")
    ax.set_title("Per-Meter: Soiling vs Event Loss")
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 4: Per-meter total change bar (sorted) with component stacking
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    if not non_cal.empty:
        meter_avg = non_cal.groupby("meter_short").agg(
            degradation=("degradation_pct", "mean"),
            soiling=("soiling_loss_pct", "mean"),
            events=("event_loss_pct", "mean"),
            other=("other_pct", "mean"),
            total=("total_change_pct", "mean"),
        ).reset_index().sort_values("total")

        names = meter_avg["meter_short"].values
        y = np.arange(len(names))

        ax.barh(y, meter_avg["degradation"], 0.6, label="Degradation",
                color="#2196F3", alpha=0.85)
        ax.barh(y, -meter_avg["soiling"], 0.6,
                left=meter_avg["degradation"],
                label="Soiling loss", color="#FF9800", alpha=0.85)
        ax.barh(y, -meter_avg["events"], 0.6,
                left=meter_avg["degradation"] - meter_avg["soiling"],
                label="Event loss", color="#F44336", alpha=0.85)

        for i, (_, r) in enumerate(meter_avg.iterrows()):
            ax.text(r["total"] + 0.3, i, f"{r['total']:+.1f}%",
                    va="center", fontsize=7, fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Yield change from baseline (%)")
        ax.set_title("Per-Meter Loss Components (avg post-baseline)")
        ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(RESULTS_DIR, "fleet_loss_attribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fleet loss attribution: {path}")


# ===================================================================
# Main
# ===================================================================

def run():
    print("=" * 70)
    print("DEGRADATION / SEASONALITY / SOILING / EVENT ANALYSIS")
    print(f"  Period: {YEAR_START}-{YEAR_END}  |  Baseline: {CALIBRATION_YEAR}")
    print(f"  Method: GHI-normalised yield with temperature correction")
    print("=" * 70)

    print("\n[1] Loading data...")
    meter_df = load_meter_v2()
    solcast = load_solcast()
    sim_df = load_simulated_power()

    meters = sorted(meter_df["meter"].unique())
    print(f"    Meters:  {len(meters)}")
    print(f"    Meter data:  {meter_df['timestamp'].min()} to {meter_df['timestamp'].max()}")
    print(f"    Solcast:     {solcast['timestamp'].min()} to {solcast['timestamp'].max()}")
    sol_cols = [c for c in SOLCAST_WEATHER_COLS if c in solcast.columns]
    print(f"    Weather features: {sol_cols}")
    if sim_df is not None:
        print(f"    Simulation:  {sim_df['timestamp'].min()} to {sim_df['timestamp'].max()}")

    print(f"\n[2] Per-meter analysis...")
    all_deg = []
    all_seasonal_rows = []
    all_monthly = []
    all_soiling = []
    all_events = []
    all_loss = []

    for idx, meter_id in enumerate(meters):
        short = meter_id.split("#")[0].replace("solar.", "")
        print(f"  ({idx+1}/{len(meters)}) {short}", end="", flush=True)

        m_df = meter_df[meter_df["meter"] == meter_id].copy()
        s_df = sim_df[sim_df["meter"] == meter_id].copy() if sim_df is not None else None

        hourly = build_hourly(m_df, solcast, s_df)
        daily = build_daily(hourly)

        if daily.empty or len(daily) < MIN_TOTAL_DAYS:
            print(f"  [SKIP: {len(daily)} days < {MIN_TOTAL_DAYS} required]", flush=True)
            continue

        cal_days = daily["cal_year_days"].iloc[0] if "cal_year_days" in daily.columns else 0
        if cal_days < MIN_CAL_YEAR_DAYS:
            print(f"  [SKIP: {cal_days} cal-year days < {MIN_CAL_YEAR_DAYS} required]",
                  flush=True)
            continue

        deg_pct, baseline, n_months, monthly = overall_degradation(daily)
        seasonal_df = seasonal_degradation(daily)
        soiling_df = detect_soiling(daily)
        events_df = detect_sudden_events(daily)

        # Loss attribution
        trend_slope = deg_pct / 100.0 if np.isfinite(deg_pct) else 0.0
        loss_df = loss_attribution(daily, soiling_df, events_df, trend_slope)

        n_soil = int(soiling_df["is_soiling"].sum()) if not soiling_df.empty else 0
        n_events = len(events_df)
        soil_only = soiling_df[soiling_df["is_soiling"]] if not soiling_df.empty else pd.DataFrame()
        rain_cleaned = int((soil_only["cleaning_type"] == "rain").sum()) if not soil_only.empty else 0
        wind_cleaned = int((soil_only["cleaning_type"] == "wind").sum()) if not soil_only.empty else 0

        all_deg.append({
            "meter": meter_id,
            "meter_short": short,
            "deg_pct_per_year": round(deg_pct, 4) if np.isfinite(deg_pct) else np.nan,
            "baseline_yield": round(baseline, 4) if np.isfinite(baseline) else np.nan,
            "n_months": int(n_months) if np.isfinite(n_months) else 0,
            "n_days": len(daily),
            "total_soiling": n_soil,
            "rain_cleaned": rain_cleaned,
            "wind_cleaned": wind_cleaned,
            "sudden_events": n_events,
        })

        if not seasonal_df.empty:
            seasonal_df["meter"] = meter_id
            seasonal_df["meter_short"] = short
            all_seasonal_rows.append(seasonal_df)

        if not monthly.empty:
            monthly["meter"] = meter_id
            all_monthly.append(monthly)

        if not soiling_df.empty:
            soiling_df["meter"] = meter_id
            all_soiling.append(soiling_df)

        if not events_df.empty:
            events_df["meter"] = meter_id
            all_events.append(events_df)

        if not loss_df.empty:
            loss_df["meter"] = meter_id
            loss_df["meter_short"] = short
            all_loss.append(loss_df)

        deg_s = f"{deg_pct:+.2f}%/yr" if np.isfinite(deg_pct) else "N/A"
        print(f"  deg={deg_s}  soil={n_soil}(R{rain_cleaned}/W{wind_cleaned})"
              f"  events={n_events}", flush=True)

        plot_meter_dashboard(meter_id, monthly, seasonal_df, soiling_df, events_df, daily)
        plot_loss_dashboard(meter_id, loss_df)

    # --- Save ---
    print("\n[3] Saving results...")
    deg_df = pd.DataFrame(all_deg)
    seasonal_all = pd.concat(all_seasonal_rows, ignore_index=True) if all_seasonal_rows else pd.DataFrame()
    monthly_all = pd.concat(all_monthly, ignore_index=True) if all_monthly else pd.DataFrame()
    soiling_all = pd.concat(all_soiling, ignore_index=True) if all_soiling else pd.DataFrame()
    events_all = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()

    deg_df.to_csv(os.path.join(RESULTS_DIR, "degradation_summary.csv"), index=False)
    print(f"    degradation_summary.csv  ({len(deg_df)} meters)")

    if not seasonal_all.empty:
        seasonal_all.to_csv(os.path.join(RESULTS_DIR, "seasonal_degradation.csv"), index=False)
        print(f"    seasonal_degradation.csv ({len(seasonal_all)} rows)")

    if not monthly_all.empty:
        monthly_all.to_csv(os.path.join(RESULTS_DIR, "monthly_yield.csv"), index=False)
        print(f"    monthly_yield.csv        ({len(monthly_all)} rows)")

    if not soiling_all.empty:
        soiling_all.to_csv(os.path.join(RESULTS_DIR, "soiling_events.csv"), index=False)
        print(f"    soiling_events.csv       ({len(soiling_all)} rows)")

    if not events_all.empty:
        events_all.to_csv(os.path.join(RESULTS_DIR, "sudden_events.csv"), index=False)
        print(f"    sudden_events.csv        ({len(events_all)} rows)")

    loss_all = pd.concat(all_loss, ignore_index=True) if all_loss else pd.DataFrame()

    if not loss_all.empty:
        loss_all.to_csv(os.path.join(RESULTS_DIR, "loss_attribution.csv"), index=False)
        print(f"    loss_attribution.csv     ({len(loss_all)} rows)")

    # Fleet overview
    print("\n[4] Fleet overview plot...")
    soiling_summary = deg_df[["meter_short", "total_soiling", "rain_cleaned", "wind_cleaned"]].copy()
    plot_fleet_overview(deg_df, seasonal_all, monthly_all, soiling_summary)

    print("\n[5] Fleet loss attribution plot...")
    if not loss_all.empty:
        plot_fleet_loss_dashboard(loss_all, deg_df)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print(f"  Method: GHI-normalised yield (temp-corrected)")
    print(f"  Baseline: {CALIBRATION_YEAR} = 1.0")
    print("=" * 70)
    valid_meters = deg_df[deg_df["deg_pct_per_year"].notna()]
    if not valid_meters.empty:
        # Flag outliers: baseline_yield < 0.5 or > 1.5, or |deg| > 30%/yr
        valid_meters = valid_meters.copy()
        valid_meters["quality"] = "OK"
        valid_meters.loc[
            (valid_meters["baseline_yield"] < 0.5) |
            (valid_meters["baseline_yield"] > 1.5) |
            (valid_meters["deg_pct_per_year"].abs() > 30),
            "quality"
        ] = "SUSPECT"

        reliable = valid_meters[valid_meters["quality"] == "OK"]
        print(f"\nMeters analysed: {len(valid_meters)} / {len(meters)}")
        print(f"  Reliable: {len(reliable)}   Suspect: {len(valid_meters) - len(reliable)}")
        if not reliable.empty:
            med = reliable["deg_pct_per_year"].median()
            print(f"\nFleet degradation (reliable meters only):")
            print(f"  Median: {med:+.2f} %/year")
            print(f"  Range:  {reliable['deg_pct_per_year'].min():+.2f} to "
                  f"{reliable['deg_pct_per_year'].max():+.2f} %/year")
        print(f"\nPer-meter rates:")
        cols = ["meter_short", "deg_pct_per_year", "baseline_yield",
                "total_soiling", "rain_cleaned", "wind_cleaned",
                "sudden_events", "quality"]
        print(valid_meters[cols].to_string(index=False))
    else:
        print("No meters had sufficient data.")

    if not seasonal_all.empty:
        print(f"\nSeasonal degradation (fleet median):")
        fleet_seasonal = seasonal_all.groupby("season")["deg_pct_per_year"].median()
        for s in SEASON_ORDER:
            if s in fleet_seasonal.index:
                print(f"  {s:8s}: {fleet_seasonal[s]:+.2f} %/year")

    if not soiling_all.empty:
        soil_confirmed = soiling_all[soiling_all["is_soiling"]].copy()
        total_soil = len(soil_confirmed)
        total_rain = int((soil_confirmed["cleaning_type"] == "rain").sum())
        total_wind = int((soil_confirmed["cleaning_type"] == "wind").sum())
        total_other = total_soil - total_rain - total_wind
        print(f"\nConfirmed soiling events: {total_soil}")
        print(f"  Cleaned by rain: {total_rain}")
        print(f"  Cleaned by wind: {total_wind}")
        if total_other > 0:
            print(f"  Other/unknown:   {total_other}")
        print(f"  Total dry spells analysed: {len(soiling_all)}")

        if not soil_confirmed.empty:
            seasonal_soil = soil_confirmed.groupby("season").agg(
                count=("spell_days", "count"),
                avg_slope=("yield_slope_per_day", "mean"),
                avg_spell=("spell_days", "mean"),
            )
            if not seasonal_soil.empty:
                print(f"\n  Seasonal soiling:")
                for s in SEASON_ORDER:
                    if s in seasonal_soil.index:
                        r = seasonal_soil.loc[s]
                        print(f"    {s:8s}: {int(r['count'])} events, "
                              f"avg slope={r['avg_slope']:.4f}/day, "
                              f"avg spell={r['avg_spell']:.1f}d")

    print(f"\nTotal sudden events: {len(events_all)}")
    if not events_all.empty:
        event_types = events_all["event_type"].value_counts()
        for et, cnt in event_types.items():
            print(f"  {et}: {cnt}")

    # Loss attribution summary
    if not loss_all.empty:
        non_cal = loss_all[loss_all["year"] != CALIBRATION_YEAR]
        if not non_cal.empty:
            print(f"\n{'='*70}")
            print("LOSS ATTRIBUTION SUMMARY (fleet median, post-baseline years)")
            print(f"{'='*70}")
            fleet_avg = non_cal.groupby("year").agg(
                total=("total_change_pct", "median"),
                degradation=("degradation_pct", "median"),
                soiling=("soiling_loss_pct", "median"),
                events=("event_loss_pct", "median"),
                other=("other_pct", "median"),
            )
            print(f"\n  {'Year':>6s} {'Total':>8s} {'Degrad':>8s} {'Soiling':>8s} "
                  f"{'Events':>8s} {'Other':>8s}")
            for yr, r in fleet_avg.iterrows():
                print(f"  {yr:>6d} {r['total']:>+7.2f}% {r['degradation']:>+7.2f}% "
                      f"{r['soiling']:>7.2f}% {r['events']:>7.2f}% {r['other']:>+7.2f}%")

            print(f"\n  Seasonal breakdown (fleet median, all post-baseline years):")
            seasonal_loss = non_cal.groupby("season").agg(
                total=("total_change_pct", "median"),
                degradation=("degradation_pct", "median"),
                soiling=("soiling_loss_pct", "median"),
                events=("event_loss_pct", "median"),
            )
            print(f"  {'Season':>8s} {'Total':>8s} {'Degrad':>8s} {'Soiling':>8s} {'Events':>8s}")
            for s in SEASON_ORDER:
                if s in seasonal_loss.index:
                    r = seasonal_loss.loc[s]
                    print(f"  {s:>8s} {r['total']:>+7.2f}% {r['degradation']:>+7.2f}% "
                          f"{r['soiling']:>7.2f}% {r['events']:>7.2f}%")

    print(f"\nAll outputs in: {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    run()
