"""
Soiling & Cleaning Effectiveness Analysis
==========================================
Analyses whether rain and wind cleaning truly restores panel yield,
considering panel orientation, tilt angle, and wind/rain direction.

Key analyses:
  1) Panel orientation profiling from simulation data
  2) Soiling spell detection with cross-meter consistency validation
  3) Rain vs wind cleaning effectiveness (confirmed by yield recovery)
  4) Wind/rain direction relative to panel facing
  5) Tilt-angle effect on soiling accumulation and cleaning

Outputs (in Results_seasonality/soiling_cleaning/):
  - panel_orientation.csv
  - soiling_spells.csv
  - cleaning_effectiveness.csv
  - cross_meter_validation.csv
  - 6-panel dashboard PNG
  - orientation_analysis PNG
  - cleaning_confirmation PNG
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
SIM_DIR = os.path.join(DATA_DIR, "simulated")

_run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_DIR = os.path.join(BASE, "Results_seasonality", "soiling_cleaning", _run_stamp)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────
METER_CSV = "SolarMeterReadings1hour_cleaned_2020_2025.csv"
SIMULATION_CSV = "solarsitesimulation.csv"
SOLCAST_CSV = "solcast_df_cleaned_2020_2025.csv"

EXCLUDE_METERS = ["bun_rd1", "bun_rd2", "bun_busstop"]
YEAR_START, YEAR_END = 2021, 2025
CALIBRATION_YEAR = 2021

ZENITH_DAYLIGHT = 85
GHI_MIN_HOURLY = 50.0
GHI_MIN_DAILY = 2.0
MIN_DAYLIGHT_HOURS = 4
MIN_CAL_YEAR_DAYS = 60

TEMP_COEFF = -0.004
T_REF = 25.0
T_CELL_OFFSET = 20.0

SEASON_MAP = {12: "Summer", 1: "Summer", 2: "Summer",
              3: "Autumn", 4: "Autumn", 5: "Autumn",
              6: "Winter", 7: "Winter", 8: "Winter",
              9: "Spring", 10: "Spring", 11: "Spring"}
SEASON_ORDER = ["Summer", "Autumn", "Winter", "Spring"]

SOILING_SPELL_MIN_DAYS = 5
SOILING_SPELL_MAX_DAYS = 60
PRECIP_THRESHOLD = 0.3     # mm/h
WIND_CLEAN_SPEED = 7.0     # m/s
SOILING_MIN_DECLINE = -0.002
SOILING_P_THRESHOLD = 0.10
ANOMALY_SLOPE_THRESHOLD = -0.05  # steeper than this -> likely meter fault

SOLCAST_WEATHER_COLS = [
    "ghi", "zenith", "precipitation_rate", "air_temp", "cloud_opacity",
    "wind_speed_10m", "wind_direction_10m", "relative_humidity",
    "snow_soiling_rooftop",
]

# ── colours ───────────────────────────────────────────────────────────
C_RAIN = "#2196F3"
C_WIND = "#FF9800"
C_NONE = "#9E9E9E"
C_YES = "#4CAF50"
C_NO = "#F44336"
C_EAST = "#E91E63"
C_NORTH = "#3F51B5"
C_STEEP = "#009688"
C_FLAT = "#FF5722"
C_MED = "#795548"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 180,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.facecolor": "white",
})

# =====================================================================
#  DATA LOADING
# =====================================================================

def _short_name(meter_full):
    """Extract short name like 'library' from full meter string."""
    import re
    m = re.search(r"bun_([a-z0-9_]+)#", meter_full, re.IGNORECASE)
    return m.group(1) if m else meter_full


def load_meter_data():
    df = pd.read_csv(os.path.join(DATA_DIR, METER_CSV))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[df["meter"].str.contains("bun_", case=False, na=False)].copy()
    df = df[(df["timestamp"].dt.year >= YEAR_START) &
            (df["timestamp"].dt.year <= YEAR_END)].copy()
    for ex in EXCLUDE_METERS:
        df = df[~df["meter"].str.contains(ex, case=False, na=False)]
    return df


def load_solcast():
    df = pd.read_csv(os.path.join(DATA_DIR, SOLCAST_CSV))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if "campus" in df.columns:
        df = df[df["campus"].str.upper().str.strip() == "BUNDOORA"].copy()
    df["timestamp"] = df["timestamp"].dt.floor("h")
    df = df[(df["timestamp"].dt.year >= YEAR_START) &
            (df["timestamp"].dt.year <= YEAR_END)].copy()
    if len(df) > 1 and df["timestamp"].diff().dropna().min() < pd.Timedelta("45min"):
        df = df.set_index("timestamp").resample("1h").mean(numeric_only=True).reset_index()
    return df


def load_simulation():
    p = os.path.join(SIM_DIR, SIMULATION_CSV)
    if not os.path.isfile(p):
        return None
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[df["meter"].str.contains("bun_", case=False, na=False)].copy()
    return df


# =====================================================================
#  1) PANEL ORIENTATION PROFILING
# =====================================================================

def profile_panel_orientations(sim_df):
    """
    Extract tilt and azimuth characteristics for each meter from
    the simulation's solar_incident_angle and POA irradiance values.
    Returns a DataFrame with one row per meter.
    """
    if sim_df is None or sim_df.empty:
        return pd.DataFrame()

    sim = sim_df.copy()
    sim["short"] = sim["meter"].apply(_short_name)
    sim["timestamp"] = pd.to_datetime(sim["timestamp"])
    sim["hour"] = sim["timestamp"].dt.hour

    required = ["solar_incident_angle", "poa_direct_irradiance",
                 "poa_diffuse_irradiance", "global_horizontal_irradiance"]
    for c in required:
        if c not in sim.columns:
            return pd.DataFrame()

    # Use clear-sky summer days for best signal
    clear_days = sim[(sim["hour"] == 12) &
                     (sim["global_horizontal_irradiance"] > 700) &
                     (sim["timestamp"].dt.month.isin([1, 12]))]
    clear_dates = set(clear_days["timestamp"].dt.date.unique())

    summer_noon = sim[(sim["hour"] == 12) &
                      (sim["timestamp"].dt.date.isin(clear_dates))]
    morning = sim[(sim["hour"] == 9) &
                  (sim["timestamp"].dt.date.isin(clear_dates))]
    afternoon = sim[(sim["hour"] == 15) &
                    (sim["timestamp"].dt.date.isin(clear_dates))]

    rows = []
    for meter_short in sim["short"].unique():
        noon_m = summer_noon[summer_noon["short"] == meter_short]
        morn_m = morning[morning["short"] == meter_short]
        aftn_m = afternoon[afternoon["short"] == meter_short]

        if noon_m.empty:
            continue

        noon_inc = noon_m["solar_incident_angle"].median()
        noon_poa_dir = noon_m["poa_direct_irradiance"].median()
        morn_poa_dir = morn_m["poa_direct_irradiance"].median() if len(morn_m) else np.nan
        aftn_poa_dir = aftn_m["poa_direct_irradiance"].median() if len(aftn_m) else np.nan
        nameplate = noon_m["nameplate_power"].median() if "nameplate_power" in noon_m.columns else np.nan

        # Infer facing direction from AM/PM asymmetry
        if pd.notna(morn_poa_dir) and pd.notna(aftn_poa_dir) and aftn_poa_dir > 0:
            am_pm_ratio = morn_poa_dir / aftn_poa_dir
        else:
            am_pm_ratio = np.nan

        if pd.notna(am_pm_ratio):
            if am_pm_ratio > 4.0:
                facing = "East/NE"
            elif am_pm_ratio < 1.0:
                facing = "West/NW"
            else:
                facing = "North"
        else:
            facing = "Unknown"

        if noon_inc < 8:
            tilt_group = "Flat (<8°)"
        elif noon_inc < 12:
            tilt_group = "Medium (8-12°)"
        else:
            tilt_group = "Steep (>12°)"

        full_meter = noon_m["meter"].iloc[0]
        rows.append({
            "meter": full_meter,
            "short": meter_short,
            "noon_incident_angle": round(noon_inc, 2),
            "noon_poa_direct": round(noon_poa_dir, 2),
            "morning_poa_direct": round(morn_poa_dir, 2) if pd.notna(morn_poa_dir) else np.nan,
            "afternoon_poa_direct": round(aftn_poa_dir, 2) if pd.notna(aftn_poa_dir) else np.nan,
            "am_pm_ratio": round(am_pm_ratio, 3) if pd.notna(am_pm_ratio) else np.nan,
            "facing": facing,
            "tilt_group": tilt_group,
            "nameplate_W": nameplate,
        })

    orient = pd.DataFrame(rows)
    for ex in EXCLUDE_METERS:
        tag = ex.replace("bun_", "")
        orient = orient[~orient["short"].str.contains(tag, case=False, na=False)]
    return orient


# =====================================================================
#  2) BUILD DAILY YIELD PER METER
# =====================================================================

def build_daily_yield(meter_df, solcast_df):
    """Return dict  meter_short -> daily DataFrame."""
    wcols = ["timestamp"] + [c for c in SOLCAST_WEATHER_COLS if c in solcast_df.columns]
    sol = solcast_df[wcols].copy()

    meter_daily = {}
    for meter_name, grp in meter_df.groupby("meter"):
        short = _short_name(meter_name)
        m = grp.copy()
        m["timestamp"] = pd.to_datetime(m["timestamp"]).dt.floor("h")
        m = m.merge(sol, on="timestamp", how="left")

        if "zenith" in m.columns:
            m.loc[m["zenith"].fillna(100) >= ZENITH_DAYLIGHT, "meter_reading"] = 0.0

        m["valid"] = (m["meter_reading"].notna() & (m["meter_reading"] > 0) &
                      m["ghi"].notna() & (m["ghi"] > GHI_MIN_HOURLY))
        if "zenith" in m.columns:
            m["valid"] = m["valid"] & (m["zenith"] < ZENITH_DAYLIGHT)

        m["yield_ghi"] = np.where(m["valid"] & (m["ghi"] > GHI_MIN_HOURLY),
                                  m["meter_reading"] / m["ghi"], np.nan)

        if "air_temp" in m.columns:
            t_cell = m["air_temp"].fillna(T_REF - T_CELL_OFFSET) + T_CELL_OFFSET
            tf = (1.0 + TEMP_COEFF * (t_cell - T_REF)).clip(0.5, 1.5)
            m["yield_tc"] = np.where(m["valid"], m["yield_ghi"] / tf, np.nan)
        else:
            m["yield_tc"] = m["yield_ghi"]

        ts = pd.to_datetime(m["timestamp"])
        m["date"] = ts.dt.date
        m["year"] = ts.dt.year
        m["month"] = ts.dt.month
        m["season"] = m["month"].map(SEASON_MAP)

        valid = m[m["valid"]].copy()
        if valid.empty:
            continue

        agg = {"meter_reading": "sum", "ghi": "sum", "yield_tc": "median", "valid": "count"}
        for c in ["precipitation_rate", "wind_speed_10m", "wind_direction_10m",
                   "relative_humidity", "air_temp", "snow_soiling_rooftop"]:
            if c in valid.columns:
                agg[c] = "mean"
        daily = valid.groupby("date").agg(agg).reset_index()
        daily.rename(columns={"valid": "n_hours"}, inplace=True)
        daily["date_dt"] = pd.to_datetime(daily["date"])
        daily["year"] = daily["date_dt"].dt.year
        daily["month"] = daily["date_dt"].dt.month
        daily["season"] = daily["month"].map(SEASON_MAP)
        daily["daily_ghi_kwh"] = daily["ghi"] / 1000.0

        daily = daily[(daily["n_hours"] >= MIN_DAYLIGHT_HOURS) &
                      (daily["daily_ghi_kwh"] >= GHI_MIN_DAILY)].copy()
        if daily.empty:
            continue

        daily["yield_daily"] = daily["meter_reading"] / daily["daily_ghi_kwh"]
        q1, q3 = daily["yield_daily"].quantile(0.05), daily["yield_daily"].quantile(0.95)
        iqr = q3 - q1
        daily.loc[~daily["yield_daily"].between(q1 - 2 * iqr, q3 + 2 * iqr), "yield_daily"] = np.nan

        cal = daily[daily["year"] == CALIBRATION_YEAR]["yield_daily"].dropna()
        norm = cal.median() if len(cal) >= MIN_CAL_YEAR_DAYS else daily["yield_daily"].dropna().median()
        if norm < 1e-6:
            norm = 1.0
        daily["yield_norm"] = daily["yield_daily"] / norm
        daily["meter"] = meter_name
        daily["short"] = short
        meter_daily[short] = daily

    return meter_daily


# =====================================================================
#  3) SOILING SPELL DETECTION (per meter)
# =====================================================================

def detect_spells(daily):
    """Detect dry-spell soiling periods and what happened after each."""
    if daily.empty or len(daily) < SOILING_SPELL_MIN_DAYS + 2:
        return pd.DataFrame()

    d = daily.dropna(subset=["yield_norm"]).sort_values("date").reset_index(drop=True)
    has_precip = "precipitation_rate" in d.columns
    has_wind = "wind_speed_10m" in d.columns

    d["rain_clean"] = (d["precipitation_rate"] >= PRECIP_THRESHOLD) if has_precip else False
    d["wind_clean"] = (d["wind_speed_10m"] >= WIND_CLEAN_SPEED) if has_wind else False
    d["is_soiling_day"] = ~(d["rain_clean"] | d["wind_clean"])

    spells, i, n = [], 0, len(d)
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
        y_s = y[ok][0]; y_e = y[ok][-1]

        recovery, cleaning_type = np.nan, "none"
        post_wind_dir, post_precip = np.nan, np.nan
        if s_end + 3 < n:
            post = d.iloc[s_end + 1:s_end + 4]
            rv = post["yield_norm"].median()
            recovery = rv - y_e if (np.isfinite(rv) and np.isfinite(y_e)) else np.nan
            if post["rain_clean"].any():
                cleaning_type = "rain"
            elif post["wind_clean"].any():
                cleaning_type = "wind"
            else:
                cleaning_type = "unknown"
            if "wind_direction_10m" in post.columns:
                post_wind_dir = post["wind_direction_10m"].mean()
            if has_precip:
                post_precip = post["precipitation_rate"].mean()

        is_soiling = (slope < SOILING_MIN_DECLINE) and (p < SOILING_P_THRESHOLD)
        is_anomaly = abs(slope) > abs(ANOMALY_SLOPE_THRESHOLD)
        cleaning_confirmed = (cleaning_type in ("rain", "wind")) and pd.notna(recovery) and (recovery > 0)
        rec_ratio = (recovery / (y_s - y_e)) if (pd.notna(recovery) and (y_s - y_e) > 0.01) else np.nan

        events.append({
            "start_date": sub.iloc[0]["date"],
            "end_date": sub.iloc[-1]["date"],
            "season": sub.iloc[0]["season"],
            "spell_days": s_len,
            "slope_per_day": round(slope, 6),
            "yield_start": round(y_s, 4),
            "yield_end": round(y_e, 4),
            "yield_drop": round(y_s - y_e, 4),
            "yield_recovery": round(recovery, 4) if pd.notna(recovery) else np.nan,
            "recovery_ratio": round(rec_ratio, 3) if pd.notna(rec_ratio) else np.nan,
            "cleaning_type": cleaning_type,
            "cleaning_confirmed": cleaning_confirmed,
            "is_soiling": is_soiling,
            "is_anomaly": is_anomaly,
            "avg_wind_speed": round(sub["wind_speed_10m"].mean(), 2) if has_wind else np.nan,
            "avg_wind_dir": round(sub["wind_direction_10m"].mean(), 1) if "wind_direction_10m" in sub.columns else np.nan,
            "post_wind_dir": round(post_wind_dir, 1) if pd.notna(post_wind_dir) else np.nan,
            "post_precip": round(post_precip, 3) if pd.notna(post_precip) else np.nan,
            "avg_humidity": round(sub["relative_humidity"].mean(), 1) if "relative_humidity" in sub.columns else np.nan,
            "r_value": round(r, 4),
            "p_value": round(p, 4),
        })
    return pd.DataFrame(events)


# =====================================================================
#  4) CROSS-METER VALIDATION
# =====================================================================

def cross_meter_validate(all_spells):
    """
    For each soiling spell start_date, check how many meters show the
    same pattern. A spell is 'fleet-confirmed' if >= 3 meters have
    a negative slope starting on the same day.
    """
    if all_spells.empty:
        return all_spells

    date_groups = all_spells.groupby("start_date")
    records = []
    for start_date, grp in date_groups:
        n_meters = grp["short"].nunique()
        n_declining = (grp["slope_per_day"] < 0).sum()
        n_soiling = grp["is_soiling"].sum()
        n_recovered = grp["cleaning_confirmed"].sum()
        avg_slope = grp["slope_per_day"].mean()
        slope_std = grp["slope_per_day"].std()

        for _, row in grp.iterrows():
            rec = row.to_dict()
            rec["fleet_meters_same_spell"] = n_meters
            rec["fleet_declining_count"] = n_declining
            rec["fleet_soiling_count"] = n_soiling
            rec["fleet_recovered_count"] = n_recovered
            rec["fleet_avg_slope"] = round(avg_slope, 6)
            rec["fleet_slope_std"] = round(slope_std, 6) if pd.notna(slope_std) else np.nan
            rec["fleet_confirmed"] = n_soiling >= 3
            rec["consistent_with_fleet"] = (
                (row["slope_per_day"] < 0) and
                (n_declining >= 3) and
                (abs(row["slope_per_day"] - avg_slope) < 2 * (slope_std if pd.notna(slope_std) else 999))
            )
            records.append(rec)

    return pd.DataFrame(records)


# =====================================================================
#  5) WIND / RAIN DIRECTION ANALYSIS
# =====================================================================

def weather_direction_profile(solcast_df):
    """Summarise wind direction patterns during daylight, rain, and strong wind."""
    sol = solcast_df.copy()
    sol["timestamp"] = pd.to_datetime(sol["timestamp"])
    daylight = sol[sol["zenith"] < ZENITH_DAYLIGHT].copy()
    daylight["month"] = daylight["timestamp"].dt.month
    daylight["season"] = daylight["month"].map(SEASON_MAP)

    bins = np.arange(0, 361, 45)
    labels_dir = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    results = {}

    for label, subset in [("all_daylight", daylight),
                          ("during_rain", daylight[daylight["precipitation_rate"] >= PRECIP_THRESHOLD]),
                          ("strong_wind", daylight[daylight["wind_speed_10m"] >= WIND_CLEAN_SPEED])]:
        if subset.empty:
            continue
        wd = subset["wind_direction_10m"].dropna()
        counts = pd.cut(wd, bins=bins, labels=labels_dir, right=False).value_counts()
        pct = (counts / counts.sum() * 100).round(1)
        results[label] = {
            "avg_direction": round(wd.mean(), 1),
            "avg_speed": round(subset["wind_speed_10m"].mean(), 2),
            "n_hours": len(subset),
            "direction_pct": pct.to_dict(),
        }

    # Seasonal breakdown
    seasonal = {}
    for s in SEASON_ORDER:
        ss = daylight[daylight["season"] == s]
        if ss.empty:
            continue
        seasonal[s] = {
            "avg_wind_dir": round(ss["wind_direction_10m"].mean(), 1),
            "avg_wind_speed": round(ss["wind_speed_10m"].mean(), 2),
            "avg_precip": round(ss["precipitation_rate"].mean(), 3),
        }
    results["seasonal"] = seasonal
    return results


# =====================================================================
#  6) PLOTS
# =====================================================================

def _wind_rose_on_ax(ax, direction_pct, title, color="#4FC3F7"):
    """Draw a simple wind-rose bar chart on a polar axis."""
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    vals = [direction_pct.get(l, 0) for l in labels]
    bars = ax.bar(angles, vals, width=0.6, alpha=0.75, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=9, pad=12, fontweight="bold")
    ax.set_yticklabels([])
    return bars


def plot_main_dashboard(all_spells, orient_df, weather_info, solcast_df):
    """
    6-panel dashboard:
      [1] Wind rose: daylight vs rain vs strong wind
      [2] Cleaning confirmation rate: rain vs wind
      [3] Soiling rate by tilt group
      [4] Cross-meter consistency heatmap
      [5] Recovery scatter: yield_drop vs recovery, coloured by cleaning type
      [6] Seasonal cleaning effectiveness
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Soiling & Cleaning Effectiveness Analysis — Bundoora Campus",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35,
                           left=0.06, right=0.96, top=0.92, bottom=0.06)

    # ── Panel 1: Wind roses ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0], polar=True)
    if "all_daylight" in weather_info:
        _wind_rose_on_ax(ax1, weather_info["all_daylight"]["direction_pct"],
                         "Prevailing Wind (daylight)", "#90CAF9")
    if "during_rain" in weather_info:
        rain_pct = weather_info["during_rain"]["direction_pct"]
        labels_d = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        vals_rain = [rain_pct.get(l, 0) for l in labels_d]
        ax1.bar(angles, vals_rain, width=0.35, alpha=0.6, color=C_RAIN,
                edgecolor="white", linewidth=0.3)
    if "strong_wind" in weather_info:
        sw_pct = weather_info["strong_wind"]["direction_pct"]
        vals_sw = [sw_pct.get(l, 0) for l in labels_d]
        ax1.bar(angles + 0.15, vals_sw, width=0.2, alpha=0.7, color=C_WIND,
                edgecolor="white", linewidth=0.3)
    ax1.legend(["Daylight", "Rain (>0.3mm/h)", "Strong wind (>7m/s)"],
               loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7)

    # ── Panel 2: Cleaning confirmation rate ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    soiling_only = all_spells[all_spells["is_soiling"] & ~all_spells["is_anomaly"]].copy()
    clean_summary = []
    for ct in ["rain", "wind"]:
        sub = soiling_only[soiling_only["cleaning_type"] == ct]
        n_total = len(sub)
        if n_total == 0:
            continue
        n_confirmed = sub["cleaning_confirmed"].sum()
        n_not = n_total - n_confirmed
        clean_summary.append({"type": ct.title(), "confirmed": n_confirmed,
                              "not_confirmed": n_not, "total": n_total})

    if clean_summary:
        cs = pd.DataFrame(clean_summary)
        x = np.arange(len(cs))
        w = 0.35
        ax2.bar(x - w / 2, cs["confirmed"], w, color=C_YES, label="Yield recovered (confirmed)")
        ax2.bar(x + w / 2, cs["not_confirmed"], w, color=C_NO, label="No recovery (not confirmed)")
        for i, row in cs.iterrows():
            pct = row["confirmed"] / row["total"] * 100
            ax2.text(i, row["total"] + 0.3, f"{pct:.0f}%", ha="center", fontsize=9, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(cs["type"])
        ax2.set_ylabel("Number of soiling events")
        ax2.legend(loc="upper right", fontsize=7)
    ax2.set_title("Cleaning Confirmation Rate\n(did yield actually recover?)", fontweight="bold")

    # ── Panel 3: Soiling by tilt group ───────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if not orient_df.empty and not soiling_only.empty:
        merged = soiling_only.merge(orient_df[["short", "tilt_group", "facing"]],
                                    on="short", how="left")
        tilt_order = ["Flat (<8°)", "Medium (8-12°)", "Steep (>12°)"]
        tilt_colors = {tilt_order[0]: C_FLAT, tilt_order[1]: C_MED, tilt_order[2]: C_STEEP}

        tilt_stats = []
        for tg in tilt_order:
            sub = merged[merged["tilt_group"] == tg]
            if sub.empty:
                tilt_stats.append({"tilt": tg, "n_events": 0, "avg_slope": 0,
                                   "pct_confirmed": 0, "n_meters": 0})
                continue
            n_m = sub["short"].nunique()
            tilt_stats.append({
                "tilt": tg,
                "n_events": len(sub),
                "avg_slope": sub["slope_per_day"].mean() * 1000,
                "pct_confirmed": sub["cleaning_confirmed"].sum() / len(sub) * 100,
                "n_meters": n_m,
            })
        ts_df = pd.DataFrame(tilt_stats)
        x = np.arange(len(ts_df))
        colors = [tilt_colors.get(t, C_NONE) for t in ts_df["tilt"]]
        bars = ax3.bar(x, ts_df["avg_slope"].abs(), color=colors, alpha=0.8, edgecolor="white")
        for i, row in ts_df.iterrows():
            ax3.text(i, row["avg_slope"] * -1 + 0.2 if row["avg_slope"] < 0 else abs(row["avg_slope"]) + 0.2,
                     f"{row['n_events']} events\n{row['n_meters']} meters\n{row['pct_confirmed']:.0f}% cleaned",
                     ha="center", fontsize=7, va="bottom")
        ax3.set_xticks(x)
        ax3.set_xticklabels(ts_df["tilt"], fontsize=8)
        ax3.set_ylabel("Avg soiling rate (|slope| × 1000/day)")
    ax3.set_title("Soiling Rate by Panel Tilt\n(excludes meter anomalies)", fontweight="bold")

    # ── Panel 4: Cross-meter consistency ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if "fleet_confirmed" in all_spells.columns:
        soiling_confirmed = soiling_only[~soiling_only["is_anomaly"]].copy()
        if not soiling_confirmed.empty:
            piv_data = []
            for _, row in soiling_confirmed.iterrows():
                piv_data.append({
                    "meter": row["short"],
                    "spell": str(row["start_date"]),
                    "consistent": 1 if row.get("consistent_with_fleet", False) else 0,
                    "fleet_count": row.get("fleet_soiling_count", 0),
                })
            piv_df = pd.DataFrame(piv_data)
            top_spells = piv_df.groupby("spell")["fleet_count"].first().nlargest(10).index
            piv_df = piv_df[piv_df["spell"].isin(top_spells)]
            if not piv_df.empty:
                hm = piv_df.pivot_table(index="meter", columns="spell",
                                        values="consistent", fill_value=-1)
                cmap = plt.cm.colors.ListedColormap([C_NONE, C_NO, C_YES])
                bounds = [-1.5, -0.5, 0.5, 1.5]
                norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
                ax4.imshow(hm.values, cmap=cmap, norm=norm, aspect="auto")
                ax4.set_xticks(range(len(hm.columns)))
                ax4.set_xticklabels([c[:10] for c in hm.columns], rotation=45, ha="right", fontsize=6)
                ax4.set_yticks(range(len(hm.index)))
                ax4.set_yticklabels(hm.index, fontsize=6)
    ax4.set_title("Cross-Meter Consistency\n(green=consistent, red=inconsistent, grey=no spell)",
                  fontweight="bold")

    # ── Panel 5: Recovery scatter ────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    plot_df = soiling_only.dropna(subset=["yield_drop", "yield_recovery"]).copy()
    if not plot_df.empty:
        color_map = {"rain": C_RAIN, "wind": C_WIND, "unknown": C_NONE, "none": C_NONE}
        for ct in ["rain", "wind", "unknown", "none"]:
            sub = plot_df[plot_df["cleaning_type"] == ct]
            if sub.empty:
                continue
            marker = "o" if ct in ("rain", "wind") else "x"
            ax5.scatter(sub["yield_drop"], sub["yield_recovery"], c=color_map[ct],
                        label=ct.title(), alpha=0.65, s=30, marker=marker, edgecolors="white", linewidths=0.3)
        ax5.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax5.axline((0, 0), slope=1, color="grey", linewidth=0.5, linestyle=":", label="Full recovery line")
        ax5.set_xlabel("Yield drop during spell")
        ax5.set_ylabel("Yield recovery after cleaning event")
        ax5.legend(fontsize=7)
        ax5.annotate("CLEANED\n(recovery > 0)", xy=(0.7, 0.85), xycoords="axes fraction",
                     fontsize=8, color=C_YES, fontweight="bold", ha="center")
        ax5.annotate("NOT CLEANED\n(recovery <= 0)", xy=(0.7, 0.15), xycoords="axes fraction",
                     fontsize=8, color=C_NO, fontweight="bold", ha="center")
    ax5.set_title("Yield Drop vs Recovery\n(above zero line = real cleaning)", fontweight="bold")

    # ── Panel 6: Seasonal cleaning effectiveness ─────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    if not soiling_only.empty:
        season_data = []
        for s in SEASON_ORDER:
            sub = soiling_only[soiling_only["season"] == s]
            if len(sub) == 0:
                continue
            n = len(sub)
            confirmed = sub["cleaning_confirmed"].sum()
            avg_drop = sub["yield_drop"].mean()
            avg_rec = sub["yield_recovery"].dropna().mean()
            season_data.append({"season": s, "n_events": n, "confirmed": confirmed,
                                "pct": confirmed / n * 100, "avg_drop": avg_drop,
                                "avg_recovery": avg_rec})
        if season_data:
            sd = pd.DataFrame(season_data)
            x = np.arange(len(sd))
            ax6.bar(x - 0.2, sd["avg_drop"], 0.35, color="#EF5350", alpha=0.8, label="Avg yield drop")
            ax6.bar(x + 0.2, sd["avg_recovery"], 0.35, color=C_YES, alpha=0.8, label="Avg yield recovery")
            ax6.axhline(0, color="black", linewidth=0.5)
            for i, row in sd.iterrows():
                ax6.text(i, max(row["avg_drop"], row["avg_recovery"]) + 0.02,
                         f"{row['n_events']} events\n{row['pct']:.0f}% confirmed",
                         ha="center", fontsize=7, va="bottom")
            ax6.set_xticks(x)
            ax6.set_xticklabels(sd["season"])
            ax6.set_ylabel("Normalised yield change")
            ax6.legend(fontsize=7)
    ax6.set_title("Seasonal Cleaning Effectiveness\n(drop vs recovery by season)", fontweight="bold")

    fig.savefig(os.path.join(RESULTS_DIR, "soiling_cleaning_dashboard.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] soiling_cleaning_dashboard.png")


def plot_orientation_analysis(all_spells, orient_df):
    """
    4-panel orientation-focused figure:
      [1] Bubble chart: incident angle vs soiling rate, sized by events
      [2] East vs North facing: cleaning rates
      [3] Per-meter soiling event count with orientation colouring
      [4] Tilt vs cleaning success scatter
    """
    if orient_df.empty or all_spells.empty:
        return

    soiling = all_spells[all_spells["is_soiling"] & ~all_spells["is_anomaly"]].copy()
    soiling = soiling.merge(orient_df[["short", "tilt_group", "facing",
                                        "noon_incident_angle"]], on="short", how="left")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Panel Orientation & Soiling Analysis", fontsize=14, fontweight="bold", y=0.98)
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # ── Panel 1: Incident angle vs avg soiling slope ─────────────────
    ax1 = axes[0, 0]
    per_meter = soiling.groupby("short").agg(
        n_events=("slope_per_day", "count"),
        avg_slope=("slope_per_day", "mean"),
        pct_confirmed=("cleaning_confirmed", "mean"),
    ).reset_index()
    per_meter = per_meter.merge(orient_df[["short", "noon_incident_angle", "facing", "tilt_group"]],
                                 on="short", how="left").dropna(subset=["noon_incident_angle"])

    face_colors = {"East/NE": C_EAST, "North": C_NORTH, "West/NW": C_WIND, "Unknown": C_NONE}
    for _, row in per_meter.iterrows():
        ax1.scatter(row["noon_incident_angle"], row["avg_slope"] * 1000,
                    s=row["n_events"] * 40 + 20,
                    c=face_colors.get(row["facing"], C_NONE),
                    alpha=0.7, edgecolors="white", linewidths=0.5)
        ax1.annotate(row["short"], (row["noon_incident_angle"], row["avg_slope"] * 1000),
                     fontsize=5.5, ha="left", va="bottom", xytext=(3, 3),
                     textcoords="offset points")
    ax1.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("Noon incident angle (°) — proxy for panel tilt")
    ax1.set_ylabel("Avg soiling rate (slope × 1000/day)")
    ax1.set_title("Panel Tilt vs Soiling Rate\n(bubble size = event count, colour = facing)", fontweight="bold")
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=l)
                    for l, c in face_colors.items() if l != "Unknown"]
    ax1.legend(handles=legend_elems, fontsize=7, loc="lower left")

    # ── Panel 2: Facing direction comparison ─────────────────────────
    ax2 = axes[0, 1]
    facing_stats = []
    for f in ["East/NE", "North"]:
        sub = soiling[soiling["facing"] == f]
        if sub.empty:
            continue
        n = len(sub)
        facing_stats.append({
            "facing": f,
            "n_events": n,
            "n_meters": sub["short"].nunique(),
            "avg_slope": sub["slope_per_day"].mean() * 1000,
            "avg_drop": sub["yield_drop"].mean(),
            "avg_recovery": sub["yield_recovery"].dropna().mean(),
            "pct_confirmed": sub["cleaning_confirmed"].sum() / n * 100,
        })
    if facing_stats:
        fs = pd.DataFrame(facing_stats)
        x = np.arange(len(fs))
        w = 0.25
        ax2.bar(x - w, fs["avg_drop"], w, color="#EF5350", alpha=0.8, label="Avg yield drop")
        ax2.bar(x, fs["avg_recovery"], w, color=C_YES, alpha=0.8, label="Avg recovery")
        ax2.bar(x + w, fs["pct_confirmed"] / 100, w, color="#42A5F5", alpha=0.8,
                label="Cleaning success rate")
        ax2.axhline(0, color="black", linewidth=0.5)
        for i, row in fs.iterrows():
            ax2.text(i, max(abs(row["avg_drop"]), abs(row["avg_recovery"])) + 0.05,
                     f"{row['n_events']} events\n{row['n_meters']} meters",
                     ha="center", fontsize=7, va="bottom")
        ax2.set_xticks(x)
        ax2.set_xticklabels(fs["facing"])
        ax2.set_ylabel("Value")
        ax2.legend(fontsize=7)
    ax2.set_title("East/NE vs North-Facing Panels\n(soiling & cleaning comparison)", fontweight="bold")

    # ── Panel 3: Per-meter event count ───────────────────────────────
    ax3 = axes[1, 0]
    meter_counts = soiling.groupby("short").agg(
        n_total=("is_soiling", "count"),
        n_confirmed_clean=("cleaning_confirmed", "sum"),
    ).reset_index()
    meter_counts = meter_counts.merge(orient_df[["short", "facing", "tilt_group"]],
                                       on="short", how="left")
    meter_counts = meter_counts.sort_values("n_total", ascending=True)
    y = np.arange(len(meter_counts))
    face_c = [face_colors.get(f, C_NONE) for f in meter_counts["facing"]]
    ax3.barh(y, meter_counts["n_total"], color=face_c, alpha=0.75, edgecolor="white", label="Total soiling events")
    ax3.barh(y, meter_counts["n_confirmed_clean"], color=C_YES, alpha=0.9, edgecolor="white",
             label="Cleaning confirmed")
    ax3.set_yticks(y)
    labels_3 = [f"{r['short']} [{r['tilt_group']}]" for _, r in meter_counts.iterrows()]
    ax3.set_yticklabels(labels_3, fontsize=6)
    ax3.set_xlabel("Number of events")
    ax3.legend(fontsize=7, loc="lower right")
    ax3.set_title("Soiling Events per Meter\n(colour = facing, label = tilt group)", fontweight="bold")

    # ── Panel 4: Recovery vs tilt angle ──────────────────────────────
    ax4 = axes[1, 1]
    plot_data = soiling.dropna(subset=["noon_incident_angle", "yield_recovery"]).copy()
    if not plot_data.empty:
        confirmed = plot_data[plot_data["cleaning_confirmed"]]
        not_confirmed = plot_data[~plot_data["cleaning_confirmed"]]
        ax4.scatter(not_confirmed["noon_incident_angle"], not_confirmed["yield_recovery"],
                    c=C_NO, alpha=0.5, s=25, label="Not cleaned", marker="x")
        ax4.scatter(confirmed["noon_incident_angle"], confirmed["yield_recovery"],
                    c=C_YES, alpha=0.7, s=40, label="Cleaning confirmed", marker="o", edgecolors="white")
        ax4.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax4.set_xlabel("Noon incident angle (°)")
        ax4.set_ylabel("Yield recovery after cleaning event")
        ax4.legend(fontsize=7)
        # Trend line
        xv = plot_data["noon_incident_angle"].values
        yv = plot_data["yield_recovery"].values
        ok = np.isfinite(xv) & np.isfinite(yv)
        if ok.sum() > 3:
            sl, ic, r, p, _ = stats.linregress(xv[ok], yv[ok])
            xr = np.linspace(xv[ok].min(), xv[ok].max(), 50)
            ax4.plot(xr, sl * xr + ic, "--", color="grey", linewidth=1,
                     label=f"Trend r={r:.2f}, p={p:.3f}")
            ax4.legend(fontsize=7)
    ax4.set_title("Panel Tilt vs Cleaning Recovery\n(does steeper tilt = better cleaning?)", fontweight="bold")

    fig.savefig(os.path.join(RESULTS_DIR, "orientation_analysis.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] orientation_analysis.png")


def plot_cleaning_confirmation(all_spells, orient_df):
    """
    Detailed 2×2 figure on cleaning confirmation:
      [1] Confirmed vs not-confirmed by cleaning type and facing
      [2] Timeline of soiling spells coloured by cleaning outcome
      [3] Recovery ratio histogram (rain vs wind)
      [4] Wind direction during spell vs during cleaning event
    """
    soiling = all_spells[all_spells["is_soiling"] & ~all_spells["is_anomaly"]].copy()
    if soiling.empty:
        return
    soiling = soiling.merge(orient_df[["short", "facing", "tilt_group"]], on="short", how="left")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Cleaning Confirmation — Detailed Analysis", fontsize=14, fontweight="bold", y=0.98)
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # ── Panel 1: Grouped bar — confirmed/not by type and facing ──────
    ax1 = axes[0, 0]
    groups = soiling.groupby(["cleaning_type", "facing"]).agg(
        n_total=("cleaning_confirmed", "count"),
        n_confirmed=("cleaning_confirmed", "sum"),
    ).reset_index()
    groups["n_not"] = groups["n_total"] - groups["n_confirmed"]
    groups = groups[groups["cleaning_type"].isin(["rain", "wind"])]
    if not groups.empty:
        groups["label"] = groups["cleaning_type"].str.title() + "\n" + groups["facing"]
        x = np.arange(len(groups))
        ax1.bar(x, groups["n_confirmed"], color=C_YES, alpha=0.8, label="Confirmed")
        ax1.bar(x, groups["n_not"], bottom=groups["n_confirmed"], color=C_NO, alpha=0.8, label="Not confirmed")
        ax1.set_xticks(x)
        ax1.set_xticklabels(groups["label"], fontsize=7)
        ax1.set_ylabel("Number of events")
        ax1.legend(fontsize=7)
    ax1.set_title("Cleaning by Type & Panel Facing\n(stacked: confirmed vs not)", fontweight="bold")

    # ── Panel 2: Timeline of spells ──────────────────────────────────
    ax2 = axes[0, 1]
    soiling["start_dt"] = pd.to_datetime(soiling["start_date"])
    soiling_sorted = soiling.sort_values("start_dt")
    meters_list = sorted(soiling_sorted["short"].unique())
    meter_idx = {m: i for i, m in enumerate(meters_list)}
    for _, row in soiling_sorted.iterrows():
        c = C_YES if row["cleaning_confirmed"] else C_NO
        start = row["start_dt"]
        end = pd.to_datetime(row["end_date"])
        y = meter_idx.get(row["short"], 0)
        ax2.barh(y, (end - start).days, left=start.toordinal(), height=0.7,
                 color=c, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax2.set_yticks(range(len(meters_list)))
    ax2.set_yticklabels(meters_list, fontsize=6)
    xticks = pd.date_range("2021-01-01", "2025-12-31", freq="YS")
    ax2.set_xticks([d.toordinal() for d in xticks])
    ax2.set_xticklabels([d.strftime("%Y") for d in xticks], fontsize=7)
    ax2.set_xlabel("Year")
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(facecolor=C_YES, label="Cleaning confirmed"),
                        Patch(facecolor=C_NO, label="Not confirmed")],
               fontsize=7, loc="upper right")
    ax2.set_title("Soiling Spell Timeline\n(green = yield recovered, red = no recovery)", fontweight="bold")

    # ── Panel 3: Recovery ratio histogram ────────────────────────────
    ax3 = axes[1, 0]
    for ct, color in [("rain", C_RAIN), ("wind", C_WIND)]:
        sub = soiling[(soiling["cleaning_type"] == ct) & soiling["recovery_ratio"].notna()]
        if sub.empty:
            continue
        vals = sub["recovery_ratio"].clip(-2, 3)
        ax3.hist(vals, bins=20, alpha=0.6, color=color, label=f"{ct.title()} (n={len(sub)})",
                 edgecolor="white")
    ax3.axvline(0, color="black", linewidth=1, linestyle="--")
    ax3.axvline(1, color="grey", linewidth=0.8, linestyle=":", label="Full recovery (ratio=1)")
    ax3.set_xlabel("Recovery ratio (recovery / drop)")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=7)
    ax3.annotate("<- No cleaning", xy=(0.05, 0.9), xycoords="axes fraction",
                 fontsize=8, color=C_NO, fontweight="bold")
    ax3.annotate("Full clean ->", xy=(0.75, 0.9), xycoords="axes fraction",
                 fontsize=8, color=C_YES, fontweight="bold")
    ax3.set_title("Recovery Ratio Distribution\n(ratio=1 means full yield restoration)", fontweight="bold")

    # ── Panel 4: Wind direction: during spell vs during cleaning ─────
    ax4 = axes[1, 1]
    wd_data = soiling.dropna(subset=["avg_wind_dir", "post_wind_dir"]).copy()
    if not wd_data.empty:
        for ct, color, marker in [("rain", C_RAIN, "o"), ("wind", C_WIND, "s")]:
            sub = wd_data[wd_data["cleaning_type"] == ct]
            if sub.empty:
                continue
            confirmed = sub[sub["cleaning_confirmed"]]
            not_conf = sub[~sub["cleaning_confirmed"]]
            ax4.scatter(not_conf["avg_wind_dir"], not_conf["post_wind_dir"],
                        c=color, alpha=0.4, s=20, marker="x")
            ax4.scatter(confirmed["avg_wind_dir"], confirmed["post_wind_dir"],
                        c=color, alpha=0.8, s=40, marker=marker, edgecolors="white",
                        label=f"{ct.title()} confirmed")
        ax4.plot([0, 360], [0, 360], "--", color="grey", linewidth=0.5, label="Same direction")
        ax4.set_xlabel("Avg wind direction during soiling spell (°)")
        ax4.set_ylabel("Wind direction during cleaning event (°)")
        ax4.set_xlim(0, 360)
        ax4.set_ylim(0, 360)
        ax4.legend(fontsize=7)
        for angle, lbl in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
            ax4.axhline(angle, color="lightgrey", linewidth=0.3)
            ax4.axvline(angle, color="lightgrey", linewidth=0.3)
    ax4.set_title("Wind Direction: During Spell vs During Cleaning\n(does direction shift matter?)",
                  fontweight="bold")

    fig.savefig(os.path.join(RESULTS_DIR, "cleaning_confirmation.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] cleaning_confirmation.png")


def plot_cross_meter_detail(all_spells, orient_df):
    """
    Focus on the best cross-meter validation dates:
    Show all meters' slope and recovery on the same spell dates.
    """
    if all_spells.empty or "fleet_soiling_count" not in all_spells.columns:
        return

    top_dates = (all_spells.groupby("start_date")["fleet_soiling_count"]
                 .first().nlargest(6).index.tolist())

    if not top_dates:
        return

    n_plots = len(top_dates)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle("Cross-Meter Validation — Same-Date Soiling Spells",
                 fontsize=14, fontweight="bold", y=1.01)
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, sd in enumerate(top_dates):
        ax = axes[idx]
        sub = all_spells[all_spells["start_date"] == sd].copy()
        sub = sub.merge(orient_df[["short", "facing"]], on="short", how="left")
        sub = sub.sort_values("slope_per_day")
        face_colors = {"East/NE": C_EAST, "North": C_NORTH, "Unknown": C_NONE}
        colors = [face_colors.get(f, C_NONE) for f in sub["facing"]]

        y = np.arange(len(sub))
        bars = ax.barh(y, sub["slope_per_day"] * 1000, color=colors, alpha=0.75,
                       edgecolor="white", linewidth=0.3)

        for i, (_, row) in enumerate(sub.iterrows()):
            rec_txt = f"rec={row['yield_recovery']:+.3f}" if pd.notna(row["yield_recovery"]) else "rec=N/A"
            flag = " *" if row.get("cleaning_confirmed", False) else ""
            ax.text(0.01, i, f"  {rec_txt}{flag}", va="center", fontsize=5.5,
                    color=C_YES if row.get("cleaning_confirmed", False) else "black")

        ax.set_yticks(y)
        ax.set_yticklabels(sub["short"], fontsize=6)
        ax.axvline(0, color="black", linewidth=0.5)
        ed = sub["end_date"].iloc[0] if len(sub) > 0 else ""
        n_soil = sub["is_soiling"].sum()
        ax.set_title(f"{sd} -> {ed}\n{len(sub)} meters, {n_soil} confirmed soiling",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Slope × 1000/day", fontsize=7)

    for idx in range(len(top_dates), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "cross_meter_validation.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] cross_meter_validation.png")


# =====================================================================
#  MAIN
# =====================================================================

def run():
    print("=" * 65)
    print("  SOILING & CLEANING EFFECTIVENESS ANALYSIS")
    print("=" * 65)
    print(f"Results -> {RESULTS_DIR}\n")

    # ── Load data ────────────────────────────────────────────────────
    print("[1/7] Loading data ...")
    meter_df = load_meter_data()
    solcast_df = load_solcast()
    sim_df = load_simulation()
    print(f"  Meter rows: {len(meter_df):,}")
    print(f"  Solcast rows: {len(solcast_df):,}")
    print(f"  Simulation rows: {len(sim_df):,}" if sim_df is not None else "  Simulation: not found")

    # ── Panel orientations ───────────────────────────────────────────
    print("\n[2/7] Profiling panel orientations ...")
    orient_df = profile_panel_orientations(sim_df)
    if not orient_df.empty:
        orient_df.to_csv(os.path.join(RESULTS_DIR, "panel_orientation.csv"), index=False)
        print(f"  Found {len(orient_df)} panels:")
        for _, r in orient_df.iterrows():
            if pd.notna(r['nameplate_W']):
                print(f"    {r['short']:25s}  tilt={r['noon_incident_angle']:5.1f}deg  "
                      f"facing={r['facing']:10s}  nameplate={r['nameplate_W']:,.0f}W")
            else:
                print(f"    {r['short']:25s}  tilt={r['noon_incident_angle']:5.1f}deg  facing={r['facing']}")
    else:
        print("  WARNING: Could not profile orientations (simulation data missing/incomplete)")

    # ── Build daily yield ────────────────────────────────────────────
    print("\n[3/7] Building daily yield per meter ...")
    meter_daily = build_daily_yield(meter_df, solcast_df)
    print(f"  Built daily data for {len(meter_daily)} meters")

    # ── Detect soiling spells ────────────────────────────────────────
    print("\n[4/7] Detecting soiling spells ...")
    all_events = []
    for short, daily in meter_daily.items():
        spells = detect_spells(daily)
        if not spells.empty:
            spells["short"] = short
            spells["meter"] = daily["meter"].iloc[0]
            all_events.append(spells)

    if not all_events:
        print("  No soiling spells detected. Exiting.")
        return

    all_spells = pd.concat(all_events, ignore_index=True)
    n_soiling = all_spells["is_soiling"].sum()
    n_anomaly = all_spells["is_anomaly"].sum()
    n_confirmed_clean = all_spells["cleaning_confirmed"].sum()
    print(f"  Total spells: {len(all_spells)}")
    print(f"  Confirmed soiling (p<0.10, slope<-0.002): {n_soiling}")
    print(f"  Suspected meter anomalies (|slope|>0.05): {n_anomaly}")
    print(f"  Cleaning confirmed (yield recovered): {n_confirmed_clean}")

    # ── Cross-meter validation ───────────────────────────────────────
    print("\n[5/7] Cross-meter consistency validation ...")
    all_spells = cross_meter_validate(all_spells)
    all_spells.to_csv(os.path.join(RESULTS_DIR, "soiling_spells.csv"), index=False)

    fleet_conf = all_spells[all_spells.get("fleet_confirmed", False) == True]
    n_fleet_conf = fleet_conf["is_soiling"].sum() if not fleet_conf.empty else 0
    print(f"  Fleet-confirmed soiling events (>=3 meters same date): {n_fleet_conf}")

    # Cleaning effectiveness summary
    soiling_real = all_spells[all_spells["is_soiling"] & ~all_spells["is_anomaly"]]
    eff_rows = []
    for ct in ["rain", "wind", "unknown", "none"]:
        sub = soiling_real[soiling_real["cleaning_type"] == ct]
        if sub.empty:
            continue
        n = len(sub)
        n_conf = sub["cleaning_confirmed"].sum()
        eff_rows.append({
            "cleaning_type": ct,
            "n_events": n,
            "n_confirmed": int(n_conf),
            "confirmation_rate_pct": round(n_conf / n * 100, 1),
            "avg_yield_drop": round(sub["yield_drop"].mean(), 4),
            "avg_yield_recovery": round(sub["yield_recovery"].dropna().mean(), 4),
            "median_recovery_ratio": round(sub["recovery_ratio"].dropna().median(), 3)
            if sub["recovery_ratio"].notna().any() else np.nan,
        })
    eff_df = pd.DataFrame(eff_rows)
    eff_df.to_csv(os.path.join(RESULTS_DIR, "cleaning_effectiveness.csv"), index=False)
    print("\n  Cleaning effectiveness summary:")
    for _, r in eff_df.iterrows():
        print(f"    {r['cleaning_type']:8s}: {r['n_events']:3d} events, "
              f"{r['confirmation_rate_pct']:5.1f}% confirmed, "
              f"avg_recovery={r['avg_yield_recovery']:+.4f}")

    # ── Weather direction profile ────────────────────────────────────
    print("\n[6/7] Analysing wind/rain direction patterns ...")
    weather_info = weather_direction_profile(solcast_df)
    if "all_daylight" in weather_info:
        wi = weather_info["all_daylight"]
        print(f"  Prevailing wind: {wi['avg_direction']}deg at {wi['avg_speed']} m/s")
    if "during_rain" in weather_info:
        wr = weather_info["during_rain"]
        print(f"  Wind during rain: {wr['avg_direction']}deg at {wr['avg_speed']} m/s")
    if "strong_wind" in weather_info:
        ws = weather_info["strong_wind"]
        print(f"  Strong wind: {ws['avg_direction']}deg at {ws['avg_speed']} m/s")

    # ── Plots ────────────────────────────────────────────────────────
    print("\n[7/7] Generating plots ...")
    plot_main_dashboard(all_spells, orient_df, weather_info, solcast_df)
    plot_orientation_analysis(all_spells, orient_df)
    plot_cleaning_confirmation(all_spells, orient_df)
    plot_cross_meter_detail(all_spells, orient_df)

    # ── Summary table ────────────────────────────────────────────────
    summary_lines = [
        "=" * 65,
        "  ANALYSIS SUMMARY",
        "=" * 65,
        f"  Meters analysed:              {len(meter_daily)}",
        f"  Total soiling spells:         {len(all_spells)}",
        f"  Confirmed soiling:            {n_soiling} ({n_soiling/len(all_spells)*100:.1f}%)",
        f"  Meter anomalies flagged:      {n_anomaly}",
        f"  Cleaning confirmed (real):    {n_confirmed_clean} ({n_confirmed_clean/max(n_soiling,1)*100:.1f}% of soiling)",
        f"  Fleet-validated soiling:      {n_fleet_conf}",
        "",
    ]

    # Key findings
    if not eff_df.empty:
        rain_row = eff_df[eff_df["cleaning_type"] == "rain"]
        wind_row = eff_df[eff_df["cleaning_type"] == "wind"]
        summary_lines.append("  KEY FINDINGS:")
        if not rain_row.empty:
            rr = rain_row.iloc[0]
            summary_lines.append(f"    Rain cleaning: {rr['confirmation_rate_pct']:.0f}% effective "
                                 f"({int(rr['n_confirmed'])}/{rr['n_events']} events)")
        if not wind_row.empty:
            wr = wind_row.iloc[0]
            summary_lines.append(f"    Wind cleaning: {wr['confirmation_rate_pct']:.0f}% effective "
                                 f"({int(wr['n_confirmed'])}/{wr['n_events']} events)")

    if "all_daylight" in weather_info:
        summary_lines.append(f"    Prevailing wind from ~{weather_info['all_daylight']['avg_direction']}deg "
                             f"(south) -- hits BACK of north-facing panels")
    if "strong_wind" in weather_info:
        summary_lines.append(f"    Strong wind from ~{weather_info['strong_wind']['avg_direction']}deg "
                             f"(west/NW) -- oblique to panel face")

    if not orient_df.empty and not soiling_real.empty:
        merged = soiling_real.merge(orient_df[["short", "tilt_group"]], on="short", how="left")
        for tg in ["Flat (<8°)", "Medium (8-12°)", "Steep (>12°)"]:
            sub = merged[merged["tilt_group"] == tg]
            if sub.empty:
                continue
            pct = sub["cleaning_confirmed"].sum() / len(sub) * 100
            summary_lines.append(f"    {tg} panels: {len(sub)} events, {pct:.0f}% cleaning confirmed")

    summary_lines.append("")
    summary_lines.append(f"  Results saved to: {RESULTS_DIR}")
    summary_lines.append("=" * 65)

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(os.path.join(RESULTS_DIR, "analysis_summary.txt"), "w") as f:
        f.write(summary_text)
    print(f"\n  [saved] analysis_summary.txt")


if __name__ == "__main__":
    run()
