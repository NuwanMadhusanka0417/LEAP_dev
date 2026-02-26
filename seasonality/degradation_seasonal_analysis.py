"""
Unified degradation, seasonality, soiling & sudden-event analysis
=================================================================
Uses simulation-trained expected power as reference (weather-normalised).
Per-meter analysis covering:
  1) Overall degradation rate (%/year) via monthly median PI trend
  2) Seasonal degradation (Summer/Autumn/Winter/Spring — southern hemisphere)
  3) Soiling detection: PI decline during dry spells, recovery after rain
  4) Sudden-event / anomaly detection in daily PI series
  5) Comprehensive per-meter and fleet-wide outputs

Inputs:
  - SolarMeterReadings1hour_cleaned_v2.csv  (analysis_valid, outage_flag)
  - simulated_power_2020_2025.csv           (XGBoost-predicted ideal power)
  - solcast_df.csv / solcast_df_cleaned.csv (weather incl. precipitation_rate)

Outputs (in Results_seasonality/):
  - degradation_summary.csv          per-meter overall + seasonal rates
  - monthly_pi.csv                   monthly median PI per meter
  - seasonal_degradation.csv         seasonal PI trends per meter
  - soiling_events.csv               detected soiling episodes
  - sudden_events.csv                detected sudden PI drops/anomalies
  - fleet_overview.png               fleet-wide dashboard
  - per-meter PNG plots              individual meter dashboards
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
SIM_DIR = os.path.join(DATA_DIR, "simulated")
RESULTS_DIR = os.path.join(BASE, "Results_seasonality")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
METER_V2_CSV = "SolarMeterReadings1hour_cleaned_2020_2025.csv"
SIMULATED_CSV = "simulated_power_2020_2025.csv"
SOLCAST_CSV = "solcast_df_cleaned_2020_2025.csv"
SOLCAST_FALLBACK = "solcast_df_cleaned_2020_2025.csv"

BUNDOORA_PREFIX = "solar.bun_"
SOLCAST_CAMPUS = "BUNDOORA"

YEAR_START = 2021
YEAR_END = 2025

ZENITH_DAYLIGHT = 90
GHI_MIN = 50.0
PI_EPS = 1e-9
OUTAGE_DAY_RATE = 0.10
MIN_VALID_PCT = 15.0
MIN_MONTHS = 6

SEASON_MAP = {
    12: "Summer", 1: "Summer", 2: "Summer",
    3: "Autumn",  4: "Autumn",  5: "Autumn",
    6: "Winter",  7: "Winter",  8: "Winter",
    9: "Spring", 10: "Spring", 11: "Spring",
}
SEASON_ORDER = ["Summer", "Autumn", "Winter", "Spring"]

# Soiling
DRY_SPELL_MIN_DAYS = 5
PRECIP_THRESHOLD = 0.5        # mm/h — below this counts as dry
SOILING_MIN_DECLINE = -0.002  # PI/day slope threshold to flag soiling

# Sudden events
ANOMALY_SIGMA = 3.0
STEP_CHANGE_WINDOW = 14       # days — compare mean PI before/after


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
    raise FileNotFoundError(f"Cannot find {SIMULATED_CSV}")


def load_solcast():
    for name in [SOLCAST_CSV, SOLCAST_FALLBACK]:
        p = os.path.join(DATA_DIR, name)
        if os.path.isfile(p):
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
    raise FileNotFoundError("Cannot find Solcast CSV")


# ===================================================================
# Calibration & PI
# ===================================================================

def calibrate(actual, expected, mask):
    """Linear calibration on valid hours: actual = a*expected + b."""
    x, y = expected[mask], actual[mask]
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if ok.sum() < 30:
        return 1.0, 0.0
    slope, intercept, _, _, _ = stats.linregress(x[ok], y[ok])
    return slope, intercept


def compute_meter_pi(meter_df, sim_df, solcast_df):
    """
    Merge actual, simulated, and weather; calibrate; compute hourly PI.
    Returns enriched DataFrame with PI, E_cal, weather, season, etc.
    """
    m = meter_df.copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"]).dt.floor("h")

    # Merge simulated
    sim = sim_df.rename(columns={"simulated_power": "E_sim"})
    sim["timestamp"] = pd.to_datetime(sim["timestamp"]).dt.floor("h")
    m = m.merge(sim[["timestamp", "E_sim"]], on="timestamp", how="left")

    # Merge weather
    weather_cols = ["timestamp"]
    for c in ["ghi", "zenith", "precipitation_rate", "air_temp", "cloud_opacity"]:
        if c in solcast_df.columns:
            weather_cols.append(c)
    sol = solcast_df[weather_cols].copy()
    m = m.merge(sol, on="timestamp", how="left")

    # Nighttime zeroing via zenith (handles datasets without prior cleaning)
    if "zenith" in m.columns:
        is_night = m["zenith"].fillna(100) >= ZENITH_DAYLIGHT
        m.loc[is_night, "meter_reading"] = 0.0

    # Valid daylight mask
    if "zenith" in m.columns and "ghi" in m.columns:
        m["valid_daylight"] = (
            (m["zenith"] < ZENITH_DAYLIGHT)
            & (m["ghi"] > GHI_MIN)
            & m["meter_reading"].notna()
            & (m["meter_reading"] > 0)
        )
    elif "analysis_valid" in m.columns:
        m["valid_daylight"] = m["analysis_valid"].astype(bool)
    else:
        m["valid_daylight"] = m["meter_reading"].notna() & (m["meter_reading"] > 0)

    if "analysis_valid" in m.columns:
        analysis = m["analysis_valid"].astype(bool).values
    else:
        analysis = m["valid_daylight"].values.copy()

    actual = m["meter_reading"].values.astype(float)
    E_sim = m["E_sim"].values.astype(float)
    E_sim = np.where(np.isfinite(E_sim), E_sim, 0.0)

    # Drop high-outage days
    m["date"] = pd.to_datetime(m["timestamp"]).dt.date
    if "outage_flag" in m.columns:
        daily_valid = m.groupby("date")["valid_daylight"].sum()
        daily_outage = m.groupby("date")["outage_flag"].sum()
        rate = daily_outage / daily_valid.replace(0, np.nan)
        bad_days = set(rate[rate > OUTAGE_DAY_RATE].index)
        m["bad_day"] = m["date"].isin(bad_days)
        analysis = analysis & (~m["bad_day"].values)
    else:
        m["bad_day"] = False

    # Calibrate (require both actual and simulated > 0)
    cal_mask = analysis & (E_sim > 0) & (actual > 0)
    a, b = calibrate(actual, E_sim, cal_mask)
    E_cal = np.maximum(a * E_sim + b, 0.0)
    m["E_cal"] = E_cal

    # PI — only where both actual power and calibrated expected are meaningful
    pi = np.full(len(m), np.nan)
    use = analysis & (E_cal > PI_EPS) & (actual > 0)
    pi[use] = actual[use] / (E_cal[use] + PI_EPS)
    # Clip extreme PI values (calibration/meter anomalies)
    pi = np.where((pi > 0.1) & (pi < 2.0), pi, np.nan)
    use = use & np.isfinite(pi)
    m["PI"] = pi
    m["pi_valid"] = use

    # Time features
    ts = pd.to_datetime(m["timestamp"])
    m["year"] = ts.dt.year
    m["month"] = ts.dt.month
    m["hour"] = ts.dt.hour
    m["day_of_year"] = ts.dt.dayofyear
    m["season"] = m["month"].map(SEASON_MAP)
    m["year_month"] = ts.dt.to_period("M")

    return m, a, b


# ===================================================================
# 1) Overall degradation
# ===================================================================

def overall_degradation(m):
    """Monthly median PI -> linear trend -> degradation %/year."""
    pi_valid = m.loc[m["pi_valid"], ["year_month", "PI"]]
    if pi_valid.empty:
        return np.nan, np.nan, np.nan, pd.DataFrame()

    monthly = pi_valid.groupby("year_month")["PI"].median().reset_index()
    monthly["month_ts"] = monthly["year_month"].dt.to_timestamp()
    monthly["years"] = (monthly["month_ts"] - monthly["month_ts"].min()).dt.days / 365.25

    if len(monthly) < MIN_MONTHS:
        return np.nan, np.nan, len(monthly), monthly

    slope, intercept, r, p, se = stats.linregress(monthly["years"], monthly["PI"])
    median_pi = np.nanmedian(monthly["PI"].values)
    deg_pct = 100.0 * slope / median_pi if median_pi > 0 else np.nan
    return deg_pct, median_pi, len(monthly), monthly


# ===================================================================
# 2) Seasonal degradation
# ===================================================================

def seasonal_degradation(m):
    """Per-season monthly median PI trend -> degradation %/year per season."""
    results = []
    pi_valid = m.loc[m["pi_valid"]].copy()
    if pi_valid.empty:
        return pd.DataFrame()

    for season in SEASON_ORDER:
        sp = pi_valid[pi_valid["season"] == season]
        if sp.empty:
            continue
        monthly = sp.groupby("year_month")["PI"].median().reset_index()
        monthly["month_ts"] = monthly["year_month"].dt.to_timestamp()
        monthly["years"] = (monthly["month_ts"] - monthly["month_ts"].min()).dt.days / 365.25

        if len(monthly) < 3:
            results.append({
                "season": season,
                "deg_pct_per_year": np.nan,
                "median_PI": np.nanmedian(sp["PI"]),
                "n_months": len(monthly),
            })
            continue

        slope, intercept, r, p, se = stats.linregress(monthly["years"], monthly["PI"])
        median_pi = np.nanmedian(monthly["PI"].values)
        deg = 100.0 * slope / median_pi if median_pi > 0 else np.nan
        results.append({
            "season": season,
            "deg_pct_per_year": round(deg, 4),
            "trend_slope": round(slope, 6),
            "median_PI": round(median_pi, 4),
            "n_months": len(monthly),
            "r_value": round(r, 4),
            "p_value": round(p, 4),
        })
    return pd.DataFrame(results)


# ===================================================================
# 3) Soiling detection
# ===================================================================

def detect_soiling(m, precip_col="precipitation_rate"):
    """
    Identify dry spells (consecutive days with low precipitation) and measure
    whether PI declines during those spells. Recovery after rain = soiling evidence.
    """
    if precip_col not in m.columns:
        return pd.DataFrame()

    pi_valid = m.loc[m["pi_valid"]].copy()
    if pi_valid.empty:
        return pd.DataFrame()

    # Daily aggregates
    daily = pi_valid.groupby("date").agg(
        daily_pi=("PI", "median"),
        daily_precip=(precip_col, "mean"),
        n_hours=("PI", "count"),
    ).reset_index()
    daily = daily[daily["n_hours"] >= 3].copy()
    daily = daily.sort_values("date").reset_index(drop=True)

    if len(daily) < DRY_SPELL_MIN_DAYS + 2:
        return pd.DataFrame()

    daily["is_dry"] = daily["daily_precip"] < PRECIP_THRESHOLD

    # Find dry spells (consecutive dry days >= DRY_SPELL_MIN_DAYS)
    spells = []
    i = 0
    while i < len(daily):
        if daily.iloc[i]["is_dry"]:
            start = i
            while i < len(daily) and daily.iloc[i]["is_dry"]:
                i += 1
            length = i - start
            if length >= DRY_SPELL_MIN_DAYS:
                spells.append((start, i - 1, length))
        else:
            i += 1

    events = []
    for s_start, s_end, s_len in spells:
        sub = daily.iloc[s_start:s_end + 1]
        x = np.arange(s_len, dtype=float)
        y = sub["daily_pi"].values
        ok = np.isfinite(y)
        if ok.sum() < 3:
            continue
        slope, intercept, r, p, se = stats.linregress(x[ok], y[ok])
        pi_before = y[0] if np.isfinite(y[0]) else np.nanmean(y[:2])
        pi_after = y[-1] if np.isfinite(y[-1]) else np.nanmean(y[-2:])

        # Check recovery: PI in the 3 days after rain
        recovery_pi = np.nan
        if s_end + 3 < len(daily):
            post = daily.iloc[s_end + 1:s_end + 4]["daily_pi"]
            recovery_pi = post.median()

        events.append({
            "start_date": sub.iloc[0]["date"],
            "end_date": sub.iloc[-1]["date"],
            "dry_days": s_len,
            "pi_slope_per_day": round(slope, 6),
            "pi_start": round(pi_before, 4),
            "pi_end": round(pi_after, 4),
            "pi_drop": round(pi_before - pi_after, 4),
            "pi_recovery_after_rain": round(recovery_pi, 4) if np.isfinite(recovery_pi) else np.nan,
            "is_soiling": slope < SOILING_MIN_DECLINE,
        })
    return pd.DataFrame(events)


# ===================================================================
# 4) Sudden event / anomaly detection
# ===================================================================

def detect_sudden_events(m):
    """
    Daily PI anomaly detection and step-change identification.
    Returns DataFrame of flagged events.
    """
    pi_valid = m.loc[m["pi_valid"]].copy()
    if pi_valid.empty:
        return pd.DataFrame()

    daily = pi_valid.groupby("date").agg(
        daily_pi=("PI", "median"),
        n_hours=("PI", "count"),
    ).reset_index()
    daily = daily[daily["n_hours"] >= 3].sort_values("date").reset_index(drop=True)
    if len(daily) < STEP_CHANGE_WINDOW * 2:
        return pd.DataFrame()

    # Smoothed baseline (rolling median)
    daily["pi_smooth"] = daily["daily_pi"].rolling(
        STEP_CHANGE_WINDOW, center=True, min_periods=3
    ).median()
    daily["pi_smooth"] = daily["pi_smooth"].bfill().ffill()
    daily["residual"] = daily["daily_pi"] - daily["pi_smooth"]

    sigma = daily["residual"].std()
    if sigma < 1e-6:
        return pd.DataFrame()
    threshold = ANOMALY_SIGMA * sigma
    daily["is_anomaly"] = daily["residual"].abs() > threshold

    events = []
    for _, row in daily[daily["is_anomaly"]].iterrows():
        idx = daily.index[daily["date"] == row["date"]]
        if len(idx) == 0:
            continue
        i = idx[0]

        # Classify: check if step change
        w = STEP_CHANGE_WINDOW
        before = daily.iloc[max(0, i - w):i]["daily_pi"].median()
        after = daily.iloc[i + 1:i + 1 + w]["daily_pi"].median()
        step = after - before if (np.isfinite(before) and np.isfinite(after)) else np.nan

        if row["residual"] < -threshold:
            etype = "sudden_drop"
        elif row["residual"] > threshold:
            etype = "sudden_spike"
        else:
            etype = "anomaly"

        if np.isfinite(step) and abs(step) > 2 * sigma:
            etype = "step_change_down" if step < 0 else "step_change_up"

        events.append({
            "date": row["date"],
            "daily_pi": round(row["daily_pi"], 4),
            "pi_smooth": round(row["pi_smooth"], 4),
            "residual": round(row["residual"], 4),
            "event_type": etype,
            "step_magnitude": round(step, 4) if np.isfinite(step) else np.nan,
        })
    return pd.DataFrame(events)


# ===================================================================
# 5) Plotting
# ===================================================================

def plot_meter_dashboard(meter_id, monthly_df, seasonal_df, soiling_df, events_df, m):
    """Create a per-meter 4-panel dashboard."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return

    short = meter_id.split("#")[0].replace("solar.", "")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Degradation Dashboard — {short}", fontsize=14, fontweight="bold")

    # Panel 1: Monthly PI trend
    ax = axes[0, 0]
    if not monthly_df.empty and "month_ts" in monthly_df.columns:
        ax.plot(monthly_df["month_ts"], monthly_df["PI"], "o-", ms=4, lw=1.5, color="steelblue")
        if len(monthly_df) >= MIN_MONTHS:
            x = (monthly_df["month_ts"] - monthly_df["month_ts"].min()).dt.days / 365.25
            s, i, _, _, _ = stats.linregress(x, monthly_df["PI"])
            ax.plot(monthly_df["month_ts"], i + s * x, "--", color="red", lw=2,
                    label=f"Trend: {100*s/monthly_df['PI'].median():.2f}%/yr")
            ax.legend(fontsize=9)
    ax.set_ylabel("Monthly median PI")
    ax.set_title("Overall Degradation Trend")
    ax.axhline(1.0, ls=":", color="green", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Panel 2: Seasonal PI box
    ax = axes[0, 1]
    pi_valid = m.loc[m["pi_valid"]]
    if not pi_valid.empty:
        season_data = [pi_valid[pi_valid["season"] == s]["PI"].dropna().values for s in SEASON_ORDER]
        season_data = [d for d in season_data if len(d) > 0]
        labels = [s for s, d in zip(SEASON_ORDER, [pi_valid[pi_valid["season"] == s]["PI"].dropna() for s in SEASON_ORDER]) if len(d) > 0]
        if season_data:
            bp = ax.boxplot(season_data, labels=labels, patch_artist=True, showfliers=False)
            colors = ["#ff9999", "#ffcc66", "#99ccff", "#99ff99"]
            for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
                patch.set_facecolor(color)
    ax.axhline(1.0, ls=":", color="green", alpha=0.5)
    ax.set_ylabel("PI")
    ax.set_title("Seasonal PI Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Daily PI with events
    ax = axes[1, 0]
    daily = pi_valid.groupby("date").agg(daily_pi=("PI", "median")).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    if not daily.empty:
        ax.scatter(daily["date"], daily["daily_pi"], s=3, alpha=0.3, color="gray")
        smooth = daily["daily_pi"].rolling(14, center=True, min_periods=3).median()
        ax.plot(daily["date"], smooth, color="blue", lw=1.5, label="14d median")
        if not events_df.empty:
            ev_dates = pd.to_datetime(events_df["date"])
            ev_pi = events_df["daily_pi"]
            drops = events_df["event_type"].str.contains("drop|step_change_down")
            if drops.any():
                ax.scatter(ev_dates[drops], ev_pi[drops], s=30, color="red",
                           zorder=5, label="Sudden drops")
        ax.legend(fontsize=8)
    ax.axhline(1.0, ls=":", color="green", alpha=0.5)
    ax.set_ylabel("Daily PI")
    ax.set_title("Daily PI & Sudden Events")
    ax.grid(True, alpha=0.3)

    # Panel 4: Soiling — PI during dry spells
    ax = axes[1, 1]
    if not soiling_df.empty and soiling_df["is_soiling"].any():
        soil = soiling_df[soiling_df["is_soiling"]].copy()
        ax.barh(range(len(soil)), soil["pi_drop"].values, color="sandybrown", edgecolor="brown")
        labels = [str(d) for d in soil["start_date"].values]
        ax.set_yticks(range(len(soil)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("PI drop during dry spell")
        ax.set_title(f"Soiling Events ({len(soil)} detected)")
    else:
        ax.text(0.5, 0.5, "No soiling events detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Soiling Events")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(RESULTS_DIR, f"dashboard_{short}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fleet_overview(all_deg, all_seasonal, all_monthly):
    """Fleet-wide summary plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Fleet Degradation Overview — Bundoora Solar", fontsize=15, fontweight="bold")

    # 1) Per-meter degradation bar
    ax = axes[0, 0]
    valid = all_deg[all_deg["deg_pct_per_year"].notna()].sort_values("deg_pct_per_year")
    if not valid.empty:
        names = [m.split("#")[0].replace("solar.", "") for m in valid["meter"]]
        colors = ["green" if v > -1 else "orange" if v > -2 else "red" for v in valid["deg_pct_per_year"]]
        ax.barh(names, valid["deg_pct_per_year"], color=colors, edgecolor="gray", alpha=0.85)
        ax.axvline(0, color="black", lw=0.5)
        ax.axvline(-0.7, ls="--", color="green", alpha=0.5, label="Typical (-0.5 to -1%/yr)")
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

    # 3) Monthly PI for all meters
    ax = axes[1, 0]
    if not all_monthly.empty and "month_ts" in all_monthly.columns:
        for meter_id in all_monthly["meter"].unique():
            sub = all_monthly[all_monthly["meter"] == meter_id]
            short = meter_id.split("#")[0].replace("solar.", "")
            ax.plot(sub["month_ts"], sub["PI"], alpha=0.6, lw=1, label=short)
        ax.axhline(1.0, ls=":", color="green", alpha=0.5)
        ax.set_ylabel("Monthly median PI")
        ax.set_title("Monthly PI — All Meters")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=5, ncol=4, loc="lower left")

    # 4) Soiling/event summary
    ax = axes[1, 1]
    if not all_deg.empty:
        ax.barh(
            [m.split("#")[0].replace("solar.", "") for m in all_deg["meter"]],
            all_deg["soiling_events"],
            color="sandybrown", edgecolor="brown", alpha=0.8, label="Soiling",
        )
        ax.barh(
            [m.split("#")[0].replace("solar.", "") for m in all_deg["meter"]],
            all_deg["sudden_events"],
            left=all_deg["soiling_events"],
            color="crimson", edgecolor="darkred", alpha=0.7, label="Sudden events",
        )
        ax.set_xlabel("Count")
        ax.set_title("Detected Events per Meter")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(RESULTS_DIR, "fleet_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fleet overview: {path}")


# ===================================================================
# Main runner
# ===================================================================

def run():
    print("=" * 70)
    print("UNIFIED DEGRADATION / SEASONALITY / SOILING / EVENT ANALYSIS")
    print(f"  Analysis period: {YEAR_START} - {YEAR_END}")
    print("=" * 70)

    # --- Load ---
    print("\n[1] Loading data...")
    meter_df = load_meter_v2()
    sim_df = load_simulated_power()
    solcast = load_solcast()

    meters = sorted(set(meter_df["meter"].unique()) & set(sim_df["meter"].unique()))
    print(f"    Meters with simulation: {len(meters)}")
    print(f"    Analysis period: {YEAR_START}-01-01 to {YEAR_END}-12-31")
    print(f"    Meter data:  {meter_df['timestamp'].min()} to {meter_df['timestamp'].max()}")
    print(f"    Simulated:   {sim_df['timestamp'].min()} to {sim_df['timestamp'].max()}")
    print(f"    Solcast:     {solcast['timestamp'].min()} to {solcast['timestamp'].max()}")

    # --- Process each meter ---
    print("\n[2] Per-meter analysis...")
    all_deg = []
    all_seasonal_rows = []
    all_monthly = []
    all_soiling = []
    all_events = []

    for idx, meter_id in enumerate(meters):
        short = meter_id.split("#")[0].replace("solar.", "")
        print(f"  ({idx+1}/{len(meters)}) {short}", end="")

        m_df = meter_df[meter_df["meter"] == meter_id].copy()
        s_df = sim_df[sim_df["meter"] == meter_id].copy()

        m, a, b = compute_meter_pi(m_df, s_df, solcast)
        valid_pct = 100.0 * m["pi_valid"].sum() / max(1, m["valid_daylight"].sum())

        # 1) Overall degradation
        deg_pct, median_pi, n_months, monthly = overall_degradation(m)

        # 2) Seasonal
        seasonal_df = seasonal_degradation(m)

        # 3) Soiling
        soiling_df = detect_soiling(m)

        # 4) Sudden events
        events_df = detect_sudden_events(m)

        # Collect
        n_soil = int(soiling_df["is_soiling"].sum()) if not soiling_df.empty else 0
        n_events = len(events_df)

        all_deg.append({
            "meter": meter_id,
            "meter_short": short,
            "deg_pct_per_year": round(deg_pct, 4) if np.isfinite(deg_pct) else np.nan,
            "median_PI": round(median_pi, 4) if np.isfinite(median_pi) else np.nan,
            "n_months": n_months if isinstance(n_months, int) else int(n_months),
            "valid_coverage_pct": round(valid_pct, 2),
            "calibration_a": round(a, 6),
            "calibration_b": round(b, 6),
            "soiling_events": n_soil,
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

        print(f"  deg={deg_pct:.2f}%/yr" if np.isfinite(deg_pct) else "  deg=N/A", end="")
        print(f"  soil={n_soil}  events={n_events}")

        # Per-meter plot
        plot_meter_dashboard(meter_id, monthly, seasonal_df, soiling_df, events_df, m)

    # --- Assemble results ---
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
        monthly_all.to_csv(os.path.join(RESULTS_DIR, "monthly_pi.csv"), index=False)
        print(f"    monthly_pi.csv           ({len(monthly_all)} rows)")

    if not soiling_all.empty:
        soiling_all.to_csv(os.path.join(RESULTS_DIR, "soiling_events.csv"), index=False)
        print(f"    soiling_events.csv       ({len(soiling_all)} rows)")

    if not events_all.empty:
        events_all.to_csv(os.path.join(RESULTS_DIR, "sudden_events.csv"), index=False)
        print(f"    sudden_events.csv        ({len(events_all)} rows)")

    # Fleet overview plot
    print("\n[4] Fleet overview plot...")
    plot_fleet_overview(deg_df, seasonal_all, monthly_all)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    valid_meters = deg_df[deg_df["deg_pct_per_year"].notna()]
    if not valid_meters.empty:
        print(f"\nMeters analysed: {len(valid_meters)} / {len(meters)}")
        print(f"Median fleet degradation: {valid_meters['deg_pct_per_year'].median():.2f} %/year")
        print(f"Range: {valid_meters['deg_pct_per_year'].min():.2f} to {valid_meters['deg_pct_per_year'].max():.2f} %/year")
        print(f"\nPer-meter rates:")
        print(valid_meters[["meter_short", "deg_pct_per_year", "median_PI", "soiling_events", "sudden_events"]].to_string(index=False))
    else:
        print("No meters had sufficient data for degradation analysis.")

    if not seasonal_all.empty:
        print(f"\nSeasonal degradation (fleet average):")
        fleet_seasonal = seasonal_all.groupby("season")["deg_pct_per_year"].median()
        for s in SEASON_ORDER:
            if s in fleet_seasonal.index:
                print(f"  {s:8s}: {fleet_seasonal[s]:+.2f} %/year")

    total_soil = soiling_all["is_soiling"].sum() if not soiling_all.empty else 0
    print(f"\nTotal soiling events: {total_soil}")
    print(f"Total sudden events: {len(events_all)}")
    print(f"\nAll outputs in: {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    run()
