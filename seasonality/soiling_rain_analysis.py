"""
Soiling & Rainfall Impact Analysis
===================================
How does power generation change after rainfall?
Quantifies soiling by comparing performance on clear-sky days
before and after rain events.

Method: GHI-normalised yield (daily kWh / daily GHI) removes weather
variation, so any remaining change is due to panel surface condition.

Key outputs:
  - Sawtooth pattern: yield declines during dry spells, recovers after rain
  - Per-event before/after comparison (on clear-sky days only)
  - Event-aligned "average event" plot (stacking all events)
  - Seasonal soiling rates
  - Annual soiling energy loss estimate

Inputs:
  - SolarMeterReadings1hour_cleaned_v2.csv
  - solcast_df.csv (with precipitation_rate, ghi)

Outputs (in Results_seasonality/):
  - soiling_rain_summary.csv         per-meter soiling rates and losses
  - soiling_rain_events.csv          every rain event with before/after yield
  - soiling_seasonal_rates.csv       soiling rate by season per meter
  - soiling_sawtooth_<meter>.png     per-meter sawtooth visualisation
  - soiling_fleet_dashboard.png      fleet-wide soiling dashboard
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "Results_seasonality")
os.makedirs(RESULTS_DIR, exist_ok=True)

METER_V2_CSV = "SolarMeterReadings1hour_cleaned_v2.csv"
SOLCAST_CSV = "solcast_df_cleaned.csv"
SOLCAST_FALLBACK = "solcast_df.csv"
SOLCAST_CAMPUS = "BUNDOORA"

# --- Thresholds ---
GHI_MIN_HOURLY = 100          # W/m² — only hours with good sunlight
ZENITH_MAX = 75               # degrees — exclude low-angle hours for cleaner signal
MIN_HOURS_PER_DAY = 4         # need this many valid sun-hours for a reliable day
MIN_DRY_DAYS = 5              # minimum consecutive dry days to qualify as a dry spell
RAIN_THRESHOLD = 0.5          # mm/h peak — above this = rain day
DRY_THRESHOLD = 0.05          # mm/h daily mean — below this = dry day
BEFORE_WINDOW = 3             # clear-sky days before rain to average
AFTER_WINDOW = 7              # days after rain to search for clear-sky days
MIN_CLEAR_AFTER = 2           # need at least 2 clear days after rain
YIELD_CLIP_Q = 0.02           # clip extreme yield outliers (top/bottom 2%)

SEASON_MAP = {
    12: "Summer", 1: "Summer", 2: "Summer",
    3: "Autumn",  4: "Autumn",  5: "Autumn",
    6: "Winter",  7: "Winter",  8: "Winter",
    9: "Spring", 10: "Spring", 11: "Spring",
}
SEASON_ORDER = ["Summer", "Autumn", "Winter", "Spring"]


def load_data():
    """Load meter readings and Solcast weather."""
    print("  Loading meter data...", end="", flush=True)
    df = pd.read_csv(os.path.join(DATA_DIR, METER_V2_CSV))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[df["meter"].str.contains("bun_", case=False, na=False)].copy()
    print(f" {len(df)} rows, {df['meter'].nunique()} meters")

    print("  Loading Solcast...", end="", flush=True)
    sol = None
    for name in [SOLCAST_CSV, SOLCAST_FALLBACK]:
        p = os.path.join(DATA_DIR, name)
        if os.path.isfile(p):
            sol = pd.read_csv(p)
            sol["timestamp"] = pd.to_datetime(sol["timestamp"], errors="coerce")
            sol = sol.dropna(subset=["timestamp"])
            if "campus" in sol.columns:
                sol = sol[sol["campus"].str.upper().str.strip() == SOLCAST_CAMPUS].copy()
            sol["timestamp"] = sol["timestamp"].dt.floor("h")
            if len(sol) > 1 and sol["timestamp"].diff().dropna().min() < pd.Timedelta("45min"):
                sol = sol.set_index("timestamp").resample("1h").mean(numeric_only=True).reset_index()
            break
    if sol is None:
        raise FileNotFoundError("Cannot find Solcast data")
    print(f" {len(sol)} hourly rows")
    return df, sol


def build_daily_yield(meter_df, sol_df):
    """
    Compute daily GHI-normalised yield for one meter.
    Yield = sum(meter_reading) / sum(ghi) for qualified hours only.
    Returns daily DataFrame with yield, precip, and clear-sky flags.
    """
    m = meter_df.copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"]).dt.floor("h")
    m = m.merge(sol_df[["timestamp", "ghi", "zenith", "precipitation_rate"]],
                on="timestamp", how="left")

    valid = (
        m["analysis_valid"].astype(bool)
        & (m["ghi"] >= GHI_MIN_HOURLY)
        & (m["zenith"] <= ZENITH_MAX)
        & (m["meter_reading"] >= 0)
    )
    m = m[valid].copy()
    if m.empty:
        return pd.DataFrame()

    m["date"] = pd.to_datetime(m["timestamp"]).dt.date

    daily = m.groupby("date").agg(
        daily_kwh=("meter_reading", "sum"),
        daily_ghi=("ghi", "sum"),
        hours=("meter_reading", "count"),
        daily_precip_mean=("precipitation_rate", "mean"),
        daily_precip_max=("precipitation_rate", "max"),
        daily_precip_sum=("precipitation_rate", "sum"),
    ).reset_index()

    daily = daily[daily["hours"] >= MIN_HOURS_PER_DAY].copy()
    daily["daily_ghi_kwh"] = daily["daily_ghi"] / 1000.0

    daily["yield_norm"] = daily["daily_kwh"] / daily["daily_ghi_kwh"]
    daily = daily[np.isfinite(daily["yield_norm"]) & (daily["yield_norm"] > 0)].copy()

    lo = daily["yield_norm"].quantile(YIELD_CLIP_Q)
    hi = daily["yield_norm"].quantile(1 - YIELD_CLIP_Q)
    daily = daily[(daily["yield_norm"] >= lo) & (daily["yield_norm"] <= hi)].copy()

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["date_dt"] = pd.to_datetime(daily["date"])
    daily["is_dry"] = daily["daily_precip_mean"] < DRY_THRESHOLD
    daily["is_rain"] = daily["daily_precip_max"] >= RAIN_THRESHOLD

    ghi_median = daily["daily_ghi_kwh"].median()
    daily["is_clear"] = daily["is_dry"] & (daily["daily_ghi_kwh"] >= ghi_median * 0.7)
    daily["month"] = daily["date_dt"].dt.month
    daily["season"] = daily["month"].map(SEASON_MAP)

    baseline = daily.loc[daily["is_clear"], "yield_norm"].median()
    daily["yield_ratio"] = daily["yield_norm"] / baseline if baseline > 0 else np.nan

    seasonal_baselines = daily.loc[daily["is_clear"]].groupby("season")["yield_norm"].median()
    daily["season_baseline"] = daily["season"].map(seasonal_baselines)
    daily["yield_season_norm"] = daily["yield_norm"] / daily["season_baseline"]

    return daily


def find_rain_events(daily):
    """
    For each rain event preceded by >= MIN_DRY_DAYS dry days, compare
    yield before and after rain using GHI-matched clear-sky days.
    This eliminates the bias where dry-spell days have higher GHI
    than post-rain "clear" days.
    """
    if daily.empty:
        return []

    n = len(daily)
    events = []

    for i in range(n):
        if not daily.iloc[i]["is_rain"]:
            continue

        dry_count = 0
        j = i - 1
        while j >= 0 and daily.iloc[j]["is_dry"]:
            dry_count += 1
            j -= 1

        if dry_count < MIN_DRY_DAYS:
            continue

        spell_start = max(0, i - dry_count)

        clear_before = daily.iloc[spell_start:i]
        clear_before = clear_before[clear_before["is_clear"]]
        if len(clear_before) < 2:
            continue

        last_clear = clear_before.tail(BEFORE_WINDOW)
        first_clear = clear_before.head(min(3, len(clear_before)))
        yield_before = last_clear["yield_norm"].median()
        yield_start = first_clear["yield_norm"].median()
        ghi_before = last_clear["daily_ghi_kwh"].median()

        after_end = min(n, i + 1 + AFTER_WINDOW)
        after_slice = daily.iloc[i + 1:after_end]
        after_clear = after_slice[after_slice["is_clear"]]
        if len(after_clear) < MIN_CLEAR_AFTER:
            continue
        after_top = after_clear.head(MIN_CLEAR_AFTER + 2)
        yield_after = after_top.head(MIN_CLEAR_AFTER)["yield_norm"].median()
        ghi_after = after_top.head(MIN_CLEAR_AFTER)["daily_ghi_kwh"].median()

        # GHI-matched recovery: only if before/after GHI within 25%
        ghi_ratio = ghi_after / ghi_before if ghi_before > 0 else np.nan
        ghi_matched = (0.75 <= ghi_ratio <= 1.25) if np.isfinite(ghi_ratio) else False
        matched_recovery = np.nan
        if ghi_matched:
            matched_recovery = (yield_after - yield_before) / (yield_before + 1e-9) * 100

        # Dry-spell soiling rate on clear days
        soiling_rate = np.nan
        if len(clear_before) >= 3:
            x = np.arange(len(clear_before), dtype=float)
            y = clear_before["yield_norm"].values
            ok = np.isfinite(y)
            if ok.sum() >= 3:
                slope, _, _, _, _ = sp_stats.linregress(x[ok], y[ok])
                soiling_rate = slope

        rain_mm = daily.iloc[i]["daily_precip_sum"]
        recovery = yield_after - yield_before

        sn_before = last_clear["yield_season_norm"].median() if "yield_season_norm" in last_clear.columns else np.nan
        sn_after = after_top.head(MIN_CLEAR_AFTER)["yield_season_norm"].median() if "yield_season_norm" in after_top.columns else np.nan
        sn_recovery = sn_after - sn_before if (np.isfinite(sn_before) and np.isfinite(sn_after)) else np.nan

        events.append({
            "rain_date": daily.iloc[i]["date"],
            "season": daily.iloc[i]["season"],
            "dry_days_before": dry_count,
            "clear_days_before": len(clear_before),
            "yield_start_of_dry": round(yield_start, 4),
            "yield_before_rain": round(yield_before, 4),
            "yield_after_rain": round(yield_after, 4),
            "yield_recovery": round(recovery, 4),
            "yield_recovery_pct": round(100 * recovery / (yield_before + 1e-9), 2),
            "ghi_matched": ghi_matched,
            "ghi_matched_recovery_pct": round(matched_recovery, 2) if np.isfinite(matched_recovery) else np.nan,
            "sn_recovery_pct": round(100 * sn_recovery, 2) if np.isfinite(sn_recovery) else np.nan,
            "ghi_before_kwh": round(ghi_before, 2),
            "ghi_after_kwh": round(ghi_after, 2),
            "ghi_ratio": round(ghi_ratio, 3) if np.isfinite(ghi_ratio) else np.nan,
            "soiling_loss_pct": round(100 * (yield_start - yield_before) / (yield_start + 1e-9), 2) if np.isfinite(yield_start) else np.nan,
            "soiling_rate_per_day": round(soiling_rate, 6) if np.isfinite(soiling_rate) else np.nan,
            "rain_mm": round(rain_mm, 2),
        })

    return events


def compute_event_aligned_yield(daily, events_df, window=14):
    """
    Stack all events and compute the average season-normalised yield
    relative to rain day (day 0). Uses yield_season_norm to remove
    seasonal bias from the aligned average.
    """
    if events_df.empty or daily.empty:
        return pd.DataFrame()

    col = "yield_season_norm" if "yield_season_norm" in daily.columns else "yield_ratio"
    lookup = daily.set_index("date_dt")[col].to_dict()
    all_curves = []

    for _, ev in events_df.iterrows():
        rain_dt = pd.to_datetime(ev["rain_date"])
        curve = {}
        for off in range(-window, window + 1):
            d = rain_dt + pd.Timedelta(days=off)
            if d in lookup and np.isfinite(lookup[d]):
                curve[off] = lookup[d]
        if len(curve) >= 5:
            all_curves.append(curve)

    if not all_curves:
        return pd.DataFrame()

    rows = []
    for off in range(-window, window + 1):
        vals = [c[off] for c in all_curves if off in c]
        if len(vals) >= 5:
            rows.append({
                "day_offset": off,
                "mean_yield": np.mean(vals),
                "median_yield": np.median(vals),
                "std_yield": np.std(vals),
                "count": len(vals),
            })
    return pd.DataFrame(rows)


def compute_dryspell_trajectory(daily, min_spell=7, max_day=21):
    """
    Find all dry spells of >= min_spell days and track how yield evolves
    day-by-day from the start (day 0). Returns a DataFrame with the
    average trajectory, which should slope downward if soiling occurs.
    """
    if daily.empty:
        return pd.DataFrame()

    n = len(daily)
    spells = []
    i = 0
    while i < n:
        if daily.iloc[i]["is_dry"]:
            start = i
            while i < n and daily.iloc[i]["is_dry"]:
                i += 1
            length = i - start
            if length >= min_spell:
                spells.append((start, i))
        else:
            i += 1

    if not spells:
        return pd.DataFrame()

    baseline = daily.loc[daily["is_clear"], "yield_norm"].median()
    if baseline <= 0:
        return pd.DataFrame()

    all_curves = []
    for s_start, s_end in spells:
        spell_data = daily.iloc[s_start:s_end]
        clear_spell = spell_data[spell_data["is_clear"]]
        if len(clear_spell) < 3:
            continue
        first_yield = clear_spell.iloc[0]["yield_norm"]
        if first_yield <= 0:
            continue

        curve = {}
        for j, (_, row) in enumerate(clear_spell.iterrows()):
            if j <= max_day:
                curve[j] = row["yield_norm"] / first_yield
        if len(curve) >= 3:
            all_curves.append(curve)

    if not all_curves:
        return pd.DataFrame()

    rows = []
    for day in range(max_day + 1):
        vals = [c[day] for c in all_curves if day in c]
        if len(vals) >= 3:
            rows.append({
                "day_in_spell": day,
                "mean_yield": np.mean(vals),
                "median_yield": np.median(vals),
                "std_yield": np.std(vals),
                "count": len(vals),
            })
    return pd.DataFrame(rows)


def seasonal_summary(events_df):
    """Average soiling metrics by season."""
    if events_df.empty:
        return pd.DataFrame()
    s = events_df.groupby("season").agg(
        n_events=("rain_date", "count"),
        avg_recovery_pct=("yield_recovery_pct", "mean"),
        median_recovery_pct=("yield_recovery_pct", "median"),
        avg_soiling_loss_pct=("soiling_loss_pct", "mean"),
        avg_dry_days=("dry_days_before", "mean"),
        avg_soiling_rate=("soiling_rate_per_day", "mean"),
    ).reset_index()
    for col in s.columns:
        if s[col].dtype == float:
            s[col] = s[col].round(4)
    return s


def estimate_annual_loss(events_df, daily):
    """Rough annual soiling energy loss estimate."""
    if events_df.empty or daily.empty:
        return np.nan, np.nan

    valid = events_df["soiling_rate_per_day"].dropna()
    if valid.empty:
        return np.nan, np.nan

    avg_rate = valid.mean()
    n_dry = daily["is_dry"].sum()
    total = len(daily)
    dry_frac = n_dry / total if total > 0 else 0
    dry_days_yr = dry_frac * 365.25

    baseline_yield = daily.loc[daily["is_clear"], "yield_norm"].median()
    if not np.isfinite(baseline_yield) or baseline_yield <= 0:
        return np.nan, np.nan

    annual_loss_pct = abs(avg_rate) / baseline_yield * dry_days_yr * 100
    avg_daily_kwh = daily["daily_kwh"].mean()
    annual_loss_kwh = abs(avg_rate) / baseline_yield * dry_days_yr * avg_daily_kwh
    return round(annual_loss_pct, 2), round(annual_loss_kwh, 2)


# ===================================================================
# Plotting
# ===================================================================

def plot_meter(meter_id, daily, events_df):
    """Per-meter 4-panel soiling dashboard."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    short = meter_id.split("#")[0].replace("solar.", "")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Soiling & Rainfall Impact — {short}", fontsize=14, fontweight="bold")

    # 1) Daily yield timeline
    ax = axes[0, 0]
    if not daily.empty:
        clear = daily[daily["is_clear"]]
        other = daily[~daily["is_clear"]]
        if not other.empty:
            ax.scatter(other["date_dt"], other["yield_norm"], s=3, alpha=0.12,
                       color="lightgray", zorder=1, label="Cloudy/wet days")
        if not clear.empty:
            ax.scatter(clear["date_dt"], clear["yield_norm"], s=8, alpha=0.45,
                       color="steelblue", zorder=2, label="Clear-sky days")
            smooth = clear.set_index("date_dt")["yield_norm"].rolling(
                "21D", min_periods=3).median()
            ax.plot(smooth.index, smooth.values, color="navy", lw=2,
                    label="21-day rolling median (clear)", zorder=3)

        rain_days = daily[daily["is_rain"]]
        for _, rd in rain_days.iterrows():
            ax.axvline(rd["date_dt"], color="dodgerblue", alpha=0.08, lw=0.5)

    ax.set_ylabel("GHI-Normalised Yield (kWh / kWh/m²)")
    ax.set_title("Daily Yield (higher = cleaner panels)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    # 2) Daily rainfall
    ax = axes[0, 1]
    if not daily.empty:
        ax.bar(daily["date_dt"], daily["daily_precip_sum"], width=1,
               color="dodgerblue", alpha=0.7)
    ax.set_ylabel("Rain (mm/day)")
    ax.set_title("Daily Rainfall")
    ax.grid(True, alpha=0.3, axis="y")

    # 3) Event-aligned yield
    ax = axes[1, 0]
    aligned = compute_event_aligned_yield(daily, events_df)
    if not aligned.empty:
        ax.fill_between(aligned["day_offset"],
                        aligned["median_yield"] - aligned["std_yield"],
                        aligned["median_yield"] + aligned["std_yield"],
                        alpha=0.15, color="steelblue")
        ax.plot(aligned["day_offset"], aligned["median_yield"],
                color="steelblue", lw=2.5, marker="o", ms=4, zorder=3)
        ax.axvline(0, color="dodgerblue", lw=2, ls="--", label="Rain day")

        pre = aligned[aligned["day_offset"] == -1]
        post = aligned[aligned["day_offset"] == 1]
        if not pre.empty and not post.empty:
            delta = post.iloc[0]["median_yield"] - pre.iloc[0]["median_yield"]
            ax.annotate(
                f"Recovery: {delta:+.4f}\n({delta*100:+.2f}%)",
                xy=(2, post.iloc[0]["median_yield"]),
                fontsize=10, fontweight="bold",
                color="green" if delta > 0 else "red",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    ax.set_xlabel("Days relative to rain")
    ax.set_ylabel("Normalised yield (1.0 = baseline)")
    ax.set_title("Average Yield Around Rain Events")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4) Dry-spell trajectory (soiling accumulation)
    ax = axes[1, 1]
    traj = compute_dryspell_trajectory(daily)
    if not traj.empty:
        ax.fill_between(traj["day_in_spell"],
                        traj["median_yield"] - traj["std_yield"],
                        traj["median_yield"] + traj["std_yield"],
                        alpha=0.15, color="sandybrown")
        ax.plot(traj["day_in_spell"], traj["median_yield"],
                color="saddlebrown", lw=2.5, marker="o", ms=4, zorder=3)
        ax.axhline(1.0, ls=":", color="green", alpha=0.5)

        if len(traj) >= 3:
            x_fit = traj["day_in_spell"].values.astype(float)
            y_fit = traj["median_yield"].values
            ok = np.isfinite(y_fit)
            if ok.sum() >= 3:
                sl, ic, _, _, _ = sp_stats.linregress(x_fit[ok], y_fit[ok])
                ax.plot(x_fit, sl * x_fit + ic, "r--", lw=1.5,
                        label=f"trend: {sl*100:+.3f}%/day")
                ax.legend(fontsize=9)

        ax.set_xlabel("Day within dry spell (0 = first dry day)")
        ax.set_ylabel("Yield relative to day 0")
        ax.set_title("Yield During Dry Spells\n(declining = soiling accumulation)")
    else:
        ax.text(0.5, 0.5, "Insufficient dry-spell data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, f"soiling_sawtooth_{short}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fleet(summary_df, seasonal_all, events_all, fleet_aligned, fleet_traj=None):
    """Fleet-wide soiling dashboard."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle("Fleet Soiling & Rainfall Impact — Bundoora Solar",
                 fontsize=15, fontweight="bold")

    # 1) Fleet event-aligned yield
    ax = axes[0, 0]
    if fleet_aligned is not None and not fleet_aligned.empty:
        ax.fill_between(fleet_aligned["day_offset"],
                        fleet_aligned["median_yield"] - fleet_aligned["std_yield"],
                        fleet_aligned["median_yield"] + fleet_aligned["std_yield"],
                        alpha=0.15, color="steelblue")
        ax.plot(fleet_aligned["day_offset"], fleet_aligned["median_yield"],
                color="steelblue", lw=3, marker="o", ms=5, label="Fleet median yield")
        ax.axvline(0, color="dodgerblue", lw=2, ls="--", label="Rain day")

        pre = fleet_aligned[fleet_aligned["day_offset"] < 0]
        post = fleet_aligned[fleet_aligned["day_offset"] > 0]
        if not pre.empty and not post.empty:
            y_before = pre.iloc[-1]["median_yield"]
            y_after = post.iloc[0]["median_yield"]
            delta = y_after - y_before
            ax.annotate(
                f"Recovery: {delta:+.4f}\n({delta*100:+.2f}%)",
                xy=(2, y_after), fontsize=11, fontweight="bold",
                color="green" if delta > 0 else "red",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    ax.set_xlabel("Days relative to rain")
    ax.set_ylabel("Normalised yield (1.0 = clean baseline)")
    ax.set_title("Fleet-Average Yield Around Rain Events\n(If soiling exists: dip before rain, rise after)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2) Per-meter season-normalised recovery
    ax = axes[0, 1]
    col_to_use = "avg_sn_recovery_pct" if "avg_sn_recovery_pct" in summary_df.columns else "avg_recovery_pct"
    if not summary_df.empty:
        s = summary_df.dropna(subset=[col_to_use]).sort_values(col_to_use)
        colors = ["green" if v > 0 else "indianred" for v in s[col_to_use]]
        ax.barh(s["meter_short"], s[col_to_use], color=colors,
                alpha=0.8, edgecolor="gray", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Season-normalised yield recovery (%)")
    ax.set_title("Per-Meter Recovery After Rain (season-adjusted)\n(positive = rain cleaned soiling)")
    ax.grid(True, alpha=0.3, axis="x")

    # 3) Seasonal soiling heatmap
    ax = axes[1, 0]
    if not seasonal_all.empty:
        pivot = seasonal_all.pivot_table(
            index="meter_short", columns="season", values="avg_recovery_pct",
        )
        cols = [s for s in SEASON_ORDER if s in pivot.columns]
        if cols:
            pivot = pivot[cols]
            try:
                sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn",
                            center=0, cbar_kws={"label": "Recovery (%)"},
                            linewidths=0.5)
            except Exception:
                pass
    ax.set_title("Seasonal Yield Recovery After Rain (%)\n(positive = soiling detected)")

    # 4) Fleet dry-spell trajectory
    ax = axes[1, 1]
    if fleet_traj is not None and not fleet_traj.empty:
        ax.fill_between(fleet_traj["day_in_spell"],
                        fleet_traj["median_yield"] - fleet_traj["std_yield"],
                        fleet_traj["median_yield"] + fleet_traj["std_yield"],
                        alpha=0.15, color="sandybrown")
        ax.plot(fleet_traj["day_in_spell"], fleet_traj["median_yield"],
                color="saddlebrown", lw=3, marker="o", ms=5, zorder=3)
        ax.axhline(1.0, ls=":", color="green", alpha=0.5)

        if len(fleet_traj) >= 3:
            x_fit = fleet_traj["day_in_spell"].values.astype(float)
            y_fit = fleet_traj["median_yield"].values
            ok = np.isfinite(y_fit)
            if ok.sum() >= 3:
                sl, ic, r, p, _ = sp_stats.linregress(x_fit[ok], y_fit[ok])
                ax.plot(x_fit, sl * x_fit + ic, "r--", lw=2,
                        label=f"trend: {sl*100:+.3f}%/day, p={p:.3f}")
                ax.legend(fontsize=9)

                if abs(sl) > 1e-6:
                    ax.annotate(
                        f"Soiling rate:\n{sl*100:.3f}%/day\n({sl*365.25*100:.1f}%/yr)",
                        xy=(fleet_traj["day_in_spell"].max() * 0.6,
                            fleet_traj["median_yield"].min()),
                        fontsize=10, fontweight="bold", color="brown",
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    ax.set_xlabel("Day within dry spell (0 = start)")
    ax.set_ylabel("Yield relative to day 0 (1.0 = clean)")
    ax.set_title("Fleet Dry-Spell Trajectory\n(declining = soiling accumulation)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(RESULTS_DIR, "soiling_fleet_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fleet dashboard: {path}")


# ===================================================================
# Main
# ===================================================================

def run():
    print("=" * 70)
    print("SOILING & RAINFALL IMPACT ANALYSIS")
    print("=" * 70)

    print("\n[1] Loading data...")
    meter_df, sol_df = load_data()
    meters = sorted(meter_df["meter"].unique())

    has_precip = "precipitation_rate" in sol_df.columns
    has_ghi = "ghi" in sol_df.columns
    print(f"    Precipitation column: {has_precip}")
    print(f"    GHI column: {has_ghi}")
    if not has_precip or not has_ghi:
        print("  ERROR: Need both precipitation_rate and ghi in Solcast data.")
        return

    print(f"\n[2] Analysing {len(meters)} meters...")
    all_summary = []
    all_events = []
    all_seasonal = []
    all_aligned = []
    all_trajectories = []

    for idx, meter_id in enumerate(meters):
        short = meter_id.split("#")[0].replace("solar.", "")
        print(f"  ({idx+1}/{len(meters)}) {short}", end="", flush=True)

        m_df = meter_df[meter_df["meter"] == meter_id].copy()
        daily = build_daily_yield(m_df, sol_df)

        if daily.empty or len(daily) < 30:
            print("  — insufficient data")
            continue

        events = find_rain_events(daily)
        ev_df = pd.DataFrame(events)
        n_ev = len(ev_df)

        avg_rec = ev_df["yield_recovery_pct"].mean() if n_ev > 0 else np.nan
        avg_sn_rec = ev_df["sn_recovery_pct"].dropna().mean() if n_ev > 0 else np.nan
        matched = ev_df["ghi_matched_recovery_pct"].dropna() if n_ev > 0 else pd.Series(dtype=float)
        avg_matched_rec = matched.mean() if len(matched) > 0 else np.nan
        n_matched = len(matched)
        avg_rate = ev_df["soiling_rate_per_day"].dropna().mean() if n_ev > 0 else np.nan
        loss_pct, loss_kwh = estimate_annual_loss(ev_df, daily)

        ssn = seasonal_summary(ev_df)
        if not ssn.empty:
            ssn["meter"] = meter_id
            ssn["meter_short"] = short
            all_seasonal.append(ssn)

        all_summary.append({
            "meter": meter_id,
            "meter_short": short,
            "n_rain_events": n_ev,
            "n_ghi_matched": n_matched,
            "avg_recovery_pct": round(avg_rec, 2) if np.isfinite(avg_rec) else np.nan,
            "avg_sn_recovery_pct": round(avg_sn_rec, 2) if np.isfinite(avg_sn_rec) else np.nan,
            "avg_ghi_matched_recovery_pct": round(avg_matched_rec, 2) if np.isfinite(avg_matched_rec) else np.nan,
            "avg_soiling_rate_per_day": round(avg_rate, 6) if np.isfinite(avg_rate) else np.nan,
            "annual_soiling_loss_pct": loss_pct,
            "annual_soiling_loss_kwh": loss_kwh,
            "total_days": len(daily),
            "dry_days": int(daily["is_dry"].sum()),
            "rain_days": int(daily["is_rain"].sum()),
            "dry_pct": round(100 * daily["is_dry"].sum() / len(daily), 1),
        })

        traj = compute_dryspell_trajectory(daily)
        if not traj.empty:
            all_trajectories.append(traj)

        if n_ev > 0:
            ev_df["meter"] = meter_id
            ev_df["meter_short"] = short
            all_events.append(ev_df)

            aligned = compute_event_aligned_yield(daily, ev_df)
            if not aligned.empty:
                all_aligned.append(aligned)

        rec_s = f"{avg_rec:+.2f}%" if np.isfinite(avg_rec) else "N/A"
        match_s = f"{avg_matched_rec:+.2f}%" if np.isfinite(avg_matched_rec) else "N/A"
        print(f"  events={n_ev}  raw_rec={rec_s}  ghi_matched({n_matched})={match_s}", flush=True)

        plot_meter(meter_id, daily, ev_df)

    # --- Assemble ---
    print("\n[3] Saving results...")
    summary = pd.DataFrame(all_summary)
    events_all = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    seasonal_all = pd.concat(all_seasonal, ignore_index=True) if all_seasonal else pd.DataFrame()

    summary.to_csv(os.path.join(RESULTS_DIR, "soiling_rain_summary.csv"), index=False)
    print(f"    soiling_rain_summary.csv   ({len(summary)} meters)")
    if not events_all.empty:
        events_all.to_csv(os.path.join(RESULTS_DIR, "soiling_rain_events.csv"), index=False)
        print(f"    soiling_rain_events.csv    ({len(events_all)} events)")
    if not seasonal_all.empty:
        seasonal_all.to_csv(os.path.join(RESULTS_DIR, "soiling_seasonal_rates.csv"), index=False)
        print(f"    soiling_seasonal_rates.csv ({len(seasonal_all)} rows)")

    # Fleet aligned
    fleet_aligned = pd.DataFrame()
    if all_aligned:
        pool = pd.concat(all_aligned, ignore_index=True)
        fleet_aligned = pool.groupby("day_offset").agg(
            median_yield=("median_yield", "mean"),
            std_yield=("std_yield", "mean"),
            count=("count", "sum"),
        ).reset_index()

    # Fleet trajectory
    fleet_traj = pd.DataFrame()
    if all_trajectories:
        pool_t = pd.concat(all_trajectories, ignore_index=True)
        fleet_traj = pool_t.groupby("day_in_spell").agg(
            median_yield=("median_yield", "mean"),
            std_yield=("std_yield", "mean"),
            count=("count", "sum"),
        ).reset_index()

    # Fleet dashboard
    print("\n[4] Fleet dashboard...")
    plot_fleet(summary, seasonal_all, events_all, fleet_aligned, fleet_traj)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("SOILING ANALYSIS RESULTS")
    print("=" * 70)

    valid = summary[summary["avg_recovery_pct"].notna()]
    if not valid.empty:
        print(f"\nMeters analysed: {len(valid)}")
        print(f"Total qualifying rain events: {len(events_all)}")

        pos_rec = events_all["yield_recovery_pct"].gt(0).sum() if not events_all.empty else 0
        neg_rec = events_all["yield_recovery_pct"].lt(0).sum() if not events_all.empty else 0
        print(f"  Events with positive recovery (soiling confirmed): {pos_rec}")
        print(f"  Events with negative recovery: {neg_rec}")

        avg_fleet_raw = valid["avg_recovery_pct"].mean()
        avg_fleet_sn = valid["avg_sn_recovery_pct"].dropna().mean()
        avg_fleet_matched = valid["avg_ghi_matched_recovery_pct"].dropna().mean()
        total_matched = valid["n_ghi_matched"].sum()

        print(f"\nFleet avg yield recovery (raw):                 {avg_fleet_raw:+.2f}%")
        print(f"Fleet avg yield recovery (season-normalised):    {avg_fleet_sn:+.2f}%")
        print(f"Fleet avg yield recovery (GHI-matched, n={int(total_matched)}): {avg_fleet_matched:+.2f}%")

        print(f"\nINTERPRETATION:")
        if avg_fleet_matched > 1.0:
            print("  SOILING CONFIRMED: When comparing days with similar GHI")
            print("  before and after rain, yield increases => rain cleans panels.")
        elif avg_fleet_matched > -2.0:
            print("  MARGINAL/LOW SOILING: The GHI-matched recovery is close to zero.")
            print("  Bundoora (Melbourne) receives regular rainfall (~650mm/yr),")
            print("  which naturally keeps panels relatively clean.")
            print("  Soiling losses are likely <2% annually - typical for")
            print("  temperate climates with regular precipitation.")
        else:
            print("  The negative recovery (even GHI-matched) suggests that")
            print("  atmospheric conditions (humidity, aerosols) after rain")
            print("  affect PV performance more than soiling between events.")
            print("  This is consistent with a low-soiling environment where")
            print("  weather effects dominate any soiling signal.")

        print(f"\nPer-meter summary:")
        cols = ["meter_short", "avg_recovery_pct", "avg_ghi_matched_recovery_pct",
                "n_rain_events", "n_ghi_matched", "dry_pct"]
        print(valid[cols].to_string(index=False))

    if not seasonal_all.empty:
        print(f"\nSeasonal recovery (fleet avg):")
        fs = seasonal_all.groupby("season")["avg_recovery_pct"].mean()
        for s in SEASON_ORDER:
            if s in fs.index:
                v = fs[s]
                label = "soiling" if v > 0 else "no soiling"
                print(f"  {s:8s}: {v:+.2f}% recovery ({label})")

    if not fleet_aligned.empty:
        pre = fleet_aligned[fleet_aligned["day_offset"] == -1]
        post = fleet_aligned[fleet_aligned["day_offset"] == 1]
        if not pre.empty and not post.empty:
            d = post.iloc[0]["median_yield"] - pre.iloc[0]["median_yield"]
            print(f"\nEvent-aligned fleet yield change (day -1 to +1): {d:+.4f} ({d*100:+.2f}%)")

    if not fleet_traj.empty and len(fleet_traj) >= 3:
        x_t = fleet_traj["day_in_spell"].values.astype(float)
        y_t = fleet_traj["median_yield"].values
        ok_t = np.isfinite(y_t)
        if ok_t.sum() >= 3:
            sl_t, _, _, p_t, _ = sp_stats.linregress(x_t[ok_t], y_t[ok_t])
            print(f"\nDry-spell soiling rate: {sl_t*100:+.4f}%/day (p={p_t:.4f})")
            print(f"  Annualised: {sl_t*365.25*100:+.2f}%/year")
            if sl_t < -0.001 and p_t < 0.05:
                print("  => SOILING ACCUMULATION DETECTED during dry spells")
            elif sl_t < 0:
                print("  => Slight downward trend, but not statistically significant")
            else:
                print("  => No soiling accumulation detected during dry spells")

    print(f"\nAll outputs in: {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    run()
