"""
Manual Cleaning Event Detection (v2) - Proper Step-Up Detection
================================================================
Detects manual cleaning by looking for dates where yield genuinely
JUMPS UP across many meters simultaneously, after removing
seasonal trends.

Method:
  1) Build daily yield_norm for each meter (GHI-normalised, temp-corrected)
  2) Remove seasonal trend using a 45-day rolling median per meter
  3) Compute per-day "step signal" = median(next 7 days) - median(prev 7 days)
  4) For each day, count how many meters show a positive step (> threshold)
  5) Rank candidate dates by (n_meters_positive, fleet_step_magnitude)
  6) Filter out dates explainable by rain or strong wind
  7) Plot top candidates with raw yield + deseasonalised yield + weather

Outputs (in Results_seasonality/manual_cleaning_events/<timestamp>/):
  - candidate_cleaning_dates.csv
  - detection_timeline.png
  - event_<N>_<date>.png  (per-event detail)
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")

_run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_DIR = os.path.join(BASE, "Results_seasonality", "manual_cleaning_events", _run_stamp)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────
METER_CSV = "SolarMeterReadings1hour_cleaned_2020_2025.csv"
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

SEASONAL_WINDOW = 45
STEP_LOOK_BACK = 7
STEP_LOOK_FORWARD = 7
STEP_THRESHOLD = 0.03
MIN_METERS_FOR_EVENT = 5
RAIN_THRESHOLD_MM = 5.0
WIND_THRESHOLD_MS = 7.0
WINDOW_DAYS = 30
TOP_N_EVENTS = 8

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ── data loading (unchanged) ──────────────────────────────────────────

def load_data():
    print("Loading meter data ...")
    mdf = pd.read_csv(os.path.join(DATA_DIR, METER_CSV))
    mdf["timestamp"] = pd.to_datetime(mdf["timestamp"], errors="coerce")
    mdf = mdf.dropna(subset=["timestamp"])
    mdf = mdf[mdf["meter"].str.contains("bun_", case=False, na=False)].copy()
    mdf = mdf[(mdf["timestamp"].dt.year >= YEAR_START) &
              (mdf["timestamp"].dt.year <= YEAR_END)].copy()
    for ex in EXCLUDE_METERS:
        mdf = mdf[~mdf["meter"].str.contains(ex, case=False, na=False)]
    print(f"  {len(mdf)} hourly rows, {mdf['meter'].nunique()} meters")

    print("Loading Solcast weather data ...")
    sol = pd.read_csv(os.path.join(DATA_DIR, SOLCAST_CSV))
    sol["timestamp"] = pd.to_datetime(sol["timestamp"], errors="coerce")
    sol = sol.dropna(subset=["timestamp"])
    sol["timestamp"] = sol["timestamp"].dt.floor("h")
    sol = sol[(sol["timestamp"].dt.year >= YEAR_START) &
              (sol["timestamp"].dt.year <= YEAR_END)].copy()
    if len(sol) > 1 and sol["timestamp"].diff().dropna().min() < pd.Timedelta("45min"):
        sol = sol.set_index("timestamp").resample("1h").mean(numeric_only=True).reset_index()
    print(f"  {len(sol)} weather rows")

    return mdf, sol


def build_meter_daily(meter_name, mdf, sol):
    """Build daily yield_norm for a single meter."""
    m = mdf[mdf["meter"] == meter_name].copy()
    m["timestamp"] = m["timestamp"].dt.floor("h")

    wcols = ["timestamp"] + [c for c in ["ghi", "zenith", "air_temp",
             "precipitation_rate", "wind_speed_10m"] if c in sol.columns]
    m = m.merge(sol[wcols], on="timestamp", how="left")

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
        temp_factor = (1.0 + TEMP_COEFF * (t_cell - T_REF)).clip(0.5, 1.5)
        m["yield_tc"] = np.where(m["valid"], m["yield_ghi"] / temp_factor, np.nan)
    else:
        m["yield_tc"] = m["yield_ghi"]

    m["date"] = m["timestamp"].dt.date
    valid = m[m["valid"]].copy()
    if valid.empty:
        return pd.DataFrame()

    daily = valid.groupby("date").agg(
        meter_reading=("meter_reading", "sum"),
        ghi=("ghi", "sum"),
        yield_tc=("yield_tc", "median"),
        n_hours=("valid", "count"),
    ).reset_index()

    daily["date_dt"] = pd.to_datetime(daily["date"])
    daily["daily_ghi_kwh"] = daily["ghi"] / 1000.0
    daily = daily[(daily["n_hours"] >= MIN_DAYLIGHT_HOURS) &
                  (daily["daily_ghi_kwh"] >= GHI_MIN_DAILY)].copy()

    daily["yield_daily"] = daily["meter_reading"] / daily["daily_ghi_kwh"]

    q1 = daily["yield_daily"].quantile(0.05)
    q3 = daily["yield_daily"].quantile(0.95)
    iqr = q3 - q1
    daily.loc[~daily["yield_daily"].between(q1 - 2 * iqr, q3 + 2 * iqr), "yield_daily"] = np.nan

    cal = daily[daily["date_dt"].dt.year == CALIBRATION_YEAR]["yield_daily"].dropna()
    norm = cal.median() if len(cal) >= MIN_CAL_YEAR_DAYS else daily["yield_daily"].dropna().median()
    if norm < 1e-6:
        norm = 1.0
    daily["yield_norm"] = daily["yield_daily"] / norm

    return daily[["date_dt", "yield_norm"]].rename(columns={"yield_norm": meter_name})


# ── step-up detection ─────────────────────────────────────────────────

def deseasonalise(all_daily, meter_cols):
    """Remove seasonal trend using a rolling median, return residuals."""
    deseason = all_daily[["date_dt"]].copy()
    for col in meter_cols:
        series = all_daily.set_index("date_dt")[col]
        seasonal = series.rolling(window=SEASONAL_WINDOW, center=True, min_periods=15).median()
        seasonal = seasonal.fillna(method="bfill").fillna(method="ffill")
        residual = series - seasonal
        deseason[col] = residual.values
    return deseason


def compute_step_signals(deseason, meter_cols):
    """For each date, compute step signal = median(forward N days) - median(backward N days)."""
    ds_indexed = deseason.set_index("date_dt")[meter_cols]

    step_forward = ds_indexed.rolling(window=STEP_LOOK_FORWARD, min_periods=3).median().shift(-STEP_LOOK_FORWARD)
    step_backward = ds_indexed.rolling(window=STEP_LOOK_BACK, min_periods=3).median()

    step_signal = step_forward - step_backward
    return step_signal


def build_daily_weather(sol):
    """Aggregate hourly weather to daily."""
    sol_daily = sol.groupby(sol["timestamp"].dt.date).agg(
        rain_mm=("precipitation_rate", "sum"),
        max_wind=("wind_speed_10m", "max"),
        mean_wind=("wind_speed_10m", "mean"),
    ).reset_index()
    sol_daily["date_dt"] = pd.to_datetime(sol_daily["timestamp"])
    return sol_daily[["date_dt", "rain_mm", "max_wind", "mean_wind"]]


def detect_cleaning_events(step_signal, meter_cols, weather_daily):
    """Find candidate cleaning dates: many meters step up, no rain/wind."""
    results = []
    dates = step_signal.index

    for d in dates:
        row = step_signal.loc[d]
        valid_steps = row.dropna()
        if len(valid_steps) < 5:
            continue

        n_positive = (valid_steps > STEP_THRESHOLD).sum()
        n_total = len(valid_steps)
        fleet_step = valid_steps.median()
        fleet_mean_step = valid_steps.mean()
        max_step = valid_steps.max()

        wx = weather_daily[weather_daily["date_dt"] == d]
        if not wx.empty:
            rain_3d = weather_daily[
                (weather_daily["date_dt"] >= d - pd.Timedelta(days=1)) &
                (weather_daily["date_dt"] <= d + pd.Timedelta(days=1))
            ]["rain_mm"].sum()
            wind_3d = weather_daily[
                (weather_daily["date_dt"] >= d - pd.Timedelta(days=1)) &
                (weather_daily["date_dt"] <= d + pd.Timedelta(days=1))
            ]["max_wind"].max()
        else:
            rain_3d = 0
            wind_3d = 0

        has_rain = rain_3d > RAIN_THRESHOLD_MM
        has_wind = wind_3d > WIND_THRESHOLD_MS

        results.append({
            "date": d,
            "n_positive": int(n_positive),
            "n_total": int(n_total),
            "pct_positive": n_positive / n_total * 100 if n_total > 0 else 0,
            "fleet_step_median": fleet_step,
            "fleet_step_mean": fleet_mean_step,
            "max_meter_step": max_step,
            "rain_3d_mm": rain_3d,
            "wind_3d_max": wind_3d,
            "has_rain": has_rain,
            "has_wind": has_wind,
            "explanation": ("rain" if has_rain else "") +
                           ("+wind" if has_wind else "") or "none",
        })

    df = pd.DataFrame(results)
    df = df.sort_values("n_positive", ascending=False)
    return df


def select_top_events(candidates, min_gap_days=14):
    """Pick top N non-overlapping events: unexplained first, then all."""
    unexplained = candidates[candidates["explanation"] == "none"].copy()
    all_cands = candidates.copy()

    selected = []
    used_dates = set()

    for source_label, source_df in [("UNEXPLAINED", unexplained), ("ALL", all_cands)]:
        for _, row in source_df.iterrows():
            if len(selected) >= TOP_N_EVENTS:
                break
            d = row["date"]
            if any(abs((d - ud).days) < min_gap_days for ud in used_dates):
                continue
            if row["n_positive"] < MIN_METERS_FOR_EVENT:
                continue
            entry = row.to_dict()
            entry["source"] = source_label
            selected.append(entry)
            used_dates.add(d)

    return pd.DataFrame(selected)


# ── plotting ──────────────────────────────────────────────────────────

def plot_detection_timeline(candidates, weather_daily, events_selected):
    """Full timeline showing the step-up detection signal."""
    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
    fig.suptitle("Manual Cleaning Detection Timeline (Deseasonalised Step-Up Signal)",
                 fontsize=15, fontweight="bold")

    ax1, ax2, ax3 = axes

    ax1.bar(candidates["date"], candidates["n_positive"],
            color="#78909C", alpha=0.6, width=1)
    ax1.set_ylabel("Meters with\npositive step-up")
    ax1.axhline(MIN_METERS_FOR_EVENT, color="red", linestyle="--", alpha=0.6,
                label=f"Threshold ({MIN_METERS_FOR_EVENT} meters)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    ax2.plot(candidates["date"], candidates["fleet_step_median"],
             color="#1565C0", linewidth=0.8, alpha=0.7, label="Fleet median step")
    ax2.axhline(0, color="grey", linewidth=0.5)
    ax2.set_ylabel("Fleet median\nstep signal")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    ax3.bar(weather_daily["date_dt"], weather_daily["rain_mm"],
            color="#2196F3", alpha=0.5, width=1, label="Rain (mm)")
    ax3_w = ax3.twinx()
    ax3_w.plot(weather_daily["date_dt"], weather_daily["max_wind"],
               color="#FF9800", linewidth=0.6, alpha=0.6, label="Max wind")
    ax3_w.axhline(WIND_THRESHOLD_MS, color="#FF9800", linestyle=":", alpha=0.4)
    ax3.set_ylabel("Rain (mm)", color="#2196F3")
    ax3_w.set_ylabel("Wind (m/s)", color="#FF9800")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_w.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
    ax3.grid(True, alpha=0.2)

    for _, ev in events_selected.iterrows():
        color = "#4CAF50" if ev["explanation"] == "none" else "#FF9800"
        lbl = "Unexplained" if ev["explanation"] == "none" else ev["explanation"]
        for ax in [ax1, ax2, ax3]:
            ax.axvline(ev["date"], color=color, linewidth=2, alpha=0.8, linestyle="-")
        ax1.text(ev["date"], ax1.get_ylim()[1] * 0.92,
                 f' {ev["date"].strftime("%Y-%m-%d")}\n {int(ev["n_positive"])} meters\n {lbl}',
                 fontsize=7, rotation=0, va="top",
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor=color, linewidth=1.5))

    ax3.set_xlabel("Date")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(RESULTS_DIR, "detection_timeline.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved detection_timeline.png")


def plot_event_detail(event_row, all_daily, deseason, sol, event_idx):
    """3-panel plot for one detected event: raw yield, deseasonalised, weather."""
    event_date = pd.Timestamp(event_row["date"])
    start = event_date - pd.Timedelta(days=WINDOW_DAYS)
    end = event_date + pd.Timedelta(days=WINDOW_DAYS)

    meter_cols = [c for c in all_daily.columns if c != "date_dt"]

    w_raw = all_daily[(all_daily["date_dt"] >= start) & (all_daily["date_dt"] <= end)].copy()
    w_ds = deseason[(deseason["date_dt"] >= start) & (deseason["date_dt"] <= end)].copy()

    if w_raw.empty:
        print("    No data, skipping.")
        return

    n_active = w_raw[meter_cols].notna().any().sum()
    n_positive = int(event_row["n_positive"])
    explanation = event_row["explanation"]
    rain_3d = event_row["rain_3d_mm"]
    wind_3d = event_row["wind_3d_max"]

    if explanation == "none":
        verdict = "NO rain/wind -> Possible MANUAL cleaning"
        title_color = "#2E7D32"
    else:
        verdict = f"Weather present ({explanation}, rain={rain_3d:.1f}mm, wind={wind_3d:.1f}m/s)"
        title_color = "#E65100"

    fig, (ax_raw, ax_ds, ax_wx) = plt.subplots(
        3, 1, figsize=(18, 14), height_ratios=[2.5, 2, 1], sharex=True)

    title = (f"Event {event_idx}: {event_date.strftime('%Y-%m-%d')}  |  "
             f"{n_positive}/{n_active} meters stepped up  |  {verdict}")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98, color=title_color)

    cmap = plt.cm.tab20
    colors = [cmap(i / max(len(meter_cols), 1)) for i in range(len(meter_cols))]

    # Panel 1: Raw yield_norm
    for i, col in enumerate(meter_cols):
        s = w_raw[["date_dt", col]].dropna()
        if s.empty:
            continue
        ax_raw.plot(s["date_dt"], s[col], linewidth=0.7, alpha=0.3, color=colors[i])

    fleet_med_raw = w_raw.set_index("date_dt")[meter_cols].median(axis=1)
    ax_raw.plot(fleet_med_raw.index, fleet_med_raw.values,
                linewidth=3, color="#D32F2F", label="Fleet Median (raw)", zorder=10)

    ax_raw.axvline(event_date, color="#2E7D32", linewidth=2.5, alpha=0.9, zorder=11)

    before_raw = fleet_med_raw.loc[fleet_med_raw.index < event_date].tail(STEP_LOOK_BACK)
    after_raw = fleet_med_raw.loc[fleet_med_raw.index >= event_date].head(STEP_LOOK_FORWARD)
    if not before_raw.empty and not after_raw.empty:
        mb = before_raw.median()
        ma = after_raw.median()
        pct = (ma - mb) / mb * 100 if mb > 0 else 0
        ax_raw.axhline(mb, xmin=0, xmax=0.47, color="#FF5722", linewidth=2,
                       linestyle="--", alpha=0.7)
        ax_raw.axhline(ma, xmin=0.53, xmax=1.0, color="#4CAF50", linewidth=2,
                       linestyle="--", alpha=0.7)
        ax_raw.annotate(
            f"Before: {mb:.3f}\nAfter:  {ma:.3f}\nChange: {pct:+.1f}%",
            xy=(event_date, (mb + ma) / 2),
            xytext=(event_date + pd.Timedelta(days=3), (mb + ma) / 2 + 0.12),
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFDE7",
                      edgecolor="#F57F17", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#F57F17", lw=1.5),
            zorder=12)

    ax_raw.set_ylabel("Raw yield_norm\n(baseline 2021 = 1.0)")
    ax_raw.legend(loc="upper left", fontsize=9)
    ax_raw.grid(True, alpha=0.25, linestyle="--")

    # Panel 2: Deseasonalised yield (residual)
    for i, col in enumerate(meter_cols):
        s = w_ds[["date_dt", col]].dropna()
        if s.empty:
            continue
        ax_ds.plot(s["date_dt"], s[col], linewidth=0.7, alpha=0.3, color=colors[i])

    fleet_med_ds = w_ds.set_index("date_dt")[meter_cols].median(axis=1)
    ax_ds.plot(fleet_med_ds.index, fleet_med_ds.values,
               linewidth=3, color="#7B1FA2", label="Fleet Median (deseasonalised)", zorder=10)

    ax_ds.axvline(event_date, color="#2E7D32", linewidth=2.5, alpha=0.9, zorder=11)
    ax_ds.axhline(0, color="grey", linewidth=0.8, alpha=0.5)

    before_ds = fleet_med_ds.loc[fleet_med_ds.index < event_date].tail(STEP_LOOK_BACK)
    after_ds = fleet_med_ds.loc[fleet_med_ds.index >= event_date].head(STEP_LOOK_FORWARD)
    if not before_ds.empty and not after_ds.empty:
        mbd = before_ds.median()
        mad = after_ds.median()
        step_val = mad - mbd
        ax_ds.annotate(
            f"Step: {step_val:+.4f}",
            xy=(event_date, (mbd + mad) / 2),
            xytext=(event_date + pd.Timedelta(days=3), (mbd + mad) / 2 + 0.05),
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EAF6",
                      edgecolor="#3F51B5", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#3F51B5", lw=1.5),
            zorder=12)

    ax_ds.set_ylabel("Deseasonalised yield\n(seasonal trend removed)")
    ax_ds.legend(loc="upper left", fontsize=9)
    ax_ds.grid(True, alpha=0.25, linestyle="--")

    # Panel 3: Weather
    sol_window = sol[(sol["timestamp"] >= start) & (sol["timestamp"] <= end)].copy()
    if not sol_window.empty:
        sol_d = sol_window.groupby(sol_window["timestamp"].dt.date).agg(
            rain_mm=("precipitation_rate", "sum"),
            max_wind=("wind_speed_10m", "max"),
        ).reset_index()
        sol_d["date_dt"] = pd.to_datetime(sol_d["timestamp"])

        ax_wx.bar(sol_d["date_dt"], sol_d["rain_mm"],
                  color="#2196F3", alpha=0.6, width=0.8, label="Rain (mm)")
        ax_wx.set_ylabel("Rain (mm)", color="#2196F3")
        ax_wx.tick_params(axis="y", labelcolor="#2196F3")

        ax_w2 = ax_wx.twinx()
        ax_w2.plot(sol_d["date_dt"], sol_d["max_wind"],
                   color="#FF9800", linewidth=1.5, marker=".", markersize=4,
                   label="Max Wind (m/s)")
        ax_w2.set_ylabel("Max Wind (m/s)", color="#FF9800")
        ax_w2.tick_params(axis="y", labelcolor="#FF9800")
        ax_w2.axhline(WIND_THRESHOLD_MS, color="#FF9800", linestyle=":", alpha=0.4)

        ax_wx.axvline(event_date, color="#2E7D32", linewidth=2.5, alpha=0.9)
        ln1, lb1 = ax_wx.get_legend_handles_labels()
        ln2, lb2 = ax_w2.get_legend_handles_labels()
        ax_wx.legend(ln1 + ln2, lb1 + lb2, loc="upper right", fontsize=9)

    ax_wx.set_xlabel("Date")
    ax_wx.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_wx.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax_wx.grid(True, alpha=0.2)
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f"event_{event_idx}_{event_date.strftime('%Y-%m-%d')}.png"
    fig.savefig(os.path.join(RESULTS_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {fname}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    mdf, sol = load_data()

    meters = sorted(mdf["meter"].unique())
    print(f"\nBuilding daily yield for {len(meters)} meters ...")

    all_frames = []
    for i, meter in enumerate(meters):
        daily = build_meter_daily(meter, mdf, sol)
        if not daily.empty:
            all_frames.append(daily.set_index("date_dt"))
        if (i + 1) % 5 == 0 or (i + 1) == len(meters):
            print(f"  {i+1}/{len(meters)} meters processed")

    all_daily = pd.concat(all_frames, axis=1).reset_index().rename(
        columns={"index": "date_dt"})
    all_daily = all_daily.sort_values("date_dt").reset_index(drop=True)
    meter_cols = [c for c in all_daily.columns if c != "date_dt"]

    print(f"\nAll-meter table: {len(all_daily)} days x {len(meter_cols)} meters")
    print(f"Date range: {all_daily['date_dt'].min().date()} to "
          f"{all_daily['date_dt'].max().date()}")

    all_daily.to_csv(os.path.join(RESULTS_DIR, "all_meters_daily_yield.csv"), index=False)
    print("Saved all_meters_daily_yield.csv")

    # Step 1: Deseasonalise
    print("\nDeseasonalising (removing 45-day rolling median) ...")
    deseason = deseasonalise(all_daily, meter_cols)

    # Step 2: Compute step signals
    print("Computing per-meter step signals (7-day forward - 7-day backward) ...")
    step_signal = compute_step_signals(deseason, meter_cols)

    # Step 3: Daily weather
    print("Building daily weather ...")
    weather_daily = build_daily_weather(sol)

    # Step 4: Detect candidate events
    print("Scanning all dates for simultaneous step-up events ...")
    candidates = detect_cleaning_events(step_signal, meter_cols, weather_daily)
    candidates.to_csv(os.path.join(RESULTS_DIR, "all_candidate_dates.csv"), index=False)

    # Step 5: Select top events (unexplained first)
    events = select_top_events(candidates)

    if events.empty:
        print("\n  No strong cleaning candidates found.")
        return

    print(f"\n{'='*70}")
    print(f"TOP {len(events)} CANDIDATE CLEANING DATES")
    print(f"{'='*70}")
    for i, (_, ev) in enumerate(events.iterrows(), 1):
        tag = "** UNEXPLAINED **" if ev["explanation"] == "none" else ev["explanation"]
        print(f"  {i}. {ev['date'].strftime('%Y-%m-%d')}  |  "
              f"{int(ev['n_positive'])}/{int(ev['n_total'])} meters stepped up  |  "
              f"fleet step={ev['fleet_step_median']:+.4f}  |  "
              f"rain={ev['rain_3d_mm']:.1f}mm  wind={ev['wind_3d_max']:.1f}m/s  |  {tag}")

    events.to_csv(os.path.join(RESULTS_DIR, "candidate_cleaning_dates.csv"), index=False)
    print("\n  Saved candidate_cleaning_dates.csv")

    # Step 6: Timeline overview
    print("\nGenerating detection timeline ...")
    plot_detection_timeline(candidates, weather_daily, events)

    # Step 7: Per-event detail plots
    print("\nGenerating per-event detail plots ...")
    for idx, (_, ev) in enumerate(events.iterrows(), 1):
        print(f"  [{idx}/{len(events)}] {ev['date'].strftime('%Y-%m-%d')}")
        plot_event_detail(ev, all_daily, deseason, sol, idx)

    print(f"\nAll outputs saved to:\n  {RESULTS_DIR}")


if __name__ == "__main__":
    main()
