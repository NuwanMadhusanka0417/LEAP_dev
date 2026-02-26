"""
Degradation analysis using expected-energy reference (no fixed 2020 baseline)
=============================================================================
Implements: ideal reference = (XGBoost expected-from-Solcast trained on simulation)
+ per-meter calibration; degradation = trend of monthly median (actual / expected_cal)
using only analysis_valid (valid daylight, no outages).

Steps:
  1) E_expected(t): XGBoost(Solcast + time -> simulated grid_power) per meter; predict for full period.
  2) Per-meter calibration on clean periods: E_expected_cal = a*E_expected + b (fit on analysis_valid).
  3) PI(t) = actual(t) / (E_expected_cal(t) + eps) only where analysis_valid.
  4) Optional: drop days with outage rate > 10%.
  5) Monthly median PI per meter -> robust trend (slope) -> degradation %/year.
  6) QC: exclude or warn for meters with low valid coverage.

Uses: SolarMeterReadings1hour_cleaned_v2.csv (outage_flag, analysis_valid),
      Solcast (Bundoora), and simulation CSVs (solarsitesimulation_2020/2021 or solarsitesimulation.csv).
"""

import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "Results_seasonality")
SIMULATED_DIR = os.path.join(BASE, "data", "simulated")

# Inputs
METER_V2_CSV = "SolarMeterReadings1hour_cleaned_v2.csv"
SOLCAST_CSV = "solcast_df_cleaned_v2.csv"
SOLCAST_FALLBACK = "solcast_df_cleaned.csv"
SIMULATION_GLOB = ["solarsitesimulation_2020.csv", "solarsitesimulation_2021.csv", "solarsitesimulation.csv"]
SIMULATED_POWER_CSV = "simulated_power_2020_2025.csv"  # optional precomputed E_expected

BUNDOORA_METER_PREFIX = "solar.bun_"
SOLCAST_CAMPUS = "BUNDOORA"

# Daylight for calibration (match data_cleaning v2)
ZENITH_DAYLIGHT = 90
GHI_MIN_DAYLIGHT = 50.0
PI_EPS = 1e-9
OUTAGE_DAY_RATE_THRESHOLD = 0.10  # drop day if >10% of valid_daylight hours are outage
MIN_VALID_COVERAGE_PCT = 15.0     # exclude meter from degradation if valid coverage < this
MIN_MONTHS_FOR_TREND = 6

# XGBoost for E_expected
WEATHER_FEATURES = ["ghi", "dni", "dhi", "zenith", "air_temp", "cloud_opacity"]
TIME_FEATURES = ["hour", "day_of_year", "month", "weekday"]
XGB_PARAMS = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_meter_v2(path=None):
    """Load V2 cleaned meter data (analysis_valid, outage_flag)."""
    path = path or os.path.join(DATA_DIR, METER_V2_CSV)
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[df["meter"].astype(str).str.strip().str.lower().str.contains("bun_", na=False)].copy()
    return df


def load_solcast_hourly(path=None):
    """Load Solcast, optionally filter BUNDOORA, resample to 1h."""
    for name in [SOLCAST_CSV, SOLCAST_FALLBACK]:
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
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("h")
    if len(df) > 1:
        delta = df["timestamp"].diff().dropna()
        if delta.min() < pd.Timedelta("45min"):
            df = df.set_index("timestamp").resample("1h").mean().reset_index()
    return df


def load_simulation_bundoora():
    """Load simulation CSVs (2020, 2021 or single); keep Bundoora meters only."""
    dfs = []
    for name in SIMULATION_GLOB:
        for base in [DATA_DIR, SIMULATED_DIR]:
            p = os.path.join(base, name)
            if os.path.isfile(p):
                d = pd.read_csv(p)
                d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
                d = d.dropna(subset=["timestamp"])
                meter_col = "meter" if "meter" in d.columns else [c for c in d.columns if "meter" in c.lower()][0]
                d = d[d[meter_col].astype(str).str.strip().str.lower().str.contains("bun_", na=False)].copy()
                if "meter" not in d.columns and meter_col != "meter":
                    d = d.rename(columns={meter_col: "meter"})
                dfs.append(d)
                break
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["timestamp", "meter"], keep="first")


def add_time_features(df, ts_col="timestamp"):
    t = pd.to_datetime(df[ts_col])
    df = df.copy()
    df["hour"] = t.dt.hour
    df["day_of_year"] = t.dt.dayofyear
    df["month"] = t.dt.month
    df["weekday"] = t.dt.weekday
    return df


def build_expected_one_meter(sim_meter, solcast_hourly, full_range, feature_cols):
    """Train XGBoost(solcast+time -> grid_power) on simulation; predict for full_range."""
    sim_meter = sim_meter.copy()
    sim_meter["timestamp"] = pd.to_datetime(sim_meter["timestamp"]).dt.floor("h")
    sim_meter = sim_meter.drop_duplicates(subset=["timestamp"], keep="first")
    merged = sim_meter.merge(solcast_hourly, on="timestamp", how="inner")
    merged = add_time_features(merged)
    use = [c for c in feature_cols if c in merged.columns]
    if "grid_power" not in merged.columns or len(use) < 2:
        return np.zeros(len(full_range))
    train = merged.dropna(subset=["grid_power"] + use)
    if len(train) < 100:
        return np.zeros(len(full_range))
    X = train[use]
    y = train["grid_power"]
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)
    pred_df = pd.DataFrame({"timestamp": full_range}).merge(solcast_hourly, on="timestamp", how="left")
    pred_df = pred_df.sort_values("timestamp").set_index("timestamp").ffill().bfill().reset_index()
    pred_df = add_time_features(pred_df)
    for c in use:
        if c not in pred_df.columns:
            pred_df[c] = 0.0
    X_pred = pred_df[[c for c in use if c in pred_df.columns]]
    pred = model.predict(X_pred)
    return np.maximum(0.0, pred)


def load_or_build_expected(meter_df, solcast_hourly, sim_df, full_range):
    """
    If simulated_power_2020_2025.csv exists and has same meters, use it as E_expected;
    else build E_expected per meter via XGBoost(simulation + Solcast).
    Returns DataFrame with columns timestamp, meter, E_expected.
    """
    for base in [SIMULATED_DIR, DATA_DIR]:
        p = os.path.join(base, SIMULATED_POWER_CSV)
        if os.path.isfile(p):
            pre = pd.read_csv(p)
            pre["timestamp"] = pd.to_datetime(pre["timestamp"], errors="coerce")
            pre = pre.dropna(subset=["timestamp"])
            if "simulated_power" in pre.columns:
                pre = pre.rename(columns={"simulated_power": "E_expected"})
            if "E_expected" not in pre.columns:
                pre = pre.rename(columns={pre.columns[-1]: "E_expected"})
            meters_in_pre = set(pre["meter"].unique())
            meters_needed = set(meter_df["meter"].unique())
            if meters_needed.issubset(meters_in_pre):
                return pre[["timestamp", "meter", "E_expected"]].copy()
    # Build from simulation + Solcast
    feature_cols = [c for c in WEATHER_FEATURES + TIME_FEATURES if c in solcast_hourly.columns]
    if not feature_cols:
        feature_cols = [c for c in solcast_hourly.columns if c != "timestamp" and np.issubdtype(solcast_hourly[c].dtype, np.number)]
    solcast_hourly = add_time_features(solcast_hourly)
    out = []
    for meter_id in meter_df["meter"].unique():
        sim_m = sim_df[sim_df["meter"] == meter_id].copy() if len(sim_df) else pd.DataFrame()
        if sim_m.empty:
            exp = np.zeros(len(full_range))
        else:
            exp = build_expected_one_meter(sim_m, solcast_hourly, full_range, feature_cols)
        for i, ts in enumerate(full_range):
            out.append({"timestamp": ts, "meter": meter_id, "E_expected": float(exp[i])})
    return pd.DataFrame(out)


def calibrate_expected(actual, E_expected, mask):
    """Fit actual = a * E_expected + b on mask; return (a, b), E_expected_cal for all."""
    x = E_expected[mask]
    y = actual[mask]
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if valid.sum() < 30:
        return 1.0, 0.0
    slope, intercept, _, _, _ = stats.linregress(x[valid], y[valid])
    E_cal = slope * E_expected + intercept
    return slope, intercept


def run():
    print("=" * 70)
    print("DEGRADATION VIA EXPECTED-REFERENCE (no 2020 baseline)")
    print("=" * 70)

    print("\n[1] Loading meter V2 and Solcast...")
    meter_df = load_meter_v2()
    solcast = load_solcast_hourly()
    solcast_hr = solcast.copy()
    solcast_hr = add_time_features(solcast_hr)
    meters = meter_df["meter"].unique().tolist()
    print(f"    Meters: {len(meters)}")
    print(f"    Solcast: {solcast['timestamp'].min()} to {solcast['timestamp'].max()}")

    ts_min = max(meter_df["timestamp"].min(), solcast["timestamp"].min())
    ts_max = min(meter_df["timestamp"].max(), solcast["timestamp"].max())
    full_range = pd.date_range(start=ts_min, end=ts_max, freq="1h")
    print(f"    Common range: {ts_min} to {ts_max} ({len(full_range)} hours)")

    print("\n[2] Loading simulation and building E_expected...")
    sim_df = load_simulation_bundoora()
    expected_df = load_or_build_expected(meter_df, solcast_hr, sim_df, full_range)
    print(f"    E_expected: {len(expected_df)} rows")

    # Merge E_expected and weather (for valid_daylight) into meter
    meter_merged = meter_df.merge(
        expected_df,
        on=["timestamp", "meter"],
        how="left",
    )
    weather_cols = ["timestamp", "ghi", "zenith"] if "ghi" in solcast.columns and "zenith" in solcast.columns else ["timestamp"]
    solcast_weather = solcast[weather_cols].copy()
    solcast_weather = solcast_weather.rename(columns={"timestamp": "ts"})
    meter_merged = meter_merged.merge(solcast_weather, left_on="timestamp", right_on="ts", how="left")
    if "zenith" in meter_merged.columns and "ghi" in meter_merged.columns:
        meter_merged["valid_daylight"] = (
            (meter_merged["zenith"] < ZENITH_DAYLIGHT) & (meter_merged["ghi"] > GHI_MIN_DAYLIGHT)
        )
    else:
        meter_merged["valid_daylight"] = meter_merged["analysis_valid"].copy()
    meter_merged["date"] = pd.to_datetime(meter_merged["timestamp"]).dt.date

    # Per-meter: calibration, PI, optional drop high-outage days
    results = []
    monthly_pi_list = []

    for meter_id in meters:
        m = meter_merged[meter_merged["meter"] == meter_id].copy()
        if m.empty:
            continue
        analysis_valid = m["analysis_valid"].values.astype(bool)
        actual = m["meter_reading"].values.astype(float)
        E_exp = m["E_expected"].values.astype(float)
        E_exp = np.where(np.isfinite(E_exp), E_exp, 0.0)

        # Optional: drop days with outage rate > 10%
        if "valid_daylight" in m.columns:
            m["valid_daylight"] = m["valid_daylight"].values
            daily_valid = m.groupby("date")["valid_daylight"].sum()
            daily_outage = m.groupby("date")["outage_flag"].sum()
            daily_rate = daily_outage / daily_valid.replace(0, np.nan)
            bad_days = set(daily_rate[daily_rate > OUTAGE_DAY_RATE_THRESHOLD].index)
            m["bad_day"] = m["date"].isin(bad_days)
            analysis_valid = analysis_valid & (~m["bad_day"].values)
        cal_mask = analysis_valid & (E_exp > 0)
        a, b = calibrate_expected(actual, E_exp, cal_mask)
        E_cal = a * E_exp + b
        E_cal = np.maximum(E_cal, 0.0)
        pi = np.full(len(m), np.nan)
        use = analysis_valid & (E_cal + PI_EPS > 0)
        pi[use] = actual[use] / (E_cal[use] + PI_EPS)

        m["E_expected_cal"] = E_cal
        m["PI"] = pi
        m["analysis_valid_pi"] = use

        total_daylight = m["valid_daylight"].sum() if "valid_daylight" in m.columns else use.sum()
        valid_pct = 100.0 * use.sum() / max(1, int(total_daylight))

        # Monthly median PI
        m["year_month"] = pd.to_datetime(m["timestamp"]).dt.to_period("M")
        pi_valid = m.loc[m["analysis_valid_pi"], ["year_month", "PI"]]
        if pi_valid.empty:
            results.append({
                "meter": meter_id,
                "degradation_pct_per_year": np.nan,
                "valid_coverage_pct": valid_pct,
                "n_months": 0,
                "excluded": "no_valid_PI",
            })
            continue
        monthly = pi_valid.groupby("year_month")["PI"].median().reset_index()
        monthly["meter"] = meter_id
        monthly["month_ts"] = monthly["year_month"].dt.to_timestamp()
        monthly_pi_list.append(monthly)

        if len(monthly) < MIN_MONTHS_FOR_TREND or valid_pct < MIN_VALID_COVERAGE_PCT:
            results.append({
                "meter": meter_id,
                "degradation_pct_per_year": np.nan,
                "valid_coverage_pct": round(valid_pct, 2),
                "n_months": len(monthly),
                "excluded": "low_coverage" if valid_pct < MIN_VALID_COVERAGE_PCT else "few_months",
            })
            continue

        # Linear trend: median PI vs months since start
        x = (monthly["month_ts"] - monthly["month_ts"].min()).dt.days / 365.25
        y = monthly["PI"].values
        slope, intercept, r, p, se = stats.linregress(x, y)
        # Degradation %/year: slope of PI per year -> percentage of median PI per year
        median_pi = np.nanmedian(y)
        if median_pi > 0:
            deg_pct_per_year = 100.0 * slope / median_pi
        else:
            deg_pct_per_year = np.nan
        results.append({
            "meter": meter_id,
            "degradation_pct_per_year": round(deg_pct_per_year, 4),
            "trend_slope_PI_per_year": round(slope, 6),
            "median_PI": round(median_pi, 4),
            "valid_coverage_pct": round(valid_pct, 2),
            "n_months": len(monthly),
            "excluded": "",
        })

    deg_df = pd.DataFrame(results)
    monthly_pi_df = pd.concat(monthly_pi_list, ignore_index=True) if monthly_pi_list else pd.DataFrame()

    # Save
    deg_path = os.path.join(RESULTS_DIR, "degradation_expected_baseline.csv")
    deg_df.to_csv(deg_path, index=False)
    print(f"\n[3] Saved: {deg_path}")

    if not monthly_pi_df.empty:
        mp_path = os.path.join(RESULTS_DIR, "monthly_median_PI_expected_baseline.csv")
        monthly_pi_df.to_csv(mp_path, index=False)
        print(f"    {mp_path}")

    # Summary
    report = deg_df[deg_df["excluded"] == ""]
    print("\n--- Degradation (expected-reference method) ---")
    if report.empty:
        print("No meters with sufficient valid coverage for trend.")
    else:
        print(report[["meter", "degradation_pct_per_year", "valid_coverage_pct", "n_months"]].to_string(index=False))

    # Pipeline checklist for thesis
    checklist_path = os.path.join(RESULTS_DIR, "degradation_expected_baseline_checklist.txt")
    with open(checklist_path, "w") as f:
        f.write("Pipeline checklist (expected-reference degradation)\n")
        f.write("Inputs: SolarMeterReadings1hour_cleaned_v2.csv, Solcast (Bundoora), simulation CSVs\n")
        f.write("1) E_expected(t) = XGBoost(Solcast+time -> sim grid_power) per meter, or load simulated_power_2020_2025.csv\n")
        f.write("2) Per-meter calibration: E_expected_cal = a*E_expected + b on analysis_valid only\n")
        f.write("3) PI(t) = actual / (E_expected_cal + eps) on analysis_valid; optional: drop days with outage rate > 10%\n")
        f.write("4) Monthly median PI -> linear trend -> slope = degradation %/year\n")
        f.write("5) QC: exclude meters with valid coverage < {}%\n".format(MIN_VALID_COVERAGE_PCT))
    print(f"    Checklist: {checklist_path}")

    # Optional QC plot: monthly median PI over time
    try:
        import matplotlib.pyplot as plt
        if not monthly_pi_df.empty and "month_ts" in monthly_pi_df.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            for meter_id in monthly_pi_df["meter"].unique():
                sub = monthly_pi_df[monthly_pi_df["meter"] == meter_id]
                ax.plot(sub["month_ts"], sub["PI"], label=meter_id.split("#")[0].replace("solar.", ""), alpha=0.8)
            ax.set_xlabel("Month")
            ax.set_ylabel("Monthly median PI")
            ax.set_title("Performance index (actual / expected_cal) — expected-reference method")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(RESULTS_DIR, "monthly_median_PI_expected_baseline.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"    Plot: {plot_path}")
    except Exception as e:
        print(f"    (QC plot skipped: {e})")

    print("\nDone.")


if __name__ == "__main__":
    run()
