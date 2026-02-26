"""
Solar meter and Solcast data cleaning (Bundoora hourly solar)
=============================================================
- Solcast alignment: try -3..+3h shift and pick best (max non-NaN GHI on merge).
- Nighttime zeroing: zenith >= 90 (or fallback hours 21-04) → force reading to 0.
- Stuck-meter detection: identical non-zero value repeating >= STUCK_MIN_HOURS → NaN.
- Daylight: zenith < 90 from Solcast; fallback hour-window. Missing-as-zero: zenith < 90
  or GHI > 20 when zenith missing; value <= ZERO_EPS → flag.
- Consecutive daylight-zero blocks (>= 3h): forced to NaN then filled by XGBoost.
- Imputation: <=2h time interp; 3-24h same-hour median; long gaps → XGBoost.
- Post-imputation: enforce night=0 again (XGBoost can predict non-zero at night).
"""

import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

# Paths (run from project root or seasonality/)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "Results_seasonality")

# Input / output file names
METER_CSV = "SolarMeterReadings1hour_2020_2025.csv"
SOLCAST_CSV = "solcast_df_2020_2025.csv"
CLEANED_METER_CSV = "SolarMeterReadings1hour_cleaned_2020_2025.csv"
CLEANED_SOLCAST_CSV = "solcast_df_cleaned_2020_2025.csv"
OUTAGE_BLOCKS_CSV = "outage_blocks_report_2020_2025.csv"

# Daylight (prefer zenith; use zenith < 90 for missing-as-zero to catch shoulder hours)
DAYLIGHT_HOURS = (6, 18)  # fallback hour window
ZENITH_DAYLIGHT = 90      # zenith < this = daylight
ZENITH_SUN_UP = 90        # zenith < 90 = sun above horizon for missing-as-zero
GHI_DAYLIGHT_THRESHOLD = 20.0
GHI_STRONG_DAYLIGHT = 50.0
ZERO_EPS = 1e-6
OUTAGE_MIN_HOURS = 3
INTERP_LIMIT = 2

# Nighttime: definite night when zenith >= 90; fallback hours for when zenith is missing
NIGHT_FALLBACK_HOURS = (21, 4)  # hours 21,22,23,0,1,2,3,4 = definite night in Victoria AU

# Stuck-meter detection: identical non-zero value repeating >= this many consecutive hours
STUCK_MIN_HOURS = 8

# Gaps
SHORT_GAP_MAX = 2
MEDIUM_GAP_MAX = 24
NEARBY_DAYS = 7
NEARBY_DAYS_MAX = 30
MIN_DONORS = 5

# Solcast time alignment (meter vs weather may differ by 0, ±1, ±2, ±3h)
SOLCAST_SHIFT_RANGE = (-3, 4)

# Outlier / cap (from clean daylight GHI>50)
HIGH_PERCENTILE_CAP = 99.5
ROBUST_CAP_QUANTILE = 99.99
IQR_OUTLIER_MULTIPLIER = 3.0

# --- V2: Flag-based (no imputation); degradation only on analysis_valid ---
DATA_VERSION_V2 = "3.0"
CLEANED_METER_V2_CSV = "SolarMeterReadings1hour_cleaned_2020_2025.csv"
CLEANED_SOLCAST_V2_CSV = "solcast_df_cleaned_2020_2025.csv"
QC_REPORT_V2_CSV = "data_cleaning_qc_report_2020_2025.csv"
OUTAGE_DAY_RATE_THRESHOLD = 0.10
VALID_DAYLIGHT_GHI = 50.0


def load_meter_data(path=None):
    """Load meter CSV; standardise column names."""
    path = path or os.path.join(DATA_DIR, METER_CSV)
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    # Normalise column names
    if "meter_reading" not in df.columns:
        cand = [c for c in df.columns if "reading" in c.lower() or "power" in c.lower()]
        if cand:
            df = df.rename(columns={cand[0]: "meter_reading"})
    meter_col = "meter" if "meter" in df.columns else [c for c in df.columns if "meter" in c.lower()][0]
    if meter_col != "meter":
        df = df.rename(columns={meter_col: "meter"})
    return df


def load_solcast_data(path=None, prefer_cleaned=True):
    """Load Solcast CSV (prefer solcast_df_cleaned.csv for robust zenith/GHI)."""
    if path is None:
        path = os.path.join(DATA_DIR, CLEANED_SOLCAST_CSV) if prefer_cleaned else os.path.join(DATA_DIR, SOLCAST_CSV)
        if prefer_cleaned and not os.path.isfile(path):
            path = os.path.join(DATA_DIR, SOLCAST_CSV)
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df


def resample_solcast_to_hourly(solcast_df):
    """Resample Solcast (30-min) to hourly (mean) for alignment with meter."""
    if "ghi" not in solcast_df.columns:
        raise ValueError("Solcast must contain 'ghi' column")
    sol = solcast_df.set_index("timestamp").sort_index()
    hourly = sol.resample("1h").mean().reset_index()
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"]).dt.floor("h")
    return hourly


def find_best_solcast_shift(meter_df, solcast_hourly, shift_range=None):
    """
    Try shifting Solcast by -3..+3 hours; pick shift that maximizes fraction of meter timestamps
    that get non-NaN GHI on merge (best alignment). Returns shift in hours (0 = no shift).
    """
    if shift_range is None:
        shift_range = range(SOLCAST_SHIFT_RANGE[0], SOLCAST_SHIFT_RANGE[1])
    meter_ts = pd.to_datetime(meter_df["timestamp"]).dt.floor("h").unique()
    meter_ts = pd.DatetimeIndex(meter_ts)
    best_shift = 0
    best_frac = 0.0
    for shift_h in shift_range:
        sc = solcast_hourly.copy()
        sc["timestamp"] = pd.to_datetime(sc["timestamp"]) + pd.Timedelta(hours=int(shift_h))
        sc["timestamp"] = sc["timestamp"].dt.floor("h")
        merged = pd.merge(
            pd.DataFrame({"timestamp": meter_ts}),
            sc[["timestamp", "ghi"]].drop_duplicates("timestamp"),
            on="timestamp",
            how="left",
        )
        frac = merged["ghi"].notna().mean()
        if frac > best_frac:
            best_frac = frac
            best_shift = shift_h
    return best_shift


def get_daylight_mask(zenith, ghi, timestamps, fallback_hours=DAYLIGHT_HOURS, zenith_daylight=ZENITH_DAYLIGHT, ghi_threshold=GHI_DAYLIGHT_THRESHOLD):
    """
    Daylight: prefer zenith < 90 from Solcast; else (GHI > threshold) OR fallback hour.
    Fallback never disabled by low/NaN GHI. Returns boolean array.
    """
    ts = pd.Series(pd.to_datetime(timestamps))
    hour = ts.dt.hour.values
    fallback = (hour >= fallback_hours[0]) & (hour <= fallback_hours[1])
    has_zenith = ~np.isnan(zenith)
    daylight = np.where(
        has_zenith,
        zenith < zenith_daylight,
        np.where(np.isnan(ghi), fallback, (ghi > ghi_threshold) | fallback),
    )
    return np.asarray(daylight, dtype=bool)


def sun_up_mask(zenith, ghi, timestamps, fallback_hours=DAYLIGHT_HOURS, zenith_sun_up=ZENITH_SUN_UP, ghi_strong=GHI_STRONG_DAYLIGHT):
    """
    Sun is up: zenith < 85 or GHI > 50, or fallback when both NaN.
    Used for missing-as-zero so we flag zeros even when GHI is NaN (zenith or fallback still apply).
    """
    ts = pd.Series(pd.to_datetime(timestamps))
    hour = ts.dt.hour.values
    fallback = (hour >= fallback_hours[0]) & (hour <= fallback_hours[1])
    has_zenith = ~np.isnan(zenith)
    has_ghi = ~np.isnan(ghi)
    sun_up = np.where(
        has_zenith,
        zenith < zenith_sun_up,
        np.where(has_ghi, ghi > ghi_strong, fallback),
    )
    return np.asarray(sun_up, dtype=bool)


def flag_missing_as_zero(vals, sun_up, eps=ZERO_EPS):
    """
    Flag as missing-as-zero when sun is up (zenith < 90 or GHI > 50 or fallback) AND value <= eps.
    Does not require GHI when zenith says sun is up (avoids ignoring NaN GHI). Returns boolean mask.
    """
    is_zero = (vals.values <= eps) if hasattr(vals, "values") else (np.asarray(vals) <= eps)
    return sun_up & is_zero


def definite_night_mask(zenith, timestamps, fallback_night=NIGHT_FALLBACK_HOURS):
    """
    True when the sun is definitely down: zenith >= 90 from Solcast, or fallback to
    hours 21-04 (Victoria AU) when zenith is missing. Used to force readings to zero.
    """
    zenith = np.asarray(zenith, dtype=float)
    ts = pd.Series(pd.to_datetime(timestamps))
    hour = ts.dt.hour.values
    has_zenith = ~np.isnan(zenith)
    lo, hi = fallback_night
    if lo > hi:
        fallback = (hour >= lo) | (hour <= hi)
    else:
        fallback = (hour >= lo) & (hour <= hi)
    return np.where(has_zenith, zenith >= 90, fallback).astype(bool)


def detect_stuck_meter(vals, min_run=STUCK_MIN_HOURS):
    """
    Detect runs of identical non-zero values lasting >= min_run consecutive hours.
    Returns boolean mask (True = stuck/faulty) and list of (start_idx, length) blocks.
    """
    v = np.asarray(vals, dtype=float)
    n = len(v)
    stuck = np.zeros(n, dtype=bool)
    blocks = []
    i = 0
    while i < n:
        if v[i] != 0 and not np.isnan(v[i]):
            start = i
            while i < n and v[i] == v[start] and not np.isnan(v[i]):
                i += 1
            run_len = i - start
            if run_len >= min_run:
                stuck[start:i] = True
                blocks.append((start, run_len))
        else:
            i += 1
    return stuck, blocks


def detect_outage_blocks(is_missing, daylight, timestamps, min_hours=OUTAGE_MIN_HOURS):
    """
    Consecutive daylight missing-as-zero blocks with length >= min_hours.
    Returns list of dicts: {start_idx, end_idx, start_ts, end_ts, length_hours}.
    """
    ts = pd.to_datetime(timestamps)
    n = len(is_missing)
    blocks = []
    i = 0
    while i < n:
        if is_missing[i] and daylight[i]:
            start = i
            while i < n and is_missing[i] and daylight[i]:
                i += 1
            length = i - start
            if length >= min_hours:
                blocks.append({
                    "start_idx": start,
                    "end_idx": i - 1,
                    "start_ts": ts.iloc[start] if hasattr(ts, "iloc") else ts[start],
                    "end_ts": ts.iloc[i - 1] if hasattr(ts, "iloc") else ts[i - 1],
                    "length_hours": length,
                })
        else:
            i += 1
    return blocks


def detect_high_outliers(series, iqr_multiplier=IQR_OUTLIER_MULTIPLIER, percentile_cap=HIGH_PERCENTILE_CAP):
    """
    Identify unreasonably large values. Returns mask of outliers and suggested cap value.
    We use percentile cap: values above percentile_cap are considered outliers; cap at that percentile.
    """
    clean = series.dropna()
    if len(clean) < 10:
        return np.zeros(len(series), dtype=bool), None
    q75 = clean.quantile(0.75)
    q25 = clean.quantile(0.25)
    iqr = q75 - q25
    if iqr <= 0:
        iqr = clean.std() or 1e-6
    upper_iqr = q75 + iqr_multiplier * iqr
    cap_value = clean.quantile(percentile_cap / 100.0)
    # Outlier = above IQR fence OR above percentile cap (use the stricter: cap_value)
    upper = min(upper_iqr, cap_value) if not np.isnan(cap_value) else upper_iqr
    mask = (series.values > upper) & (series.notna().values)
    return mask, cap_value


def apply_outlier_cap(df, value_col="meter_reading", cap_value=None, percentile=99.5):
    """Replace values above cap with cap; if cap_value is None, use percentile of non-NaN series."""
    out = df.copy()
    if cap_value is None:
        cap_value = out[value_col].quantile(percentile / 100.0)
    out.loc[out[value_col] > cap_value, value_col] = cap_value
    return out


def classify_gaps(is_missing):
    """
    Given boolean array (True = missing), return list of (start_idx, length) for each gap.
    """
    gaps = []
    i = 0
    n = len(is_missing)
    while i < n:
        if is_missing[i]:
            start = i
            while i < n and is_missing[i]:
                i += 1
            gaps.append((start, i - start))
        else:
            i += 1
    return gaps


def impute_short_gaps_time(series, timestamps, gap_ranges):
    """Fill gaps of length 1–2 with time interpolation (method='time', limit=2)."""
    idx = pd.DatetimeIndex(pd.to_datetime(timestamps))
    s = pd.Series(series.values.copy(), index=idx)
    s = s.sort_index()
    s = s.interpolate(method="time", limit=SHORT_GAP_MAX, limit_direction="both")
    s = s.reindex(idx)  # back to original order
    series.iloc[:] = s.values
    return series


def impute_medium_gaps_similar_time(series, timestamps, gap_ranges, ghi, nearby_days=NEARBY_DAYS, nearby_days_max=NEARBY_DAYS_MAX, min_donors=MIN_DONORS, eps=ZERO_EPS, ghi_min=GHI_STRONG_DAYLIGHT):
    """
    Fill gaps of length 3–24h with median of same hour on nearby days.
    Donor pool: value > eps and GHI > ghi_min. If too few donors, expand window up to nearby_days_max.
    """
    ts = pd.to_datetime(timestamps)
    ghi = np.asarray(ghi) if ghi is not None else np.full(len(ts), np.nan)
    df = pd.DataFrame({"value": series.values, "ts": ts, "ghi": ghi})
    df["hour"] = df["ts"].dt.hour
    df["date"] = df["ts"].dt.normalize()
    donor_ok = (df["value"] > eps) & (df["ghi"] > ghi_min)
    df["donor"] = donor_ok

    for start, length in gap_ranges:
        if length < 3 or length > MEDIUM_GAP_MAX:
            continue
        end = start + length
        gap_dates = set(df.iloc[start:end]["date"].unique())
        for window_days in [nearby_days, (nearby_days + nearby_days_max) // 2, nearby_days_max]:
            if gap_dates:
                min_gap = min(gap_dates)
                max_gap = max(gap_dates)
                window_start = min_gap - pd.Timedelta(days=window_days)
                window_end = max_gap + pd.Timedelta(days=window_days)
                mask = (
                    (df["date"] >= window_start)
                    & (df["date"] <= window_end)
                    & (~df["date"].isin(gap_dates))
                    & df["donor"]
                )
            else:
                mask = df["donor"].copy()
            donor_df = df.loc[mask]
            medians_by_hour = donor_df.groupby("hour")["value"].median().to_dict()
            # Check if we have enough donors per hour for this gap
            filled = 0
            for i in range(start, end):
                h = df.iloc[i]["hour"]
                med = medians_by_hour.get(h)
                if med is not None and not np.isnan(med) and med > eps:
                    series.iloc[i] = med
                    filled += 1
            donor_count = donor_df.groupby("hour").size()
            min_hour_donors = donor_count.min() if len(donor_count) else 0
            if filled == length or min_hour_donors >= min_donors or window_days >= nearby_days_max:
                break
    return series


def _build_long_gap_features(solcast_hourly, ts_ser):
    """Build feature df: ghi, dni, dhi, zenith, air_temp, cloud_opacity + hour_sin/cos + doy_sin/cos."""
    sc = solcast_hourly.copy()
    sc["timestamp"] = pd.to_datetime(sc["timestamp"]).dt.floor("h")
    cols = ["timestamp", "ghi", "dni", "dhi", "zenith", "air_temp", "cloud_opacity"]
    avail = [c for c in cols if c in sc.columns]
    if "ghi" not in avail:
        return None, []
    sc = sc[avail].groupby("timestamp").first().reset_index()
    ts_ser = pd.to_datetime(ts_ser).dt.floor("h")
    df = ts_ser.to_frame(name="ts")
    df = df.merge(sc.rename(columns={"timestamp": "ts"}), on="ts", how="left")
    df["hour"] = ts_ser.dt.hour
    df["doy"] = ts_ser.dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    feat = ["ghi", "dni", "dhi", "zenith", "air_temp", "cloud_opacity", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    use = [f for f in feat if f in df.columns]
    for f in use:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    return df, use


def impute_long_gaps_model(series, timestamps, gap_ranges, solcast_hourly, max_gap_hours=24):
    """
    Fill gaps using XGBoost (weather + time features). When max_gap_hours=0, fill ALL ranges
    (used for outage blocks 3–24h). Otherwise only ranges with length > max_gap_hours.
    """
    ts = pd.to_datetime(timestamps)
    if solcast_hourly is None or len(solcast_hourly) == 0:
        return series
    feat_df, X_cols = _build_long_gap_features(solcast_hourly, ts)
    if feat_df is None or not X_cols:
        return series
    feat_df["value"] = series.values
    clean = feat_df.dropna(subset=["value"] + X_cols)
    clean = clean[clean["value"] > ZERO_EPS]
    if len(clean) < 50:
        return series
    X_train = clean[X_cols].fillna(0)
    y_train = clean["value"]
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    for start, length in gap_ranges:
        if max_gap_hours > 0 and length <= max_gap_hours:
            continue
        end = start + length
        pred_df = feat_df.iloc[start:end].copy()
        pred_df = pred_df[X_cols].fillna(0)
        if pred_df.empty:
            continue
        pred = model.predict(pred_df)
        for ii in range(length):
            series.iloc[start + ii] = max(0.0, pred[ii])
    return series


def clean_one_meter(meter_df, meter_id, solcast_hourly, options=None):
    """
    Clean one meter:
    0) Night zeroing (zenith >= 90 or fallback hours 21-04)
    0b) Stuck-meter detection (identical value >= 8h → NaN)
    1) Missing-as-zero during daylight
    2) Outage blocks → NaN
    3) Outlier cap
    4-5) Gap imputation (short/medium/long)
    6) Post-imputation night enforcement (force night = 0 again)
    7) Non-negative + robust cap
    """
    options = options or {}
    daylight_hours = options.get("daylight_hours", DAYLIGHT_HOURS)
    ghi_threshold = options.get("ghi_threshold", GHI_DAYLIGHT_THRESHOLD)
    ghi_strong = options.get("ghi_strong_daylight", GHI_STRONG_DAYLIGHT)
    eps = options.get("zero_eps", ZERO_EPS)
    short_max = options.get("short_gap_max", SHORT_GAP_MAX)
    medium_max = options.get("medium_gap_max", MEDIUM_GAP_MAX)
    nearby_days = options.get("nearby_days", NEARBY_DAYS)
    iqr_mult = options.get("iqr_outlier_multiplier", IQR_OUTLIER_MULTIPLIER)
    perc_cap = options.get("high_percentile_cap", HIGH_PERCENTILE_CAP)
    cap_quantile = options.get("robust_cap_quantile", ROBUST_CAP_QUANTILE)

    df = meter_df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"]
    vals = df["meter_reading"].copy()

    # Merge Solcast so we have ghi, zenith per row
    sc = solcast_hourly.copy()
    sc["timestamp"] = pd.to_datetime(sc["timestamp"]).dt.floor("h")
    merge_cols = ["timestamp"] + [c for c in ["ghi", "zenith", "dni", "dhi", "air_temp", "cloud_opacity"] if c in sc.columns]
    sc = sc[merge_cols].groupby("timestamp").first().reset_index()
    df_work = df.merge(sc, on="timestamp", how="left")
    ghi = df_work["ghi"].values if "ghi" in df_work.columns else np.full(len(df), np.nan)
    zenith = df_work["zenith"].values if "zenith" in df_work.columns else np.full(len(df), np.nan)

    # --- 0) Force nighttime readings to zero ---
    is_night = definite_night_mask(zenith, ts)
    night_nonzero_count = int((is_night & (vals > 0)).sum())
    vals.loc[is_night] = 0.0

    # --- 0b) Stuck-meter detection: identical non-zero values repeating >= 8h → NaN ---
    stuck_mask, stuck_blocks = detect_stuck_meter(vals.values, min_run=STUCK_MIN_HOURS)
    stuck_count = int(stuck_mask.sum())
    vals.loc[stuck_mask] = np.nan

    # Daylight and sun-up masks
    daylight = get_daylight_mask(zenith, ghi, ts, daylight_hours, ZENITH_DAYLIGHT, ghi_threshold)
    sun_up = sun_up_mask(zenith, ghi, ts, daylight_hours, ZENITH_SUN_UP, ghi_strong)

    # 1) Missing-as-zero: sun is up AND value <= eps
    missing_as_zero = flag_missing_as_zero(vals, sun_up, eps)
    vals.loc[missing_as_zero] = np.nan

    # 2) Consecutive daylight-zero blocks (>= 3h): force to NaN before imputation
    is_missing = vals.isna()
    outage_blocks = detect_outage_blocks(is_missing.values, daylight, ts, OUTAGE_MIN_HOURS)
    for blk in outage_blocks:
        for i in range(blk["start_idx"], blk["end_idx"] + 1):
            vals.iloc[i] = np.nan
    is_missing = vals.isna()

    # 3) High outliers: cap at percentile
    outlier_mask, cap_val = detect_high_outliers(vals, iqr_mult, perc_cap)
    if outlier_mask.any() and cap_val is not None:
        vals.loc[outlier_mask] = cap_val

    # 4) Gap classification
    gaps = classify_gaps(is_missing.values)
    short = [(s, L) for s, L in gaps if 1 <= L <= short_max]
    medium = [(s, L) for s, L in gaps if short_max < L <= medium_max]
    long_gaps = [(s, L) for s, L in gaps if L > medium_max]

    outage_ranges = [(blk["start_idx"], blk["length_hours"]) for blk in outage_blocks]
    def overlaps_outage(gap_start, gap_len):
        gap_end = gap_start + gap_len
        for blk in outage_blocks:
            blk_end = blk["end_idx"] + 1
            if gap_start < blk_end and blk["start_idx"] < gap_end:
                return True
        return False
    medium_no_outage = [(s, L) for s, L in medium if not overlaps_outage(s, L)]
    model_ranges = long_gaps + outage_ranges

    # 5) Impute: short → time interp, medium → median, long/outage → XGBoost
    impute_short_gaps_time(vals, ts, short)
    impute_medium_gaps_similar_time(vals, ts, medium_no_outage, ghi, nearby_days, eps=eps, ghi_min=ghi_strong)
    impute_long_gaps_model(vals, ts, model_ranges, solcast_hourly, max_gap_hours=0)

    # Remaining NaN: time interpolation limited to <=2h
    idx = pd.DatetimeIndex(pd.to_datetime(ts))
    s = pd.Series(vals.values, index=idx).sort_index()
    s = s.interpolate(method="time", limit=INTERP_LIMIT, limit_direction="both")
    s = s.reindex(idx)
    vals = pd.Series(s.values, index=vals.index)

    # 6) Post-imputation: non-negative + enforce night=0 again
    vals = vals.clip(lower=0.0)
    vals.loc[is_night] = 0.0

    # 7) Robust cap: 99.99% quantile from clean daylight (GHI > 50)
    clean_daylight_mask = daylight & (~np.isnan(ghi)) & (ghi > ghi_strong)
    clean_vals = vals[clean_daylight_mask]
    clean_vals = clean_vals[clean_vals > eps]
    if len(clean_vals) >= 10:
        cap_val_robust = clean_vals.quantile(cap_quantile / 100.0)
        vals = vals.clip(upper=cap_val_robust)

    out = pd.DataFrame({"timestamp": ts, "meter_reading": vals.values, "meter": meter_id})
    stats = {
        "meter": meter_id,
        "night_values_zeroed": night_nonzero_count,
        "stuck_meter_hours_removed": stuck_count,
        "stuck_meter_blocks": len(stuck_blocks),
        "unexpected_zeros_replaced": int(missing_as_zero.sum()),
        "high_outliers_capped": int(outlier_mask.sum()),
        "short_gaps_imputed": len(short),
        "medium_gaps_imputed": len(medium_no_outage),
        "long_gaps_imputed": len(long_gaps),
        "outage_blocks_model_filled": len(outage_ranges),
        "outage_blocks": outage_blocks,
    }
    return out, stats


def clean_all_meters(meter_df, solcast_hourly, options=None):
    """Clean each meter separately and concatenate. Returns (cleaned_df, list of stats dicts)."""
    cleaned = []
    all_stats = []
    for meter_id, group in meter_df.groupby("meter"):
        out, stats = clean_one_meter(group, meter_id, solcast_hourly, options)
        cleaned.append(out)
        all_stats.append(stats)
    return pd.concat(cleaned, ignore_index=True), all_stats


def build_outage_blocks_report(all_stats):
    """Build a single DataFrame of outage blocks (meter, start_ts, end_ts, length_hours) for CSV."""
    rows = []
    for stats in all_stats:
        for blk in stats.get("outage_blocks", []):
            rows.append({
                "meter": stats["meter"],
                "start_ts": blk["start_ts"],
                "end_ts": blk["end_ts"],
                "length_hours": blk["length_hours"],
            })
    return pd.DataFrame(rows)


# =============================================================================
# V2: Flag-based cleaning — daylight zeros never treated as real; analysis only on analysis_valid
# =============================================================================

def valid_daylight_mask_v2(zenith, ghi, ghi_min=VALID_DAYLIGHT_GHI):
    """
    valid_daylight = (zenith < 90) & (GHI > 50). Fallback: if zenith missing, use (GHI > 50) only.
    """
    zenith = np.asarray(zenith)
    ghi = np.asarray(ghi)
    has_zenith = ~np.isnan(zenith)
    ghi_ok = (~np.isnan(ghi)) & (ghi > ghi_min)
    return np.where(has_zenith, (zenith < 90) & ghi_ok, ghi_ok)


def build_expected_series_xgb(df_merged, solcast_hourly, value_col="meter_reading", train_mask=None):
    """
    XGBoost on Solcast + time features trained on clean hours (train_mask & value > 0); predict for all.
    Returns expected_power array same length as df_merged.
    """
    ts_ser = pd.Series(pd.to_datetime(df_merged["timestamp"]).dt.floor("h").values)
    feat_df, X_cols = _build_long_gap_features(solcast_hourly, ts_ser)
    if feat_df is None or not X_cols:
        return np.full(len(df_merged), np.nan)
    X_all = feat_df[X_cols].fillna(0)
    if train_mask is None:
        train_mask = (df_merged[value_col].notna()) & (df_merged[value_col] > ZERO_EPS)
    else:
        train_mask = np.asarray(train_mask) & (df_merged[value_col].values > ZERO_EPS)
    if train_mask.sum() < 50:
        return np.full(len(df_merged), np.nan)
    idx = np.where(train_mask)[0]
    X_train = X_all.iloc[idx].fillna(0)
    y_train = df_merged[value_col].values[idx]
    if len(X_train) < 50:
        return np.full(len(df_merged), np.nan)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    expected = model.predict(X_all.fillna(0))
    return np.maximum(0.0, expected)


def process_one_meter_v2(meter_df, meter_id, solcast_hourly, build_expected=True, ghi_min=VALID_DAYLIGHT_GHI):
    """
    V2: Merge Solcast, zero nighttime readings, detect stuck meters, add valid_daylight,
    outage_flag, analysis_valid. Optionally add expected_power and PR.
    Returns (df_out, qc_dict).
    """
    df = meter_df.sort_values("timestamp").reset_index(drop=True)
    sc = solcast_hourly.copy()
    sc["timestamp"] = pd.to_datetime(sc["timestamp"]).dt.floor("h")
    merge_cols = ["timestamp"] + [c for c in ["ghi", "zenith", "dni", "dhi", "air_temp", "cloud_opacity"] if c in sc.columns]
    sc = sc[merge_cols].groupby("timestamp").first().reset_index()
    df = df.merge(sc, on="timestamp", how="left")
    ghi = df["ghi"].values if "ghi" in df.columns else np.full(len(df), np.nan)
    zenith = df["zenith"].values if "zenith" in df.columns else np.full(len(df), np.nan)
    reading = df["meter_reading"].values.copy()

    # Force nighttime readings to zero (zenith >= 90 or fallback hours)
    is_night = definite_night_mask(zenith, df["timestamp"])
    night_nonzero_count = int((is_night & (reading > 0)).sum())
    reading[is_night] = 0.0

    # Stuck-meter detection: identical non-zero value repeating >= 8h → set to 0
    stuck_mask, stuck_blocks = detect_stuck_meter(reading, min_run=STUCK_MIN_HOURS)
    stuck_count = int(stuck_mask.sum())
    reading[stuck_mask] = 0.0

    valid_daylight = valid_daylight_mask_v2(zenith, ghi, ghi_min)
    is_zero = reading <= ZERO_EPS
    outage_flag = valid_daylight & is_zero
    analysis_valid = valid_daylight & (~outage_flag)

    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "meter": meter_id,
        "meter_reading": reading,
        "outage_flag": outage_flag,
        "analysis_valid": analysis_valid,
    })
    out["data_version"] = DATA_VERSION_V2

    if build_expected and "ghi" in df.columns:
        expected = build_expected_series_xgb(df, solcast_hourly, "meter_reading", train_mask=analysis_valid)
        out["expected_power"] = expected
        pr = np.full(len(out), np.nan)
        av = analysis_valid & (expected > ZERO_EPS)
        pr[av] = reading[av] / np.maximum(expected[av], 1e-9)
        out["PR"] = pr
    else:
        out["expected_power"] = np.nan
        out["PR"] = np.nan

    valid_hours = valid_daylight.sum()
    daylight_zero_pct = (outage_flag.sum() / valid_hours * 100.0) if valid_hours > 0 else 0.0
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["outage_flag"] = outage_flag
    df["valid_daylight"] = valid_daylight
    daily_valid = df.groupby("date")["valid_daylight"].sum()
    daily_outage = df.groupby("date")["outage_flag"].sum()
    daily_rate = daily_outage / daily_valid.replace(0, np.nan)
    outage_days = list(daily_rate[daily_rate > OUTAGE_DAY_RATE_THRESHOLD].index) if len(daily_rate) else []
    qc = {
        "meter": meter_id,
        "night_values_zeroed": night_nonzero_count,
        "stuck_meter_hours_removed": stuck_count,
        "stuck_meter_blocks": len(stuck_blocks),
        "daylight_zero_pct": round(daylight_zero_pct, 2),
        "outage_days_count": len(outage_days),
        "outage_days": outage_days,
        "analysis_valid_hours": int(analysis_valid.sum()),
        "valid_daylight_hours": int(valid_daylight.sum()),
    }
    return out, qc


def run_cleaning_v2(
    meter_path=None,
    solcast_path=None,
    out_meter_path=None,
    out_solcast_path=None,
    build_expected=True,
):
    """
    V2: Flag-based cleaning. No imputation; add outage_flag and analysis_valid.
    For degradation/seasonality use only analysis_valid (or drop days with outage rate > 10%).
    Saves *_v2.csv and QC report. Prints version before saving.
    """
    print(f"[V2] Data version: {DATA_VERSION_V2}")
    meter_path = meter_path or os.path.join(DATA_DIR, METER_CSV)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_meter_path = out_meter_path or os.path.join(OUT_DIR, CLEANED_METER_V2_CSV)
    out_solcast_path = out_solcast_path or os.path.join(OUT_DIR, CLEANED_SOLCAST_V2_CSV)
    qc_path = os.path.join(RESULTS_DIR, QC_REPORT_V2_CSV)

    print("Loading data...")
    meter_df = load_meter_data(meter_path)
    solcast_df = load_solcast_data(solcast_path, prefer_cleaned=(solcast_path is None))

    print("Resampling Solcast to hourly...")
    solcast_hourly = resample_solcast_to_hourly(solcast_df)
    shift_h = find_best_solcast_shift(meter_df, solcast_hourly)
    if shift_h != 0:
        solcast_hourly["timestamp"] = pd.to_datetime(solcast_hourly["timestamp"]) + pd.Timedelta(hours=shift_h)
        solcast_hourly["timestamp"] = solcast_hourly["timestamp"].dt.floor("h")
        print(f"  Solcast shift: {shift_h:+d} h")

    print("Processing meters (v2: night zeroing + stuck detection + flags)...")
    all_out = []
    all_qc = []
    for meter_id, group in meter_df.groupby("meter"):
        out, qc = process_one_meter_v2(group, meter_id, solcast_hourly, build_expected=build_expected)
        all_out.append(out)
        all_qc.append(qc)
        night_z = qc.get("night_values_zeroed", 0)
        stuck_h = qc.get("stuck_meter_hours_removed", 0)
        if night_z > 0 or stuck_h > 0:
            print(f"  {meter_id}: night_zeroed={night_z}, stuck_removed={stuck_h}h")

    out_df = pd.concat(all_out, ignore_index=True)
    qc_rows = []
    for qc in all_qc:
        row = {k: v for k, v in qc.items() if k != "outage_days"}
        row["outage_days_list"] = ",".join(str(d) for d in qc["outage_days"]) if qc.get("outage_days") else ""
        qc_rows.append(row)
    qc_df = pd.DataFrame(qc_rows)

    print(f"[V2] Saving version {DATA_VERSION_V2} — do not use old CSV.")
    out_meter_path = os.path.abspath(out_meter_path)
    out_df.to_csv(out_meter_path, index=False)
    print(f"  Meter v2: {out_meter_path}")

    cleaned_solcast = clean_solcast(solcast_df)
    cleaned_solcast.to_csv(os.path.abspath(out_solcast_path), index=False)
    print(f"  Solcast v2: {out_solcast_path}")

    qc_df.to_csv(qc_path, index=False)
    print(f"  QC report: {qc_path}")

    # Monthly median PR for degradation trend (per meter, if PR exists)
    if "PR" in out_df.columns and out_df["PR"].notna().any():
        out_df["month"] = pd.to_datetime(out_df["timestamp"]).dt.to_period("M")
        pr_valid = out_df[out_df["analysis_valid"] & out_df["PR"].notna()]
        if len(pr_valid):
            monthly_pr = pr_valid.groupby(["meter", "month"])["PR"].median().reset_index()
            monthly_pr.to_csv(os.path.join(RESULTS_DIR, "monthly_median_PR_v2.csv"), index=False)
            print(f"  Monthly median PR: Results_seasonality/monthly_median_PR_v2.csv")

    return out_df, cleaned_solcast, {"version": DATA_VERSION_V2, "qc": all_qc}


def clean_solcast(solcast_df):
    """
    Solcast is reference/weather data; we only resample to hourly and drop duplicates.
    Optionally drop rows with all-zero GHI/DNI/DHI if you want to keep only useful rows.
    """
    df = solcast_df.copy()
    df = df.dropna(subset=["timestamp"])
    # Resample to hourly for consistency with meter
    if "ghi" in df.columns:
        df = df.set_index("timestamp").resample("1h").mean().reset_index()
    return df


def run_cleaning(
    meter_path=None,
    solcast_path=None,
    out_meter_path=None,
    out_solcast_path=None,
    options=None,
):
    """
    Load meter and Solcast data, clean meter data (per meter), save cleaned outputs.
    Returns (cleaned_meter_df, cleaned_solcast_df, report_dict).
    """
    meter_path = meter_path or os.path.join(DATA_DIR, METER_CSV)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_meter_path = out_meter_path or os.path.join(OUT_DIR, CLEANED_METER_CSV)
    out_solcast_path = out_solcast_path or os.path.join(OUT_DIR, CLEANED_SOLCAST_CSV)

    print("Loading data...")
    meter_df = load_meter_data(meter_path)
    # Prefer solcast_df_cleaned.csv for robust zenith/GHI when no path given
    solcast_df = load_solcast_data(solcast_path, prefer_cleaned=(solcast_path is None))

    print("Resampling Solcast to hourly...")
    solcast_hourly = resample_solcast_to_hourly(solcast_df)
    # Align Solcast to meter: try -3..+3h shift and pick best (reduces NaN/ wrong zenith)
    shift_h = find_best_solcast_shift(meter_df, solcast_hourly)
    if shift_h != 0:
        solcast_hourly["timestamp"] = pd.to_datetime(solcast_hourly["timestamp"]) + pd.Timedelta(hours=shift_h)
        solcast_hourly["timestamp"] = solcast_hourly["timestamp"].dt.floor("h")
        print(f"  Applied Solcast time shift: {shift_h:+d} h for better alignment")

    print("Cleaning meter data (night zeroing + stuck detection + imputation)...")
    cleaned_meter, cleaning_stats = clean_all_meters(meter_df, solcast_hourly, options)

    total_night = sum(s.get("night_values_zeroed", 0) for s in cleaning_stats)
    total_stuck = sum(s.get("stuck_meter_hours_removed", 0) for s in cleaning_stats)
    print(f"  Total night values zeroed: {total_night}")
    print(f"  Total stuck-meter hours removed: {total_stuck}")

    print("Cleaning Solcast (hourly only)...")
    cleaned_solcast = clean_solcast(solcast_df)

    out_meter_path = os.path.abspath(out_meter_path)
    out_solcast_path = os.path.abspath(out_solcast_path)
    cleaned_meter.to_csv(out_meter_path, index=False)
    cleaned_solcast.to_csv(out_solcast_path, index=False)
    print("Cleaned files saved (originals unchanged):")
    print(f"  Meter:  {out_meter_path}")
    print(f"  Solcast: {out_solcast_path}")

    report = {
        "meter_rows_raw": len(meter_df),
        "meter_rows_cleaned": len(cleaned_meter),
        "meters": cleaned_meter["meter"].nunique(),
        "solcast_hourly_rows": len(cleaned_solcast),
        "cleaning_stats": cleaning_stats,
    }
    # Per-meter summary (exclude nested outage_blocks for CSV)
    if cleaning_stats:
        stats_flat = [{k: v for k, v in s.items() if k != "outage_blocks"} for s in cleaning_stats]
        stats_df = pd.DataFrame(stats_flat)
        report_path = os.path.join(OUT_DIR, "data_cleaning_report.csv")
        stats_df.to_csv(report_path, index=False)
        print(f"Saved report: {report_path}")
        # Outage blocks report (start/end per meter)
        outage_df = build_outage_blocks_report(cleaning_stats)
        if not outage_df.empty:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            outage_path = os.path.join(RESULTS_DIR, OUTAGE_BLOCKS_CSV)
            outage_df.to_csv(outage_path, index=False)
            print(f"Saved outage blocks: {outage_path}")
    return cleaned_meter, cleaned_solcast, report


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].lower() == "v2":
        print("Running V2 (flag-based, no imputation; analysis_valid for degradation)...")
        out_df, solcast, report = run_cleaning_v2(build_expected=True)
        print("Report version:", report["version"])
    else:
        cleaned_meter, cleaned_solcast, report = run_cleaning()
        print("Report:", {k: v for k, v in report.items() if k != "cleaning_stats"})
