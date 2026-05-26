#!/usr/bin/env python3
"""
7-day hourly PV generation: PVLib physics + XGBoost trained on past weather + meter.

Processes every meter key in ``config._BUILDING_PVLIB_GEOMETRY`` (via
``config.analysis_meter_keys()``) unless ``--building-key`` selects one site.

**Training:** historical cleaned Solcast (local CSV) + hourly meter readings where timestamps overlap.

**Forecast window (default):** from **00:00 on the calendar day after** the last meter reading,
using **all available forward weather** up to ~7 days (target 168 h), capped at 10 days (240 h).
If fewer hours exist (e.g. 6 days), that span is used; no fixed 168 h requirement.
Use ``--backtest`` only for evaluation on the weather tail.

Outputs in ``data_for_viz/`` per key ``<key>``:
  forecast_7d_pvlib_{key}.csv, forecast_7d_xgboost_{key}.csv,
  forecast_7d_combined_{key}.csv, forecast_7d_run_meta_{key}.csv

Legacy: ``forecast_7d_combined_library.csv`` when ``library`` is processed.

Usage::
  cd PV_analysis
  python 4_forecast_7d_pvlib_xgboost.py --azure-live --campus BUNDOORA
  python 4_forecast_7d_pvlib_xgboost.py
  python 4_forecast_7d_pvlib_xgboost.py --backtest
  python 4_forecast_7d_pvlib_xgboost.py --building-key dmw
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

import config

try:
    import xgboost as xgb
except ImportError as e:
    raise SystemExit(
        "Missing dependency: xgboost. Install with: pip install xgboost\n" + str(e)
    ) from e

try:
    from sklearn.metrics import mean_absolute_error
except ImportError as e:
    raise SystemExit(
        "Missing dependency: scikit-learn. Install with: pip install scikit-learn\n" + str(e)
    ) from e

FORECAST_HOURS_TARGET = 168  # ~7 days (ideal)
FORECAST_HOURS_MAX = 240  # ~10 days (cap)
FORECAST_HOURS_MIN = 1
FORECAST_HOURS = FORECAST_HOURS_TARGET  # backtest tail length

_WEATHER_REQ = ("ghi", "dni", "dhi", "air_temp")

DEFAULT_AZURE_ACCOUNT_URL = "https://leapdata.blob.core.windows.net"
DEFAULT_AZURE_CONTAINER = "solar-forecasts-solcast"

_DEFAULT_AZURE_SAS_FROM_NOTEBOOK = (
    "UmQIm94uLMGZkl8vOie0B1omByZBzJmP6tNodMe9HVHlgWAgprw2OX62wXCZqmJ4jAH04IBlfM5xNukhc3x9rQ=="
)


def _import_pvlib_builder():
    mod_path = Path(__file__).resolve().parent / "3_expected_power_pvlib.py"
    spec = importlib.util.spec_from_file_location("pvlib_expected_internal", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_weather_local(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Weather CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    if len(df) > 1 and df["timestamp"].diff().dt.total_seconds().median() < 2000:
        df = df.set_index("timestamp").resample("1h").mean(numeric_only=True).reset_index()
    return _ensure_pvlib_weather_columns(df)


def _ensure_pvlib_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def _first_present(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    ghi_src = _first_present(["ghi", "global_horizontal_irradiance", "GHI"])
    if ghi_src and ghi_src != "ghi":
        df["ghi"] = pd.to_numeric(df[ghi_src], errors="coerce")
    dni_src = _first_present(["dni", "direct_normal_irradiance", "DNI"])
    if dni_src and dni_src != "dni":
        df["dni"] = pd.to_numeric(df[dni_src], errors="coerce")
    dhi_src = _first_present(["dhi", "diffuse_horizontal_irradiance", "DHI"])
    if dhi_src and dhi_src != "dhi":
        df["dhi"] = pd.to_numeric(df[dhi_src], errors="coerce")
    air_src = _first_present(["air_temp", "air_temperature", "temp_air", "temperature"])
    if air_src and air_src != "air_temp":
        df["air_temp"] = pd.to_numeric(df[air_src], errors="coerce")

    req = ("ghi", "dni", "dhi", "air_temp")
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Weather frame missing columns {miss}. Have: {list(df.columns)}")
    return df


def _load_solcast_blobs_from_azure(
    *,
    account_url: str,
    container_name: str,
    sas_token: str,
    name_predicate,
) -> pd.DataFrame:
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as e:
        raise SystemExit(
            "Azure mode requires: pip install azure-storage-blob\n" + str(e)
        ) from e

    container_client = BlobServiceClient(
        account_url=account_url, credential=sas_token
    ).get_container_client(container_name)
    csv_files = [
        b.name
        for b in container_client.list_blobs()
        if name_predicate(b.name)
    ]
    if not csv_files:
        return pd.DataFrame()

    dataframes = []
    for name in csv_files:
        blob_client = container_client.get_blob_client(name)
        raw = blob_client.download_blob().readall()
        text = raw.decode("utf-8", errors="replace")
        temp_df = pd.read_csv(StringIO(text))
        temp_df["source_file"] = name
        dataframes.append(temp_df)
    return pd.concat(dataframes, ignore_index=True)


def load_solcast_live_from_azure(
    *,
    account_url: str,
    container_name: str,
    sas_token: str,
    campus: str,
    live_min_timestamp: str | None = None,
    include_forecast_blobs: bool = True,
) -> pd.DataFrame:
    live_df = _load_solcast_blobs_from_azure(
        account_url=account_url,
        container_name=container_name,
        sas_token=sas_token,
        name_predicate=lambda n: "live" in n and n.endswith(".csv"),
    )
    if live_df.empty:
        raise RuntimeError(
            f"No blobs matching 'live' + .csv in container {container_name!r}."
        )

    parts = [live_df]
    if include_forecast_blobs:
        fc_df = _load_solcast_blobs_from_azure(
            account_url=account_url,
            container_name=container_name,
            sas_token=sas_token,
            name_predicate=lambda n: "forecast" in n and n.endswith(".csv"),
        )
        if not fc_df.empty:
            print(f"  Also loaded forecast blobs: {len(fc_df):,} rows")
            parts.append(fc_df)

    solcast_live_df = pd.concat(parts, ignore_index=True)
    if "accessed_on" in solcast_live_df.columns:
        solcast_live_df["accessed_on"] = pd.to_datetime(solcast_live_df["accessed_on"], errors="coerce")
    solcast_live_df["timestamp"] = pd.to_datetime(solcast_live_df["timestamp"], errors="coerce")
    solcast_live_df = solcast_live_df.dropna(subset=["timestamp"])
    if "accessed_on" in solcast_live_df.columns and "campus" in solcast_live_df.columns:
        solcast_live_df = solcast_live_df.sort_values(by="accessed_on").drop_duplicates(
            subset=["timestamp", "campus"], keep="last"
        )
    if "campus" not in solcast_live_df.columns:
        raise ValueError("Live Solcast CSVs must include a 'campus' column.")

    if live_min_timestamp:
        solcast_live_df = solcast_live_df[
            solcast_live_df["timestamp"] > pd.to_datetime(live_min_timestamp)
        ]

    solcast_live_df = solcast_live_df.sort_values(by="timestamp")
    drop_cols = [c for c in ("accessed_on", "source_file", "timestamp_utc") if c in solcast_live_df.columns]
    if drop_cols:
        solcast_live_df = solcast_live_df.drop(columns=drop_cols)

    campus_u = campus.strip().upper()
    solcast_live_df["campus"] = solcast_live_df["campus"].astype(str).str.strip().str.upper()
    solcast_live_df = solcast_live_df[solcast_live_df["campus"] == campus_u].copy()
    if solcast_live_df.empty:
        raise ValueError(f"No live weather rows for campus {campus_u!r}.")

    return _ensure_pvlib_weather_columns(solcast_live_df)


def _load_meter_hourly(path: str, meter_key_full: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Meter CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[df["meter"].astype(str).str.strip() == meter_key_full].copy()
    if df.empty:
        raise ValueError(f"No rows for meter '{meter_key_full}' in {path}")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")
    if "meter_reading" not in df.columns:
        raise ValueError(f"Column meter_reading missing in {path}")
    out = df[["timestamp", "meter_reading"]].copy()
    out["actual_kwh"] = pd.to_numeric(out["meter_reading"], errors="coerce")
    return out[["timestamp", "actual_kwh"]]


def _attach_weather_extras(exp: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    extra = ["zenith", "air_temp", "wind_speed_10m"]
    cols = ["timestamp"] + [c for c in extra if c in weather.columns]
    w = weather[cols].copy()
    return exp.merge(w, on="timestamp", how="left")


def _add_calendar_features(ts: pd.Series) -> pd.DataFrame:
    t = pd.to_datetime(ts)
    hour = t.dt.hour.astype(np.float64)
    doy = t.dt.dayofyear.astype(np.float64)
    dow = t.dt.dayofweek.astype(np.float64)
    return pd.DataFrame(
        {
            "hour": hour,
            "doy": doy,
            "dow": dow,
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        }
    )


def _feature_matrix(base: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cal = _add_calendar_features(base["timestamp"])
    parts = pd.concat(
        [
            base[
                [
                    "expected_kwh",
                    "ghi_wm2",
                    "dni_wm2",
                    "dhi_wm2",
                    "poa_global_wm2",
                    "temp_cell_c",
                ]
            ].copy(),
            cal,
        ],
        axis=1,
    )
    if "zenith" in base.columns:
        parts["zenith"] = pd.to_numeric(base["zenith"], errors="coerce").fillna(90.0)
    else:
        parts["zenith"] = 90.0
    if "air_temp" in base.columns:
        parts["air_temp"] = pd.to_numeric(base["air_temp"], errors="coerce")
    else:
        parts["air_temp"] = np.nan
    if "wind_speed_10m" in base.columns:
        parts["wind_speed_10m"] = pd.to_numeric(base["wind_speed_10m"], errors="coerce").fillna(
            2.0
        )
    else:
        parts["wind_speed_10m"] = 2.0

    feature_names = list(parts.columns)
    return parts, feature_names


def _last_reading_from_meter_df(meter_df: pd.DataFrame) -> pd.Timestamp:
    m = meter_df.copy()
    m["actual_kwh"] = pd.to_numeric(m["actual_kwh"], errors="coerce")
    valid = m[m["actual_kwh"].notna()]
    if valid.empty:
        raise ValueError("No meter readings found in meter CSV for this meter")
    return pd.Timestamp(valid["timestamp"].max())


def _last_reading_from_hourly_master(
    path: str,
    *,
    meter_key: str,
    building_key: str,
) -> pd.Timestamp | None:
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "meter_id" in df.columns:
        mk = str(meter_key).strip()
        df = df[df["meter_id"].astype(str).str.strip() == mk]
    elif "building_key" in df.columns:
        bk = building_key.strip().lower()
        df = df[df["building_key"].astype(str).str.strip().str.lower() == bk]
    if df.empty or "timestamp" not in df.columns:
        return None
    if "actual_kwh" in df.columns:
        actual = pd.to_numeric(df["actual_kwh"], errors="coerce")
        with_data = df[actual.notna()]
        if not with_data.empty:
            return pd.Timestamp(with_data["timestamp"].max())
    return pd.Timestamp(df["timestamp"].max())


def _resolve_per_meter_forecast_start(
    *,
    meter_df: pd.DataFrame,
    hourly_master_path: str | None,
    meter_key: str,
    building_key: str,
) -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    """
    Per-meter anchor: last reading timestamp and forecast start at midnight the next calendar day.
    Returns (forecast_start, last_meter_reading, anchor_source, hourly_master_path_or_empty).
    """
    last_meter = _last_reading_from_meter_df(meter_df)
    source = "meter_csv"
    master_path = ""

    if hourly_master_path and os.path.isfile(hourly_master_path):
        master_last = _last_reading_from_hourly_master(
            hourly_master_path,
            meter_key=meter_key,
            building_key=building_key,
        )
        master_path = os.path.abspath(hourly_master_path)
        if master_last is not None and master_last > last_meter:
            last_meter = master_last
            source = "hourly_master"

    last_day = pd.Timestamp(last_meter).normalize()
    fs = last_day + pd.Timedelta(days=1)
    return fs, last_meter, source, master_path


def _backtest_forecast_start(hist_weather: pd.DataFrame) -> pd.Timestamp:
    w_max = pd.Timestamp(hist_weather["timestamp"].max()).floor("h")
    return w_max - pd.Timedelta(hours=FORECAST_HOURS - 1)


def _validate_backtest_weather(hist_weather: pd.DataFrame, fs: pd.Timestamp, fe: pd.Timestamp) -> None:
    w = hist_weather["timestamp"]
    w_min, w_max = w.min(), w.max()
    last_needed = fe - pd.Timedelta(hours=1)
    if fs < w_min:
        raise ValueError(f"forecast_start {fs} is before weather min {w_min}")
    if w_max < last_needed:
        raise ValueError(
            f"Weather ends at {w_max} but backtest needs data through {last_needed} "
            f"(last of {FORECAST_HOURS} hours ending before {fe}). "
            "Extend the weather CSV or use default forward mode (no --backtest)."
        )


def _slice_forecast_hours(
    weather: pd.DataFrame,
    forecast_start: pd.Timestamp,
    *,
    allow_gaps: bool = True,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, int]:
    """
    Use available weather from preferred start forward (up to 10 d cap), not a fixed 168 h grid.
    If the anchor is after all weather, use the trailing window ending at the last hour.
    """
    df = weather.sort_values("timestamp").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("h")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    if df.empty:
        raise RuntimeError("Weather frame is empty.")

    lo = pd.Timestamp(df["timestamp"].min()).floor("h")
    hi = pd.Timestamp(df["timestamp"].max()).floor("h")
    fs_pref = pd.Timestamp(forecast_start).floor("h")

    n_avail = int((hi - lo).total_seconds() / 3600) + 1
    if n_avail < FORECAST_HOURS_MIN:
        raise RuntimeError(
            f"Weather only spans {n_avail} hour(s) ({lo} → {hi}); "
            f"need at least {FORECAST_HOURS_MIN}."
        )

    window_note = "forward_from_anchor"
    if fs_pref <= hi:
        end_cap = fs_pref + pd.Timedelta(hours=FORECAST_HOURS_MAX - 1)
        end_ts = min(hi, end_cap)
        idx = pd.date_range(start=fs_pref, end=end_ts, freq="h")
    else:
        n_take = min(n_avail, FORECAST_HOURS_MAX)
        fs_pref = hi - pd.Timedelta(hours=n_take - 1)
        idx = pd.date_range(start=fs_pref, periods=n_take, freq="h")
        window_note = "trailing_before_anchor"
        print(
            f"  Note: preferred start {pd.Timestamp(forecast_start).floor('h')} is after "
            f"weather end {hi}; using last {n_take} h of available weather."
        )

    df_idx = df.set_index("timestamp").sort_index()
    block = df_idx.reindex(idx)
    if block[list(_WEATHER_REQ)].isna().all(axis=None):
        raise RuntimeError(
            f"No weather overlaps forecast grid starting {idx[0]}. "
            "Extend Solcast (live/forecast blobs) or pass --forecast-start."
        )

    has_data = block[list(_WEATHER_REQ)].notna().any(axis=1)
    if not has_data.any():
        raise RuntimeError("No usable weather hours in forecast window.")
    first_good = has_data[has_data].index[0]
    last_good = has_data[has_data].index[-1]
    block = block.loc[first_good:last_good]

    if allow_gaps:
        block = block.interpolate(limit_direction="both").ffill().bfill()
    elif block[list(_WEATHER_REQ)].isna().any().any():
        missing = block[block[list(_WEATHER_REQ)].isna().any(axis=1)].index
        raise RuntimeError(
            f"Forecast window has {len(missing)} hour(s) without weather "
            f"(first gap {missing[0]})."
        )

    if block[list(_WEATHER_REQ)].isna().any().any():
        raise RuntimeError(
            "Could not fill all weather hours for the forecast window."
        )

    fs_out = pd.Timestamp(block.index[0])
    fe_out = pd.Timestamp(block.index[-1]) + pd.Timedelta(hours=1)
    n_hours = len(block)

    if n_hours < FORECAST_HOURS_MIN:
        raise RuntimeError(
            f"Only {n_hours} forecast hour(s) after trim; need at least {FORECAST_HOURS_MIN}."
        )

    print(
        f"  Forecast span: {n_hours} h (~{n_hours / 24:.1f} d)  "
        f"[{fs_out} → {fe_out})  ({window_note})"
    )
    if n_hours < FORECAST_HOURS_TARGET:
        print(
            f"  Note: less than target {FORECAST_HOURS_TARGET} h (~7 d); "
            "using all available forward weather."
        )
    elif n_hours >= FORECAST_HOURS_MAX:
        print(f"  Note: capped at {FORECAST_HOURS_MAX} h (~10 d) of weather.")

    out = block.reset_index()
    if "timestamp" not in out.columns:
        out = out.rename(columns={out.columns[0]: "timestamp"})
    for c in _WEATHER_REQ:
        if c not in out.columns:
            raise ValueError(f"Weather slice missing {c}")
    extra = [c for c in ("zenith", "wind_speed_10m", "albedo") if c in df.columns]
    for c in extra:
        if c not in out.columns and c in df_idx.columns:
            out[c] = df_idx.reindex(block.index)[c].values
    return out, fs_out, fe_out, n_hours


def _slice_historical_forecast_hours(
    weather: pd.DataFrame,
    forecast_start: pd.Timestamp,
    *,
    allow_gaps: bool,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, int]:
    """Strict 168 h tail for --backtest; flexible span for forward local mode."""
    if allow_gaps:
        return _slice_forecast_hours(weather, forecast_start, allow_gaps=True)

    fs = pd.Timestamp(forecast_start).floor("h")
    fe = fs + pd.Timedelta(hours=FORECAST_HOURS)
    idx = pd.date_range(start=fs, periods=FORECAST_HOURS, freq="h")
    df_idx = (
        weather.sort_values("timestamp")
        .assign(timestamp=lambda d: pd.to_datetime(d["timestamp"]).dt.floor("h"))
        .drop_duplicates(subset=["timestamp"], keep="last")
        .set_index("timestamp")
        .sort_index()
    )
    block = df_idx.reindex(idx)
    if block[list(_WEATHER_REQ)].isna().any().any():
        missing = block[block[list(_WEATHER_REQ)].isna().any(axis=1)].index
        raise RuntimeError(
            f"Backtest window has {len(missing)} hour(s) without weather "
            f"(first gap {missing[0]})."
        )
    out = block.reset_index()
    if "timestamp" not in out.columns:
        out = out.rename(columns={out.columns[0]: "timestamp"})
    return out, fs, fe, FORECAST_HOURS


def _slice_live_forecast_hours(
    live_weather: pd.DataFrame,
    forecast_start: pd.Timestamp | None,
    *,
    align_auto: bool = True,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, int]:
    if forecast_start is None and align_auto:
        fs_in = pd.Timestamp.now().replace(minute=0, second=0, microsecond=0)
    elif forecast_start is None:
        fs_in = pd.Timestamp.now().replace(minute=0, second=0, microsecond=0)
    else:
        fs_in = pd.Timestamp(forecast_start)
    return _slice_forecast_hours(live_weather, fs_in, allow_gaps=True)


def _hourly_master_csv_path(building_key: str, out_dir: str) -> str:
    slug = building_key.strip().lower().replace(" ", "_")
    return os.path.join(out_dir, f"hourly_{slug}_master.csv")


def _default_campus_for_building(bldg: dict) -> str:
    c = str(bldg.get("campus", "")).strip().upper().replace(" ", "")
    if "BUNDOORA" in c or c == "BUN":
        return "BUNDOORA"
    if "WODONGA" in c or "ALBURY" in c:
        return "WODONGA"
    if "BENDIGO" in c:
        return "BENDIGO"
    if "MILDURA" in c:
        return "MILDURA"
    if "SHEPPARTON" in c:
        return "SHEPPARTON"
    return "BUNDOORA"


def _train_xgboost(train_df: pd.DataFrame) -> tuple[xgb.XGBRegressor, float]:
    X_all, _ = _feature_matrix(train_df)
    y_all = train_df["actual_kwh"].astype(float).values
    n = len(X_all)
    cut = max(int(n * 0.9), n - 2000)
    X_tr, X_va = X_all.iloc[:cut], X_all.iloc[cut:]
    y_tr, y_va = y_all[:cut], y_all[cut:]
    model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=40,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pred_va = model.predict(X_va)
    mae_va = float(mean_absolute_error(y_va, pred_va))
    return model, mae_va


def process_building_forecast(
    building_key: str,
    *,
    hist_weather: pd.DataFrame,
    pv_mod,
    meter_csv: str,
    out_dir: str,
    historical_weather_csv: str,
    azure_live: bool,
    live_raw: pd.DataFrame | None,
    campus_override: str | None,
    forecast_start_arg: str | None,
    hourly_master_csv: str | None,
    backtest: bool,
    azure_container: str,
    save_model: bool,
) -> dict:
    key = building_key.strip().lower()
    bldg = config.get_building_config(key)
    meter_key = bldg["meter_key_full"]
    campus = (campus_override or _default_campus_for_building(bldg)).strip().upper()

    print(f"\n{'=' * 55}")
    print(f"  [{key}] {bldg['building_name']}")
    print(f"  Meter: {meter_key} | campus: {campus}")
    print(
        f"  DC: {bldg['system_kwp']:.2f} kWp | tilt/azimuth: "
        f"{bldg['surface_tilt_deg']:.0f}° / {bldg['surface_azimuth_deg']:.0f}°"
    )

    exp_hist = pv_mod.build_expected(
        hist_weather,
        system_dc_w=bldg["system_dc_w"],
        inverter_ac_w=bldg["inverter_ac_w"],
        surface_tilt_deg=bldg["surface_tilt_deg"],
        surface_azimuth_deg=bldg["surface_azimuth_deg"],
    )
    exp_hist = _attach_weather_extras(exp_hist, hist_weather)

    meter = _load_meter_hourly(meter_csv, meter_key)
    meter_max = meter["timestamp"].max()
    print(f"  Meter rows: {len(meter):,}  max ts: {meter_max}")

    merged = exp_hist.merge(meter, on="timestamp", how="inner")
    merged_valid = merged.dropna(subset=["actual_kwh", "expected_kwh"]).copy()
    print(f"  Trainable overlap hours: {len(merged_valid):,}")

    forecast_anchor_source = ""
    hourly_master_used = ""
    last_meter_reading = ""
    hourly_master_path = hourly_master_csv or _hourly_master_csv_path(key, out_dir)

    if azure_live:
        if live_raw is None:
            raise RuntimeError("live_raw required for azure_live mode")
        if forecast_start_arg:
            fs_in = pd.to_datetime(forecast_start_arg)
            forecast_anchor_source = "cli"
            align_auto = False
            print(f"  Forecast start (--forecast-start): {fs_in}")
        else:
            fs_in, last_ts, forecast_anchor_source, hourly_master_used = (
                _resolve_per_meter_forecast_start(
                    meter_df=meter,
                    hourly_master_path=hourly_master_path,
                    meter_key=meter_key,
                    building_key=key,
                )
            )
            last_meter_reading = last_ts.isoformat()
            align_auto = False
            print(
                f"  Last meter reading: {last_ts} (day {last_ts.normalize().date()})"
            )
            print(
                f"  Forecast anchor ({forecast_anchor_source}): "
                f"from {fs_in} (midnight after last reading day), flexible span"
            )
        live_7d, fs, fe, n_forecast_hours = _slice_live_forecast_hours(
            live_raw, fs_in, align_auto=align_auto
        )
        mode = "azure_live"
        print(f"  Forecast window: [{fs}, {fe})  mode={mode}")

        train_df = merged_valid[merged_valid["timestamp"] < fs].copy()
        if len(train_df) < 500:
            raise ValueError(
                f"Too few training hours before forecast start ({len(train_df)}). "
                "Use an earlier --forecast-start or ensure meter/history overlap."
            )

        forecast_exp = pv_mod.build_expected(
            live_7d,
            system_dc_w=bldg["system_dc_w"],
            inverter_ac_w=bldg["inverter_ac_w"],
            surface_tilt_deg=bldg["surface_tilt_deg"],
            surface_azimuth_deg=bldg["surface_azimuth_deg"],
        )
        forecast_base = _attach_weather_extras(forecast_exp, live_7d)
        weather_meta = f"azure:{azure_container}"
    else:
        if backtest:
            fs = _backtest_forecast_start(hist_weather)
            fe = fs + pd.Timedelta(hours=FORECAST_HOURS)
            mode = "backtest"
            forecast_anchor_source = "weather_tail"
            _validate_backtest_weather(hist_weather, fs, fe)
            print(
                "  Forecast mode: backtest (Solcast CSV tail — not per-meter; "
                "omit --backtest for production)"
            )
        elif forecast_start_arg:
            fs = pd.to_datetime(forecast_start_arg)
            fe = fs + pd.Timedelta(hours=FORECAST_HOURS)
            mode = "explicit"
            forecast_anchor_source = "cli"
            print(f"  Forecast start (--forecast-start): {fs}")
        else:
            fs, last_ts, forecast_anchor_source, hourly_master_used = (
                _resolve_per_meter_forecast_start(
                    meter_df=meter,
                    hourly_master_path=hourly_master_path,
                    meter_key=meter_key,
                    building_key=key,
                )
            )
            mode = "forward"
            last_meter_reading = last_ts.isoformat()
            print(
                f"  Last meter reading: {last_ts} (day {last_ts.normalize().date()})"
            )
            print(
                f"  Forecast anchor ({forecast_anchor_source}): "
                f"from {fs} (midnight after last reading day), flexible span"
            )

        train_df = merged_valid[merged_valid["timestamp"] < fs].copy()
        if len(train_df) < 500:
            raise ValueError(
                f"Too few training hours before forecast start ({len(train_df)}). "
                "Check meter/weather overlap or use --backtest with longer history."
            )

        local_weather, fs, fe, n_forecast_hours = _slice_historical_forecast_hours(
            hist_weather,
            fs,
            allow_gaps=(mode != "backtest"),
        )
        print(f"  Forecast mode: {mode}  window: [{fs}, {fe})")
        forecast_exp = pv_mod.build_expected(
            local_weather,
            system_dc_w=bldg["system_dc_w"],
            inverter_ac_w=bldg["inverter_ac_w"],
            surface_tilt_deg=bldg["surface_tilt_deg"],
            surface_azimuth_deg=bldg["surface_azimuth_deg"],
        )
        forecast_base = _attach_weather_extras(forecast_exp, local_weather)
        weather_meta = os.path.abspath(historical_weather_csv)

    print(f"  Training XGBoost ({len(train_df):,} hours) …")
    model, mae_va = _train_xgboost(train_df)
    print(f"  Holdout MAE (kWh/h): {mae_va:.4f}")

    X_fcast, _ = _feature_matrix(forecast_base)
    y_xgb = model.predict(X_fcast)

    out_pv = forecast_base[["timestamp"]].copy()
    out_pv["expected_kwh_pvlib"] = forecast_base["expected_kwh"].values
    out_pv["building_key"] = key
    out_pv["forecast_mode"] = mode

    out_xgb = forecast_base[["timestamp"]].copy()
    out_xgb["predicted_kwh_xgboost"] = y_xgb
    out_xgb["building_key"] = key
    out_xgb["forecast_mode"] = mode

    out_combo = forecast_base[["timestamp"]].copy()
    out_combo["expected_kwh_pvlib"] = forecast_base["expected_kwh"].values
    out_combo["predicted_kwh_xgboost"] = y_xgb
    out_combo["building_key"] = key
    out_combo["forecast_mode"] = mode

    p_pv = os.path.join(out_dir, f"forecast_7d_pvlib_{key}.csv")
    p_xgb = os.path.join(out_dir, f"forecast_7d_xgboost_{key}.csv")
    p_combo = os.path.join(out_dir, f"forecast_7d_combined_{key}.csv")
    p_meta = os.path.join(out_dir, f"forecast_7d_run_meta_{key}.csv")

    out_pv.to_csv(p_pv, index=False)
    out_xgb.to_csv(p_xgb, index=False)
    out_combo.to_csv(p_combo, index=False)

    if key == "library":
        legacy_combo = os.path.join(out_dir, "forecast_7d_combined_library.csv")
        out_combo.to_csv(legacy_combo, index=False)
        print(f"  Wrote {legacy_combo} (dashboard legacy name)")

    print(f"  Wrote {p_pv}")
    print(f"  Wrote {p_xgb}")
    print(f"  Wrote {p_combo}")

    meta_row = {
        "building_key": key,
        "meter_key_full": meter_key,
        "building_name": bldg["building_name"],
        "campus": campus,
        "forecast_mode": mode,
        "forecast_start": fs.isoformat(),
        "forecast_end": fe.isoformat(),
        "forecast_hours": n_forecast_hours,
        "forecast_hours_target": FORECAST_HOURS_TARGET,
        "forecast_hours_max": FORECAST_HOURS_MAX,
        "train_hours_used": len(train_df),
        "val_mae_kwh_per_h": mae_va,
        "historical_weather_csv": os.path.abspath(historical_weather_csv),
        "forecast_weather_source": weather_meta,
        "meter_csv": os.path.abspath(meter_csv),
        "out_pvlib_csv": os.path.abspath(p_pv),
        "out_xgboost_csv": os.path.abspath(p_xgb),
        "out_combined_csv": os.path.abspath(p_combo),
    }
    if forecast_anchor_source:
        meta_row["forecast_anchor_source"] = forecast_anchor_source
    if last_meter_reading:
        meta_row["last_meter_reading"] = last_meter_reading
    if hourly_master_used:
        meta_row["hourly_master_csv"] = hourly_master_used
    pd.DataFrame([meta_row]).to_csv(p_meta, index=False)
    print(f"  Wrote {p_meta}")

    if save_model:
        mpath = os.path.join(out_dir, f"xgb_forecast_model_{key}.json")
        model.get_booster().save_model(mpath)
        print(f"  Saved model: {mpath}")

    return meta_row


def main() -> None:
    config.load_credentials()
    ap = argparse.ArgumentParser(
        description="7d PVLib + XGBoost forecast for meters in config._BUILDING_PVLIB_GEOMETRY"
    )
    ap.add_argument(
        "--building-key",
        default=None,
        help="Process one meter only (default: all keys from config.analysis_meter_keys()).",
    )
    ap.add_argument(
        "--historical-weather-csv",
        default=config.SOLCAST_CLEANED_V2,
        help="Local Solcast CSV for training (PVLib + meter overlap)",
    )
    ap.add_argument("--meter-csv", default=config.METER_READINGS)
    ap.add_argument("--out-dir", default=config.DATA_FOR_VIZ_DIR)
    ap.add_argument(
        "--forecast-start",
        default=None,
        help="Override: same 7d start for every meter (default: per-meter day after last reading).",
    )
    ap.add_argument(
        "--hourly-master-csv",
        default=None,
        help="Override hourly master path (single --building-key run only).",
    )
    ap.add_argument(
        "--backtest",
        action="store_true",
        help="[local] Evaluate on last 168h of Solcast CSV (NOT per-meter; do not use for ops)",
    )
    ap.add_argument(
        "--azure-live",
        action="store_true",
        help="Forecast using live Solcast blobs from Azure",
    )
    ap.add_argument(
        "--campus",
        default=None,
        help="Solcast campus for live blobs (default: per building from panel_data)",
    )
    ap.add_argument(
        "--azure-account-url",
        default=os.environ.get("AZURE_STORAGE_ACCOUNT_URL", DEFAULT_AZURE_ACCOUNT_URL),
    )
    ap.add_argument(
        "--azure-container",
        default=os.environ.get("AZURE_STORAGE_CONTAINER", DEFAULT_AZURE_CONTAINER),
    )
    ap.add_argument(
        "--live-min-timestamp",
        default=None,
        help="Optional: drop live rows with timestamp <= this",
    )
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()

    if args.azure_live:
        if args.backtest:
            ap.error("With --azure-live, do not use --backtest.")
    elif args.backtest and args.forecast_start:
        ap.error("Use only one of --backtest or --forecast-start.")

    if args.building_key:
        keys = [args.building_key.strip().lower()]
    else:
        keys = config.analysis_meter_keys()

    if not keys:
        print("ERROR: No meter keys in config._BUILDING_PVLIB_GEOMETRY", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.historical_weather_csv):
        print(f"ERROR: Weather file not found: {args.historical_weather_csv}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Meters to forecast ({len(keys)}): {', '.join(keys)}")
    if args.backtest:
        print("Mode: backtest on shared Solcast CSV tail (not per-meter).")
    elif args.forecast_start:
        if len(keys) > 1:
            print(
                f"Mode: explicit start {args.forecast_start} for ALL meters "
                "(omit --forecast-start for per-meter anchors)."
            )
    else:
        print(
            "Mode: per-meter — flexible span (target ~7 d, max 10 d) from midnight after "
            "each meter's last reading day."
        )

    print("\nLoading historical weather (training) …")
    hist_weather = _load_weather_local(args.historical_weather_csv)
    print(
        f"  historical rows: {len(hist_weather):,}  "
        f"{hist_weather['timestamp'].min()} → {hist_weather['timestamp'].max()}"
    )

    pv_mod = _import_pvlib_builder()

    live_raw: pd.DataFrame | None = None
    if args.azure_live:
        campus_live = (args.campus or "BUNDOORA").strip().upper()
        sas = config.env_value("AZURE_STORAGE_SAS_TOKEN") or _DEFAULT_AZURE_SAS_FROM_NOTEBOOK
        print(f"\nLoading live Solcast from Azure ({args.azure_container}) campus={campus_live} …")
        live_raw = load_solcast_live_from_azure(
            account_url=args.azure_account_url.rstrip("/"),
            container_name=args.azure_container,
            sas_token=sas,
            campus=campus_live,
            live_min_timestamp=args.live_min_timestamp,
        )
        print(
            f"  live rows: {len(live_raw):,}  "
            f"{live_raw['timestamp'].min()} → {live_raw['timestamp'].max()}"
        )

    meta_rows: list[dict] = []
    errors: list[str] = []

    for key in keys:
        try:
            meta = process_building_forecast(
                key,
                hist_weather=hist_weather,
                pv_mod=pv_mod,
                meter_csv=args.meter_csv,
                out_dir=args.out_dir,
                historical_weather_csv=args.historical_weather_csv,
                azure_live=args.azure_live,
                live_raw=live_raw,
                campus_override=args.campus,
                forecast_start_arg=args.forecast_start,
                hourly_master_csv=args.hourly_master_csv if args.building_key else None,
                backtest=args.backtest,
                azure_container=args.azure_container,
                save_model=args.save_model,
            )
            meta_rows.append(meta)
        except Exception as e:
            errors.append(f"{key}: {e}")
            print(f"  ERROR [{key}]: {e}", file=sys.stderr)

    if meta_rows:
        meta_path = os.path.join(args.out_dir, "forecast_7d_runs_meta_all.csv")
        pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
        print(f"\nWrote {meta_path} ({len(meta_rows)} site(s))")

    if errors:
        print("\nFailed meters:", file=sys.stderr)
        for msg in errors:
            print(f"  - {msg}", file=sys.stderr)
        sys.exit(1 if not meta_rows else 2)

    print("\nDone.")


if __name__ == "__main__":
    main()
