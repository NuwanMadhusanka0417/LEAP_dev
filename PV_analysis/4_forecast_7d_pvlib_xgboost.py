#!/usr/bin/env python3
"""
7-day hourly PV generation: PVLib physics + XGBoost trained on past weather + meter.

**Training:** historical cleaned Solcast (local CSV) + hourly meter readings where timestamps overlap.

**Forecast (--azure-live):** loads live Solcast blobs from Azure (same pattern as ``forecasting.ipynb``).
By default the **first forecast hour** is the hour **after** the last timestamp in the viz merge file
``{out_dir}/hourly_{building}_master.csv`` (e.g. ``hourly_library_master.csv`` for ``--building-key library``),
matching the dashboard hourly series; if that file is missing, the hour after the last row in ``--meter-csv``
is used. Override with ``--forecast-start``. PVLib and XGBoost use **only** that 168 h live-weather window.

**Credentials:** matches ``forecasting.ipynb`` — a default SAS is embedded so ``--azure-live`` runs without
setup. Prefer overriding with ``AZURE_STORAGE_SAS_TOKEN`` (and rotate the notebook token if this repo is shared).

  AZURE_STORAGE_SAS_TOKEN        (optional; overrides embedded default)
  AZURE_STORAGE_ACCOUNT_URL      (default: https://leapdata.blob.core.windows.net)
  AZURE_STORAGE_CONTAINER        (default: solar-forecasts-solcast)

Outputs in ``PV_analysis/data_for_viz/`` (unless ``--out-dir``):
  forecast_7d_pvlib_{key}.csv, forecast_7d_xgboost_{key}.csv,
  forecast_7d_combined_{key}.csv, forecast_7d_run_meta_{key}.csv

Usage::
  pip install xgboost scikit-learn azure-storage-blob
  cd PV_analysis
  python 4_forecast_7d_pvlib_xgboost.py --building-key library --azure-live --campus BUNDOORA
  python 4_forecast_7d_pvlib_xgboost.py --building-key library --backtest   # local CSV only
"""
from __future__ import annotations

import argparse
import importlib.util
import os
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

FORECAST_HOURS = 168

DEFAULT_AZURE_ACCOUNT_URL = "https://leapdata.blob.core.windows.net"
DEFAULT_AZURE_CONTAINER = "solar-forecasts-solcast"

# Same value as forecasting.ipynb BlobServiceClient credential=…; env AZURE_STORAGE_SAS_TOKEN overrides.
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
    """Ensure ghi, dni, dhi, air_temp exist (names as expected by build_expected)."""
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


def load_solcast_live_from_azure(
    *,
    account_url: str,
    container_name: str,
    sas_token: str,
    campus: str,
    live_min_timestamp: str | None = None,
) -> pd.DataFrame:
    """Load and merge 'live' CSV blobs; same idea as forecasting.ipynb (without storing secrets in repo)."""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as e:
        raise SystemExit(
            "Azure mode requires: pip install azure-storage-blob\n" + str(e)
        ) from e

    blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)
    container_client = blob_service_client.get_container_client(container_name)
    csv_files = [
        b.name
        for b in container_client.list_blobs()
        if "live" in b.name and b.name.endswith(".csv")
    ]
    if not csv_files:
        raise RuntimeError(
            f"No blobs matching 'live' + .csv in container {container_name!r}."
        )

    dataframes = []
    for name in csv_files:
        blob_client = container_client.get_blob_client(name)
        # Older azure-storage-blob: content_as_text() has no ``errors=`` kwarg; decode bytes explicitly.
        raw = blob_client.download_blob().readall()
        text = raw.decode("utf-8", errors="replace")
        temp_df = pd.read_csv(StringIO(text))
        temp_df["source_file"] = name
        dataframes.append(temp_df)

    solcast_live_df = pd.concat(dataframes, ignore_index=True)
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


def _resolve_forecast_window_local(
    weather: pd.DataFrame,
    meter_max: pd.Timestamp,
    *,
    backtest: bool,
    forward: bool,
    forecast_start_arg: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    w = weather["timestamp"].sort_values()
    w_min, w_max = w.iloc[0], w.iloc[-1]

    if forecast_start_arg:
        fs = pd.to_datetime(forecast_start_arg)
        mode = "explicit"
    elif forward:
        fs = pd.Timestamp(meter_max).floor("h") + pd.Timedelta(hours=1)
        mode = "forward"
    else:
        fs = pd.Timestamp(w_max).floor("h") - pd.Timedelta(hours=FORECAST_HOURS - 1)
        mode = "backtest"

    fe = fs + pd.Timedelta(hours=FORECAST_HOURS)
    if fs < w_min:
        raise ValueError(f"forecast_start {fs} is before weather min {w_min}")
    if w_max < fe - pd.Timedelta(seconds=1):
        raise ValueError(
            f"Weather ends at {w_max} but forecast needs hours through {fe}. "
            "Use --backtest, shorten horizon, or provide weather that extends far enough."
        )
    if forward and fs <= meter_max:
        raise ValueError(
            f"Forward mode: forecast_start {fs} must be after last meter time {meter_max}"
        )
    return fs, fe, mode


def _align_fs_to_live_span(
    fs: pd.Timestamp,
    lo: pd.Timestamp,
    hi: pd.Timestamp,
    *,
    auto_chosen_start: bool,
) -> pd.Timestamp:
    """Ensure [fs, fs + (FORECAST_HOURS-1)h] lies within [lo, hi] (hourly live feed bounds)."""
    span_needed = pd.Timedelta(hours=FORECAST_HOURS - 1)
    if hi - lo < span_needed:
        raise RuntimeError(
            f"Live weather only spans {(hi - lo).total_seconds() / 3600:.0f} h ({lo} → {hi}); "
            f"need at least {FORECAST_HOURS} consecutive hours."
        )

    # Match naive vs tz-aware live timestamps
    if lo.tzinfo is None and fs.tzinfo is not None:
        fs = fs.tz_localize(None)
    elif lo.tzinfo is not None and fs.tzinfo is None:
        fs = fs.tz_localize(lo.tzinfo, ambiguous=True, nonexistent="shift_forward")
    elif lo.tzinfo is not None and fs.tzinfo is not None:
        fs = fs.tz_convert(lo.tzinfo)

    fs = fs.floor("h")

    if fs + span_needed > hi:
        if auto_chosen_start:
            fs = hi - span_needed
            print(
                f"  Note: live feed ends {hi}; using last {FORECAST_HOURS} h "
                f"(forecast start {fs})."
            )
        else:
            raise RuntimeError(
                f"A 168 h window from --forecast-start {fs} extends past live data (last hour {hi}). "
                f"Use --forecast-start on or before {(hi - span_needed)}."
            )

    if fs < lo:
        if auto_chosen_start:
            fs = lo
            print(f"  Note: adjusted forecast start to first live hour {lo}.")
        else:
            raise RuntimeError(
                f"--forecast-start {fs} is before live data ({lo})."
            )

    return fs


def _slice_live_forecast_hours(
    live_weather: pd.DataFrame,
    forecast_start: pd.Timestamp | None,
    *,
    align_auto: bool = True,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Take exactly FORECAST_HOURS rows from live weather on an hourly grid [fs, fe)."""
    df = live_weather.sort_values("timestamp").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("h")
    # Multiple raw timestamps can map to the same hour → unique index required for reindex().
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    lo = df["timestamp"].min()
    hi = df["timestamp"].max()
    auto = align_auto and forecast_start is None
    if forecast_start is None:
        fs = pd.Timestamp.now().replace(minute=0, second=0, microsecond=0)
    else:
        fs = pd.Timestamp(forecast_start)

    fs = _align_fs_to_live_span(fs, lo, hi, auto_chosen_start=auto)

    fe = fs + pd.Timedelta(hours=FORECAST_HOURS)
    idx = pd.date_range(start=fs, periods=FORECAST_HOURS, freq="h")
    df_idx = df.set_index("timestamp").sort_index()
    block = df_idx.reindex(idx)
    if block[["ghi", "dni", "dhi", "air_temp"]].isna().all(axis=None):
        raise RuntimeError(
            f"No live weather overlaps forecast grid starting {fs}. "
            "Check blob data range or pass --forecast-start."
        )
    block = block.interpolate(limit_direction="both")
    block = block.ffill().bfill()
    if block[["ghi", "dni", "dhi", "air_temp"]].isna().any().any():
        raise RuntimeError(
            "Could not fill all weather hours for the 7-day window; missing data after reindex."
        )

    out = block.reset_index()
    if "timestamp" not in out.columns:
        out = out.rename(columns={out.columns[0]: "timestamp"})
    req = ["ghi", "dni", "dhi", "air_temp"]
    for c in req:
        if c not in out.columns:
            raise ValueError(f"Live slice missing {c}")
    extra = [c for c in ("zenith", "wind_speed_10m", "albedo") if c in df.columns]
    for c in extra:
        if c not in out.columns and c in df_idx.columns:
            out[c] = df_idx.reindex(idx)[c].values
    return out, fs, fe


def _hourly_master_csv_path(building_key: str, out_dir: str) -> str:
    slug = building_key.strip().lower().replace(" ", "_")
    return os.path.join(out_dir, f"hourly_{slug}_master.csv")


def _last_timestamp_hourly_master(path: str, meter_key: str) -> pd.Timestamp | None:
    """Latest timestamp in viz hourly master for this meter (same file as JS dashboard)."""
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "meter_id" in df.columns:
        df = df[df["meter_id"].astype(str).str.strip() == str(meter_key).strip()]
    if df.empty or "timestamp" not in df.columns:
        return None
    return pd.Timestamp(df["timestamp"].max())


def _forecast_start_after_viz_or_meter(
    *,
    hourly_master_path: str | None,
    meter_key: str,
    meter_max: pd.Timestamp,
) -> tuple[pd.Timestamp, str, str]:
    """Return (forecast_start, anchor_label, path_or_note). Anchor = hour after last obs."""
    if hourly_master_path and os.path.isfile(hourly_master_path):
        last = _last_timestamp_hourly_master(hourly_master_path, meter_key)
        if last is not None:
            fs = last.floor("h") + pd.Timedelta(hours=1)
            return fs, "hourly_master", os.path.abspath(hourly_master_path)
    fs = pd.Timestamp(meter_max).floor("h") + pd.Timedelta(hours=1)
    return fs, "meter_csv_max", ""


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


def main() -> None:
    ap = argparse.ArgumentParser(description="7d PVLib + XGBoost forecast export")
    ap.add_argument("--building-key", default="library", help="Short key, e.g. library")
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
        help="Start of 7d window (naive local ISO). For --azure-live, default is hour after last "
        "timestamp in hourly_{building}_master.csv (viz file) or meter CSV.",
    )
    ap.add_argument(
        "--hourly-master-csv",
        default=None,
        help="Optional path to hourly_*_master.csv; default is {out_dir}/hourly_{building}_master.csv",
    )
    ap.add_argument(
        "--forward",
        action="store_true",
        help="[local mode] Forecast after last meter timestamp",
    )
    ap.add_argument(
        "--backtest",
        action="store_true",
        help="[local mode] Last 168h of historical CSV",
    )
    ap.add_argument(
        "--azure-live",
        action="store_true",
        help="Forecast using live Solcast blobs from Azure (PVLib + XGB on that weather only)",
    )
    ap.add_argument(
        "--campus",
        default=None,
        help="Solcast campus code for live blobs, e.g. BUNDOORA (default: from panel_data campus)",
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
        help="Optional: drop live rows with timestamp <= this (e.g. 2024-07-13 for notebook parity)",
    )
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()

    if args.azure_live:
        if args.forward or args.backtest:
            ap.error("With --azure-live, do not use --forward/--backtest (local forecast modes).")
    else:
        if args.forward and args.backtest:
            ap.error("Use only one of --forward or --backtest")
        if not args.forward and not args.forecast_start:
            args.backtest = True

    bldg = config.get_building_config(args.building_key.strip().lower())
    meter_key = bldg["meter_key_full"]
    campus = (args.campus or _default_campus_for_building(bldg)).strip().upper()
    os.makedirs(args.out_dir, exist_ok=True)

    hourly_master_path = args.hourly_master_csv or _hourly_master_csv_path(
        args.building_key, args.out_dir
    )
    forecast_anchor_source = ""
    hourly_master_used = ""

    print("Loading historical weather (training) …")
    hist_weather = _load_weather_local(args.historical_weather_csv)
    print(
        f"  historical rows: {len(hist_weather):,}  "
        f"{hist_weather['timestamp'].min()} → {hist_weather['timestamp'].max()}"
    )

    pv_mod = _import_pvlib_builder()
    print("PVLib build_expected (historical weather) …")
    exp_hist = pv_mod.build_expected(
        hist_weather,
        system_dc_w=bldg["system_dc_w"],
        inverter_ac_w=bldg["inverter_ac_w"],
    )
    exp_hist = _attach_weather_extras(exp_hist, hist_weather)

    print(f"Loading meter: {meter_key} …")
    meter = _load_meter_hourly(args.meter_csv, meter_key)
    meter_max = meter["timestamp"].max()
    print(f"  meter rows: {len(meter):,}  max ts: {meter_max}")

    merged = exp_hist.merge(meter, on="timestamp", how="inner")
    merged_valid = merged.dropna(subset=["actual_kwh", "expected_kwh"]).copy()
    print(f"  trainable overlap hours: {len(merged_valid):,}")

    if args.azure_live:
        sas = (os.environ.get("AZURE_STORAGE_SAS_TOKEN") or "").strip() or _DEFAULT_AZURE_SAS_FROM_NOTEBOOK
        print(f"Loading live Solcast from Azure ({args.azure_container}) campus={campus} …")
        live_raw = load_solcast_live_from_azure(
            account_url=args.azure_account_url.rstrip("/"),
            container_name=args.azure_container,
            sas_token=sas,
            campus=campus,
            live_min_timestamp=args.live_min_timestamp,
        )
        print(
            f"  live rows: {len(live_raw):,}  "
            f"{live_raw['timestamp'].min()} → {live_raw['timestamp'].max()}"
        )
        if args.forecast_start:
            fs_in = pd.to_datetime(args.forecast_start)
            forecast_anchor_source = "cli"
            align_auto = False
            print(f"  Forecast start from --forecast-start: {fs_in}")
        else:
            fs_in, forecast_anchor_source, hourly_master_used = _forecast_start_after_viz_or_meter(
                hourly_master_path=hourly_master_path,
                meter_key=meter_key,
                meter_max=meter_max,
            )
            align_auto = False
            last_note = (
                f"last row in {hourly_master_used}"
                if forecast_anchor_source == "hourly_master"
                else f"last meter row in {os.path.abspath(args.meter_csv)}"
            )
            print(
                f"  Forecast anchor ({forecast_anchor_source}): {last_note} → "
                f"first forecast hour {fs_in}"
            )
        live_7d, fs, fe = _slice_live_forecast_hours(
            live_raw, fs_in, align_auto=align_auto
        )
        mode = "azure_live"
        print(f"Forecast window: [{fs}, {fe})  ({FORECAST_HOURS} h)  mode={mode}")

        train_df = merged_valid[merged_valid["timestamp"] < fs].copy()
        if len(train_df) < 500:
            raise SystemExit(
                f"Too few training hours before forecast start ({len(train_df)}). "
                "Use an earlier --forecast-start or ensure meter/history overlap."
            )

        print("PVLib build_expected (live 7d weather only) …")
        forecast_exp = pv_mod.build_expected(
            live_7d,
            system_dc_w=bldg["system_dc_w"],
            inverter_ac_w=bldg["inverter_ac_w"],
        )
        forecast_base = _attach_weather_extras(forecast_exp, live_7d)
        weather_meta = f"azure:{args.azure_container}"
    else:
        fs, fe, mode = _resolve_forecast_window_local(
            hist_weather,
            meter_max,
            backtest=args.backtest,
            forward=args.forward,
            forecast_start_arg=args.forecast_start,
        )
        print(f"Forecast mode: {mode}  window: [{fs}, {fe})  ({FORECAST_HOURS} h)")

        train_df = merged_valid[merged_valid["timestamp"] < fs].copy()
        if len(train_df) < 500:
            raise SystemExit(
                f"Too few training hours before forecast start ({len(train_df)}). "
                "Check meter/weather overlap or use --backtest with longer history."
            )

        fcast_mask = (exp_hist["timestamp"] >= fs) & (exp_hist["timestamp"] < fe)
        forecast_base = exp_hist.loc[fcast_mask].copy()
        if len(forecast_base) != FORECAST_HOURS:
            raise SystemExit(
                f"Expected {FORECAST_HOURS} forecast hours, got {len(forecast_base)}. "
                "Check weather continuity (hourly grid)."
            )
        weather_meta = os.path.abspath(args.historical_weather_csv)

    X_all, _ = _feature_matrix(train_df)
    y_all = train_df["actual_kwh"].astype(float).values

    n = len(X_all)
    cut = max(int(n * 0.9), n - 2000)
    X_tr, X_va = X_all.iloc[:cut], X_all.iloc[cut:]
    y_tr, y_va = y_all[:cut], y_all[cut:]

    print(f"Training XGBoost  (train {len(X_tr):,}  val {len(X_va):,}) …")
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
    print(f"  Holdout MAE (kWh/h): {mae_va:.4f}")

    X_fcast, _ = _feature_matrix(forecast_base)
    y_xgb = model.predict(X_fcast)

    out_pv = forecast_base[["timestamp"]].copy()
    out_pv["expected_kwh_pvlib"] = forecast_base["expected_kwh"].values
    out_pv["building_key"] = args.building_key
    out_pv["forecast_mode"] = mode

    out_xgb = forecast_base[["timestamp"]].copy()
    out_xgb["predicted_kwh_xgboost"] = y_xgb
    out_xgb["building_key"] = args.building_key
    out_xgb["forecast_mode"] = mode

    out_combo = forecast_base[["timestamp"]].copy()
    out_combo["expected_kwh_pvlib"] = forecast_base["expected_kwh"].values
    out_combo["predicted_kwh_xgboost"] = y_xgb
    out_combo["building_key"] = args.building_key
    out_combo["forecast_mode"] = mode

    key = args.building_key.strip().lower().replace(" ", "_")
    p_pv = os.path.join(args.out_dir, f"forecast_7d_pvlib_{key}.csv")
    p_xgb = os.path.join(args.out_dir, f"forecast_7d_xgboost_{key}.csv")
    p_combo = os.path.join(args.out_dir, f"forecast_7d_combined_{key}.csv")
    p_meta = os.path.join(args.out_dir, f"forecast_7d_run_meta_{key}.csv")

    out_pv.to_csv(p_pv, index=False)
    out_xgb.to_csv(p_xgb, index=False)
    out_combo.to_csv(p_combo, index=False)

    meta_row = {
        "building_key": args.building_key,
        "meter_key_full": meter_key,
        "campus": campus,
        "forecast_mode": mode,
        "forecast_start": fs.isoformat(),
        "forecast_end": fe.isoformat(),
        "forecast_hours": FORECAST_HOURS,
        "train_hours_used": len(train_df),
        "val_mae_kwh_per_h": mae_va,
        "historical_weather_csv": os.path.abspath(args.historical_weather_csv),
        "forecast_weather_source": weather_meta,
        "meter_csv": os.path.abspath(args.meter_csv),
        "out_pvlib_csv": os.path.abspath(p_pv),
        "out_xgboost_csv": os.path.abspath(p_xgb),
        "out_combined_csv": os.path.abspath(p_combo),
    }
    if args.azure_live:
        meta_row["forecast_anchor_source"] = forecast_anchor_source
        meta_row["hourly_master_csv"] = hourly_master_used
    meta = pd.DataFrame([meta_row])
    meta.to_csv(p_meta, index=False)

    if args.save_model:
        mpath = os.path.join(args.out_dir, f"xgb_forecast_model_{key}.json")
        model.get_booster().save_model(mpath)
        print(f"Saved model: {mpath}")

    print("Wrote:")
    print(f"  {p_pv}")
    print(f"  {p_xgb}")
    print(f"  {p_combo}")
    print(f"  {p_meta}")


if __name__ == "__main__":
    main()
