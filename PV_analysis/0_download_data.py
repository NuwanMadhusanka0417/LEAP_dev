#!/usr/bin/env python3
"""
Incremental download of meter readings (SQL Server), Solcast weather (Azure blobs),
and HSU soiling (Open-Meteo air quality + pvlib).

Meters: append only rows newer than the last timestamp per meter.
Weather: re-fetch from Azure for the last ``SOLCAST_REFRESH_DAYS`` (14) before the
file's last timestamp per campus, plus any newer hours — live values replace stale forecast rows.
HSU soiling: PM2.5/PM10 from Open-Meteo CAMS aligned to merged Solcast rain (BUNDOORA).
Writes ``data_raw/hsu_soiling_output.csv`` and publishes ``data_for_viz/hsu_soiling_bundoora.csv``.
PM values are converted µg/m³ → g/m³ before pvlib.soiling.hsu.

Credentials: copy ``.env.example`` → ``.env`` in this folder (auto-loaded), or export env vars.

Usage (from PV_analysis/)::
  python 0_download_data.py
  python 0_download_data.py --weather-only
  python 0_download_data.py --meters-only
  python 0_download_data.py --soiling-only
  python 0_download_data.py --no-soiling
  python 0_download_data.py --campus BUNDOORA
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.parse
from io import StringIO
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

import config

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE, "data_raw")

METER_RAW_PRIMARY = "SolarMeterReadings1hour.csv"
# Live/forecast rows appended by Azure download (often starts ~2024 when blobs began).
SOLCAST_RAW_PRIMARY = "solcast_df_2020_2025.csv"
# Static multi-year Solcast archive (typically 2020 → earlier cut-off); not updated by Step 0.
SOLCAST_ARCHIVE = "solcast_df.csv"
HSU_SOILING_RAW = "hsu_soiling_output.csv"
HSU_DEFAULT_CAMPUS = "BUNDOORA"
METER_OUTPUT_COLS = ("timestamp", "meter", "meter_reading")

# Bundoora grid point for Open-Meteo air quality (matches Solcast BUNDOORA weather)
HSU_AIR_QUALITY_LAT = -37.72
HSU_AIR_QUALITY_LON = 145.05
HSU_CLEANING_THRESHOLD_MM = 0.5
# pvlib.soiling.hsu expects PM in g/m³; Open-Meteo returns µg/m³.
PM_UGM3_TO_GM3 = 1e-6
# Open-Meteo cams_global PM2.5/PM10 (Bundoora); earlier hours have timestamps but null PM values.
OPENMETEO_CAMS_PM_START = pd.Timestamp("2022-08-04", tz="UTC")
OPENMETEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

DEFAULT_AZURE_ACCOUNT_URL = "https://leapdata.blob.core.windows.net"
DEFAULT_AZURE_CONTAINER = "solar-forecasts-solcast"
# Same token as forecasting.ipynb / 4_forecast (dev fallback if .env not filled in)
_DEV_AZURE_SAS_FALLBACK = (
    "UmQIm94uLMGZkl8vOie0B1omByZBzJmP6tNodMe9HVHlgWAgprw2OX62wXCZqmJ4jAH04IBlfM5xNukhc3x9rQ=="
)
# Dev fallbacks when .env still has placeholders (same as forecasting.ipynb)
_DEV_DB = {
    "LEAP_DB_SERVER": "23.101.237.42",
    "LEAP_DB_DATABASE": "leap",
    "LEAP_DB_USER": "sa",
    "LEAP_DB_PASSWORD": "NjOzAi^GpU!!",
}


def raw_meter_path() -> str:
    return os.path.join(DATA_RAW_DIR, METER_RAW_PRIMARY)


def raw_solcast_path() -> str:
    return os.path.join(DATA_RAW_DIR, SOLCAST_RAW_PRIMARY)


def raw_solcast_archive_path() -> str:
    return os.path.join(DATA_RAW_DIR, SOLCAST_ARCHIVE)


def raw_hsu_soiling_path() -> str:
    return os.path.join(DATA_RAW_DIR, HSU_SOILING_RAW)


def hsu_soiling_viz_path() -> str:
    os.makedirs(config.DATA_FOR_VIZ_DIR, exist_ok=True)
    return os.path.join(config.DATA_FOR_VIZ_DIR, config.HSU_SOILING_VIZ_CSV)


def publish_hsu_for_viz(raw_path: str) -> None:
    """Copy canonical raw HSU CSV into data_for_viz/ for the JS dashboard."""
    import shutil

    if not os.path.isfile(raw_path):
        return
    viz_path = hsu_soiling_viz_path()
    shutil.copy2(raw_path, viz_path)
    print(f"  Published dashboard copy -> {viz_path}")


def ensure_raw_dir() -> None:
    os.makedirs(DATA_RAW_DIR, exist_ok=True)


def _compute_diff(series: pd.Series) -> pd.Series:
    return series.diff()


def _handle_outliers(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return series
    threshold = float(np.nanquantile(valid, 0.999))
    if not np.isfinite(threshold):
        return series
    return series.where((series <= threshold) & (series >= 0), np.nan)


def sql_readings_to_hourly_kwh(df: pd.DataFrame, meter_id: str) -> pd.DataFrame:
    """Cumulative MeterReading → diff → outliers → 30min sum → hourly sum (forecasting.ipynb)."""
    if df.empty:
        return pd.DataFrame(columns=list(METER_OUTPUT_COLS))

    work = df.copy()
    work = work.drop_duplicates(subset=["ReadingTimestamp"])
    work["MeterReading"] = pd.to_numeric(work["MeterReading"], errors="coerce")
    work["MeterReading"] = work["MeterReading"].interpolate()
    work["Difference"] = work["MeterReading"].transform(_compute_diff)
    work["meter_reading"] = work["Difference"].transform(_handle_outliers)
    work["meter_reading"] = work["meter_reading"].interpolate(method="linear")
    work["timestamp"] = pd.to_datetime(work["ReadingTimestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp")

    work = work.set_index("timestamp")
    half_hour = work.resample("30min").agg({"meter_reading": "sum"}).reset_index()
    half_hour = half_hour.sort_values("timestamp").set_index("timestamp")
    hourly = (
        half_hour["meter_reading"]
        .resample("h", label="right", closed="right")
        .sum(min_count=1)
        .reset_index()
    )
    hourly["meter"] = meter_id
    hourly["meter_reading"] = pd.to_numeric(hourly["meter_reading"], errors="coerce")
    return hourly[list(METER_OUTPUT_COLS)]


def last_timestamp_by_meter(path: str) -> dict[str, pd.Timestamp]:
    if not os.path.isfile(path):
        return {}
    df = pd.read_csv(path, usecols=["timestamp", "meter"], parse_dates=["timestamp"])
    if df.empty:
        return {}
    return (
        df.groupby("meter", dropna=False)["timestamp"]
        .max()
        .astype("datetime64[ns]")
        .to_dict()
    )


def last_timestamp_by_campus(path: str, *, campus_col: str = "campus") -> dict[str, pd.Timestamp]:
    if not os.path.isfile(path):
        return {}
    try:
        df = pd.read_csv(path, usecols=["timestamp", campus_col], parse_dates=["timestamp"])
    except ValueError:
        return {}
    if df.empty or campus_col not in df.columns:
        return {}
    df[campus_col] = df[campus_col].astype(str).str.strip().str.upper()
    return (
        df.groupby(campus_col, dropna=False)["timestamp"]
        .max()
        .astype("datetime64[ns]")
        .to_dict()
    )


def merge_meter_frames(existing: pd.DataFrame | None, new_rows: pd.DataFrame) -> pd.DataFrame:
    parts = []
    if existing is not None and not existing.empty:
        parts.append(existing[list(METER_OUTPUT_COLS)])
    if new_rows is not None and not new_rows.empty:
        parts.append(new_rows[list(METER_OUTPUT_COLS)])
    if not parts:
        return pd.DataFrame(columns=list(METER_OUTPUT_COLS))
    out = pd.concat(parts, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.floor("h")
    out = out.dropna(subset=["timestamp", "meter"])
    return (
        out.sort_values(["meter", "timestamp"])
        .drop_duplicates(subset=["meter", "timestamp"], keep="last")
        .reset_index(drop=True)
    )


def normalize_solcast_live(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if "accessed_on" in out.columns:
        out["accessed_on"] = pd.to_datetime(out["accessed_on"], errors="coerce")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if "accessed_on" in out.columns and "campus" in out.columns:
        out = out.sort_values("accessed_on").drop_duplicates(
            subset=["timestamp", "campus"], keep="last"
        )
    drop_cols = [c for c in ("accessed_on", "source_file", "timestamp_utc") if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    if "campus" in out.columns:
        out["campus"] = out["campus"].astype(str).str.strip().str.upper()
    out = out.sort_values(["campus", "timestamp"] if "campus" in out.columns else ["timestamp"])
    return out.reset_index(drop=True)


def merge_solcast_frames(
    existing: pd.DataFrame | None,
    new_rows: pd.DataFrame,
    *,
    subset: Iterable[str] | None = None,
) -> pd.DataFrame:
    parts = []
    if existing is not None and not existing.empty:
        parts.append(existing)
    if new_rows is not None and not new_rows.empty:
        parts.append(new_rows)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if subset is None:
        subset = ("timestamp", "campus") if "campus" in out.columns else ("timestamp",)
    return (
        out.sort_values(list(subset))
        .drop_duplicates(subset=list(subset), keep="last")
        .reset_index(drop=True)
    )


def filter_solcast_refresh_window(
    live: pd.DataFrame,
    last_by_campus: dict[str, pd.Timestamp],
    *,
    refresh_days: int | None = None,
) -> pd.DataFrame:
    """Keep Azure rows from (last_timestamp - refresh_days) onward for each campus."""
    if live.empty:
        return live
    refresh_days = config.SOLCAST_REFRESH_DAYS if refresh_days is None else refresh_days
    window = pd.Timedelta(days=int(refresh_days))

    if "campus" not in live.columns:
        cutoff = max(last_by_campus.values()) if last_by_campus else None
        if cutoff is None:
            return live
        refresh_from = cutoff - window
        return live[live["timestamp"] >= refresh_from].copy()

    chunks = []
    for campus, grp in live.groupby("campus", sort=False):
        campus_u = str(campus).strip().upper()
        cutoff = last_by_campus.get(campus_u)
        if cutoff is not None:
            refresh_from = cutoff - window
            grp = grp[grp["timestamp"] >= refresh_from]
        if not grp.empty:
            chunks.append(grp)
    if not chunks:
        return live.iloc[0:0].copy()
    return pd.concat(chunks, ignore_index=True)


def _azure_sas_token() -> str:
    tok = config.env_value("AZURE_STORAGE_SAS_TOKEN")
    if tok:
        return tok
    print(
        "WARNING: AZURE_STORAGE_SAS_TOKEN not set in .env — using dev fallback "
        "(same as forecasting.ipynb). Add the token to PV_analysis/.env.",
        file=sys.stderr,
    )
    return _DEV_AZURE_SAS_FALLBACK


def build_db_engine():
    server = config.env_value("LEAP_DB_SERVER")
    database = config.env_value("LEAP_DB_DATABASE") or "leap"
    username = config.env_value("LEAP_DB_USER")
    password = config.env_value("LEAP_DB_PASSWORD")
    driver = os.environ.get("LEAP_DB_ODBC_DRIVER", "ODBC Driver 18 for SQL Server").strip()

    if not server:
        server = _DEV_DB["LEAP_DB_SERVER"]
        print("WARNING: LEAP_DB_SERVER not in .env — using forecasting.ipynb default.", file=sys.stderr)
    if not username:
        username = _DEV_DB["LEAP_DB_USER"]
        print("WARNING: LEAP_DB_USER not in .env — using forecasting.ipynb default.", file=sys.stderr)
    if not password:
        password = _DEV_DB["LEAP_DB_PASSWORD"]
        print("WARNING: LEAP_DB_PASSWORD not in .env — using forecasting.ipynb default.", file=sys.stderr)

    odbc = (
        f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
        f"UID={username};PWD={password};TrustServerCertificate=yes"
    )
    url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(odbc)
    return create_engine(url)


def meter_ids_for_pipeline(
    *,
    all_realenergy: bool,
    building_keys: list[str] | None,
    engine,
) -> list[str]:
    if all_realenergy:
        q = text(
            "SELECT DISTINCT [MeterKeyString] AS meter "
            "FROM [dbo].[SolarMeterReadings] "
            "WHERE [MeterKeyString] LIKE '%realenergyintotheload%'"
        )
        df = pd.read_sql_query(q, engine)
        return sorted(df["meter"].astype(str).str.strip().unique().tolist())

    keys = building_keys if building_keys else config.analysis_meter_keys()
    return [config.get_building_config(k.strip().lower())["meter_key_full"] for k in keys]


def fetch_meter_sql_chunk(engine, meter_id: str, since_ts: pd.Timestamp | None) -> pd.DataFrame:
    if since_ts is None:
        q = text(
            "SELECT [MeterKeyString], [ReadingTimestamp], [MeterReading] "
            "FROM [dbo].[SolarMeterReadings] "
            "WHERE [MeterKeyString] = :meter "
            "ORDER BY [ReadingTimestamp]"
        )
        return pd.read_sql_query(q, engine, params={"meter": meter_id})

    q = text(
        "SELECT [MeterKeyString], [ReadingTimestamp], [MeterReading] "
        "FROM [dbo].[SolarMeterReadings] "
        "WHERE [MeterKeyString] = :meter AND [ReadingTimestamp] > :since "
        "ORDER BY [ReadingTimestamp]"
    )
    new_df = pd.read_sql_query(q, engine, params={"meter": meter_id, "since": since_ts})
    if new_df.empty:
        return new_df

    q_anchor = text(
        "SELECT TOP 1 [MeterKeyString], [ReadingTimestamp], [MeterReading] "
        "FROM [dbo].[SolarMeterReadings] "
        "WHERE [MeterKeyString] = :meter AND [ReadingTimestamp] <= :since "
        "ORDER BY [ReadingTimestamp] DESC"
    )
    anchor = pd.read_sql_query(q_anchor, engine, params={"meter": meter_id, "since": since_ts})
    if anchor.empty:
        return new_df
    return pd.concat([anchor, new_df], ignore_index=True).drop_duplicates(
        subset=["ReadingTimestamp"], keep="last"
    )


def download_meters(engine, meter_ids: list[str], *, out_path: str, dry_run: bool) -> int:
    existing = None
    last_by_meter = last_timestamp_by_meter(out_path)
    if os.path.isfile(out_path):
        existing = pd.read_csv(out_path, parse_dates=["timestamp"])

    new_parts: list[pd.DataFrame] = []
    n_added = 0

    for meter_id in meter_ids:
        since = last_by_meter.get(meter_id)
        print(f"  {meter_id}  since {since if since is not None else '(full history)'}")

        try:
            raw = fetch_meter_sql_chunk(engine, meter_id, since)
        except Exception as e:
            print(f"    ERROR: {e}", file=sys.stderr)
            continue
        if raw.empty:
            print("    no new rows")
            continue

        hourly = sql_readings_to_hourly_kwh(raw, meter_id)
        if since is not None:
            hourly = hourly[hourly["timestamp"] > since].copy()
        if hourly.empty:
            print("    no new hourly rows after filter")
            continue

        print(f"    +{len(hourly):,} hourly rows  ({hourly['timestamp'].min()} → {hourly['timestamp'].max()})")
        new_parts.append(hourly)
        n_added += len(hourly)

    if dry_run:
        print(f"  [dry-run] would append {n_added:,} rows")
        return n_added

    merged = merge_meter_frames(
        existing, pd.concat(new_parts, ignore_index=True) if new_parts else None
    )
    if merged.empty and existing is None:
        print("  WARNING: no meter data written (empty result).")
        return 0

    merged.to_csv(out_path, index=False)
    print(f"  Wrote {out_path} ({len(merged):,} rows total)")
    return n_added


def _fetch_azure_solcast_frames(container_client, *, blob_filter: str) -> list[pd.DataFrame]:
    """blob_filter: 'live' or 'forecast' (substring match in blob name)."""
    csv_files = [
        b.name
        for b in container_client.list_blobs()
        if blob_filter in b.name and b.name.endswith(".csv")
    ]
    frames = []
    for name in csv_files:
        raw = container_client.get_blob_client(name).download_blob().readall()
        temp = pd.read_csv(StringIO(raw.decode("utf-8", errors="replace")))
        temp["source_file"] = name
        frames.append(temp)
    return frames


def download_weather(*, out_path: str, campus_filter: str | None, dry_run: bool) -> int:
    sas = _azure_sas_token()
    account_url = (
        config.env_value("AZURE_STORAGE_ACCOUNT_URL") or DEFAULT_AZURE_ACCOUNT_URL
    ).rstrip("/")
    container = config.env_value("AZURE_STORAGE_CONTAINER") or DEFAULT_AZURE_CONTAINER

    existing = None
    last_by_campus = last_timestamp_by_campus(out_path)
    if os.path.isfile(out_path):
        existing = pd.read_csv(out_path, parse_dates=["timestamp"])
        print(
            f"  Existing weather: {len(existing):,} rows, "
            f"{existing['timestamp'].min()} → {existing['timestamp'].max()}"
        )
    else:
        print("  No existing raw weather file — will store all Azure weather rows.")

    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as e:
        raise SystemExit("pip install azure-storage-blob") from e

    container_client = BlobServiceClient(
        account_url=account_url, credential=sas
    ).get_container_client(container)

    live_frames = _fetch_azure_solcast_frames(container_client, blob_filter="live")
    if not live_frames:
        raise RuntimeError(f"No live CSV blobs in container {container!r}")

    forecast_frames = _fetch_azure_solcast_frames(container_client, blob_filter="forecast")
    if forecast_frames:
        print(f"  Forecast blobs: {len(forecast_frames)} file(s)")
    else:
        print("  Forecast blobs: none found (live only)")

    combined = normalize_solcast_live(
        pd.concat(live_frames + forecast_frames, ignore_index=True)
    )
    new_rows = filter_solcast_refresh_window(combined, last_by_campus)
    if campus_filter:
        campus_u = campus_filter.strip().upper()
        new_rows = new_rows[new_rows["campus"].astype(str).str.upper() == campus_u].copy()

    if new_rows.empty:
        print("  No weather rows in refresh window.")
        return 0

    refresh_note = (
        f"(refresh last {config.SOLCAST_REFRESH_DAYS} d before file max per campus)"
        if last_by_campus
        else "(initial load)"
    )
    print(
        f"  Weather rows to merge: {len(new_rows):,}  "
        f"{new_rows['timestamp'].min()} → {new_rows['timestamp'].max()}  {refresh_note}"
    )
    if dry_run:
        print("  [dry-run] would append weather rows")
        return len(new_rows)

    subset = ("timestamp", "campus") if "campus" in new_rows.columns else ("timestamp",)
    merge_solcast_frames(existing, new_rows, subset=subset).to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    return len(new_rows)


def _as_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def hsu_surface_tilt_deg() -> float:
    try:
        return float(config.surface_geometry_for_meter_key("library")[0])
    except Exception:
        return float(config.SURFACE_TILT_DEG)


def load_solcast_hourly_rainfall(path: str, *, campus: str | None = None) -> pd.Series:
    """Hourly accumulated rainfall (mm) from Solcast precipitation_rate, UTC index."""
    df = pd.read_csv(path)
    if campus and "campus" in df.columns:
        campus_u = campus.strip().upper()
        df = df[df["campus"].astype(str).str.strip().str.upper() == campus_u]
    if df.empty:
        raise ValueError(f"No Solcast rows in {path}" + (f" for campus {campus}" if campus else ""))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    step_hours = df.index.to_series().diff().dropna().median().total_seconds() / 3600.0
    if not np.isfinite(step_hours) or step_hours <= 0:
        step_hours = 0.5
    df["rain_mm"] = df["precipitation_rate"].fillna(0.0) * step_hours
    return df["rain_mm"].resample("1h").sum()


def load_merged_solcast_hourly_rainfall(*, campus: str | None = None) -> pd.Series:
    """Rain from archive ``solcast_df.csv`` (2020+) plus live ``solcast_df_2020_2025.csv``."""
    parts: list[pd.Series] = []
    archive = raw_solcast_archive_path()
    live = raw_solcast_path()
    if os.path.isfile(archive):
        parts.append(load_solcast_hourly_rainfall(archive, campus=campus))
        print(f"  Rain archive: {archive}  ({parts[-1].index.min()} → {parts[-1].index.max()})")
    if os.path.isfile(live):
        live_rain = load_solcast_hourly_rainfall(live, campus=campus)
        parts.append(live_rain)
        print(f"  Rain live file: {live}  ({live_rain.index.min()} → {live_rain.index.max()})")
    if not parts:
        raise FileNotFoundError(
            f"No Solcast rain source in {DATA_RAW_DIR} "
            f"(expected {SOLCAST_ARCHIVE} and/or {SOLCAST_RAW_PRIMARY})"
        )
    rain = pd.concat(parts).sort_index()
    rain.index = pd.Index([_as_utc(t) for t in rain.index])
    return rain[~rain.index.duplicated(keep="last")]


def fetch_openmeteo_pm(
    lat: float,
    lon: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """PM2.5 / PM10 (µg/m³) from Open-Meteo Air Quality API, hourly UTC."""
    import requests

    start = _as_utc(start)
    end = _as_utc(end)
    if start > end:
        return pd.DataFrame(columns=["pm2_5", "pm10"])

    frames: list[pd.DataFrame] = []
    for yr in range(start.year, end.year + 1):
        s = max(pd.Timestamp(f"{yr}-01-01", tz="UTC"), start)
        e = min(pd.Timestamp(f"{yr}-12-31 23:00:00", tz="UTC"), end)
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm2_5,pm10",
            "start_date": s.strftime("%Y-%m-%d"),
            "end_date": e.strftime("%Y-%m-%d"),
            "timezone": "GMT",
            "domains": "cams_global",
        }
        r = requests.get(OPENMETEO_AIR_QUALITY_URL, params=params, timeout=120)
        r.raise_for_status()
        h = r.json().get("hourly", {})
        if not h.get("time"):
            continue
        chunk = pd.DataFrame(h)
        chunk["time"] = pd.to_datetime(chunk["time"], utc=True)
        chunk = chunk.set_index("time")
        frames.append(chunk[["pm2_5", "pm10"]])
        print(f"    PM {yr}: {len(chunk):,} hourly rows")

    if not frames:
        return pd.DataFrame(columns=["pm2_5", "pm10"])
    pm = pd.concat(frames).sort_index()
    return pm[~pm.index.duplicated(keep="first")]


def extend_pm_with_climatology(
    pm: pd.DataFrame,
    full_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, int]:
    """
    Reindex PM to ``full_index`` and fill gaps with month-of-year × hour-of-day medians
    from observed Open-Meteo CAMS rows (needed before ~Aug 2022 when PM is unavailable).
    """
    out = pm.reindex(full_index)
    obs = out.dropna(subset=["pm2_5", "pm10"])
    if obs.empty:
        return out, 0

    missing = out["pm2_5"].isna()
    n_missing = int(missing.sum())
    if n_missing == 0:
        return out, 0

    cal = obs.copy()
    cal["_month"] = cal.index.month
    cal["_hour"] = cal.index.hour
    medians = cal.groupby(["_month", "_hour"], observed=True)[["pm2_5", "pm10"]].median()
    global_med = obs[["pm2_5", "pm10"]].median()

    miss_idx = out.index[missing]
    lookup = pd.DataFrame({"_month": miss_idx.month, "_hour": miss_idx.hour}, index=miss_idx)
    filled = lookup.join(medians, on=["_month", "_hour"])
    out.loc[missing, "pm2_5"] = filled["pm2_5"].fillna(global_med["pm2_5"]).to_numpy()
    out.loc[missing, "pm10"] = filled["pm10"].fillna(global_med["pm10"]).to_numpy()
    return out, n_missing


def load_existing_hsu_soiling(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty:
        return None
    df.index = pd.Index([_as_utc(t) for t in df.index], name=df.index.name)
    return df.sort_index()


def download_hsu_soiling(
    *,
    out_path: str,
    campus_filter: str | None,
    dry_run: bool,
    pm_climatology: bool = True,
) -> int:
    """Incremental Open-Meteo PM fetch + full pvlib HSU recompute on aligned rain/PM."""
    campus = (campus_filter or HSU_DEFAULT_CAMPUS).strip().upper()
    try:
        rain_h = load_merged_solcast_hourly_rainfall(campus=campus)
    except (FileNotFoundError, ValueError) as e:
        print(f"  Skip HSU soiling: {e}")
        return 0

    try:
        import pvlib
    except ImportError as e:
        raise SystemExit("HSU soiling requires pvlib: pip install pvlib") from e

    period_start = _as_utc(rain_h.index.min())
    period_end = min(_as_utc(pd.Timestamp.now(tz="UTC")).floor("h"), _as_utc(rain_h.index.max()))
    print(
        f"  Merged rain span ({campus}): {period_start} → {period_end}  "
        f"(HSU through {period_end})"
    )

    existing = load_existing_hsu_soiling(out_path)
    if existing is None:
        legacy_path = os.path.join(BASE, HSU_SOILING_RAW)
        if os.path.isfile(legacy_path):
            existing = load_existing_hsu_soiling(legacy_path)
            if existing is not None:
                print(f"  Using legacy HSU file: {legacy_path} ({len(existing):,} rows)")

    prev_len = len(existing) if existing is not None else 0
    pm_parts: list[pd.DataFrame] = []

    if existing is not None:
        first_hsu = _as_utc(existing.index.min())
        last_hsu = _as_utc(existing.index.max())
        print(f"  Existing HSU file: {prev_len:,} rows  {first_hsu} → {last_hsu}")
        if first_hsu > period_start:
            backfill_end = first_hsu - pd.Timedelta(hours=1)
            print(
                f"  Backfilling PM before existing file: "
                f"{period_start.date()} → {backfill_end.date()}"
            )
            if not dry_run:
                pm_parts.append(
                    fetch_openmeteo_pm(
                        HSU_AIR_QUALITY_LAT,
                        HSU_AIR_QUALITY_LON,
                        period_start,
                        backfill_end,
                    )
                )
        pm_fetch_start = last_hsu + pd.Timedelta(hours=1)
    else:
        last_hsu = None
        pm_fetch_start = period_start
        print("  No existing HSU file — initial PM download from merged rain start")

    pm_fetch_end = period_end
    if pm_fetch_start <= pm_fetch_end:
        print(
            f"  Fetching air quality PM: {pm_fetch_start.date()} → {pm_fetch_end.date()}"
        )
        if not dry_run:
            pm_parts.append(
                fetch_openmeteo_pm(
                    HSU_AIR_QUALITY_LAT,
                    HSU_AIR_QUALITY_LON,
                    pm_fetch_start,
                    pm_fetch_end,
                )
            )
        else:
            print("  [dry-run] would fetch Open-Meteo PM")
            return 0
    elif existing is not None:
        print(f"  PM already current through {last_hsu}")

    if dry_run:
        return 0

    new_pm = pd.concat(pm_parts).sort_index() if pm_parts else pd.DataFrame(columns=["pm2_5", "pm10"])
    if not new_pm.empty:
        print(f"  Downloaded {len(new_pm):,} PM hourly rows")

    if existing is not None and {"pm2_5", "pm10"}.issubset(existing.columns):
        pm = pd.concat([existing[["pm2_5", "pm10"]], new_pm]).sort_index()
    elif not new_pm.empty:
        pm = new_pm
    else:
        pm = fetch_openmeteo_pm(
            HSU_AIR_QUALITY_LAT,
            HSU_AIR_QUALITY_LON,
            period_start,
            period_end,
        )

    pm = pm[~pm.index.duplicated(keep="last")]
    if pm.empty:
        print("  WARNING: no PM data available for HSU soiling.")
        return 0

    align_end = min(period_end, rain_h.index.max())
    rain_slice = rain_h.loc[period_start:align_end]
    if rain_slice.empty:
        print("  WARNING: no Solcast rain rows in HSU period.")
        return 0

    if pm_climatology:
        pm_aligned, n_clim = extend_pm_with_climatology(pm, rain_slice.index)
        if n_clim:
            print(
                f"  PM climatology fill: {n_clim:,} hours before Open-Meteo CAMS "
                f"(~{OPENMETEO_CAMS_PM_START.date()}) — rain-only deposition proxy"
            )
        df = pd.DataFrame({"rainfall": rain_slice}).join(pm_aligned, how="left")
    else:
        pm_aligned = pm.loc[period_start:align_end]
        df = pd.DataFrame({"rainfall": rain_slice}).join(pm_aligned, how="inner")

    if df.empty:
        print("  WARNING: no overlapping rain + PM rows for HSU.")
        return 0

    df["pm2_5"] = df["pm2_5"].interpolate(limit=6)
    df["pm10"] = df["pm10"].interpolate(limit=6)
    df = df.dropna(subset=["pm2_5", "pm10"])
    df["rainfall"] = df["rainfall"].fillna(0.0)

    surface_tilt = hsu_surface_tilt_deg()
    soiling_ratio = pvlib.soiling.hsu(
        rainfall=df["rainfall"],
        cleaning_threshold=HSU_CLEANING_THRESHOLD_MM,
        surface_tilt=surface_tilt,
        pm2_5=df["pm2_5"] * PM_UGM3_TO_GM3,
        pm10=df["pm10"] * PM_UGM3_TO_GM3,
        depo_veloc=None,
        rain_accum_period=pd.Timedelta("1h"),
    )

    out = df.copy()
    out["soiling_ratio"] = soiling_ratio
    out["soiling_loss_pct"] = (1.0 - out["soiling_ratio"]) * 100.0
    out.to_csv(out_path)
    publish_hsu_for_viz(out_path)

    added = len(out) - prev_len
    daily = out["soiling_ratio"].resample("1D").mean()
    print(
        f"  Wrote {out_path} ({len(out):,} rows, +{max(added, 0):,} vs prior)  "
        f"tilt={surface_tilt}°  mean SR={out['soiling_ratio'].mean():.4f}"
    )
    if not daily.empty:
        print(f"  Worst daily SR: {daily.min():.4f} on {daily.idxmin().date()}")
    return max(added, 0)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Incremental download: SQL meters + Azure Solcast + HSU soiling."
    )
    ap.add_argument("--weather-only", action="store_true")
    ap.add_argument("--meters-only", action="store_true")
    ap.add_argument("--soiling-only", action="store_true")
    ap.add_argument("--no-soiling", action="store_true")
    ap.add_argument(
        "--campus",
        default=None,
        help="Only append weather for this campus (e.g. BUNDOORA). Default: all campuses in blobs.",
    )
    ap.add_argument("--all-realenergy-meters", action="store_true")
    ap.add_argument("--building-key", action="append", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--no-pm-climatology",
        action="store_true",
        help="HSU soiling: only use measured Open-Meteo PM (~Aug 2022+); skip pre-CAMS hours.",
    )
    return ap.parse_args()


def main() -> int:
    loaded = config.load_credentials()
    if loaded:
        print("Loaded credentials from:", ", ".join(loaded))

    args = parse_args()
    mode_flags = sum(
        bool(x) for x in (args.weather_only, args.meters_only, args.soiling_only)
    )
    if mode_flags > 1:
        print(
            "ERROR: use at most one of --weather-only / --meters-only / --soiling-only",
            file=sys.stderr,
        )
        return 1

    ensure_raw_dir()
    meter_path = raw_meter_path()
    weather_path = raw_solcast_path()
    hsu_path = raw_hsu_soiling_path()

    print("PV_analysis incremental download")
    print(f"  data_raw/: {DATA_RAW_DIR}")

    run_weather = not args.meters_only and not args.soiling_only
    run_soiling = not args.meters_only and not args.weather_only and not args.no_soiling
    if args.soiling_only:
        run_weather = False
        run_soiling = True

    if run_weather:
        print("\n[Weather] Azure live Solcast →", weather_path)
        try:
            download_weather(out_path=weather_path, campus_filter=args.campus, dry_run=args.dry_run)
        except SystemExit as e:
            print(e, file=sys.stderr)
            return 1

    if run_soiling:
        print("\n[HSU soiling] Open-Meteo PM + pvlib ->", hsu_path)
        try:
            download_hsu_soiling(
                out_path=hsu_path,
                campus_filter=args.campus,
                dry_run=args.dry_run,
                pm_climatology=not args.no_pm_climatology,
            )
        except SystemExit as e:
            print(e, file=sys.stderr)
            return 1

    if not args.weather_only and not args.soiling_only:
        print("\n[Meters] SQL Server →", meter_path)
        engine = build_db_engine()
        try:
            meter_ids = meter_ids_for_pipeline(
                all_realenergy=args.all_realenergy_meters,
                building_keys=args.building_key,
                engine=engine,
            )
            print(f"  Meters ({len(meter_ids)})")
            download_meters(engine, meter_ids, out_path=meter_path, dry_run=args.dry_run)
        finally:
            engine.dispose()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
