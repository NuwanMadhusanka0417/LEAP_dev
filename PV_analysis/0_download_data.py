#!/usr/bin/env python3
"""
Incremental download of meter readings (SQL Server) and Solcast weather (Azure blobs).

Appends only rows newer than the last timestamp in existing raw CSVs under data_raw/:
  - SolarMeterReadings1hour_2020_2025.csv  (columns: timestamp, meter, meter_reading)
  - solcast_df_2020_2025.csv

Credentials: copy ``.env.example`` → ``.env`` in this folder (auto-loaded), or export env vars.

Usage (from PV_analysis/)::
  python 0_download_data.py
  python 0_download_data.py --weather-only
  python 0_download_data.py --meters-only
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

METER_RAW_PRIMARY = "SolarMeterReadings1hour_2020_2025.csv"
SOLCAST_RAW_PRIMARY = "solcast_df_2020_2025.csv"
METER_OUTPUT_COLS = ("timestamp", "meter", "meter_reading")

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


def filter_solcast_after_last(
    live: pd.DataFrame,
    last_by_campus: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    if live.empty:
        return live
    if "campus" not in live.columns:
        cutoff = max(last_by_campus.values()) if last_by_campus else None
        if cutoff is None:
            return live
        return live[live["timestamp"] > cutoff].copy()

    chunks = []
    for campus, grp in live.groupby("campus", sort=False):
        campus_u = str(campus).strip().upper()
        cutoff = last_by_campus.get(campus_u)
        if cutoff is not None:
            grp = grp[grp["timestamp"] > cutoff]
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
    new_rows = filter_solcast_after_last(combined, last_by_campus)
    if campus_filter:
        campus_u = campus_filter.strip().upper()
        new_rows = new_rows[new_rows["campus"].astype(str).str.upper() == campus_u].copy()

    if new_rows.empty:
        print("  No new weather rows after last timestamp.")
        return 0

    print(
        f"  New weather rows: {len(new_rows):,}  "
        f"{new_rows['timestamp'].min()} → {new_rows['timestamp'].max()}"
    )
    if dry_run:
        print("  [dry-run] would append weather rows")
        return len(new_rows)

    subset = ("timestamp", "campus") if "campus" in new_rows.columns else ("timestamp",)
    merge_solcast_frames(existing, new_rows, subset=subset).to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    return len(new_rows)


def parse_args():
    ap = argparse.ArgumentParser(description="Incremental download: SQL meters + Azure Solcast.")
    ap.add_argument("--weather-only", action="store_true")
    ap.add_argument("--meters-only", action="store_true")
    ap.add_argument(
        "--campus",
        default=None,
        help="Only append weather for this campus (e.g. BUNDOORA). Default: all campuses in blobs.",
    )
    ap.add_argument("--all-realenergy-meters", action="store_true")
    ap.add_argument("--building-key", action="append", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    loaded = config.load_credentials()
    if loaded:
        print("Loaded credentials from:", ", ".join(loaded))

    args = parse_args()
    if args.weather_only and args.meters_only:
        print("ERROR: use at most one of --weather-only / --meters-only", file=sys.stderr)
        return 1

    ensure_raw_dir()
    meter_path = raw_meter_path()
    weather_path = raw_solcast_path()

    print("PV_analysis incremental download")
    print(f"  data_raw/: {DATA_RAW_DIR}")

    if not args.meters_only:
        print("\n[Weather] Azure live Solcast →", weather_path)
        try:
            download_weather(out_path=weather_path, campus_filter=args.campus, dry_run=args.dry_run)
        except SystemExit as e:
            print(e, file=sys.stderr)
            return 1

    if not args.weather_only:
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
