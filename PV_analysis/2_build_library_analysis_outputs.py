"""
Build library-only analysis CSVs from:
  • Cleaned meter CSV — actual (+ optional legacy expected_power)
  • Expected hourly kWh: **live** PVLib from Solcast (default — uses ``panel_data`` DC/AC for ``--building-key``).
    Optional: ``--precomputed`` loads ``data_pvlib/expected_power_pvlib_cleaned_v2.csv`` only if you built it
    with matching ``python 3_expected_power_pvlib.py --building <same key>``.
    Old precomputed CSVs may still reflect legacy 100 kWp / 95 kW runs (without ``--building``).
  • panel_data.csv — DC/AC nameplate via get_building_config() (metadata columns on output)

Outputs (under PV_analysis/data_for_viz/ by default):
  1. hourly_library_master.csv
  2. daily_library_metrics.csv
  3. library_kpis_summary.csv

Usage:
    cd PV_analysis
    python 2_build_library_analysis_outputs.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

import config


def _load_precomputed_expected(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    if "expected_kwh" not in df.columns:
        raise ValueError(f"Precomputed file must include 'expected_kwh': {path}")
    return df


def _warn_if_precomputed_wrong_scale(
    pvlib_hourly: pd.DataFrame, bldg: dict, building_key: str
) -> None:
    """Detect precomputed CSV built with ``3_expected_power_pvlib.py`` **without**
    ``--building <key>``. Stale CSVs may still be from older defaults or without ``--building``.
    """
    ac_kw = bldg.get("inverter_ac_kw")
    if ac_kw is None or not np.isfinite(float(ac_kw)) or float(ac_kw) <= 0:
        return
    mx = float(pd.to_numeric(pvlib_hourly["expected_kwh"], errors="coerce").max())
    cap = float(ac_kw)
    if cap > 150 and mx < 0.35 * cap:
        bk = building_key.strip().lower()
        print(
            f"\n*** WARNING: Precomputed expected_kwh peaks at ~{mx:.0f} kWh/h but "
            f"{bldg.get('building_name', bk)} has ~{cap:.0f} kW AC.\n"
            "    This usually means the CSV was not built with ``--building library`` (or is an old 100 kWp / 95 kW file), "
            f"not panel_data for ``--building {bk}``.\n"
            "    Fix:  python 2_build_library_analysis_outputs.py --compute-pvlib\n"
            f"    Or:   python 3_expected_power_pvlib.py --building {bk}\n"
        )


def _import_pvlib_module():
    """Load ``3_expected_power_pvlib.py`` (filename cannot be a normal import)."""
    mod_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3_expected_power_pvlib.py")
    spec = importlib.util.spec_from_file_location("expected_power_pvlib_internal", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PVLib module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_library_meter(path: str, meter_key_full: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[df["meter"].astype(str).str.strip() == meter_key_full].copy()
    if df.empty:
        raise ValueError(f"No rows for meter '{meter_key_full}' in {path}")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")
    out = pd.DataFrame({"timestamp": df["timestamp"]})
    if "meter_reading" not in df.columns:
        raise ValueError(f"Column 'meter_reading' not found in {path}")
    out["actual_kwh"] = pd.to_numeric(df["meter_reading"], errors="coerce")
    if "expected_power" in df.columns:
        out["legacy_expected_kwh"] = pd.to_numeric(df["expected_power"], errors="coerce")
    for col in ("outage_flag", "analysis_valid", "data_version"):
        if col in df.columns:
            out[col] = df[col]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Library master CSVs: meter + PVLib + daily/KPI")
    ap.add_argument("--meter-csv", default=config.METER_READINGS)
    ap.add_argument("--weather-csv", default=config.SOLCAST_CLEANED_V2)
    ap.add_argument(
        "--expected-csv",
        default=None,
        help="Precomputed PVLib hourly CSV (timestamp, expected_kwh). Must match --building-key.",
    )
    ap.add_argument(
        "--precomputed",
        action="store_true",
        help=f"Shorthand for --expected-csv {os.path.basename(config.EXPECTED_PVLIB_CLEANED_V2)} "
        "(only if that file was built with the same --building key).",
    )
    ap.add_argument(
        "--compute-pvlib",
        action="store_true",
        help="Force live PVLib even if --expected-csv / --precomputed is set.",
    )
    ap.add_argument("--out-dir", default=config.LIBRARY_ANALYSIS_DIR)
    ap.add_argument("--building-key", default="library")
    args = ap.parse_args()

    if not os.path.isfile(args.meter_csv):
        print(f"ERROR: Meter file not found: {args.meter_csv}")
        sys.exit(1)

    if not os.path.isfile(config.PANEL_DATA):
        print(f"ERROR: panel_data.csv not found at:\n  {config.PANEL_DATA}")
        sys.exit(1)
    print(f"panel_data: {os.path.abspath(config.PANEL_DATA)}")

    bldg = config.get_building_config(args.building_key)
    meter_key = bldg["meter_key_full"]

    print(f"Building: {bldg['building_name']}")
    print(f"Meter:    {meter_key}")
    print(
        f"DC:       {bldg['system_kwp']:.2f} kWp ({bldg['system_dc_w']:,.0f} W) | "
        f"AC: {bldg['inverter_ac_kw']} kW"
    )
    if args.building_key.strip().lower() == "library" and bldg["system_kwp"] < 300:
        print(
            "WARNING: Library is ~384 kWp in catalogue; DC below 300 suggests wrong panel_data row or file."
        )

    print("Loading meter …")
    meter = _load_library_meter(args.meter_csv, meter_key)

    expected_path = args.expected_csv
    if args.precomputed and expected_path is None:
        expected_path = config.EXPECTED_PVLIB_CLEANED_V2

    if expected_path and not args.compute_pvlib:
        if not os.path.isfile(expected_path):
            print(f"ERROR: Precomputed expected file not found: {expected_path}")
            sys.exit(1)
        print(f"Loading precomputed PVLib expected: {expected_path}")
        pvlib_hourly = _load_precomputed_expected(expected_path)
        _warn_if_precomputed_wrong_scale(pvlib_hourly, bldg, args.building_key)
    else:
        if not os.path.isfile(args.weather_csv):
            print(
                f"ERROR: Weather file not found: {args.weather_csv}\n"
                "Pass --expected-csv, or place precomputed data at:\n"
                f"  {config.EXPECTED_PVLIB_CLEANED_V2}\n"
                "or use --weather-csv with --compute-pvlib."
            )
            sys.exit(1)
        print("Loading weather & running PVLib (3_expected_power_pvlib.py) …")
        pvlib_mod = _import_pvlib_module()
        weather = pvlib_mod._load_cleaned(args.weather_csv)
        pvlib_hourly = pvlib_mod.build_expected(
            weather,
            system_dc_w=bldg["system_dc_w"],
            inverter_ac_w=bldg["inverter_ac_w"],
        )

    merged = meter.merge(pvlib_hourly, on="timestamp", how="left")
    n_exp = int(merged["expected_kwh"].notna().sum())
    print(f"Merge: {n_exp} / {len(merged)} hourly rows have PVLib expected_kwh (rest NaN = timestamp mismatch).")

    merged["building_name"] = bldg["building_name"]
    merged["meter_id"] = meter_key
    merged["system_kwp"] = bldg["system_kwp"]
    merged["inverter_ac_kw"] = (
        bldg["inverter_ac_kw"] if bldg["inverter_ac_kw"] is not None else np.nan
    )
    merged["panel_type"] = bldg["panel_type"]
    merged["campus"] = bldg.get("campus", "Bundoora")

    os.makedirs(args.out_dir, exist_ok=True)
    hourly_path = os.path.join(args.out_dir, "hourly_library_master.csv")
    merged.to_csv(hourly_path, index=False)
    print(f"Wrote {hourly_path}  ({len(merged)} rows)")

    d = merged.copy()
    d["date"] = pd.to_datetime(d["timestamp"]).dt.normalize()
    g = d.groupby("date", dropna=False)
    daily = pd.DataFrame({
        "date": g["actual_kwh"].sum().index,
        "actual_kwh_sum": g["actual_kwh"].sum().values,
        "pvlib_expected_kwh_sum": g["expected_kwh"].sum().values,
    })
    if "legacy_expected_kwh" in d.columns:
        daily["legacy_expected_kwh_sum"] = g["legacy_expected_kwh"].sum().values
    daily["diff_pvlib_minus_actual_kwh"] = (
        daily["pvlib_expected_kwh_sum"] - daily["actual_kwh_sum"]
    )
    eps = 1e-6
    daily["PR_vs_pvlib"] = np.where(
        daily["pvlib_expected_kwh_sum"] > eps,
        daily["actual_kwh_sum"] / daily["pvlib_expected_kwh_sum"],
        np.nan,
    )
    if "legacy_expected_kwh_sum" in daily.columns:
        daily["PR_vs_legacy"] = np.where(
            daily["legacy_expected_kwh_sum"] > eps,
            daily["actual_kwh_sum"] / daily["legacy_expected_kwh_sum"],
            np.nan,
        )

    daily_path = os.path.join(args.out_dir, "daily_library_metrics.csv")
    daily.to_csv(daily_path, index=False)
    print(f"Wrote {daily_path}  ({len(daily)} days)")

    both = merged.dropna(subset=["actual_kwh", "expected_kwh"])
    sub = both.loc[both["actual_kwh"].notna() & both["expected_kwh"].notna()]
    total_act = float(sub["actual_kwh"].sum())
    total_pv = float(sub["expected_kwh"].sum())
    ratio = total_act / total_pv if total_pv > eps else np.nan
    corr = sub["actual_kwh"].corr(sub["expected_kwh"]) if len(sub) > 1 else np.nan

    used_weather = os.path.abspath(args.weather_csv)
    if expected_path and not args.compute_pvlib:
        used_weather = ""

    kpi = {
        "meter_id": meter_key,
        "building_name": bldg["building_name"],
        "system_kwp": bldg["system_kwp"],
        "inverter_ac_kw": bldg["inverter_ac_kw"],
        "panel_data_csv": os.path.abspath(config.PANEL_DATA),
        "meter_csv": os.path.abspath(args.meter_csv),
        "weather_csv": used_weather,
        "expected_source": (
            "precomputed:" + os.path.abspath(expected_path)
            if expected_path and not args.compute_pvlib
            else "pvlib_live:" + used_weather
        ),
        "hourly_rows_meter": len(meter),
        "hourly_rows_merged": len(merged),
        "hours_with_pvlib": int(sub["expected_kwh"].notna().sum()),
        "first_timestamp": str(merged["timestamp"].min()),
        "last_timestamp": str(merged["timestamp"].max()),
        "sum_actual_kwh_overlap": total_act,
        "sum_pvlib_expected_kwh_overlap": total_pv,
        "actual_over_expected_ratio": ratio,
        "correlation_actual_vs_pvlib": corr,
    }
    if "legacy_expected_kwh" in merged.columns:
        leg = merged.dropna(subset=["actual_kwh", "legacy_expected_kwh"])
        t_leg = float(leg["legacy_expected_kwh"].sum())
        kpi["sum_legacy_expected_kwh_overlap"] = t_leg
        kpi["actual_over_legacy_ratio"] = (
            float(leg["actual_kwh"].sum()) / t_leg if t_leg > eps else np.nan
        )

    kpi_path = os.path.join(args.out_dir, "library_kpis_summary.csv")
    pd.DataFrame([kpi]).to_csv(kpi_path, index=False)
    print(f"Wrote {kpi_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
