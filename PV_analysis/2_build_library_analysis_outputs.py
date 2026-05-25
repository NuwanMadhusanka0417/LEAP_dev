"""
Build per-meter analysis CSVs for all keys in ``config._BUILDING_PVLIB_GEOMETRY``.

  • Cleaned meter CSV — actual (+ optional legacy expected_power)
  • Expected hourly kWh: live PVLib from Solcast (per-building DC/AC + tilt/azimuth from config)
  • panel_data.csv — nameplate via get_building_config()

Outputs (under ``data_for_viz/`` by default), **per meter key** ``<key>``:
  hourly_<key>_master.csv, daily_<key>_metrics.csv
Plus combined: sites_kpis_summary.csv

Legacy alias: ``hourly_library_master.csv`` when ``library`` is processed.

Usage:
    cd PV_analysis
    python 2_build_library_analysis_outputs.py
    python 2_build_library_analysis_outputs.py --building-key dmw   # one meter only
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


def _precomputed_path_for_key(building_key: str) -> str:
    """Per-building PVLib CSV under data_pvlib/; library may use legacy v2 filename."""
    key = building_key.strip().lower()
    legacy = config.EXPECTED_PVLIB_CLEANED_V2
    if key == "library" and os.path.isfile(legacy):
        return legacy
    return os.path.join(config.BASE, "data_pvlib", f"expected_power_pvlib_{key}.csv")


def _warn_if_precomputed_wrong_scale(
    pvlib_hourly: pd.DataFrame, bldg: dict, building_key: str
) -> None:
    ac_kw = bldg.get("inverter_ac_kw")
    if ac_kw is None or not np.isfinite(float(ac_kw)) or float(ac_kw) <= 0:
        return
    mx = float(pd.to_numeric(pvlib_hourly["expected_kwh"], errors="coerce").max())
    cap = float(ac_kw)
    if cap > 150 and mx < 0.35 * cap:
        bk = building_key.strip().lower()
        print(
            f"\n*** WARNING [{bk}]: Precomputed expected_kwh peaks at ~{mx:.0f} kWh/h but "
            f"{bldg.get('building_name', bk)} has ~{cap:.0f} kW AC.\n"
            "    Re-run with --compute-pvlib or build with matching --building.\n"
        )


def _import_pvlib_module():
    mod_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3_expected_power_pvlib.py")
    spec = importlib.util.spec_from_file_location("expected_power_pvlib_internal", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PVLib module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_meter_slice(path: str, meter_key_full: str) -> pd.DataFrame:
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


def _resolve_pvlib_hourly(
    building_key: str,
    bldg: dict,
    *,
    weather_csv: str,
    expected_csv: str | None,
    use_precomputed: bool,
    force_compute: bool,
    weather_df: pd.DataFrame | None,
    pvlib_mod,
) -> tuple[pd.DataFrame, str]:
    """Return (pvlib_hourly, expected_source_label)."""
    key = building_key.strip().lower()
    pre_path = expected_csv or (_precomputed_path_for_key(key) if use_precomputed else None)

    if pre_path and not force_compute and os.path.isfile(pre_path):
        print(f"  [{key}] Loading precomputed: {pre_path}")
        pvlib_hourly = _load_precomputed_expected(pre_path)
        _warn_if_precomputed_wrong_scale(pvlib_hourly, bldg, key)
        return pvlib_hourly, "precomputed:" + os.path.abspath(pre_path)

    if weather_df is None:
        if not os.path.isfile(weather_csv):
            raise FileNotFoundError(f"Weather file not found: {weather_csv}")
        if pvlib_mod is None:
            pvlib_mod = _import_pvlib_module()
        weather_df = pvlib_mod._load_cleaned(weather_csv)

    print(f"  [{key}] PVLib build_expected (tilt={bldg['surface_tilt_deg']:.0f}°, az={bldg['surface_azimuth_deg']:.0f}°) …")
    pvlib_hourly = pvlib_mod.build_expected(
        weather_df,
        system_dc_w=bldg["system_dc_w"],
        inverter_ac_w=bldg["inverter_ac_w"],
        surface_tilt_deg=bldg["surface_tilt_deg"],
        surface_azimuth_deg=bldg["surface_azimuth_deg"],
    )
    return pvlib_hourly, "pvlib_live:" + os.path.abspath(weather_csv)


def process_building(
    building_key: str,
    *,
    meter_csv: str,
    weather_csv: str,
    out_dir: str,
    expected_csv: str | None,
    use_precomputed: bool,
    force_compute: bool,
    weather_df: pd.DataFrame | None,
    pvlib_mod,
) -> dict:
    key = building_key.strip().lower()
    bldg = config.get_building_config(key)
    meter_key = bldg["meter_key_full"]

    print(f"\n{'=' * 55}")
    print(f"  [{key}] {bldg['building_name']}")
    print(f"  Meter: {meter_key}")
    print(
        f"  DC: {bldg['system_kwp']:.2f} kWp | AC: {bldg['inverter_ac_kw']} kW | "
        f"tilt/azimuth: {bldg['surface_tilt_deg']:.0f}° / {bldg['surface_azimuth_deg']:.0f}°"
    )
    if key == "library" and bldg["system_kwp"] < 300:
        print("  WARNING: Library catalogue ~384 kWp; DC below 300 — check panel_data.")

    meter = _load_meter_slice(meter_csv, meter_key)
    pvlib_hourly, expected_source = _resolve_pvlib_hourly(
        key,
        bldg,
        weather_csv=weather_csv,
        expected_csv=expected_csv,
        use_precomputed=use_precomputed,
        force_compute=force_compute,
        weather_df=weather_df,
        pvlib_mod=pvlib_mod,
    )

    merged = meter.merge(pvlib_hourly, on="timestamp", how="left")
    n_exp = int(merged["expected_kwh"].notna().sum())
    print(f"  Merge: {n_exp} / {len(merged)} rows with expected_kwh")

    merged["building_key"] = key
    merged["building_name"] = bldg["building_name"]
    merged["meter_id"] = meter_key
    merged["system_kwp"] = bldg["system_kwp"]
    merged["inverter_ac_kw"] = (
        bldg["inverter_ac_kw"] if bldg["inverter_ac_kw"] is not None else np.nan
    )
    merged["panel_type"] = bldg["panel_type"]
    merged["campus"] = bldg.get("campus", "Bundoora")

    os.makedirs(out_dir, exist_ok=True)
    hourly_path = os.path.join(out_dir, f"hourly_{key}_master.csv")
    merged.to_csv(hourly_path, index=False)
    print(f"  Wrote {hourly_path} ({len(merged)} rows)")

    if key == "library":
        legacy_path = os.path.join(out_dir, "hourly_library_master.csv")
        merged.to_csv(legacy_path, index=False)
        print(f"  Wrote {legacy_path} (dashboard legacy name)")

    d = merged.copy()
    d["date"] = pd.to_datetime(d["timestamp"]).dt.normalize()
    g = d.groupby("date", dropna=False)
    daily = pd.DataFrame({
        "date": g["actual_kwh"].sum().index,
        "building_key": key,
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

    daily_path = os.path.join(out_dir, f"daily_{key}_metrics.csv")
    daily.to_csv(daily_path, index=False)
    print(f"  Wrote {daily_path} ({len(daily)} days)")

    if key == "library":
        daily.to_csv(os.path.join(out_dir, "daily_library_metrics.csv"), index=False)

    both = merged.dropna(subset=["actual_kwh", "expected_kwh"])
    sub = both.loc[both["actual_kwh"].notna() & both["expected_kwh"].notna()]
    total_act = float(sub["actual_kwh"].sum())
    total_pv = float(sub["expected_kwh"].sum())
    ratio = total_act / total_pv if total_pv > eps else np.nan
    corr = sub["actual_kwh"].corr(sub["expected_kwh"]) if len(sub) > 1 else np.nan

    used_weather = os.path.abspath(weather_csv)
    if expected_source.startswith("precomputed:"):
        used_weather = ""

    kpi = {
        "building_key": key,
        "meter_id": meter_key,
        "building_name": bldg["building_name"],
        "system_kwp": bldg["system_kwp"],
        "inverter_ac_kw": bldg["inverter_ac_kw"],
        "surface_tilt_deg": bldg["surface_tilt_deg"],
        "surface_azimuth_deg": bldg["surface_azimuth_deg"],
        "panel_data_csv": os.path.abspath(config.PANEL_DATA),
        "meter_csv": os.path.abspath(meter_csv),
        "weather_csv": used_weather,
        "expected_source": expected_source,
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
    return kpi


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build hourly/daily/KPI CSVs for meters in config._BUILDING_PVLIB_GEOMETRY"
    )
    ap.add_argument("--meter-csv", default=config.METER_READINGS)
    ap.add_argument("--weather-csv", default=config.SOLCAST_CLEANED_V2)
    ap.add_argument(
        "--expected-csv",
        default=None,
        help="Precomputed PVLib CSV for a single --building-key run only.",
    )
    ap.add_argument(
        "--precomputed",
        action="store_true",
        help="Use data_pvlib/expected_power_pvlib_<key>.csv per meter (library: legacy v2 if present).",
    )
    ap.add_argument(
        "--compute-pvlib",
        action="store_true",
        help="Force live PVLib for all meters (ignore precomputed files).",
    )
    ap.add_argument("--out-dir", default=config.LIBRARY_ANALYSIS_DIR)
    ap.add_argument(
        "--building-key",
        default=None,
        help="Process one meter only (default: all keys in config._BUILDING_PVLIB_GEOMETRY).",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.meter_csv):
        print(f"ERROR: Meter file not found: {args.meter_csv}")
        sys.exit(1)
    if not os.path.isfile(config.PANEL_DATA):
        print(f"ERROR: panel_data.csv not found at:\n  {config.PANEL_DATA}")
        sys.exit(1)

    if args.building_key:
        keys = [args.building_key.strip().lower()]
    else:
        keys = config.analysis_meter_keys()

    if not keys:
        print("ERROR: No meter keys in config._BUILDING_PVLIB_GEOMETRY")
        sys.exit(1)

    print(f"panel_data: {os.path.abspath(config.PANEL_DATA)}")
    print(f"Meters to process ({len(keys)}): {', '.join(keys)}")

    weather_df = None
    pvlib_mod = None
    need_live = args.compute_pvlib or not args.precomputed
    if args.expected_csv:
        need_live = args.compute_pvlib
    if need_live or any(
        args.compute_pvlib
        or not os.path.isfile(_precomputed_path_for_key(k))
        for k in keys
    ):
        if not os.path.isfile(args.weather_csv):
            print(f"ERROR: Weather file not found: {args.weather_csv}")
            sys.exit(1)
        print(f"\nLoading weather once: {args.weather_csv}")
        pvlib_mod = _import_pvlib_module()
        weather_df = pvlib_mod._load_cleaned(args.weather_csv)

    kpi_rows: list[dict] = []
    errors: list[str] = []

    for key in keys:
        try:
            kpi = process_building(
                key,
                meter_csv=args.meter_csv,
                weather_csv=args.weather_csv,
                out_dir=args.out_dir,
                expected_csv=args.expected_csv if args.building_key else None,
                use_precomputed=args.precomputed,
                force_compute=args.compute_pvlib,
                weather_df=weather_df,
                pvlib_mod=pvlib_mod,
            )
            kpi_rows.append(kpi)
        except Exception as e:
            errors.append(f"{key}: {e}")
            print(f"  ERROR [{key}]: {e}", file=sys.stderr)

    if kpi_rows:
        kpi_df = pd.DataFrame(kpi_rows)
        sites_path = os.path.join(args.out_dir, "sites_kpis_summary.csv")
        kpi_df.to_csv(sites_path, index=False)
        print(f"\nWrote {sites_path} ({len(kpi_df)} site(s))")
        lib = kpi_df[kpi_df["building_key"] == "library"]
        if not lib.empty:
            legacy_kpi = os.path.join(args.out_dir, "library_kpis_summary.csv")
            lib.to_csv(legacy_kpi, index=False)
            print(f"Wrote {legacy_kpi}")

    if errors:
        print("\nFailed meters:", file=sys.stderr)
        for msg in errors:
            print(f"  - {msg}", file=sys.stderr)
        sys.exit(1 if not kpi_rows else 2)

    print("\nDone.")


if __name__ == "__main__":
    main()
