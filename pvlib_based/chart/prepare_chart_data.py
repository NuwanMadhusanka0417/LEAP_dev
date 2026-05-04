"""
Merge library actual readings, old expected power, and pvlib expected power
into a single JSON file for the interactive chart.

Usage:
    python prepare_chart_data.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

METER_KEY = "solar.bun_library#realenergyintotheload#kwh"


def main() -> None:
    # 1. Old meter CSV — actual readings + old expected
    print("Loading meter readings …")
    meter = pd.read_csv(config.METER_READINGS, parse_dates=["timestamp"])
    lib = meter[meter["meter"] == METER_KEY].copy()
    lib = lib.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")
    lib = lib.rename(columns={
        "meter_reading": "actual_kwh",
        "expected_power": "old_expected_kwh",
    })
    lib = lib[["timestamp", "actual_kwh", "old_expected_kwh"]]

    # 2. PVLib expected
    pvlib_path = os.path.join(config.OUTPUT_DIR, "expected_power_pvlib_library.csv")
    if not os.path.isfile(pvlib_path):
        print(f"ERROR: {pvlib_path} not found. Run expected_power_pvlib.py --building library first.")
        sys.exit(1)
    print("Loading pvlib expected …")
    pvlib_df = pd.read_csv(pvlib_path, parse_dates=["timestamp"])
    pvlib_df = pvlib_df[["timestamp", "expected_kwh"]].rename(
        columns={"expected_kwh": "pvlib_expected_kwh"}
    )

    # 3. Merge on timestamp (inner join — only 2021 overlap)
    merged = lib.merge(pvlib_df, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp")

    print(f"  Merged rows: {len(merged)}")
    print(f"  Range: {merged['timestamp'].iloc[0]} to {merged['timestamp'].iloc[-1]}")

    # 4. Round to reduce JSON size
    merged["actual_kwh"] = merged["actual_kwh"].round(2)
    merged["old_expected_kwh"] = merged["old_expected_kwh"].round(2)
    merged["pvlib_expected_kwh"] = merged["pvlib_expected_kwh"].round(2)
    merged["timestamp"] = merged["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

    # 5. Write JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chart_data.json")
    records = {
        "timestamp": merged["timestamp"].tolist(),
        "actual_kwh": merged["actual_kwh"].tolist(),
        "old_expected_kwh": merged["old_expected_kwh"].tolist(),
        "pvlib_expected_kwh": merged["pvlib_expected_kwh"].tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(records, f)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Wrote {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
