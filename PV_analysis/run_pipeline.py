#!/usr/bin/env python3
"""
Run the full PV_analysis refresh: download → clean → build viz → forecast.

Usage (from PV_analysis/)::
  python run_pipeline.py

  python run_pipeline.py --skip-download
  python run_pipeline.py --skip-forecast
  python run_pipeline.py --download-only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import config

HERE = os.path.dirname(os.path.abspath(__file__))


def run_step(label: str, argv: list[str]) -> None:
    cmd = [sys.executable] + argv
    print(f"\n{'=' * 60}\n{label}\n  {' '.join(cmd)}\n{'=' * 60}")
    subprocess.run(cmd, cwd=HERE, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="PV_analysis end-to-end: download, clean, build dashboard CSVs, forecast."
    )
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-forecast", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument(
        "--azure-live",
        action="store_true",
        help="Step 4 re-fetches Azure weather (default: use cleaned CSV from Step 0/1).",
    )
    ap.add_argument("--campus", default="BUNDOORA", help="Campus for weather download + forecast.")
    ap.add_argument("--clean-v2", action="store_true", help="Run 1_data_cleaning.py v2.")
    ap.add_argument(
        "--no-compute-pvlib",
        action="store_true",
        help="Use precomputed PVLib CSVs in step 2 if present.",
    )
    args = ap.parse_args()

    loaded = config.load_credentials()
    if loaded:
        print("Loaded credentials from:", ", ".join(loaded))

    if not args.skip_download:
        dl_argv = ["0_download_data.py", "--campus", args.campus]
        run_step("Step 0 — Download / merge raw data", dl_argv)

    if args.download_only:
        print("\n--download-only: stopping after download.")
        return 0

    clean_argv = ["1_data_cleaning.py"]
    if args.clean_v2:
        clean_argv.append("v2")
    run_step("Step 1 — Clean meter + weather", clean_argv)

    build_argv = ["2_build_library_analysis_outputs.py"]
    if not args.no_compute_pvlib:
        build_argv.append("--compute-pvlib")
    run_step("Step 2 — Build dashboard CSVs", build_argv)

    if not args.skip_forecast:
        # Default: local cleaned weather (Step 0 already merged live + forecast blobs).
        fc_argv = ["4_forecast_7d_pvlib_xgboost.py", "--campus", args.campus]
        if args.azure_live:
            fc_argv.append("--azure-live")
        run_step("Step 3 — Forecast (flexible span, local weather)", fc_argv)

    print("\nPipeline complete. Start dashboard:\n  python serve_js_dashboard.py")
    print("Then open http://127.0.0.1:8080/JS_viz/ and refresh the browser.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        print(f"\nPipeline failed (exit {e.returncode}).", file=sys.stderr)
        raise SystemExit(e.returncode) from e
