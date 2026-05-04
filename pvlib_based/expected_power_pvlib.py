"""
Expected AC/DC power from Solcast weather via pvlib (Hay–Davies POA, Faiman T_cell, PVWatts DC).

Can generate expected power for a specific building (e.g. Library) using specs
from panel_data.csv, then optionally compare against actual meter readings.

Usage:
  # Generic (uses config.py defaults):
  python expected_power_pvlib.py

  # Building-specific (loads DC/AC from panel_data.csv):
  python expected_power_pvlib.py --building library

  # Raw multi-campus weather:
  python expected_power_pvlib.py --weather raw --campus BUNDOORA

  # Custom output directory:
  python expected_power_pvlib.py --building library --output-dir ../Results_expected_power
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import pvlib
from pvlib import irradiance, location, pvsystem, temperature

import config

REQUIRED = ("ghi", "dni", "dhi", "air_temp")


def _load_cleaned(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if len(df) > 1 and df["timestamp"].diff().dt.total_seconds().median() < 2000:
        df = df.set_index("timestamp").resample("1h").mean(numeric_only=True).reset_index()
    return df


def _load_raw_campus(path: str, campus: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "campus" not in df.columns:
        raise ValueError("Raw weather file must include a campus column")
    df = df.dropna(subset=["timestamp"])
    df = df[df["campus"].str.upper().str.strip() == campus.upper()].sort_values("timestamp")
    return df.set_index("timestamp").resample("1h").mean(numeric_only=True).reset_index()


def _localize_times(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize(config.TIMEZONE, ambiguous=True, nonexistent="shift_forward")
    return idx.tz_convert(config.TIMEZONE)


def _poa(
    idx: pd.DatetimeIndex,
    zenith: pd.Series,
    azimuth: pd.Series,
    ghi: pd.Series,
    dni: pd.Series,
    dhi: pd.Series,
    albedo: pd.Series | None,
) -> pd.Series:
    alb = albedo.astype(float).clip(0.05, 0.4) if albedo is not None else pd.Series(0.2, index=idx)
    dni_extra = pvlib.irradiance.get_extra_radiation(idx.dayofyear)
    return irradiance.get_total_irradiance(
        surface_tilt=config.SURFACE_TILT_DEG,
        surface_azimuth=config.SURFACE_AZIMUTH_DEG,
        solar_zenith=zenith,
        solar_azimuth=azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        albedo=alb,
        model="haydavies",
    )["poa_global"]


def build_expected(
    df: pd.DataFrame,
    system_dc_w: float | None = None,
    inverter_ac_w: float | None = None,
) -> pd.DataFrame:
    """One row per weather timestamp; index is tz-aware site time.

    If ``system_dc_w`` / ``inverter_ac_w`` are provided they override the
    config.py fallback values, allowing building-specific modelling.
    """
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Weather data missing columns: {missing}")

    dc_w = system_dc_w if system_dc_w is not None else config.SYSTEM_DC_W
    ac_w = inverter_ac_w if inverter_ac_w is not None else config.INVERTER_AC_W

    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")
    times = _localize_times(pd.DatetimeIndex(df["timestamp"]))
    df = df.copy()
    df.index = times

    loc = location.Location(
        config.LATITUDE, config.LONGITUDE, tz=config.TIMEZONE, altitude=config.ALTITUDE_M
    )
    solpos = loc.get_solarposition(times)
    zenith = solpos["apparent_zenith"].clip(0, 90)
    sun_az = solpos["azimuth"]

    ghi = df["ghi"].astype(float)
    dni = df["dni"].astype(float)
    dhi = df["dhi"].astype(float)
    alb = df["albedo"] if "albedo" in df.columns else None

    poa = _poa(times, zenith, sun_az, ghi, dni, dhi, alb)
    air = df["air_temp"].astype(float)
    wind = df["wind_speed_10m"].astype(float) if "wind_speed_10m" in df.columns else pd.Series(2.0, index=times)
    wind = wind.fillna(2.0)
    t_cell = temperature.faiman(poa, air, wind, u0=25.0, u1=6.84)

    p_dc = pvsystem.pvwatts_dc(
        effective_irradiance=poa,
        temp_cell=t_cell,
        pdc0=dc_w,
        gamma_pdc=config.GAMMA_PDC,
    ).clip(lower=0.0)

    p_ac = p_dc if ac_w is None or ac_w <= 0 else p_dc.clip(upper=ac_w)

    # Strip tz label but keep wall-clock values in Melbourne local time.
    # (tz_convert(None) would shift to UTC — wrong for meter comparison.)
    ts_out = times.tz_localize(None) if times.tz is not None else times
    return pd.DataFrame(
        {
            "timestamp": ts_out,
            "ghi_wm2": ghi.values,
            "dni_wm2": dni.values,
            "dhi_wm2": dhi.values,
            "poa_global_wm2": poa.values,
            "temp_cell_c": t_cell.values,
            "p_dc_w": p_dc.values,
            "p_ac_w": p_ac.values,
            "expected_kwh": (p_ac.values / 1000.0),
        }
    )


def _load_meter_readings(meter_key_full: str) -> pd.DataFrame | None:
    """Load actual hourly meter readings for a single building."""
    path = config.METER_READINGS
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[df["meter"] == meter_key_full].copy()
    if df.empty:
        return None
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")
    df = df.rename(columns={"meter_reading": "actual_kwh"})
    return df[["timestamp", "actual_kwh"]]


def run(
    weather_mode: str,
    campus: str,
    building: str | None = None,
    output_dir: str | None = None,
) -> tuple[str, pd.DataFrame]:
    # ── load weather ──
    if weather_mode == "cleaned":
        path = config.SOLCAST_CLEANED_V2
        tag = "cleaned_v2"
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        raw = _load_cleaned(path)
    else:
        path = config.SOLCAST_RAW_MULTI
        tag = f"raw_{campus.lower()}"
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        raw = _load_raw_campus(path, campus)

    # ── resolve building-specific params ──
    bldg_cfg = None
    if building:
        bldg_cfg = config.get_building_config(building)
        tag = building.lower()
        print("=" * 55)
        print(f"  Building   : {bldg_cfg['building_name']}")
        print(f"  Network    : {bldg_cfg['network']}")
        print(f"  Campus     : {bldg_cfg['campus']}")
        print(f"  DC cap     : {bldg_cfg['system_kwp']:.2f} kWp")
        print(f"  AC cap     : {bldg_cfg['inverter_ac_kw'] or 'N/A'} kW")
        print(f"  Panels     : {bldg_cfg['no_panels'] or 'N/A'} x {bldg_cfg['panel_type']}")
        print(f"  Inverter   : {bldg_cfg['inverter_type']}")
        print(f"  Optimisers : {bldg_cfg['optimisers']}")
        print(f"  Meter key  : {bldg_cfg['meter_key_full']}")
        print("=" * 55)

    dc_w = bldg_cfg["system_dc_w"] if bldg_cfg else None
    ac_w = bldg_cfg["inverter_ac_w"] if bldg_cfg else None

    out = build_expected(raw, system_dc_w=dc_w, inverter_ac_w=ac_w)

    # ── merge actual meter readings when available ──
    if bldg_cfg:
        actual = _load_meter_readings(bldg_cfg["meter_key_full"])
        if actual is not None:
            out = out.merge(actual, on="timestamp", how="left")
            matched = out["actual_kwh"].notna().sum()
            print(f"  Meter  : {matched} hours matched from {config.METER_READINGS}")
        else:
            print("  Meter  : no meter file found — expected-only output")

    # ── save ──
    out_dir = os.path.abspath(output_dir) if output_dir else config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"expected_power_pvlib_{tag}.csv")
    out.to_csv(out_path, index=False)
    return out_path, out


def _print_comparison_analysis(out: pd.DataFrame) -> None:
    """Print a detailed expected-vs-actual comparison report."""
    # Only consider hours where both values exist and there's meaningful production
    MIN_KWH = 0.5
    valid = out.dropna(subset=["actual_kwh", "expected_kwh"]).copy()
    daytime = valid[(valid["expected_kwh"] > MIN_KWH) | (valid["actual_kwh"] > MIN_KWH)]

    if daytime.empty:
        print("\n  No overlapping daytime hours to compare.")
        return

    total_hours = len(daytime)
    exp_sum = daytime["expected_kwh"].sum()
    act_sum = daytime["actual_kwh"].sum()

    # Hours where expected < actual (model underestimates)
    under = daytime[daytime["expected_kwh"] < daytime["actual_kwh"]]
    # Hours where expected >= actual (model meets or overestimates)
    over = daytime[daytime["expected_kwh"] >= daytime["actual_kwh"]]

    pct_under = len(under) / total_hours * 100
    pct_over = len(over) / total_hours * 100

    corr = daytime["expected_kwh"].corr(daytime["actual_kwh"])
    overall_ratio = act_sum / exp_sum if exp_sum > 0 else float("nan")

    # Mean absolute percentage error on daytime hours
    nonzero_exp = daytime[daytime["expected_kwh"] > MIN_KWH].copy()
    if not nonzero_exp.empty:
        nonzero_exp["pct_err"] = (
            (nonzero_exp["actual_kwh"] - nonzero_exp["expected_kwh"])
            / nonzero_exp["expected_kwh"] * 100
        )
        mean_pct_err = nonzero_exp["pct_err"].mean()
        median_pct_err = nonzero_exp["pct_err"].median()
    else:
        mean_pct_err = median_pct_err = float("nan")

    # Monthly breakdown
    daytime_m = daytime.copy()
    daytime_m["month"] = pd.to_datetime(daytime_m["timestamp"]).dt.month

    print("\n" + "=" * 60)
    print("  EXPECTED vs ACTUAL  —  Comparison Analysis")
    print("=" * 60)
    print(f"  Daytime hours compared  : {total_hours}")
    print(f"  Total expected (kWh)    : {exp_sum:,.1f}")
    print(f"  Total actual   (kWh)    : {act_sum:,.1f}")
    print(f"  Overall actual/expected : {overall_ratio:.4f}")
    print(f"  Correlation (r)         : {corr:.4f}")
    print("-" * 60)
    print(f"  Expected < Actual  (under-estimate) : {len(under):>5} hours  ({pct_under:.1f}%)")
    print(f"  Expected >= Actual (over-estimate)   : {len(over):>5} hours  ({pct_over:.1f}%)")
    print("-" * 60)
    print(f"  Mean  (actual-expected)/expected     : {mean_pct_err:+.1f}%")
    print(f"  Median (actual-expected)/expected    : {median_pct_err:+.1f}%")
    print("-" * 60)

    print("  Monthly breakdown:")
    print(f"  {'Month':>7} | {'Exp kWh':>10} | {'Act kWh':>10} | {'Act/Exp':>8} | {'Under%':>7} | {'Over%':>7}")
    for m in range(1, 13):
        mdf = daytime_m[daytime_m["month"] == m]
        if mdf.empty:
            continue
        m_exp = mdf["expected_kwh"].sum()
        m_act = mdf["actual_kwh"].sum()
        m_ratio = m_act / m_exp if m_exp > 0 else float("nan")
        m_under = (mdf["expected_kwh"] < mdf["actual_kwh"]).sum() / len(mdf) * 100
        m_over = 100.0 - m_under
        month_name = pd.Timestamp(2021, m, 1).strftime("%b")
        print(f"  {month_name:>7} | {m_exp:>10,.1f} | {m_act:>10,.1f} | {m_ratio:>8.3f} | {m_under:>6.1f}% | {m_over:>6.1f}%")

    print("=" * 60)

    if overall_ratio > 1.05:
        print("\n  NOTE: Actual production EXCEEDS expected by "
              f"{(overall_ratio - 1) * 100:.1f}% overall.")
        print("  Possible reasons:")
        print("    1. Solcast weather model predicted more clouds than reality")
        print("       (satellite-derived irradiance != ground truth)")
        print("    2. PVWatts model applies conservative system losses")
        print("       (soiling, mismatch, wiring) that may overstate losses")
        print("    3. Panel tilt/azimuth in config may differ from as-built")
        print("       geometry, reducing modelled POA irradiance")
        print("    4. Temperature coefficient (gamma) may be too aggressive")
    elif overall_ratio < 0.95:
        print("\n  NOTE: Actual production is BELOW expected by "
              f"{(1 - overall_ratio) * 100:.1f}% overall.")
        print("  Possible reasons:")
        print("    1. Real system losses (soiling, shading, degradation)")
        print("    2. Inverter clipping or outages in actual data")
        print("    3. Meter outage periods flagged in the data")
    else:
        print(f"\n  NOTE: Expected and actual are within 5% — good agreement.")


def main() -> None:
    ap = argparse.ArgumentParser(description="PVLib expected power from Solcast weather")
    ap.add_argument(
        "--weather",
        choices=("cleaned", "raw"),
        default="cleaned",
        help="cleaned: solcast_df_cleaned_v2.csv; raw: solcast_df.csv + --campus",
    )
    ap.add_argument("--campus", default="BUNDOORA", help="Used with --weather raw")
    ap.add_argument(
        "--building",
        default=None,
        metavar="KEY",
        help="Meter key for a specific building (e.g. library, bg, hs1). "
             "Loads DC/AC capacity from panel_data.csv and merges meter readings.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=f"Folder for CSV (default: {config.OUTPUT_DIR})",
    )
    args = ap.parse_args()

    path, out = run(
        args.weather, args.campus,
        building=args.building,
        output_dir=args.output_dir,
    )
    t0, t1 = out["timestamp"].iloc[0], out["timestamp"].iloc[-1]
    print(f"\nWrote {path}")
    print(f"  rows: {len(out)} | range: {t0} to {t1}")

    if "actual_kwh" in out.columns:
        _print_comparison_analysis(out)


if __name__ == "__main__":
    main()
