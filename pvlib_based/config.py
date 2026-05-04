"""
Site and PV system defaults — edit for your as-built array.

Building-specific parameters (DC capacity, inverter AC limit) are loaded
from pvlib_based/data/panel_data.csv via ``get_building_config()``.
"""
import os
import re

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")

# Weather: prefer cleaned hourly Bundoora series
SOLCAST_CLEANED_V2 = os.path.join(DATA_DIR, "solcast_df_cleaned_v2.csv")
SOLCAST_RAW_MULTI = os.path.join(DATA_DIR, "solcast_df.csv")

# Meter readings (hourly, all buildings)
METER_READINGS = os.path.join(DATA_DIR, "SolarMeterReadings1hour_cleaned_v2_2021.csv")
# Long-range hourly meter file (2020–2025) — library batch analysis
METER_READINGS_2020_2025 = os.path.join(DATA_DIR, "SolarMeterReadings1hour_cleaned_2020_2025.csv")

# Outputs: library-only merged hourly + daily + KPI CSVs (build_library_analysis_outputs.py)
LIBRARY_ANALYSIS_DIR = os.path.join(BASE, "Results_library")

# Panel / inverter catalogue for every building
PANEL_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "panel_data.csv")

# La Trobe Bundoora (approximate — adjust if array GPS differs)
LATITUDE = -37.7183
LONGITUDE = 145.0454
ALTITUDE_M = 85.0
TIMEZONE = "Australia/Melbourne"

# Library as-built: 10° tilt, north-facing. PVLib: surface_azimuth 0° = north (clockwise from north).
SURFACE_TILT_DEG = 10.0
SURFACE_AZIMUTH_DEG = 0.0

# Fallback DC nameplate (W) when no building is specified
SYSTEM_DC_W = 100_000.0
GAMMA_PDC = -0.004
# Fallback inverter AC limit (W)
INVERTER_AC_W = 95_000.0

# Output
OUTPUT_DIR = os.path.join(BASE, "data", "pvlib_expected")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── building-specific helpers ────────────────────────────────────────────

# Maps the short meter key (e.g. "library") to the Building column in panel_data.csv.
# Only entries where the match isn't obvious need to be listed here.
_METER_TO_BUILDING = {
    "bg": "Beth Gleeson",
    "dmw": "David Myers",
    "dw": "Donald Whitehead",
    "hs1": "Health Sciences 1",
    "hs2": "Health Sciences 2",
    "hs3": "Health Sciences 3",
    "hsc": "Health Sciences Clinic",
    "hu3": "Humanities",
    "library": "Library",
    "lims1": "LIMS1",
    "ltss_pv_lgcmeter": None,
    "ltusa_north_solar": None,
    "ltusa_south_solar": None,
    "lwso": None,
    "mc": "Menzies College",
    "pe": "Peribolos East",
    "ps2": "Physical Sciences 2",
    "pw": "Peribolos West",
    "rlr": "R L Reid",
    "sw": "Sylvia Walton",
    "tlc": "The Learning Commons",
    "union": "Union Building",
    "wlt": "West Lecture Theatre",
    "carm1": "Caval building - Carm 1",
    "carm2": "Caval building - Carm 2",
    "cs1": "Campus Services 1",
    "rd1": "RD1",
    "rd2": "RD2",
    "busstop": "Bus Stop",
    "ccr": None,
}


def _parse_inverter_kw(text: str) -> float:
    """Sum all inverter ratings from strings like '4 x SolarEdge SE82.8K'."""
    total = 0.0
    for match in re.finditer(r"(\d+)\s*x\s*\w+\s*SE([\d.]+)K", text, re.IGNORECASE):
        total += int(match.group(1)) * float(match.group(2))
    if total > 0:
        return total
    for match in re.finditer(r"SE([\d.]+)K", text, re.IGNORECASE):
        total += float(match.group(1))
    return total


def get_building_config(meter_key: str) -> dict:
    """Return {system_dc_w, inverter_ac_w, building_name} for a meter key.

    ``meter_key`` is the short name used in the meter CSV, e.g. "library",
    "bg", "hs1".  Raises ``ValueError`` if the building is not found in
    panel_data.csv.
    """
    key = meter_key.strip().lower()
    building_prefix = _METER_TO_BUILDING.get(key)
    if building_prefix is None:
        raise ValueError(
            f"No panel_data mapping for meter key '{key}'. "
            f"Add it to _METER_TO_BUILDING in config.py."
        )

    pdf = pd.read_csv(PANEL_DATA)
    match = pdf[pdf["Building"].str.strip().str.lower().str.startswith(building_prefix.lower())]
    if match.empty:
        raise ValueError(f"Building '{building_prefix}' not found in {PANEL_DATA}")
    row = match.iloc[0]

    kwp_str = str(row["kWp"]).strip().replace(",", "")
    dc_w = float(kwp_str) * 1000.0

    inv_text = str(row.get("Inverter", ""))
    inv_kw = _parse_inverter_kw(inv_text)
    ac_w = inv_kw * 1000.0 if inv_kw > 0 else None

    no_panels = row.get("No_panels", "")
    try:
        no_panels = int(float(str(no_panels).strip()))
    except (ValueError, TypeError):
        no_panels = None

    return {
        "building_name": row["Building"].strip(),
        "network": str(row.get("Network", "")).strip(),
        "campus": str(row.get("Campus", "")).strip(),
        "system_dc_w": dc_w,
        "system_kwp": dc_w / 1000.0,
        "inverter_ac_w": ac_w,
        "inverter_ac_kw": (ac_w / 1000.0) if ac_w else None,
        "no_panels": no_panels,
        "panel_type": str(row.get("Panel", "")).strip(),
        "inverter_type": str(row.get("Inverter", "")).strip(),
        "optimisers": str(row.get("Optimsiers", "")).strip(),
        "meter_key_full": f"solar.bun_{key}#realenergyintotheload#kwh",
    }
