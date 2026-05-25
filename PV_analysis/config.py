"""
Site and PV system defaults — edit for your as-built array.

All input CSVs live under ``PV_analysis/data_cleaned/``.
Visualization / pipeline outputs go to ``PV_analysis/data_for_viz/``.

Building-specific parameters (DC capacity, inverter AC limit, optional PVLib
tilt/azimuth) are loaded from ``panel_data.csv`` (prefers ``data_cleaned/``,
else ``data_pvlib/``). Tilt/azimuth for known keys are overridden by
``_BUILDING_PVLIB_GEOMETRY``.

Global PVLib fallbacks (``SYSTEM_DC_W``, ``INVERTER_AC_W``) match **Library (L)**
from ``panel_data.csv`` so ``3_expected_power_pvlib.py`` without ``--building``
uses the same nameplate as ``get_building_config("library")``.
"""
import os
import re

import pandas as pd

# PV_analysis package root (this folder)
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data_cleaned")
DATA_FOR_VIZ_DIR = os.path.join(BASE, "data_for_viz")

# Weather: cleaned hourly Bundoora series (place file in data_cleaned/)
SOLCAST_CLEANED_V2 = os.path.join(DATA_DIR, "solcast_df_cleaned_2020_2025.csv")
SOLCAST_RAW_MULTI = os.path.join(DATA_DIR, "solcast_df.csv")

# Meter readings (hourly, all buildings)
# METER_READINGS = os.path.join(DATA_DIR, "SolarMeterReadings1hour_cleaned_v2_2021.csv")
METER_READINGS = os.path.join(DATA_DIR, "SolarMeterReadings1hour_cleaned.csv")

# Long-range hourly meter file (2020–2025) — library batch analysis
# METER_READINGS_2020_2025 = os.path.join(DATA_DIR, "SolarMeterReadings1hour_cleaned_2020_2025.csv")

# Outputs: library merged hourly + daily + KPI (build_library_analysis_outputs.py)
LIBRARY_ANALYSIS_DIR = DATA_FOR_VIZ_DIR

# Panel / inverter catalogue for every building (repo copy often lives under data_pvlib/)
_PANEL_CLEANED = os.path.join(DATA_DIR, "panel_data.csv")
_PANEL_PVLIB = os.path.join(BASE, "data_pvlib", "panel_data.csv")
PANEL_DATA = _PANEL_CLEANED if os.path.isfile(_PANEL_CLEANED) else _PANEL_PVLIB

# La Trobe Bundoora (approximate — adjust if array GPS differs)
LATITUDE = -37.7183
LONGITUDE = 145.0454
ALTITUDE_M = 85.0
TIMEZONE = "Australia/Melbourne"

# Default PVLib plane orientation when a meter key has no entry in _BUILDING_PVLIB_GEOMETRY.
# PVLib: surface_azimuth clockwise from north (0° = north, 180° = south).
SURFACE_TILT_DEG = 10.0
SURFACE_AZIMUTH_DEG = 0.0

# Per-meter overrides (tilt°, azimuth°). Library, DMW, DW: 10° tilt, 180° azimuth (south-facing plane).
_BUILDING_PVLIB_GEOMETRY: dict[str, tuple[float, float]] = {
    "library": (10.0, 180.0),
    "dmw": (10.0, 180.0),
    "dw": (10.0, 180.0),
}


def surface_geometry_for_meter_key(meter_key: str) -> tuple[float, float]:
    """Return (surface_tilt_deg, surface_azimuth_deg) for PVLib POA modelling."""
    key = meter_key.strip().lower()
    return _BUILDING_PVLIB_GEOMETRY.get(key, (SURFACE_TILT_DEG, SURFACE_AZIMUTH_DEG))


def analysis_meter_keys() -> list[str]:
    """Short meter keys for ``2_build_library_analysis_outputs.py`` (edit ``_BUILDING_PVLIB_GEOMETRY``)."""
    return sorted(_BUILDING_PVLIB_GEOMETRY.keys())

# ── Library (L) catalogue row (panel_data.csv) — primary site for library analysis ──
# kWp 384.12 | 1164 × Trina 330W | 4 × SolarEdge SE82.8K | Bundoora
LIBRARY_SYSTEM_KWP = 384.12
LIBRARY_INVERTER_AC_KW = 4.0 * 82.8  # 331.2 kW AC nameplate
LIBRARY_NO_PANELS = 1164

# PVLib fallbacks when ``build_expected(..., system_dc_w=None)`` (same as get_building_config("library"))
GAMMA_PDC = -0.004
SYSTEM_DC_W = LIBRARY_SYSTEM_KWP * 1000.0  # 384_120 W
INVERTER_AC_W = LIBRARY_INVERTER_AC_KW * 1000.0  # 331_200 W

# Precomputed hourly PVLib CSV (must be built with matching ``--building library`` if used for Library)
EXPECTED_PVLIB_CLEANED_V2 = os.path.join(BASE, "data_pvlib", "expected_power_pvlib_cleaned_v2.csv")

# Optional PVLib standalone CSV output (expected_power_pvlib.py)
OUTPUT_DIR = os.path.join(DATA_FOR_VIZ_DIR, "pvlib_expected")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_FOR_VIZ_DIR, exist_ok=True)


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
    """Return DC/AC nameplate, PVLib tilt/azimuth, and metadata for a meter key.

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

    surf_tilt, surf_az = surface_geometry_for_meter_key(key)

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
        "surface_tilt_deg": surf_tilt,
        "surface_azimuth_deg": surf_az,
    }
