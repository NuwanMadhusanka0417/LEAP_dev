# PV Analysis — La Trobe Bundoora solar pipeline

Solar analysis for multiple buildings (Library, DMW, DW, …).  
All commands below assume the **`solarazure`** Python environment and are run from **`PV_analysis/`**.

```bash
conda activate solarazure
cd PV_analysis
```

Copy `.env.example` → `.env` and fill in SQL + Azure credentials (do not commit `.env`).

---

## Python scripts (one line each)

| Script | What it does |
|--------|----------------|
| **`config.py`** | Paths, site defaults, meter keys, `panel_data.csv` mappings, and PVLib tilt/azimuth per building. |
| **`0_download_data.py`** | Incrementally downloads SQL meter readings and Azure Solcast weather into `data_raw/`. |
| **`1_data_cleaning.py`** | Cleans meters (night zeros, stuck detection, imputation) and aligns Solcast into `data_cleaned/`. |
| **`2_build_library_analysis_outputs.py`** | Builds per-meter dashboard CSVs (hourly master, daily metrics, KPIs) in `data_for_viz/`. |
| **`3_expected_power_pvlib.py`** | Computes PVLib expected kWh from Solcast weather — **called automatically** by steps 2 and 4; run manually only for debugging. |
| **`4_forecast_7d_pvlib_xgboost.py`** | Trains XGBoost on history and writes ~7-day PVLib + ML forecasts per meter. |
| **`run_pipeline.py`** | Runs the full refresh: download → clean → build → forecast (one command). |
| **`serve_js_dashboard.py`** | Serves `JS_viz/` and `data_for_viz/` on http://127.0.0.1:8080 for the browser dashboard. |

---

## Whole pipeline (recommended)

```bash
conda activate solarazure
cd PV_analysis

# 1) Full data refresh (download → clean → dashboard CSVs → 7-day forecast)
python run_pipeline.py

# 2) Start dashboard (keep this terminal open)
python serve_js_dashboard.py
```

Open **http://127.0.0.1:8080/JS_viz/** and hard-refresh (`Ctrl+F5`) after each pipeline run.

### `run_pipeline.py` options

| Flag | Effect |
|------|--------|
| `--skip-download` | Skip Step 0 (use existing `data_raw/` CSVs). |
| `--skip-forecast` | Skip Step 4 (7-day forecast). |
| `--download-only` | Only download; stop before clean/build/forecast. |
| `--azure-live` | Step 4 reads live Solcast from Azure instead of local cleaned CSV. |
| `--campus BUNDOORA` | Campus filter for weather download + forecast (default). |
| `--clean-v2` | Run `1_data_cleaning.py v2` instead of default cleaning. |
| `--no-compute-pvlib` | Step 2 uses precomputed PVLib CSVs if present. |

---

## Run each step individually

### Step 0 — Download / merge raw data

```bash
python 0_download_data.py --campus BUNDOORA
```

**Writes:** `data_raw/SolarMeterReadings1hour_2020_2025.csv`, `data_raw/solcast_df_2020_2025.csv`

- **Meters:** appends new SQL rows per meter.
- **Weather:** re-fetches the last **14 days** from Azure (`live` + `forecast` blobs) and replaces stale forecast rows in that window.

Options: `--weather-only`, `--meters-only`, `--all-realenergy-meters`, `--building-key hs1` (repeatable).

### Step 1 — Clean meter + weather

```bash
python 1_data_cleaning.py
# or
python 1_data_cleaning.py v2
```

**Writes:** `data_cleaned/SolarMeterReadings1hour_cleaned_2020_2025.csv`, `data_cleaned/solcast_df_cleaned_2020_2025.csv`, `results_cleaning/outage_blocks_report_2020_2025.csv`

### Step 2 — Build dashboard CSVs

```bash
python 2_build_library_analysis_outputs.py --compute-pvlib
```

Processes every key in **`config._BUILDING_PVLIB_GEOMETRY`**.

**Writes** (per meter `<key>`): `data_for_viz/hourly_<key>_master.csv`, `daily_<key>_metrics.csv`, plus `sites_kpis_summary.csv`.

Option: `--building-key dmw` (one site only).

### Step 3 — PVLib expected power *(optional manual)*

Usually **not needed** — step 2 calls this internally.

```bash
python 3_expected_power_pvlib.py --building library
```

### Step 4 — 7-day forecast

Run **after Step 2** (needs `hourly_<key>_master.csv` for anchor times).

```bash
python 4_forecast_7d_pvlib_xgboost.py --campus BUNDOORA
# or live Azure weather:
python 4_forecast_7d_pvlib_xgboost.py --azure-live --campus BUNDOORA
```

**Writes:** `forecast_7d_pvlib_<key>.csv`, `forecast_7d_combined_<key>.csv`, etc. in `data_for_viz/`.

---

## Dashboard

```bash
conda activate solarazure
cd PV_analysis
python serve_js_dashboard.py
```

→ **http://127.0.0.1:8080/JS_viz/**

Use the **Meter** dropdown to switch sites. Data comes from `data_for_viz/hourly_<key>_master.csv` and `forecast_7d_combined_<key>.csv`.

| # | Tab | What it shows |
|---|-----|---------------|
| 1 | Daily | Daily actual vs PVLib expected kWh |
| 2 | Difference | PVLib − actual diff, health ratio H |
| 3 | PVLib chart | Hourly actual vs PVLib overlay |
| 4 | Season Analysis | Performance ratio H by season |
| 5 | Season + irradiance | Daily GHI + H by season |
| 6 | Meter degradation | Annual/monthly H trend for selected meter |
| 7 | All meters degradation | Side-by-side bar chart comparing all sites |
| 8 | 7-day forecast | PVLib + XGBoost forward forecast |

Stop the server with `Ctrl+C`.

---

## Required inputs

Place in `data_raw/` (or let Step 0 download them):

| File | Description |
|------|-------------|
| `SolarMeterReadings1hour_2020_2025.csv` | Hourly meter kWh (column `meter` = short key, e.g. `library`) |
| `solcast_df_2020_2025.csv` | Hourly Solcast weather (GHI, DNI, DHI, air temp, …) |

`panel_data.csv` must exist in `data_cleaned/` or `data_pvlib/` (DC/AC nameplate per building).

---

## Adding meters (including all sites)

The pipeline has **two separate meter lists**:

1. **Download (Step 0)** — which meters are pulled from SQL into the raw CSV.  
2. **Analysis (Steps 2 & 4)** — which meters get dashboard CSVs and forecasts (`config._BUILDING_PVLIB_GEOMETRY`).

### A. Download all SQL meters at once

```bash
python 0_download_data.py --all-realenergy-meters --campus BUNDOORA
```

This downloads every meter in `[dbo].[SolarMeterReadings]` whose key contains `realenergyintotheload`.

Then run the rest without re-downloading:

```bash
python run_pipeline.py --skip-download
```

### B. Add a meter to analysis + dashboard

For each new short key (e.g. `hs1`, `bg`, `union`):

**1. `panel_data.csv`** — row must exist with Building name, kWp, inverter, etc.

**2. `config.py` → `_METER_TO_BUILDING`** — map short key → Building column prefix:

```python
_METER_TO_BUILDING = {
    "hs1": "Health Sciences 1",
    # ...
}
```

**3. `config.py` → `_BUILDING_PVLIB_GEOMETRY`** — add key with PVLib surface tilt (°) and azimuth (°):

```python
_BUILDING_PVLIB_GEOMETRY = {
    "library": (10.0, 180.0),
    "dmw": (10.0, 180.0),
    "dw": (10.0, 180.0),
    "hs1": (10.0, 180.0),   # example: add new sites here
}
```

Only keys in `_BUILDING_PVLIB_GEOMETRY` appear in the dashboard dropdown and get `hourly_*_master.csv` / forecasts.

**4. Re-run pipeline:**

```bash
python run_pipeline.py
python serve_js_dashboard.py
```

### C. Download one extra meter without all sites

```bash
python 0_download_data.py --building-key hs1 --building-key bg
```

### Known meter keys (from `config._METER_TO_BUILDING`)

`bg`, `dmw`, `dw`, `hs1`, `hs2`, `hs3`, `hsc`, `hu3`, `library`, `lims1`, `mc`, `pe`, `ps2`, `pw`, `rlr`, `sw`, `tlc`, `union`, `wlt`, `carm1`, `carm2`, `cs1`, `rd1`, `rd2`, `busstop`, …  

Keys mapped to `None` in `_METER_TO_BUILDING` need a `panel_data.csv` row and a mapping before analysis will work.

SQL meter column format: `solar.bun_<key>#realenergyintotheload#kwh` (see `get_building_config()` in `config.py`).

---

## Credentials (`.env`)

| Variable | Purpose |
|----------|---------|
| `LEAP_DB_SERVER`, `LEAP_DB_USER`, `LEAP_DB_PASSWORD` | SQL meter readings |
| `AZURE_STORAGE_SAS_TOKEN` | Solcast `live` + `forecast` blobs |

---

## Folder layout

| Folder | Contents |
|--------|----------|
| `data_raw/` | Downloaded meter + Solcast CSVs |
| `data_cleaned/` | Cleaned meter/weather, `panel_data.csv` |
| `data_for_viz/` | Dashboard + forecast CSVs |
| `data_pvlib/` | Optional precomputed PVLib outputs |
| `results_cleaning/` | QC / outage reports |
| `JS_viz/` | Browser dashboard (HTML + JS) |
