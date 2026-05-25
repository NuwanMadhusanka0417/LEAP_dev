# PV Analysis — Single Meter Pipeline

Solar analysis for La Trobe Bundoora (Library, DMW, DW, …).  
All commands are run from the **`PV_analysis/`** directory.

---

> **Note on `3_expected_power_pvlib.py`:** you do **not** need to run this script manually.
> Both `2_build_library_analysis_outputs.py` (Step 2) and `4_forecast_7d_pvlib_xgboost.py` (Step 3)
> import and execute it automatically when they need PVLib expected power values.

---

## Required raw inputs

Place these files in `PV_analysis/data_raw/` before running:

| File | Description |
|------|-------------|
| `SolarMeterReadings1hour_2020_2025.csv` | Hourly meter kWh readings (all buildings) |
| `solcast_df_2020_2025.csv` | Hourly Solcast weather (GHI, DNI, DHI, air temp, zenith, azimuth) |

`panel_data.csv` (DC/AC nameplate per building) must be in `data_cleaned/` or `data_pvlib/`.

---

## Running order

### Step 1 — Clean meter + weather data
```bash
python 1_data_cleaning.py
```
**Reads:** `data_raw/SolarMeterReadings1hour_2020_2025.csv`, `data_raw/solcast_df_2020_2025.csv`  
**Writes:**
- `data_cleaned/SolarMeterReadings1hour_cleaned_2020_2025.csv` — imputed, capped, nighttime-zeroed meter
- `data_cleaned/solcast_df_cleaned_2020_2025.csv` — aligned Solcast hourly
- `results_cleaning/outage_blocks_report_2020_2025.csv` — outage/stuck-meter report

---

### Step 2 — Build dashboard data files
```bash
python 2_build_library_analysis_outputs.py
```
Processes **every meter key** in `config._BUILDING_PVLIB_GEOMETRY` (default: `library`, `dmw`, `dw`).  
Edit that dict in `config.py` to add/remove sites or change PVLib tilt/azimuth.

**Reads:** `data_cleaned/` meter + weather CSVs, `panel_data.csv`  
**Writes** (to `data_for_viz/`), per key `<key>`:
- `hourly_<key>_master.csv` — merged hourly actual + PVLib expected (+ GHI from weather merge)
- `daily_<key>_metrics.csv` — daily aggregates
- `sites_kpis_summary.csv` — one row per site

Legacy names kept for **library**: `hourly_library_master.csv`, `daily_library_metrics.csv`, `library_kpis_summary.csv`.

Optional: `--building-key dmw` (one site only); `--compute-pvlib` (ignore precomputed PVLib CSVs).

---

### Step 3 — 7-day forecast *(optional)*
Run **after Step 2** (needs `hourly_<key>_master.csv` for Azure anchor times).

Processes all keys in `config._BUILDING_PVLIB_GEOMETRY` (default: `library`, `dmw`, `dw`).

```bash
# Live Azure Solcast weather (all configured meters):
python 4_forecast_7d_pvlib_xgboost.py --azure-live --campus BUNDOORA

# Local (default): 7 days per meter from midnight after that meter's last reading day:
python 4_forecast_7d_pvlib_xgboost.py

# Evaluation only — last 168h of shared Solcast file (NOT per-meter):
python 4_forecast_7d_pvlib_xgboost.py --backtest
```

Optional: `--building-key dmw` (one site only).

**Reads:** `data_cleaned/` meter + weather, `panel_data.csv`, `hourly_<key>_master.csv` (Step 2)  
**Writes** (to `data_for_viz/`), per key `<key>`:
- `forecast_7d_pvlib_<key>.csv`, `forecast_7d_xgboost_<key>.csv`, `forecast_7d_combined_<key>.csv`, `forecast_7d_run_meta_<key>.csv`
- `forecast_7d_runs_meta_all.csv` — combined run summary

Legacy for **library**: `forecast_7d_combined_library.csv` (dashboard).

---

## Visualising the dashboard

After Step 2 (and optionally Steps 3–4), start the local server:

```bash
python serve_js_dashboard.py
```

Then open **[http://127.0.0.1:8080/JS_viz/](http://127.0.0.1:8080/JS_viz/)** in your browser.

Use the **Meter** dropdown in the header to switch between **library**, **dmw**, and **dw** (from `sites_kpis_summary.csv` or the same keys as `config._BUILDING_PVLIB_GEOMETRY`).

### Dashboard tabs

| # | Tab | What it shows |
|---|-----|---------------|
| 1 | **Daily** | Daily actual vs PVLib expected kWh, trend line |
| 2 | **Difference** | Daily PVLib − actual diff, 7-day median, health ratio H |
| 3 | **Library (PVLib chart)** | Full hourly actual vs PVLib overlay with date zoom |
| 4 | **Season Analysis** | Performance ratio by season |
| 5 | **Season + irradiance** | H faceted by season and GHI band |
| 6 | **Meter degradation** | Annual H trend (OLS), monthly H detail |
| 7 | **7-day forecast** | PVLib + XGBoost 7-day hourly forecast *(requires Step 4)* |

> **Note:** The server must stay running while you use the dashboard. Stop it with `Ctrl+C`.
