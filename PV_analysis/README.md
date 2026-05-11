# PV Analysis — Single Meter Pipeline

Single-meter solar analysis for La Trobe Bundoora (Library).  
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
**Reads:** `data_cleaned/` meter + weather CSVs, `panel_data.csv`  
**Writes** (to `data_for_viz/`):
- `hourly_library_master.csv` — merged hourly: actual kWh, PVLib expected kWh, GHI
- `daily_library_metrics.csv` — daily aggregates and difference metrics
- `library_kpis_summary.csv` — single-row KPI summary (PR, correlation, system size)

> **Tip:** add `--compute-pvlib` to re-run PVLib on-the-fly instead of using a cached file.

---

### Step 3 — 7-day forecast *(optional)*
```bash
# Live Azure Solcast weather:
python 4_forecast_7d_pvlib_xgboost.py --building-key library --azure-live --campus BUNDOORA

# Local historical data only (backtest / offline):
python 4_forecast_7d_pvlib_xgboost.py --building-key library --backtest
```
**Reads:** `data_cleaned/` meter + weather CSVs, `panel_data.csv`  
**Writes** (to `data_for_viz/`):
- `forecast_7d_pvlib_library.csv` — PVLib physics forecast
- `forecast_7d_xgboost_library.csv` — XGBoost ML forecast
- `forecast_7d_combined_library.csv` — merged PVLib vs XGBoost (used by dashboard)
- `forecast_7d_run_meta_library.csv` — run metadata

---

## Visualising the dashboard

After Step 2 (and optionally Steps 3–4), start the local server:

```bash
python serve_js_dashboard.py
```

Then open **[http://127.0.0.1:8080/JS_viz/](http://127.0.0.1:8080/JS_viz/)** in your browser.

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
