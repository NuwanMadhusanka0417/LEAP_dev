# Solar PV Baselines + Forecasting + Predictive Maintenance (Quick Dev Guide)

This repo/pipeline generates **per-meter time series CSVs** (actual + model predictions) and provides:
1) a **Streamlit dashboard** to explore/compare meters over time (hourly + daily views), and  
2) a **batch “predictive maintenance”** script that turns model-vs-actual differences into **daily KPIs + alerts**.

---

## Folder / Data Conventions

### Input / Generated data
- **`Results/`**  
  Contains per-meter CSV outputs used by *both* Streamlit and predictive maintenance:
  - `*_series_new.csv`

### Output (maintenance analytics)
- **`Maintenance/`** (created by `predictive_maintenance.py`)
  - `daily_kpis_<meter>.csv`
  - `alerts_<meter>.csv`

### Typical expected columns in `Results/*_series_new.csv`
Minimum required:
- `timestamp` (datetime)
- `meter` (meter identifier)
- `campus`
- `Meter_reading` *(actual measured energy/power)*
- `Real_model_prediction` *(model prediction aligned to actual)*
- `Simulated_model` *(engineering simulation baseline)*

Optional (if available):
- `Simulated_degraded`
- `global_horizontal_irradiance`
- `zenith`

> Note: most scripts also accept these alternative column names and normalize them:
> - `Meter_reading (actual)` → `Meter_reading`  
> - `Real model prediction` → `Real_model_prediction`  
> - `Simulated model` → `Simulated_model`  
> - `Simulated (degraded)` → `Simulated_degraded`

---

## Code Overview (What each file is for)

### 1) `forecasting hourly.ipynb`
**Purpose:** Build/refresh baseline models from Solcast + internal simulation data, then produce / update forecasting-related outputs.

**What it does (high-level):**
- Loads historical Solcast data (CSV per campus) to build baseline relationships.
- Loads “live/current” Solcast data from Azure blob storage (container appears to include `live` files).
- Connects to a SQL Server DB (simulation/source data).
- Trains baseline regression models (e.g., `XGBRegressor`, `LinearRegression`) and performs preprocessing (diff/outlier handling, date alignment, etc.).

**Important dev note (security):**
- This notebook currently contains **hard-coded credentials** (DB server/user/password, SAS token).
  - Move these into environment variables or a secrets store before sharing publicly.

**Outputs / downstream:**
- The Streamlit + maintenance scripts assume you already have `Results/*_series_new.csv`.
- If this notebook is the place you generate those CSVs, keep that output format stable.

---

### 2) `streamlit_main.py`
**Purpose:** The **Streamlit entry point** (the “app shell”).

**Key responsibilities:**
- Scans `Results/` for `*_series_new.csv` and builds a meter list.
- Computes a **global min/max date** across all meters.
- Provides **global controls** in the sidebar:
  - global meter (`st.session_state["global_meter"]`)
  - global date range (`st.session_state["global_date"]`)
- Launches 4 tab modules using `runpy.run_path()` and passes a `_tab_prefix` for unique widget keys.

**Run it:**
```bash
streamlit run streamlit_main.py
```

---

### 3) `tab1.py`  (Series Viewer + Analytics + “Next Few Days” Forecast)
**Purpose:** The most detailed per-meter exploration tab.

**Key outputs on this tab:**
- **Main time-series chart** (selected series):
  - Actual (`Meter_reading`)
  - Real model prediction (`Real_model_prediction`)
  - Simulation baseline (`Simulated_model`)
  - Degraded simulation (`Simulated_degraded`, if present)
- **Analytics** (daily / daylight-focused):
  - Health-style ratios (e.g., actual vs simulated, actual vs real-model)
  - “Model gap” summary (Simulated − Real) using daily median and rolling medians
- **Forecasting (hourly)** — “Next Few Days — Forecast (Real vs Simulated)”
  - Finds the last timestamp with actual readings, then starts from the **next hour**
  - For the forecast window:
    - hourly gap: `Simulated_model − Real_model_prediction`
    - smoothed gap (7-hour median)
    - cumulative gap
    - daily sums and percent differences

**Why it matters:** This is where the **“forecasting hourly”** idea is visualized (not trained).

---

### 4) `tab2.py`  (Daily “PowerBI-style” view)
**Purpose:** A clean **daily aggregation** view for a selected meter.

**Core logic:**
- Loads the selected meter’s `*_series_new.csv`
- Applies the **global date range** from the main app
- Aggregates to daily totals and plots daily comparisons (focused on a fixed “prediction source”, currently set to `Simulated_model` in code)

---

### 5) `tab3.py`  (Daily Difference view: Prediction − Actual)
**Purpose:** Visualize **daily differences** between prediction and actual.

**Core logic:**
- Loads meter series and uses global date range
- Computes daily (Prediction − Actual) to highlight bias / mismatch patterns
- The prediction source is currently fixed to `Simulated_model`

---

### 6) `tab4.py`  (Fleet Trend ranking)
**Purpose:** Fleet-level ranking of meters by **trend/slope**.

**Core logic:**
- Iterates through all meters in `Results/`
- Computes a daily difference series, then fits a simple slope:
  - slope ≈ “kWh per day” change in mismatch
- Ranks meters to find those degrading faster (or improving)

---

### 7) `predictive_maintenance.py`
**Purpose:** Batch job to convert `Results/*_series_new.csv` into **daily KPIs** and **alerts** per meter.

**Key computations:**
- Daylight detection:
  - preferred: `global_horizontal_irradiance > 5` AND `zenith < 90`
  - fallback: `Simulated_model > 0.05`
- Pointwise metrics (daylight points):
  - `HS = Actual / Simulated_model`   (health vs simulation baseline)
  - `HR = Actual / Real_model_prediction` (health vs “real model”)
  - `GAP = Simulated_model − Real_model_prediction`
- Daily aggregation (median/sum) and rolling 30-day baselines:
  - rolling medians for `HS_ref`, `HR_ref`
  - deltas `ΔHS`, `ΔHR`
  - `Lost_kWh` estimate from `(1 - HS) * SimEnergy`
- Rule-based alerts (tunable):
  - **Soiling**: HS < 0.90 for 3 consecutive days
  - **RapidFault**: HS < 0.75 OR ActEnergy < 0.60 × SimEnergy
  - **ModelDrift**: HR < 0.95 for 5 consecutive days

**Run it:**
```bash
python predictive_maintenance.py
```

---

## “Whole pipeline” mental model (for future dev)

1. **Model/baseline generation** (likely in `forecasting hourly.ipynb`)  
   → produces or updates **`Results/*_series_new.csv`** with consistent columns.

2. **Exploration / validation** (Streamlit)  
   - `streamlit_main.py` orchestrates
   - `tab1..tab4.py` visualize different aspects (hourly forecast window, daily totals, differences, fleet ranking)

3. **Operational analytics** (Predictive Maintenance)  
   - `predictive_maintenance.py` reads Results and writes:
     - `Maintenance/daily_kpis_<meter>.csv`
     - `Maintenance/alerts_<meter>.csv`

---

## Development Tips / Gotchas

- Keep the `Results/*_series_new.csv` schema stable (column naming is the main integration contract).
- Prefer storing secrets in:
  - environment variables (`os.environ[...]`)
  - `.streamlit/secrets.toml` (for Streamlit)
  - or a vault/key store for production
- If adding new model outputs, add them as new columns (don’t break the existing required columns).
- If you change daylight rules, keep them consistent between Streamlit analytics and predictive maintenance (or clearly document differences).

