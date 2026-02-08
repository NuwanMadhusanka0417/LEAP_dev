# tab2.py — PowerBI-style Daily, tab-safe and global-meter aware
import os, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

# ---- call set_page_config only once (works when embedded in tabs) ----
if not st.session_state.get("_pc_done"):
    st.set_page_config(page_title="Solar PV Baselines — Daily (All Meters)", layout="wide")
    st.session_state["_pc_done"] = True

st.title("Solar PV Baselines (based on Solar Eng. Simulation Data) — Daily")

# ---- tab-unique widget key builder ----
def K(name: str) -> str:
    return f"{st.session_state.get('_tab_prefix','root')}::{name}"

RESULTS_DIR = "Results"

@st.cache_data
def list_meter_files(results_dir: str):
    pattern = os.path.join(results_dir, "*_series_new.csv")
    files = sorted(glob.glob(pattern))
    meter_to_path = {}
    for f in files:
        base = Path(f).stem  # e.g., "..._series_new"
        if base.endswith("_series_new"):
            base = base[:-len("_series_new")]
        meter_name = base.split("#", 1)[0]  # "solar.bun_hs1"
        meter_to_path[meter_name] = f
    return meter_to_path

@st.cache_data
def load_series(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Meter_reading (actual)": "Meter_reading",
        "Real model prediction": "Real_model_prediction",
        "Simulated model": "Simulated_model",
        "Simulated (degraded)": "Simulated_degraded",
    })

    needed = ["timestamp","meter","Meter_reading","Real_model_prediction","Simulated_model"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {Path(csv_path).name}: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    for c in ["Meter_reading","Real_model_prediction","Simulated_model","Simulated_degraded",
              "global_horizontal_irradiance","zenith"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (df
          .drop_duplicates(subset=["timestamp"])
          .sort_values("timestamp")
          .set_index("timestamp"))
    return df

# -------- Sidebar: pick meter (follow global) & prediction source --------
meter_map = list_meter_files(RESULTS_DIR)
if not meter_map:
    st.error(f"No files found in `{RESULTS_DIR}` matching *_series_new.csv")
    st.stop()

meters = sorted(meter_map.keys())

# follow_global = st.sidebar.checkbox("Follow global meter", value=True, key=K("follow_global"))
follow_global = True
global_m = st.session_state.get("global_meter")

if follow_global and global_m in meter_map:
    selected_meter = global_m
    st.sidebar.caption(f"Using global meter: **{selected_meter}**")
else:
    default_idx = meters.index(global_m) if global_m in meters else 0
    selected_meter = st.sidebar.selectbox("Meter", meters, index=default_idx, key=K("meter"))

# Let the user choose which prediction we call “Prediction”
# pred_source = st.sidebar.selectbox(
#     "Prediction source",
#     ["Simulated_model", "Real_model_prediction"],
#     index=0,
#     key=K("pred_src"),
# )

pred_source = "Simulated_model"

csv_path = meter_map[selected_meter]
df = load_series(csv_path)
df = df[df["meter"].notna()]  # safety

# ----- Date filter per meter -----
# min_d = pd.to_datetime(df["timestamp"].min()).date()
# max_d = pd.to_datetime(df["timestamp"].max()).date()


gdates = st.session_state.get("global_date")
min_d_all, max_d_all = df.index.min().date(), df.index.max().date()

if isinstance(gdates, (list, tuple)) and len(gdates) == 2 and gdates[0] and gdates[1]:
    m0 = pd.to_datetime(gdates[0])
    m1 = pd.to_datetime(gdates[1]) + pd.Timedelta(days=1)  # inclusive end
else:
    # fallback to full available range
    m0 = pd.to_datetime(min_d_all)
    m1 = pd.to_datetime(max_d_all) + pd.Timedelta(days=1)

# dfm = df[(df["timestamp"] >= d0) & (df["timestamp"] < d1)].copy()
dfm = df.loc[(df.index >= m0) & (df.index < m1)]

if dfm.empty:
    st.warning("No data in the selected date range.")
    st.stop()

st.subheader(f"Meter: {selected_meter}")
st.caption(f"Source file: {Path(csv_path).name}")

# ----- KPI cards -----
# dfm = dfm.set_index("timestamp")
act  = dfm["Meter_reading"].fillna(0)
sim  = dfm["Simulated_model"].fillna(0)
real = dfm["Real_model_prediction"].fillna(0)

prediction_series = sim if pred_source == "Simulated_model" else real
diff_kwh = float((prediction_series - act).sum())

# “Degradation % (vs Sim)” = (Sim - Actual)/Sim
deg_pct = float(((sim.sum() - act.sum()) / sim.sum()) * 100) if sim.sum() != 0 else np.nan

k1, k2 = st.columns(2)
k1.metric("Degradation % (vs Sim)", f"{deg_pct:,.2f}%" if np.isfinite(deg_pct) else "—")
k2.metric("Difference (kWh)", f"{diff_kwh:,.0f}")

# ----- Build daily series -----
locator   = mdates.AutoDateLocator(minticks=4, maxticks=8)
formatter = mdates.ConciseDateFormatter(locator)

ghi_daily = None
if "global_horizontal_irradiance" in dfm.columns:
    ghi_daily = dfm["global_horizontal_irradiance"].resample("D").mean()

daily = pd.DataFrame(index=pd.date_range(m0.normalize(), (m1 - pd.Timedelta(days=1)).normalize(), freq="D"))
daily["Prediction_kWh"] = prediction_series.resample("D").sum(min_count=1)
daily["Actual_kWh"]     = act.resample("D").sum(min_count=1)

# ----- Plot 1: GHI daily mean -----
fig1, ax1 = plt.subplots(figsize=(12, 3.8))
if ghi_daily is not None and not ghi_daily.empty:
    ax1.plot(ghi_daily.index, ghi_daily.values, lw=1.6, color="C0")
    ax1.set_ylabel("W/m²")
else:
    ax1.text(0.5, 0.5, "No GHI column in CSV", ha="center", va="center", transform=ax1.transAxes)

ax1.set_title("Global Horizontal Irradiance (W/m²)")
ax1.set_xlabel("")
ax1.xaxis.set_major_locator(locator); ax1.xaxis.set_major_formatter(formatter); fig1.autofmt_xdate()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# ----- Plot 2: Solar Generation (Prediction vs Actual) -----
fig2, ax2 = plt.subplots(figsize=(12, 4.2))
pred = daily["Prediction_kWh"]
actd = daily["Actual_kWh"]
c_pred, c_act = "tab:blue", "tab:orange"

if pred.notna().any():
    ax2.fill_between(daily.index, 0, pred.values, alpha=0.25, color=c_pred, label="Prediction (kWh)", zorder=1)
    ax2.plot(daily.index, pred.values, lw=1.8, color=c_pred, zorder=2)
if actd.notna().any():
    ax2.fill_between(daily.index, 0, actd.values, alpha=0.25, color=c_act, label="Actual (kWh)", zorder=1)
    ax2.plot(daily.index, actd.values,  lw=1.8, color=c_act,  zorder=2)

ax2.set_title("Solar Generation (kWh)")
ax2.set_ylabel("kWh"); ax2.set_xlabel("")
ax2.xaxis.set_major_locator(locator); ax2.xaxis.set_major_formatter(formatter); fig2.autofmt_xdate()
ax2.grid(True, alpha=0.3)
ax2.legend()
st.pyplot(fig2)
