# tab4.py — Fleet view: rank meters by degradation trend (slope)
import os, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---- call set_page_config only once (safe under tabs) ----
if not st.session_state.get("_pc_done"):
    st.set_page_config(page_title="Solar — Fleet Trend (All Meters)", layout="wide")
    st.session_state["_pc_done"] = True

# Unique widget keys per tab
def K(name: str) -> str:
    return f"{st.session_state.get('_tab_prefix','root')}::{name}"

st.title("Solar PV Baselines (based on Solar Eng. Simulation Data) — Daily")

RESULTS_DIR = "Results"

# ---------- helpers ----------
@st.cache_data
def list_meter_files(results_dir: str):
    pattern = os.path.join(results_dir, "*_series_new.csv")
    files = sorted(glob.glob(pattern))
    m = {}
    for f in files:
        base = Path(f).stem
        if base.endswith("_series_new"):
            base = base[:-len("_series_new")]
        meter_name = base.split("#", 1)[0]
        m[meter_name] = f
    return m

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
    need = ["timestamp","meter","Meter_reading","Real_model_prediction","Simulated_model"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in {Path(csv_path).name}: {miss}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    for c in ["Meter_reading","Real_model_prediction","Simulated_model","Simulated_degraded",
              "global_horizontal_irradiance","zenith"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_slope(diff_daily: pd.Series) -> float:
    """Return slope (kWh per day) of daily difference, NaN if too short."""
    s = diff_daily.dropna()
    if len(s) < 10:
        return np.nan
    t = (s.index - s.index.min()).days.astype(float)
    slope, _ = np.polyfit(t, s.values, 1)
    return float(slope)

@st.cache_data
def global_minmax(m2p: dict):
    mins, maxs = [], []
    for p in m2p.values():
        df = load_series(p)
        if not df.empty:
            mins.append(df["timestamp"].min())
            maxs.append(df["timestamp"].max())
    if not mins or not maxs:
        return None, None
    return pd.to_datetime(min(mins)).date(), pd.to_datetime(max(maxs)).date()

# ---------- inputs ----------
# st.sidebar.header("Controls")

meter_map = list_meter_files(RESULTS_DIR)
if not meter_map:
    st.error(f"No files found in `{RESULTS_DIR}` matching *_series_new.csv")
    st.stop()

# Use MAIN app's date range (saved by streamlit_main.py as ('YYYY-MM-DD','YYYY-MM-DD'))
global_range = st.session_state.get("global_date")  # (start_date, end_date) from main
gmin, gmax = global_minmax(meter_map)

if isinstance(global_range, (list, tuple)) and len(global_range) == 2:
    d0 = pd.to_datetime(global_range[0])
    d1 = pd.to_datetime(global_range[1]) + pd.Timedelta(days=1)
else:
    # fallback: full available range
    if gmin is None:
        st.warning("No timestamps in source CSVs.")
        st.stop()
    d0 = pd.to_datetime(gmin)
    d1 = pd.to_datetime(gmax) + pd.Timedelta(days=1)

# pred_source = st.sidebar.selectbox(
#     "Prediction source",
#     ["Simulated_model", "Real_model_prediction", "Simulated_degraded"],
#     index=0, key=K("pred_src")
# )

pred_source = "Simulated_model"
# daylight_only = st.sidebar.checkbox(
#     "Daylight only (GHI>5 & zenith<90°)", value=True, key=K("day")
# )
daylight_only = False
# ---------- compute per-meter slopes ----------
rows = []
for meter, path in meter_map.items():
    df = load_series(path)
    if df.empty:
        continue

    df = df[(df["timestamp"] >= d0) & (df["timestamp"] < d1)].copy()
    if df.empty:
        continue

    df = df.set_index("timestamp").sort_index()
    if daylight_only and {"global_horizontal_irradiance","zenith"} <= set(df.columns):
        is_day = (df["global_horizontal_irradiance"] > 5.0) & (df["zenith"] < 90.0)
    else:
        is_day = pd.Series(True, index=df.index)

    act  = df.loc[is_day, "Meter_reading"].astype(float)
    if pred_source not in df.columns:
        continue
    pred = df.loc[is_day, pred_source].astype(float)

    act_d  = act.resample("D").sum(min_count=1)
    pred_d = pred.resample("D").sum(min_count=1)
    diff_d = (pred_d - act_d)

    slope = compute_slope(diff_d)
    total_diff = float(diff_d.sum(skipna=True))
    n_days = int(diff_d.notna().sum())

    rows.append({
        "Meter": meter,
        "Slope_kWh_per_day": slope,
        "Total_Difference_kWh": total_diff,
        "Points": n_days,
        "SourceFile": Path(path).name
    })

res = pd.DataFrame(rows).sort_values("Slope_kWh_per_day", ascending=False, na_position="last")

st.subheader("Slope of Degradation Trend")
st.caption(
    f"Prediction series: **{pred_source}**   •   Window: **{d0.date()} → {(d1 - pd.Timedelta(days=1)).date()}**   •   Daylight: **{daylight_only}**"
)

if res.empty:
    st.warning("No meters have data in the selected window.")
    st.stop()

# ---------- bar chart (horizontal) ----------
# top_n = st.sidebar.slider(
#     "Top N", min_value=5, max_value=min(50, len(res)), value=min(20, len(res)), key=K("topn")
# )

top_n = len(res)
plot_df = res.head(top_n)

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(plot_df["Meter"][::-1], plot_df["Slope_kWh_per_day"][::-1], color="#1f77b4")
ax.set_xlabel("Slope (kWh/day)")
ax.set_ylabel("Meter")
ax.set_title("Slope of Degradation Trend (Prediction − Actual)")
ax.grid(axis="x", alpha=0.3)
st.pyplot(fig)

# ---------- data table + download ----------
with st.expander("Show / Download table"):
    st.dataframe(res, use_container_width=True)
    csv = res.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"fleet_trend_{pred_source}_{d0.date()}_{(d1 - pd.Timedelta(days=1)).date()}.csv",
        mime="text/csv",
        key=K("dl")
    )
