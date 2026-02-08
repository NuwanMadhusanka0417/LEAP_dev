# tab3.py  — Difference (Prediction − Actual), tab-safe & global-meter aware
import os, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

# Call set_page_config only once (safe when embedded in tabs)
if not st.session_state.get("_pc_done"):
    st.set_page_config(page_title="Solar — Difference (Prediction − Actual)", layout="wide")
    st.session_state["_pc_done"] = True

# Build unique widget keys per tab
def K(name: str) -> str:
    return f"{st.session_state.get('_tab_prefix','root')}::{name}"

st.title("Solar PV Baselines (based on Solar Eng. Simulation Data) — Daily")

RESULTS_DIR = "Results"

# ----------------- helpers -----------------
@st.cache_data
def list_meter_files(results_dir: str):
    pattern = os.path.join(results_dir, "*_series_new.csv")
    files = sorted(glob.glob(pattern))
    meter_to_path = {}
    for f in files:
        base = Path(f).stem  # e.g., "solar.bun_x#realenergyintotheload#kwh_series_new"
        if base.endswith("_series_new"):
            base = base[:-len("_series_new")]
        meter_name = base.split("#", 1)[0]  # "solar.bun_x"
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
    need = ["timestamp", "meter", "Meter_reading", "Real_model_prediction", "Simulated_model"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in {Path(csv_path).name}: {miss}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    for c in ["Meter_reading","Real_model_prediction","Simulated_model",
              "global_horizontal_irradiance","zenith"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (df
          .drop_duplicates(subset=["timestamp"])
          .sort_values("timestamp")
          .set_index("timestamp"))
    return df

# ----------------- sidebar controls -----------------
# st.sidebar.header("Controls")

meter_map = list_meter_files(RESULTS_DIR)
if not meter_map:
    st.error(f"No files found in `{RESULTS_DIR}` matching *_series_new.csv")
    st.stop()

meters = sorted(meter_map.keys())

# Follow global meter by default (from streamlit_main.py)
# follow_global = st.sidebar.checkbox("Follow global meter", value=True, key=K("follow_global"))
global_m = st.session_state.get("global_meter")
follow_global = True
if follow_global and global_m in meter_map:
    selected_meter = global_m
    st.sidebar.caption(f"Using global meter: **{selected_meter}**")
else:
    default_idx = meters.index(global_m) if global_m in meters else 0
    selected_meter = st.sidebar.selectbox("Meter", meters, index=default_idx, key=K("meter"))
# selected_meter = global_m
# Choose which prediction to compare against actual
# pred_source = st.sidebar.selectbox(
#     "Prediction source", ["Simulated_model", "Real_model_prediction"], index=0, key=K("pred_src")
# )

pred_source = "Simulated_model"

csv_path = meter_map[selected_meter]
df = load_series(csv_path)

# min_d = pd.to_datetime(df["timestamp"].min()).date()
# max_d = pd.to_datetime(df["timestamp"].max()).date()
# date_range = st.sidebar.date_input(
#     "Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d, key=K("date")
# )

# if isinstance(date_range, tuple) and len(date_range) == 2:
#     d0 = pd.to_datetime(date_range[0])
#     d1 = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
# else:
#     d0 = pd.to_datetime(date_range)
#     d1 = d0 + pd.Timedelta(days=1)

gdates = st.session_state.get("global_date")
min_d_all, max_d_all = df.index.min().date(), df.index.max().date()

if isinstance(gdates, (list, tuple)) and len(gdates) == 2 and gdates[0] and gdates[1]:
    d0 = pd.to_datetime(gdates[0])
    d1 = pd.to_datetime(gdates[1]) + pd.Timedelta(days=1)  # inclusive end
else:
    # fallback to full available range
    d0 = pd.to_datetime(min_d_all)
    d1 = pd.to_datetime(max_d_all) + pd.Timedelta(days=1)



st.subheader(f"Meter: {selected_meter}")
st.caption(f"Source file: {Path(csv_path).name}")

# ----------------- filter & build daily series -----------------
# df = df[(df["timestamp"] >= d0) & (df["timestamp"] < d1)].copy()
df = df.loc[(df.index >= d0) & (df.index < d1)]

if df.empty:
    st.warning("No data in selected date range.")
    st.stop()

# df = df.set_index("timestamp")

# daylight mask (fallback to all True)
if {"global_horizontal_irradiance","zenith"} <= set(df.columns):
    is_day = (df["global_horizontal_irradiance"] > 5) & (df["zenith"] < 90)
else:
    is_day = pd.Series(True, index=df.index)

ghi_daily = (df.loc[is_day, "global_horizontal_irradiance"].resample("D").mean()
             if "global_horizontal_irradiance" in df.columns else pd.Series(dtype=float))

act_d  = df.loc[is_day, "Meter_reading"].resample("D").sum(min_count=1)
pred_d = df.loc[is_day, pred_source].resample("D").sum(min_count=1)

diff_d = (pred_d - act_d).rename("Diff_kWh").dropna()
med7   = diff_d.rolling(7, min_periods=3).median()

# slope & trendline
if len(diff_d) >= 10:
    t = (diff_d.index - diff_d.index.min()).days.astype(float)
    slope, intercept = np.polyfit(t, diff_d.values, 1)  # kWh/day
    trendline = pd.Series(intercept + slope * t, index=diff_d.index)
else:
    slope, intercept, trendline = np.nan, np.nan, None

total_diff = float(diff_d.sum())

# ----------------- KPIs -----------------
k1, k2 = st.columns(2)
k1.metric("Slope (kWh/day)", f"{slope:,.2f}" if np.isfinite(slope) else "—")
k2.metric("Difference (kWh)", f"{total_diff:,.0f}")

# ----------------- plots -----------------
locator   = mdates.AutoDateLocator(minticks=4, maxticks=8)
formatter = mdates.ConciseDateFormatter(locator)

# Top: GHI
fig1, ax1 = plt.subplots(figsize=(12, 3.6))
if not ghi_daily.empty:
    ax1.plot(ghi_daily.index, ghi_daily.values, lw=1.6, color="C0")
    ax1.set_ylabel("W/m²")
ax1.set_title("Global Horizontal Irradiance (W/m²)")
ax1.set_xlabel("")
ax1.xaxis.set_major_locator(locator); ax1.xaxis.set_major_formatter(formatter); fig1.autofmt_xdate()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# Bottom: Difference bars + trend
fig2, ax2 = plt.subplots(figsize=(12, 4.6))
pos = diff_d.clip(lower=0.0)
neg = diff_d.clip(upper=0.0)

ax2.bar(pos.index, pos.values, width=1.0, color="#e74c3c", alpha=0.9, label="Pred − Actual (>0)")
ax2.bar(neg.index, neg.values, width=1.0, color="#3498db", alpha=0.9, label="Pred − Actual (<0)")

if trendline is not None:
    ax2.plot(trendline.index, trendline.values, ls="--", lw=2.0, color="black", label="Trend")
ax2.plot(med7.index, med7.values, lw=1.6, color="#2c3e50", alpha=0.9, label="7-day median")

ax2.axhline(0, color="grey", lw=1.0, ls="--")
ax2.set_title(f"Difference (kWh) — {pred_source} minus Actual")
ax2.set_ylabel("kWh"); ax2.set_xlabel("Timestamp")
ax2.xaxis.set_major_locator(locator); ax2.xaxis.set_major_formatter(formatter); fig2.autofmt_xdate()
ax2.grid(True, alpha=0.3)
ax2.legend()
st.pyplot(fig2)

# ----------------- table & download -----------------
with st.expander("Show / Download daily table"):
    out = pd.DataFrame({
        "Actual_kWh": act_d, "Prediction_kWh": pred_d,
        "Diff_kWh": diff_d, "Diff_7dMedian": med7
    })
    st.dataframe(out)
    csv = out.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV",
                       data=csv,
                       file_name=f"{selected_meter}_daily_difference.csv",
                       mime="text/csv",
                       key=K("dl"))
