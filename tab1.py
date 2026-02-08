# app2.py — tab-safe
import os
from pathlib import Path
import glob
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import matplotlib.dates as mdates

# --- Guard: only call set_page_config once, even when embedded in tabs ---
if not st.session_state.get("_pc_done"):
    st.set_page_config(page_title="Meter Series Viewer", layout="wide")
    st.session_state["_pc_done"] = True

st.title("Meter Series Viewer")

RESULTS_DIR = "Results"

# Tab-unique widget key builder
def K(name: str) -> str:
    return f"{st.session_state.get('_tab_prefix','root')}::{name}"

COLORS = {
    "actual": "green",
    "real":   "C0",      # blue
    "sim":    "C1",      # orange
    "degr":   "0.8",     # light gray
}

LABELS = {
    "actual": "Meter (Actual)",
    "real":   "Model (Real)",
    "sim":    "Model (Simulated)",
    "degr":   "Simulated (degraded)",
}

@st.cache_data
def list_meter_files(results_dir: str):
    pattern = os.path.join(results_dir, "*_series_new.csv")
    files = sorted(glob.glob(pattern))
    meter_to_path = {}
    for f in files:
        base = Path(f).stem  # e.g., "solar.bun_hs1#realenergyintotheload#kwh_series_new"
        if base.endswith("_series_new"):
            base = base[:-len("_series_new")]
        meter_name = base.split("#", 1)[0]  # "solar.bun_hs1"
        meter_to_path[meter_name] = f
    return meter_to_path

@st.cache_data
def load_series(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        "Meter_reading (actual)": "Meter_reading",
        "Real model prediction": "Real_model_prediction",
        "Simulated model": "Simulated_model",
        "Simulated (degraded)": "Simulated_degraded",
        "Meter reading": "Meter_reading",
    }
    df = df.rename(columns=rename_map)

    needed = ["timestamp","meter","campus",
              "Meter_reading","Real_model_prediction","Simulated_model","Simulated_degraded"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {csv_path}: {missing}")

    for c in ["Meter_reading","Real_model_prediction","Simulated_model","Simulated_degraded"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (df
          .drop_duplicates(subset=["timestamp"])
          .sort_values("timestamp")
          .set_index("timestamp"))
    return df

# ---------------- Load files ----------------
meter_map = list_meter_files(RESULTS_DIR)

meters = sorted(meter_map.keys())

# tab-unique key helper (safe across tabs)
def K(name: str) -> str:
    return f"{st.session_state.get('_tab_prefix','root')}::{name}"

# Follow the global meter from streamlit_main.py by default
follow_global = st.sidebar.checkbox("Follow global meter", value=True, key=K("follow_global"))

global_m = st.session_state.get("global_meter")
if follow_global and global_m in meter_map:
    selected_meter = global_m
    st.sidebar.caption(f"Using global meter: **{selected_meter}**")
else:
    # If user wants to override, preselect the global meter when available
    default_idx = meters.index(global_m) if global_m in meters else 0
    selected_meter = st.sidebar.selectbox("Meter", meters, index=default_idx, key=K("meter"))

# Use this selected_meter everywhere below
# csv_path = meter_map[selected_meter]


if not meter_map:
    st.warning(f"No *_series_new.csv files found in {RESULTS_DIR}/")
    st.stop()

# Sidebar: choose meter + series
# st.sidebar.header("Options")
# meter_names = sorted(meter_map.keys())
# selected_meter = st.sidebar.selectbox("Meter", meter_names, index=0, key=K("meter"))

ALL_SERIES = ["Meter_reading", "Real_model_prediction", "Simulated_model", "Simulated_degraded"]
selected_series = st.sidebar.multiselect(
    "Graph types to show",
    options=ALL_SERIES,
    default=ALL_SERIES,
    key=K("series_select")
)

# Load ALL data once; we'll make 2 filtered copies
csv_path = meter_map[selected_meter]
df_all = load_series(csv_path)

gdates = st.session_state.get("global_date")
min_d_all, max_d_all = df_all.index.min().date(), df_all.index.max().date()

if isinstance(gdates, (list, tuple)) and len(gdates) == 2 and gdates[0] and gdates[1]:
    m0 = pd.to_datetime(gdates[0])
    m1 = pd.to_datetime(gdates[1]) + pd.Timedelta(days=1)  # inclusive end
else:
    # fallback to full available range
    m0 = pd.to_datetime(min_d_all)
    m1 = pd.to_datetime(max_d_all) + pd.Timedelta(days=1)

df_main = df_all.loc[(df_all.index >= m0) & (df_all.index < m1)]

if df_main.empty:
    st.warning("No data in the GLOBAL main date range.")
    st.stop()

st.subheader(f"Meter: {selected_meter}")
st.caption(f"Source file: {Path(csv_path).name}")

# ---------------- MAIN CHART ----------------
if not selected_series:
    st.info("Select at least one graph type from the sidebar to display a chart.")
else:
    fig, ax = plt.subplots(figsize=(12, 5))
    plotted_any = False

    # Gray fill only if Simulated_degraded is selected
    if "Simulated_degraded" in selected_series and df_main["Simulated_degraded"].notna().any():
        s_sim_deg = df_main["Simulated_degraded"].dropna().sort_index()
        if not s_sim_deg.empty:
            ax.fill_between(
                s_sim_deg.index, 0, s_sim_deg.values,
                color="gray", alpha=0.35, label="Degradation gap", zorder=1, interpolate=True
            )
            plotted_any = True

    # Lines
    if "Real_model_prediction" in selected_series and df_main["Real_model_prediction"].notna().any():
        ax.plot(df_main.index, df_main["Real_model_prediction"],color=COLORS["real"],  label="Real model prediction", zorder=2)
        plotted_any = True

    if "Simulated_model" in selected_series and df_main["Simulated_model"].notna().any():
        ax.plot(df_main.index, df_main["Simulated_model"], color=COLORS["sim"],   label="Simulated model prediction", linestyle="-", zorder=2)
        plotted_any = True

    if "Meter_reading" in selected_series and df_main["Meter_reading"].notna().any():
        s_act = df_main["Meter_reading"].dropna().sort_index()
        if not s_act.empty:
            ax.plot(s_act.index, s_act.values, label="Meter reading (actual)",linewidth=1.5, color=COLORS["actual"],   alpha=0.95, zorder=3, markersize=2.5)
            plotted_any = True

    ax.set_title("Forecast comparison")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Predicted grid_power_kWh")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    if plotted_any:
        ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

# (Optional) data preview for MAIN
with st.expander("Show data (main graph)"):
    cols = ["meter","campus"] + selected_series
    cols = [c for c in cols if c in df_main.columns or c in ["meter","campus"]]
    st.dataframe(df_main[cols])

# ==================== ANALYTICS (3 CHARTS) ====================
st.markdown("---")
st.subheader("Analytics")

# ---------------- Date picker 2: ANALYTICS ----------------
min_d_ana, max_d_ana = df_all.index.min().date(), df_all.index.max().date()
picked_ana = st.sidebar.date_input(
    "Analytics date range",
    value=(min_d_ana, max_d_ana),
    min_value=min_d_ana, max_value=max_d_ana,
    key=K("analytics_date_range")  # unique per tab
)

# Build df_ana from df_all (independent of df_main)
df_ana = df_all
if isinstance(picked_ana, tuple) and len(picked_ana) == 2:
    a0, a1 = picked_ana
    if a0 and a1:
        if a1 < a0:
            a0, a1 = a1, a0
        a0 = pd.to_datetime(a0)
        a1 = pd.to_datetime(a1) + pd.Timedelta(days=1)
        df_ana = df_all.loc[(df_all.index >= a0) & (df_all.index < a1)]
elif picked_ana:
    a0 = pd.to_datetime(picked_ana)
    a1 = a0 + pd.Timedelta(days=1)
    df_ana = df_all.loc[(df_all.index >= a0) & (df_all.index < a1)]

if df_ana.empty:
    st.warning("No data in the selected ANALYTICS date range.")
    st.stop()

# ---------- Build analytics series ----------
if {"global_horizontal_irradiance","zenith"} <= set(df_ana.columns):
    ghi = pd.to_numeric(df_ana["global_horizontal_irradiance"], errors="coerce").fillna(0.0)
    zen = pd.to_numeric(df_ana["zenith"], errors="coerce").fillna(180.0)
    is_day = (ghi > 5.0) & (zen < 90.0)
else:
    is_day = (df_ana["Simulated_model"].astype(float) > 0.05)

dtmp = pd.DataFrame(
    {
        "Act":  df_ana["Meter_reading"].astype(float),
        "Sim":  df_ana["Simulated_model"].astype(float),
        "Real": df_ana["Real_model_prediction"].astype(float),
        "is_day": is_day.values,
    },
    index=df_ana.index,
)

d_day = dtmp[dtmp["is_day"]].copy()
daily_energy = d_day.resample("D").sum()[["Act","Sim","Real"]]

daily_energy["H"] = np.divide(
    daily_energy["Act"], daily_energy["Sim"],
    out=np.full(len(daily_energy), np.nan, dtype=float),
    where=(daily_energy["Sim"] > 0)
)
daily_energy["T"] = np.divide(
    daily_energy["Act"], daily_energy["Real"],
    out=np.full(len(daily_energy), np.nan, dtype=float),
    where=(daily_energy["Real"] > 0)
)

daily_energy["H_7d_med"] = daily_energy["H"].rolling(7, min_periods=3).median()
daily_energy["T_7d_med"] = daily_energy["T"].rolling(7, min_periods=3).median()

model_gap = (df_ana["Simulated_degraded"].astype(float) - df_ana["Real_model_prediction"].astype(float))
tidx = df_ana.index

c1, c2, c3 = st.columns(3, gap="large")

locator   = mdates.AutoDateLocator(minticks=4, maxticks=8)
formatter = mdates.ConciseDateFormatter(locator)

st.sidebar.markdown("### What’s what?")
st.sidebar.write(
"""**Meter (Actual)**: Site meter readings.

**Model (Real)**: Predictions from the model trained on real (meter) labels.

**Model (Simulated)**: Predictions from the model trained on simulated labels.

**Simulated (degraded)**: Simulated predictions adjusted by panel degradation."""
)

# 1) Health Ratio
with c1:
    fig_hs, ax_hs = plt.subplots(figsize=(5.5, 3.6))
    ax_hs.plot(daily_energy.index, daily_energy["H"], ".", alpha=0.6, label="Daily H (energy-based)")
    if daily_energy["H_7d_med"].notna().any():
        ax_hs.plot(daily_energy.index, daily_energy["H_7d_med"], linewidth=1.6, label="7-day median")
    ax_hs.axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.8, label="1.00")
    ax_hs.axhline(0.90, linestyle="--", linewidth=0.8, alpha=0.8, label="0.90")
    ax_hs.set_title("Health Ratio — (Actual ÷ Simulated)")
    ax_hs.set_xlabel("Date"); ax_hs.set_ylabel("H")
    ax_hs.xaxis.set_major_locator(locator); ax_hs.xaxis.set_major_formatter(formatter); fig_hs.autofmt_xdate()
    ax_hs.grid(True, alpha=0.3)
    ax_hs.set_ylim(0, 2)
    ax_hs.legend(loc="upper right", fontsize=8)
    fig_hs.tight_layout()
    st.pyplot(fig_hs)

# 2) Tracking Ratio
with c2:
    fig_hr, ax_hr = plt.subplots(figsize=(5.5, 3.6))
    ax_hr.plot(daily_energy.index, daily_energy["T"], ".", alpha=0.6, label="Daily T (energy-based)")
    if daily_energy["T_7d_med"].notna().any():
        ax_hr.plot(daily_energy.index, daily_energy["T_7d_med"], linewidth=1.6, label="7-day median")
    ax_hr.axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.8, label="1.00")
    ax_hr.set_title("Tracking Ratio — (Actual ÷ Real)")
    ax_hr.set_xlabel("Date"); ax_hr.set_ylabel("T")
    ax_hr.xaxis.set_major_locator(locator); ax_hr.xaxis.set_major_formatter(formatter); fig_hr.autofmt_xdate()
    ax_hr.grid(True, alpha=0.3)
    ax_hr.set_ylim(0, 2)
    ax_hr.legend(loc="upper right", fontsize=8)
    fig_hr.tight_layout()
    st.pyplot(fig_hr)

# 3) Model Gap — daily band
with c3:
    if "is_day" in locals() and isinstance(is_day, pd.Series):
        day_mask_gap = is_day.reindex(tidx).fillna(False).values
    else:
        day_mask_gap = np.ones(len(tidx), dtype=bool)

    gtmp = pd.DataFrame({"gap": model_gap, "is_day": day_mask_gap}, index=tidx)
    gday = gtmp[gtmp["is_day"]].copy()

    def p10(x): return np.nanpercentile(x, 10)
    def p90(x): return np.nanpercentile(x, 90)

    daily_gap = gday.resample("D")["gap"].agg(["median", p10, p90]).rename(
        columns={"median": "med", "p10": "p10", "p90": "p90"}
    )
    daily_gap["med_7d"] = daily_gap["med"].rolling(7, min_periods=3).median()

    lo = np.nanpercentile(daily_gap[["p10","p90"]].values, 5)
    hi = np.nanpercentile(daily_gap[["p10","p90"]].values, 95)
    y_min, y_max = max(-10, lo), min(10, hi)

    fig_gap, ax_gap = plt.subplots(figsize=(5.5, 3.6))
    ax_gap.fill_between(daily_gap.index, daily_gap["p10"], daily_gap["p90"],
                        alpha=0.25, label="Daylight gap range (10–90%)")
    ax_gap.plot(daily_gap.index, daily_gap["med"], ".", alpha=0.6, label="Daily median")
    if daily_gap["med_7d"].notna().any():
        ax_gap.plot(daily_gap.index, daily_gap["med_7d"], lw=1.8, label="7-day median")

    ax_gap.axhline(0.0, ls="--", lw=1.0, alpha=0.8, color="grey")
    ax_gap.set_title("Model Gap — (Simulated − Real)")
    ax_gap.set_xlabel("Date"); ax_gap.set_ylabel("Gap (kWh)")
    ax_gap.xaxis.set_major_locator(locator); ax_gap.xaxis.set_major_formatter(formatter); fig_hs.autofmt_xdate()
    ax_gap.grid(True, alpha=0.3)
    ax_gap.set_ylim(y_min, y_max)
    ax_gap.legend(loc="upper right", fontsize=8)

    fig_gap.tight_layout()
    st.pyplot(fig_gap)

# (Optional) data preview for ANALYTICS
with st.expander("Show data (analytics calculations)"):
    show_cols = ["meter","campus","Meter_reading","Real_model_prediction","Simulated_model","Simulated_degraded"]
    show_cols = [c for c in show_cols if c in df_ana.columns]
    st.dataframe(df_ana[show_cols])

#################################################################################################################
###########  Next 7 day analysis  ###############################################################################
#################################################################################################################

st.markdown("---")
st.subheader("Next Few Days — Forecast (Model: Real vs Simulated)")

# Unique slider key
h_days = st.sidebar.slider("Forecast horizon (days)", min_value=3, max_value=14, value=7, key=K("forecast_horizon_days"))

# Start from the last available Meter reading timestamp (next hour)
if "Meter_reading" in df_all.columns:
    last_actual_ts = df_all.loc[df_all["Meter_reading"].astype(float).notna()].index.max()
else:
    last_actual_ts = None

if pd.isna(last_actual_ts):
    st.info("No meter readings found. Showing forecast from the first available prediction timestamp.")
    start_ts = df_all.index.min().ceil("H")
else:
    start_ts = (pd.to_datetime(last_actual_ts)).ceil("H")

end_ts = start_ts + pd.Timedelta(days=h_days)

# Extract forecast slice
cols_needed = [c for c in ["Real_model_prediction", "Simulated_model", "global_horizontal_irradiance", "zenith"] if c in df_all.columns]
df_fc = df_all.loc[(df_all.index >= start_ts) & (df_all.index < end_ts), cols_needed].copy()

if df_fc.empty or (df_fc[["Real_model_prediction","Simulated_model"]].dropna(how="all").empty):
    st.warning("No prediction data available in the selected forecast window.")
else:
    # Daytime mask for the forecast
    if {"global_horizontal_irradiance","zenith"} <= set(df_fc.columns):
        ghi = pd.to_numeric(df_fc["global_horizontal_irradiance"], errors="coerce").fillna(0.0)
        zen = pd.to_numeric(df_fc["zenith"], errors="coerce").fillna(180.0)
        is_day_fc = (ghi > 5.0) & (zen < 90.0)
    else:
        is_day_fc = (df_fc[["Real_model_prediction","Simulated_model"]].max(axis=1) > 0.05)

    s_real = pd.to_numeric(df_fc["Real_model_prediction"], errors="coerce")
    s_sim  = pd.to_numeric(df_fc["Simulated_model"],      errors="coerce")
    df_fc["gap"] = s_sim - s_real
    df_fc["gap_med7h"] = df_fc["gap"].rolling(7, min_periods=1, center=True).median()
    df_fc["cum_gap"] = df_fc["gap"].cumsum()

    daily_fc = pd.DataFrame({"Real_model": s_real, "Sim_model": s_sim})
    daily_fc = daily_fc.resample("D").sum(min_count=1)
    daily_fc["pct_diff"] = 100.0 * (daily_fc["Real_model"] - daily_fc["Sim_model"]) / daily_fc["Sim_model"].replace(0, np.nan)

    c1, c2, c3 = st.columns(3, gap="large")

    try:
        _loc = locator; _fmt = formatter
    except NameError:
        _loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
        _fmt = mdates.ConciseDateFormatter(_loc)

    # Panel 1: Daily energy bars
    with c1:
        fig1, ax1 = plt.subplots(figsize=(5.5, 3.6))
        x = np.arange(len(daily_fc.index))
        w = 0.38
        ax1.bar(x - w/2, daily_fc["Real_model"].values, width=w, label="Model (Real)", alpha=0.9)
        ax1.bar(x + w/2, daily_fc["Sim_model"].values, width=w, label="Model (Simulated)", alpha=0.9)
        for i, (xr, xs, pdiff) in enumerate(zip(daily_fc["Real_model"], daily_fc["Sim_model"], daily_fc["pct_diff"])):
            if np.isfinite(pdiff):
                ax1.text(i + w/2, max(xr, xs) * (1.02 if max(xr, xs) > 0 else 0.02),
                         f"{pdiff:+.0f}%", ha="center", va="bottom", fontsize=9)
        ax1.set_xticks(x, [d.strftime("%b %d") for d in daily_fc.index])
        ax1.set_ylabel("Daily energy (kWh)")
        ax1.set_title("Daily Energy — Next Days")
        ax1.grid(True, axis="y", alpha=0.3)
        ax1.legend()
        fig1.tight_layout()
        st.pyplot(fig1)

    # Panel 2: Hourly gap
    with c2:
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.6))
        dm = is_day_fc.astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, dm.values, 0]) != 0)
        for start, end in edges.reshape(-1, 2):
            if dm.iloc[start] == 1:
                ax2.axvspan(df_fc.index[start], df_fc.index[end-1], color="C0", alpha=0.08)
        ax2.plot(df_fc.index, df_fc["gap"], ".", ms=3, alpha=0.5, label="Hourly gap (Sim − Real)")
        ax2.plot(df_fc.index, df_fc["gap_med7h"], lw=2, label="7-hour median")
        ax2.axhline(0, color="0.6", lw=1, ls="--")
        ax2.set_ylabel("Gap (kWh)")
        ax2.set_title("Hourly Gap (Simulated - Real)")
        ax2.xaxis.set_major_locator(_loc); ax2.xaxis.set_major_formatter(_fmt); fig2.autofmt_xdate()
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        fig2.tight_layout()
        st.pyplot(fig2)

    # Panel 3: Cumulative difference
    with c3:
        fig3, ax3 = plt.subplots(figsize=(5.5, 3.6))
        ax3.plot(df_fc.index, df_fc["cum_gap"], lw=2, color="C1")
        ax3.axhline(0, color="0.6", lw=1, ls="--")
        ax3.set_ylabel("Cumulative gap (kWh)")
        ax3.set_title("Cumulative Difference — Simulated vs Real")
        ax3.xaxis.set_major_locator(_loc); ax3.xaxis.set_major_formatter(_fmt); fig3.autofmt_xdate()
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        st.pyplot(fig3)

    weekly_real = float(daily_fc["Real_model"].sum())
    weekly_sim  = float(daily_fc["Sim_model"].sum())
    weekly_diff = weekly_sim - weekly_real
    weekly_pcnt = (100.0 * weekly_diff / weekly_real) if weekly_real else np.nan

    st.caption(
        f"**Forecast window:** {start_ts:%Y-%m-%d %H:%M} → {(end_ts - pd.Timedelta(hours=1)):%Y-%m-%d %H:%M}  \n"
        f"**Totals:** Real = {weekly_real:.1f} kWh, Simulated = {weekly_sim:.1f} kWh "
        f"(**Δ = {weekly_diff:+.1f} kWh, {weekly_pcnt:+.1f}%**)."
    )
