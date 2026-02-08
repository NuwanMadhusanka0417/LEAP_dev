# streamlit_main.py
import os, glob
from pathlib import Path
import pandas as pd
import streamlit as st
import runpy

st.set_page_config(page_title="Solar Dashboards", layout="wide")
st.title("Solar Dashboards")

# ---------- discovery ----------
@st.cache_data
def list_meter_files(results_dir: str = "Results"):
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
def global_minmax_timestamps(meter_to_path: dict):
    """Return (min_date, max_date) across all CSVs, or (None, None) if empty."""
    mins, maxs = [], []
    for p in meter_to_path.values():
        try:
            df = pd.read_csv(p, usecols=["timestamp"])
            if df.empty: 
                continue
            t = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
            if not t.empty:
                mins.append(t.min())
                maxs.append(t.max())
        except Exception:
            continue
    if not mins or not maxs:
        return None, None
    return mins[0].date() if mins else None, max(maxs).date() if maxs else None

meter_map_global = list_meter_files("Results")
meters_global    = sorted(meter_map_global.keys()) or ["<no files found>"]

# ---------- GLOBAL SIDEBAR CONTROLS ----------
st.sidebar.header("Global controls")

# Global meter (used by tabs unless they override)
st.session_state["global_meter"] = st.sidebar.selectbox(
    "Meter (applies to all tabs)",
    meters_global,
    index=0,
    key="GLOBAL_METER_SELECT",
)

# Global date range (tabs read st.session_state['global_date'])
gmin, gmax = global_minmax_timestamps(meter_map_global)
if gmin is None or gmax is None:
    # sensible fallback if no files
    gmin = pd.to_datetime("2000-01-01").date()
    gmax = pd.to_datetime("today").date()

global_range = st.sidebar.date_input(
    "Main Date Range",
    value=(gmin, gmax),
    min_value=gmin,
    max_value=gmax,
    key="GLOBAL_MAIN_DATERANGE",
)
# Normalize and save for tabs
if isinstance(global_range, (list, tuple)) and len(global_range) == 2:
    st.session_state["global_date"] = (pd.to_datetime(global_range[0]).date(),
                                       pd.to_datetime(global_range[1]).date())
else:
    # single-date selection fallback -> one day window
    d0 = pd.to_datetime(global_range).date()
    st.session_state["global_date"] = (d0, d0)

# ---------- run helpers ----------
def run_child(path: str, prefix: str):
    # give the child a unique namespace for widget keys
    st.session_state["_tab_prefix"] = prefix
    try:
        runpy.run_path(path, run_name="__main__")
    finally:
        st.session_state.pop("_tab_prefix", None)

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Difference", "PowerBI-style Daily", "Series Viewer", "Fleet Trend"]
)

with tab1:
    st.markdown("### Difference (Prediction − Actual)")
    run_child("tab1.py", "diff")

with tab2:
    st.markdown("### PowerBI-style Daily")
    run_child("tab2.py", "pbi")

with tab3:
    st.markdown("### Series Viewer")
    run_child("tab3.py", "series")

with tab4:
    st.markdown("### Fleet Trend (All Meters)")
    run_child("tab4.py", "t4")
