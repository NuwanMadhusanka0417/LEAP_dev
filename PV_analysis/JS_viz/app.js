/**
 * Tabbed dashboard: loads PV_analysis/data_for_viz CSVs and renders Plotly charts.
 */
import {
  parseCSV,
  parseHour,
  fetchDataVizFile,
  purgeAllPlotlyInRoot,
  tryFetchPvlibFile,
} from "./utils.js";
import { renderDifference } from "./tab_difference.js";
import { renderPowerBIDaily } from "./tab_powerbi_daily.js";
import { renderSeries } from "./tab_series.js";
import { renderFleet } from "./tab_fleet.js";
import { renderLibraryPVLib } from "./tab_library_pvlib.js";
import { renderSeasonPerf } from "./tab_season_perf.js";
import { renderForecast } from "./tab_forecast.js";

const PRECOMPUTED_PVLIB_HOURLY = "expected_power_pvlib_cleaned_v2.csv";

let state = {
  hourly: [],
  kpiRows: [],
  /** Rows from forecast_7d_combined_library.csv (PVLib vs XGBoost 7d window). */
  forecastCombined: [],
  dateFrom: "",
  dateTo: "",
  /** yyyy-mm-dd from loaded hourly */
  dataMin: "",
  dataMax: "",
  /** True when hourly rows have a second-model kWh (CSV column or merged precomputed file). */
  hasSecondModelBaseline: false,
};

function isoFromDate(d) {
  return d.toISOString().slice(0, 10);
}

function addDays(isoStr, n) {
  const d = new Date(isoStr + "T12:00:00");
  d.setDate(d.getDate() + n);
  return isoFromDate(d);
}

function addMonths(isoStr, n) {
  const d = new Date(isoStr + "T12:00:00");
  d.setMonth(d.getMonth() + n);
  return isoFromDate(d);
}

function normalizeHourly(rows) {
  const out = [];
  for (const r of rows) {
    const ts = parseHour(r.timestamp);
    if (!ts) continue;
    const day = r.timestamp.trim().slice(0, 10);
    const legRaw =
      r.legacy_expected_kwh ??
      r.expected_power ??
      r.real_expected_kwh ??
      r.Real_model_prediction ??
      "";
    const legacy =
      legRaw !== undefined && legRaw !== "" && String(legRaw).trim() !== ""
        ? Number(legRaw)
        : NaN;
    out.push({
      ts,
      tsStr: r.timestamp.trim(),
      day,
      actual: Number(r.actual_kwh),
      expected: Number(r.expected_kwh),
      ghi: Number(r.ghi_wm2),
      legacy: Number.isFinite(legacy) ? legacy : NaN,
    });
  }
  return out.sort((a, b) => a.ts - b.ts);
}

/**
 * When hourly_library_master has no legacy/Real column (tab1 Real_model_prediction),
 * join precomputed PVLib hourly expected_kwh by timestamp so T = Actual÷Real and gap = Sim−Real can run.
 */
function mergeLegacyFromPrecomputedHourly(hourly, pvlibRows) {
  const map = new Map();
  for (const r of pvlibRows) {
    const k = (r.timestamp ?? "").trim();
    if (!k) continue;
    const v = Number(r.expected_kwh);
    if (Number.isFinite(v)) map.set(k, v);
  }
  let filled = 0;
  for (const row of hourly) {
    if (Number.isFinite(row.legacy)) continue;
    const v = map.get(row.tsStr);
    if (Number.isFinite(v)) {
      row.legacy = v;
      filled++;
    }
  }
  return filled;
}

function refreshSecondModelFlag() {
  state.hasSecondModelBaseline = state.hourly.some((r) => Number.isFinite(r.legacy));
}

function normalizeForecastCombined(rows) {
  const out = [];
  for (const r of rows) {
    const tsStr = (r.timestamp ?? "").trim();
    const ts = parseHour(tsStr);
    if (!ts) continue;
    const exp = Number(r.expected_kwh_pvlib);
    const pred = Number(r.predicted_kwh_xgboost);
    if (!Number.isFinite(exp) || !Number.isFinite(pred)) continue;
    out.push({
      ts,
      tsStr,
      day: tsStr.slice(0, 10),
      expected: exp,
      predicted: pred,
      gap: exp - pred,
    });
  }
  return out.sort((a, b) => a.ts - b.ts);
}

function normalizeKpi(rows) {
  return rows.map((r) => ({
    meter_id: r.meter_id,
    building_name: r.building_name,
    system_kwp: r.system_kwp,
    actual_over_expected_ratio: r.actual_over_expected_ratio,
    correlation_actual_vs_pvlib: r.correlation_actual_vs_pvlib,
  }));
}

function setDateBoundsAndDefault() {
  if (!state.hourly.length) return;
  const days = state.hourly.map((r) => r.day);
  const mn = days.reduce((a, b) => (a < b ? a : b));
  const mx = days.reduce((a, b) => (a > b ? a : b));
  state.dataMin = mn;
  state.dataMax = mx;

  const fromInput = document.getElementById("global-date-from");
  const toInput = document.getElementById("global-date-to");
  fromInput.min = mn;
  fromInput.max = mx;
  toInput.min = mn;
  toInput.max = mx;

  // Default: first ~month of data (same idea as library_power_chart.html initial window)
  let from = mn;
  let to = addMonths(mn, 1);
  if (to > mx) to = mx;
  if (from > to) to = mx;

  state.dateFrom = from;
  state.dateTo = to;
  fromInput.value = from;
  toInput.value = to;
}

function applyGlobalPreset(preset) {
  const mn = state.dataMin;
  const mx = state.dataMax;
  if (!mn || !mx) return;

  let from = mn;
  let to = mx;

  if (preset === "week") {
    to = addDays(mn, 7);
    if (to > mx) to = mx;
  } else if (preset === "month") {
    to = addMonths(mn, 1);
    if (to > mx) to = mx;
  } else if (preset === "quarter") {
    to = addMonths(mn, 3);
    if (to > mx) to = mx;
  } else if (preset === "all") {
    from = mn;
    to = mx;
  }

  document.getElementById("global-date-from").value = from;
  document.getElementById("global-date-to").value = to;
  state.dateFrom = from;
  state.dateTo = to;
  showTab(activeTab);
}

let activeTab = "difference";

function showTab(id) {
  activeTab = id;
  document.querySelectorAll(".tabs button").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === id);
  });
  document.querySelectorAll(".panel").forEach((p) => {
    p.classList.toggle("active", p.id === `panel-${id}`);
  });

  const mainEl = document.querySelector("main");
  if (mainEl) purgeAllPlotlyInRoot(mainEl);

  state.dateFrom = document.getElementById("global-date-from").value;
  state.dateTo = document.getElementById("global-date-to").value;

  const df = state.dateFrom;
  const dt = state.dateTo;
  const rangeCtx = {
    dateFrom: df,
    dateTo: dt,
    dataMin: state.dataMin,
    dataMax: state.dataMax,
  };

  if (id === "difference") {
    renderDifference(document.getElementById("panel-difference"), state.hourly, df, dt);
  } else if (id === "powerbi") {
    renderPowerBIDaily(document.getElementById("panel-powerbi"), state.hourly, df, dt);
  } else if (id === "series") {
    renderSeries(document.getElementById("panel-series"), state.hourly, df, dt);
  } else if (id === "fleet") {
    renderFleet(document.getElementById("panel-fleet"), state.hourly, state.kpiRows, df, dt);
  } else if (id === "library") {
    renderLibraryPVLib(
      document.getElementById("panel-library"),
      state.hourly,
      rangeCtx
    );
  } else if (id === "seasonperf") {
    renderSeasonPerf(document.getElementById("panel-seasonperf"), state.hourly);
  } else if (id === "forecast") {
    renderForecast(
      document.getElementById("panel-forecast"),
      state.forecastCombined
    );
  }
}

async function init() {
  const status = document.getElementById("load-status");
  try {
    status.textContent = "Loading CSVs…";
    const { text: hourlyText } = await fetchDataVizFile("hourly_library_master.csv");
    const { rows: hRows } = parseCSV(hourlyText);
    state.hourly = normalizeHourly(hRows);

    const pv = await tryFetchPvlibFile(PRECOMPUTED_PVLIB_HOURLY);
    if (pv) {
      const { rows: pRows } = parseCSV(pv.text);
      const nFill = mergeLegacyFromPrecomputedHourly(state.hourly, pRows);
      if (nFill > 0) {
        console.info(
          `[Difference tab] Filled Real baseline from ${PRECOMPUTED_PVLIB_HOURLY} for ${nFill.toLocaleString()} hourly rows (${pv.url}).`
        );
      }
    }
    refreshSecondModelFlag();

    try {
      const { text: kpiText } = await fetchDataVizFile("library_kpis_summary.csv");
      const { rows: kRows } = parseCSV(kpiText);
      state.kpiRows = normalizeKpi(kRows);
    } catch {
      state.kpiRows = [];
    }

    try {
      const { text: fcText } = await fetchDataVizFile(
        "forecast_7d_combined_library.csv"
      );
      const { rows: fcRows } = parseCSV(fcText);
      state.forecastCombined = normalizeForecastCombined(fcRows);
    } catch {
      state.forecastCombined = [];
    }

    if (!state.hourly.length) throw new Error("No rows in hourly_library_master.csv");

    setDateBoundsAndDefault();
    status.textContent = `Loaded ${state.hourly.length.toLocaleString()} hourly rows · ${state.kpiRows.length} KPI row(s)${
      state.forecastCombined.length
        ? ` · Forecast: ${state.forecastCombined.length} h`
        : " · Forecast CSV: not found"
    }${
      state.hasSecondModelBaseline
        ? " · Real baseline: CSV column and/or precomputed PVLib"
        : " · Real baseline: missing (T/gap need legacy_expected_kwh or data_pvlib CSV)"
    }`;

    document.querySelectorAll(".tabs button").forEach((btn) => {
      btn.addEventListener("click", () => showTab(btn.dataset.tab));
    });

    document.getElementById("global-apply").addEventListener("click", () => {
      state.dateFrom = document.getElementById("global-date-from").value;
      state.dateTo = document.getElementById("global-date-to").value;
      showTab(activeTab);
    });

    document.getElementById("preset-week").addEventListener("click", () => applyGlobalPreset("week"));
    document.getElementById("preset-month").addEventListener("click", () => applyGlobalPreset("month"));
    document.getElementById("preset-quarter").addEventListener("click", () => applyGlobalPreset("quarter"));
    document.getElementById("preset-all").addEventListener("click", () => applyGlobalPreset("all"));

    showTab("difference");
  } catch (e) {
    status.textContent = "Error: " + e.message;
    console.error(e);
  }
}

init();
