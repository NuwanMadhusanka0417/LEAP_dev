/**
 * Tabbed dashboard: loads PV_analysis/data_for_viz CSVs per selected meter.
 */
import {
  parseCSV,
  parseHour,
  tryFetchPvlibFile,
  purgeAllPlotlyInRoot,
} from "./utils.js";
import {
  loadMeterCatalog,
  fetchHourlyMasterText,
  tryFetchForecastCombinedText,
  hourlyMasterFilename,
} from "./meters.js";
import { renderDifference } from "./tab_difference.js";
import { renderPowerBIDaily } from "./tab_powerbi_daily.js";
import { renderLibraryPVLib } from "./tab_library_pvlib.js";
import { renderSeasonPerf } from "./tab_season_perf.js";
import { renderSeasonIrradiance } from "./tab_season_irradiance.js";
import { renderMeterDegradation } from "./tab_degradation.js";
import { renderForecast } from "./tab_forecast.js";

const PRECOMPUTED_PVLIB_HOURLY = "expected_power_pvlib_cleaned_v2.csv";

let state = {
  meters: [],
  meterKey: "library",
  meterLabel: "Library",
  hourly: [],
  forecastCombined: [],
  dateFrom: "",
  dateTo: "",
  dataMin: "",
  dataMax: "",
  hasSecondModelBaseline: false,
};

function siteCtx() {
  return { key: state.meterKey, label: state.meterLabel };
}

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

  let from = mn;
  let to = addMonths(mn, 1);
  if (to > mx) to = mx;
  if (from > to) to = mx;

  state.dateFrom = from;
  state.dateTo = to;
  fromInput.value = from;
  toInput.value = to;
}

function updateLoadStatus(extra = "") {
  const status = document.getElementById("load-status");
  if (!status) return;
  const site = state.meterLabel || state.meterKey;
  if (!state.hourly.length) {
    status.textContent = extra || `No data for ${site}`;
    return;
  }
  status.textContent =
    `${site} · ${state.hourly.length.toLocaleString()} hourly rows` +
    (state.forecastCombined.length
      ? ` · Forecast: ${state.forecastCombined.length} h`
      : " · Forecast: not found") +
    (state.hasSecondModelBaseline
      ? " · Real baseline: yes"
      : "") +
    (extra ? ` · ${extra}` : "");
}

async function loadMeterData(meterKey) {
  const key = meterKey.trim().toLowerCase();
  const m = state.meters.find((x) => x.key === key);
  state.meterKey = key;
  state.meterLabel = m?.label ?? key;

  const { text: hourlyText } = await fetchHourlyMasterText(key);
  const { rows: hRows } = parseCSV(hourlyText);
  state.hourly = normalizeHourly(hRows);

  const pv = await tryFetchPvlibFile(PRECOMPUTED_PVLIB_HOURLY);
  if (pv) {
    const { rows: pRows } = parseCSV(pv.text);
    const nFill = mergeLegacyFromPrecomputedHourly(state.hourly, pRows);
    if (nFill > 0) {
      console.info(
        `[${key}] Filled Real baseline from ${PRECOMPUTED_PVLIB_HOURLY} for ${nFill.toLocaleString()} rows.`
      );
    }
  }
  refreshSecondModelFlag();

  const fcRes = await tryFetchForecastCombinedText(key);
  if (fcRes) {
    const { rows: fcRows } = parseCSV(fcRes.text);
    state.forecastCombined = normalizeForecastCombined(fcRows);
  } else {
    state.forecastCombined = [];
  }

  if (!state.hourly.length) {
    throw new Error(`No rows in ${hourlyMasterFilename(key)}`);
  }

  setDateBoundsAndDefault();
}

function populateMeterSelect() {
  const sel = document.getElementById("global-meter");
  if (!sel) return;
  sel.innerHTML = state.meters
    .map(
      (m) =>
        `<option value="${m.key}" ${m.key === state.meterKey ? "selected" : ""}>${m.label}</option>`
    )
    .join("");
}

async function onMeterChange() {
  const sel = document.getElementById("global-meter");
  const key = sel?.value || state.meterKey;
  const status = document.getElementById("load-status");
  try {
    status.textContent = `Loading ${key}…`;
    await loadMeterData(key);
    updateLoadStatus();
    showTab(activeTab);
  } catch (e) {
    state.hourly = [];
    state.forecastCombined = [];
    updateLoadStatus(e.message);
    console.error(e);
    const panel = document.getElementById(`panel-${activeTab}`);
    if (panel) {
      panel.innerHTML = `<p style="color:#f87171;padding:2rem">Failed to load ${hourlyMasterFilename(key)}: ${e.message}</p>`;
    }
  }
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

let activeTab = "powerbi";

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

  if (!state.hourly.length) {
    const panel = document.getElementById(`panel-${id}`);
    if (panel) {
      panel.innerHTML = `<p style="color:#94a3b8;padding:2rem">Select a meter and ensure <code>${hourlyMasterFilename(state.meterKey)}</code> exists (run script 2).</p>`;
    }
    return;
  }

  state.dateFrom = document.getElementById("global-date-from").value;
  state.dateTo = document.getElementById("global-date-to").value;

  const df = state.dateFrom;
  const dt = state.dateTo;
  const ctx = siteCtx();
  const rangeCtx = {
    dateFrom: df,
    dateTo: dt,
    dataMin: state.dataMin,
    dataMax: state.dataMax,
    site: ctx,
  };

  if (id === "powerbi") {
    renderPowerBIDaily(
      document.getElementById("panel-powerbi"),
      state.hourly,
      df,
      dt,
      ctx,
    );
  } else if (id === "difference") {
    renderDifference(
      document.getElementById("panel-difference"),
      state.hourly,
      df,
      dt,
      ctx,
    );
  } else if (id === "library") {
    renderLibraryPVLib(
      document.getElementById("panel-library"),
      state.hourly,
      rangeCtx,
    );
  } else if (id === "seasonperf") {
    renderSeasonPerf(
      document.getElementById("panel-seasonperf"),
      state.hourly,
      ctx,
    );
  } else if (id === "seasonirr") {
    renderSeasonIrradiance(
      document.getElementById("panel-seasonirr"),
      state.hourly,
      ctx,
    );
  } else if (id === "degradation") {
    renderMeterDegradation(
      document.getElementById("panel-degradation"),
      state.hourly,
      df,
      dt,
      ctx,
    );
  } else if (id === "forecast") {
    renderForecast(
      document.getElementById("panel-forecast"),
      state.forecastCombined,
      ctx,
    );
  }
}

async function init() {
  const status = document.getElementById("load-status");
  try {
    status.textContent = "Loading meter list…";
    state.meters = await loadMeterCatalog();
    if (!state.meters.length) {
      throw new Error("No meters in catalog");
    }
    state.meterKey = state.meters[0].key;
    populateMeterSelect();

    await loadMeterData(state.meterKey);
    updateLoadStatus();

    document.querySelectorAll(".tabs button").forEach((btn) => {
      btn.addEventListener("click", () => showTab(btn.dataset.tab));
    });

    document.getElementById("global-meter")?.addEventListener("change", onMeterChange);

    document.getElementById("global-apply").addEventListener("click", () => {
      state.dateFrom = document.getElementById("global-date-from").value;
      state.dateTo = document.getElementById("global-date-to").value;
      showTab(activeTab);
    });

    document.getElementById("preset-week").addEventListener("click", () => applyGlobalPreset("week"));
    document.getElementById("preset-month").addEventListener("click", () => applyGlobalPreset("month"));
    document.getElementById("preset-quarter").addEventListener("click", () => applyGlobalPreset("quarter"));
    document.getElementById("preset-all").addEventListener("click", () => applyGlobalPreset("all"));

    showTab("powerbi");
  } catch (e) {
    status.textContent = "Error: " + e.message;
    console.error(e);
  }
}

init();
