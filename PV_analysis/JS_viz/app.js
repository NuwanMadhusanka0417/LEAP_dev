/**
 * Tabbed dashboard: loads PV_analysis/data_for_viz CSVs per selected meter.
 */
import { parseCSV, parseHour, purgeAllPlotlyInRoot } from "./utils.js";
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
import { renderAllMetersDegradation } from "./tab_all_meters_degradation.js";
import { renderHealthMonitor } from "./tab_health_monitor.js";
import { renderForecast } from "./tab_forecast.js";

let state = {
  meters: [],
  meterKey: "library",
  meterLabel: "Library",
  hourly: [],
  forecastCombined: [],
  forecastStaleHint: "",
  dateFrom: "",
  dateTo: "",
  dataMin: "",
  dataMax: "",
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

/** Last ``n`` calendar months ending at ``endIso`` (clamped to ``minIso``). */
function trailingMonthsRange(endIso, minIso, n) {
  const to = endIso;
  let from = addMonths(endIso, -n);
  if (from < minIso) from = minIso;
  return { from, to };
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
    const rainRaw = r.rain_mm ?? r.precipitation_mm ?? r.precipitation_rate;
    const tempRaw = r.temp_c ?? r.temp_cell_c ?? r.air_temp;
    const rain = Number(rainRaw);
    const temp = Number(tempRaw);
    out.push({
      ts,
      tsStr: r.timestamp.trim(),
      day,
      actual: Number(r.actual_kwh),
      expected: Number(r.expected_kwh),
      ghi: Number(r.ghi_wm2),
      legacy: Number.isFinite(legacy) ? legacy : NaN,
      rain_mm: Number.isFinite(rain) ? rain : NaN,
      temp_c: Number.isFinite(temp) ? temp : NaN,
      temp_cell_c: Number.isFinite(temp) ? temp : NaN,
    });
  }
  return out.sort((a, b) => a.ts - b.ts);
}

/** Midnight on the calendar day after the last hourly meter timestamp. */
function forecastAnchorFromHourly(hourly) {
  if (!hourly.length) return null;
  const last = hourly[hourly.length - 1].ts;
  const d = new Date(last.getTime());
  d.setDate(d.getDate() + 1);
  d.setHours(0, 0, 0, 0);
  return d;
}

/**
 * Keep only forward forecast hours (after last meter reading day).
 * Drops stale backtest CSV rows (e.g. February) when a fresh May–June file exists.
 */
function filterForwardForecast(hourly, forecastRows) {
  if (!forecastRows.length) {
    return { rows: [], staleHint: "" };
  }
  const anchor = forecastAnchorFromHourly(hourly);
  if (!anchor) {
    return { rows: forecastRows, staleHint: "" };
  }
  const forward = forecastRows.filter((r) => r.ts >= anchor);
  if (forward.length) {
    return { rows: forward, staleHint: "" };
  }
  const t0 = forecastRows[0].tsStr;
  const t1 = forecastRows[forecastRows.length - 1].tsStr;
  const anchorStr = anchor.toISOString().slice(0, 16).replace("T", " ");
  return {
    rows: [],
    staleHint:
      `Loaded forecast file ends at ${t1}, but the current meter data expects forward hours from ${anchorStr}. ` +
      "Re-run: python 4_forecast_7d_pvlib_xgboost.py --campus BUNDOORA (or python run_pipeline.py).",
  };
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

  const { from, to } = trailingMonthsRange(mx, mn, 3);

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
      ? ` · Forecast: ${state.forecastCombined.length} h forward`
      : state.forecastStaleHint
        ? " · Forecast: stale/missing"
        : " · Forecast: not found") +
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

  const fcRes = await tryFetchForecastCombinedText(key);
  state.forecastStaleHint = "";
  if (fcRes) {
    const { rows: fcRows } = parseCSV(fcRes.text);
    const normalized = normalizeForecastCombined(fcRows);
    const { rows, staleHint } = filterForwardForecast(state.hourly, normalized);
    state.forecastCombined = rows;
    state.forecastStaleHint = staleHint;
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
    state.forecastStaleHint = "";
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
    from = addDays(mx, -7);
    if (from < mn) from = mn;
    to = mx;
  } else if (preset === "month") {
    ({ from, to } = trailingMonthsRange(mx, mn, 1));
  } else if (preset === "quarter") {
    ({ from, to } = trailingMonthsRange(mx, mn, 3));
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
  /** Degradation tabs always use the full meter history (ignore header date filter). */
  const fullFrom = state.dataMin;
  const fullTo = state.dataMax;
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
      fullFrom,
      fullTo,
      ctx,
    );
  } else if (id === "alldegradation") {
    const panel = document.getElementById("panel-alldegradation");
    if (panel) {
      panel.innerHTML = `<p style="color:#94a3b8;padding:2rem">Loading all meters…</p>`;
      renderAllMetersDegradation(panel, fullFrom, fullTo).catch((e) => {
        panel.innerHTML = `<p style="color:#f87171;padding:2rem">${e.message}</p>`;
        console.error(e);
      });
    }
  } else if (id === "health") {
    renderHealthMonitor(
      document.getElementById("panel-health"),
      state.hourly,
      ctx,
    );
  } else if (id === "forecast") {
    renderForecast(
      document.getElementById("panel-forecast"),
      state.forecastCombined,
      { ...ctx, staleHint: state.forecastStaleHint },
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
