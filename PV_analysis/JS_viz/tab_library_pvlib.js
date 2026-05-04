/**
 * Same visualization as pvlib_based/chart/library_power_chart.html
 * — actual vs old expected vs PVLib expected.
 * Loads optional ../data_for_viz/library_chart_data.json; if missing, builds two-series
 * chart from hourly CSV (no legacy "old expected" column in master file).
 */
import {
  plotlyDarkTheme,
  tryFetchDataVizFile,
  nextPlotDomId,
  purgePlotlyInContainer,
  runAfterTabLayout,
  measureChartBox,
  PLOTLY_STATIC,
} from "./utils.js";

function updateStats(DATA, from, to) {
  const d0 = new Date(from);
  const d1 = new Date(to);
  let sumA = 0,
    sumO = 0,
    sumP = 0,
    count = 0;
  for (let i = 0; i < DATA.timestamp.length; i++) {
    const t = new Date(DATA.timestamp[i].replace(" ", "T"));
    if (t >= d0 && t <= d1) {
      sumA += DATA.actual_kwh[i] || 0;
      sumO += DATA.old_expected_kwh[i] || 0;
      sumP += DATA.pvlib_expected_kwh[i] || 0;
      count++;
    }
  }
  const set = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.textContent = v;
  };
  set("lib-stat-hours", count.toLocaleString());
  set("lib-stat-actual", (sumA / 1000).toFixed(1) + " MWh");
  set("lib-stat-old", (sumO / 1000).toFixed(1) + " MWh");
  set("lib-stat-pvlib", (sumP / 1000).toFixed(1) + " MWh");
}

function addDays(isoStr, n) {
  const d = new Date(isoStr + "T12:00:00");
  d.setDate(d.getDate() + n);
  return d.toISOString().slice(0, 10);
}

function addMonths(isoStr, n) {
  const d = new Date(isoStr + "T12:00:00");
  d.setMonth(d.getMonth() + n);
  return d.toISOString().slice(0, 10);
}

/**
 * @param {HTMLElement} container
 * @param {Array} hourly
 * @param {{ dateFrom?: string, dateTo?: string, dataMin?: string, dataMax?: string }} [rangeCtx] global range from main header
 */
export async function renderLibraryPVLib(container, hourly, rangeCtx = {}) {
  purgePlotlyInContainer(container);

  let DATA = null;

  const jsonRes = await tryFetchDataVizFile("library_chart_data.json");
  if (jsonRes) {
    try {
      DATA = JSON.parse(jsonRes.text);
    } catch {
      DATA = null;
    }
  }

  if (!DATA || !DATA.timestamp?.length) {
    const t = hourly.map((r) => r.tsStr);
    DATA = {
      timestamp: t,
      actual_kwh: hourly.map((r) => r.actual),
      old_expected_kwh: hourly.map(() => null),
      pvlib_expected_kwh: hourly.map((r) => r.expected),
    };
  }

  const plotLibId = nextPlotDomId("plot-library-main");
  container.innerHTML = `
    <h2>Library — actual vs expected (PVLib chart)</h2>
    <p class="note">Three-way compare needs <code>data_for_viz/library_chart_data.json</code> (from <code>pvlib_based/chart/prepare_chart_data.py</code> or copy). Otherwise only actual + PVLib from hourly master are shown.</p>
    <div class="controls" style="margin-bottom:12px;">
      <label>From <input type="date" id="lib-date-from" /></label>
      <label>To <input type="date" id="lib-date-to" /></label>
      <button type="button" id="lib-apply" class="btn btn-primary">Apply</button>
      <button type="button" id="lib-w1" class="btn btn-secondary">1 Week</button>
      <button type="button" id="lib-m1" class="btn btn-secondary">1 Month</button>
      <button type="button" id="lib-q1" class="btn btn-secondary">3 Months</button>
      <button type="button" id="lib-all" class="btn btn-secondary">Full range</button>
    </div>
    <div class="stats-bar" style="display:flex;flex-wrap:wrap;gap:20px;font-size:0.82rem;margin-bottom:10px;background:#1e293b;padding:10px 14px;border-radius:8px;border:1px solid #334155;">
      <span><span style="color:#22c55e">●</span> Actual</span>
      <span><span style="color:#f97316">●</span> Old Expected</span>
      <span><span style="color:#3b82f6">●</span> PVLib Expected</span>
      <span>Hours: <strong id="lib-stat-hours">—</strong></span>
      <span>Σ Actual: <strong id="lib-stat-actual">—</strong></span>
      <span>Σ Old: <strong id="lib-stat-old">—</strong></span>
      <span>Σ PVLib: <strong id="lib-stat-pvlib">—</strong></span>
    </div>
    <div id="${plotLibId}" class="chart-box chart-box--tall"></div>
  `;

  const traces = [
    {
      x: DATA.timestamp,
      y: DATA.actual_kwh,
      name: "Actual Power (kWh)",
      type: "scatter",
      mode: "lines",
      line: { color: "#22c55e", width: 1.5 },
      hovertemplate:
        "<b>Actual</b>: %{y:.1f} kWh<br>%{x}<extra></extra>",
    },
  ];
  const hasOld = DATA.old_expected_kwh.some(
    (v) => v !== null && v !== undefined && Number.isFinite(Number(v))
  );
  if (hasOld) {
    traces.push({
      x: DATA.timestamp,
      y: DATA.old_expected_kwh,
      name: "Old Expected (kWh)",
      type: "scatter",
      mode: "lines",
      line: { color: "#f97316", width: 1.5, dash: "dot" },
      hovertemplate:
        "<b>Old Expected</b>: %{y:.1f} kWh<br>%{x}<extra></extra>",
    });
  }
  traces.push({
    x: DATA.timestamp,
    y: DATA.pvlib_expected_kwh,
    name: "PVLib Expected (kWh)",
    type: "scatter",
    mode: "lines",
    line: { color: "#3b82f6", width: 1.5 },
    hovertemplate:
      "<b>PVLib Expected</b>: %{y:.1f} kWh<br>%{x}<extra></extra>",
  });

  const t0 = DATA.timestamp[0]?.slice(0, 10) || "2020-01-01";
  const tEnd =
    DATA.timestamp[DATA.timestamp.length - 1]?.slice(0, 10) || "2021-12-31";
  const dataMin = rangeCtx.dataMin || t0;
  const dataMax = rangeCtx.dataMax || tEnd;

  const fromEl = document.getElementById("lib-date-from");
  const toEl = document.getElementById("lib-date-to");
  fromEl.min = dataMin;
  fromEl.max = dataMax;
  toEl.min = dataMin;
  toEl.max = dataMax;

  if (rangeCtx.dateFrom && rangeCtx.dateTo) {
    fromEl.value = rangeCtx.dateFrom;
    toEl.value = rangeCtx.dateTo;
  } else {
    const defaultTo =
      addDays(t0, 7) > tEnd ? tEnd : addDays(t0, 7);
    fromEl.value = t0;
    toEl.value = defaultTo;
  }

  const tLib = plotlyDarkTheme();
  const layoutBase = {
    ...tLib,
    margin: { t: 24, r: 28, b: 56, l: 56 },
    autosize: false,
    xaxis: {
      ...tLib.xaxis,
      type: "date",
      range: [fromEl.value, toEl.value],
      rangeslider: { visible: true, bgcolor: "#1e293b", thickness: 0.08 },
    },
    yaxis: { ...tLib.yaxis, title: "kWh" },
    legend: { orientation: "h", y: 1.08, x: 0.5, xanchor: "center" },
    hovermode: "x unified",
    dragmode: "zoom",
  };

  runAfterTabLayout(container, () => {
    const chartDiv = document.getElementById(plotLibId);
    if (!chartDiv) return;
    const sz = measureChartBox(container, chartDiv);
    const layout = { ...layoutBase, width: sz.width, height: sz.height };
    Plotly.newPlot(chartDiv, traces, layout, PLOTLY_STATIC);

    function applyRange(from, to) {
      fromEl.value = from;
      toEl.value = to;
      Plotly.relayout(chartDiv, { "xaxis.range": [from, to] });
      updateStats(DATA, from, to);
    }

    document.getElementById("lib-apply").onclick = () =>
      applyRange(fromEl.value, toEl.value);
    document.getElementById("lib-w1").onclick = () => {
      const from = fromEl.value;
      let to = addDays(from, 7);
      if (to > dataMax) to = dataMax;
      applyRange(from, to);
    };
    document.getElementById("lib-m1").onclick = () => {
      const from = fromEl.value;
      let to = addMonths(from, 1);
      if (to > dataMax) to = dataMax;
      applyRange(from, to);
    };
    document.getElementById("lib-q1").onclick = () => {
      const from = fromEl.value;
      let to = addMonths(from, 3);
      if (to > dataMax) to = dataMax;
      applyRange(from, to);
    };
    document.getElementById("lib-all").onclick = () => {
      applyRange(dataMin, dataMax);
    };

    if (typeof chartDiv.on === "function") {
      chartDiv.on("plotly_relayout", (ev) => {
        let x0 = ev["xaxis.range[0]"] || ev["xaxis.range"]?.[0];
        let x1 = ev["xaxis.range[1]"] || ev["xaxis.range"]?.[1];
        if (x0 && x1) updateStats(DATA, x0, x1);
      });
    }

    updateStats(DATA, fromEl.value, toEl.value);
  });
}
