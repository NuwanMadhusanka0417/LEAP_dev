/**
 * Difference tab: PVLib daily diff + health ratio (H) + forecast.
 * H = daytime Σ Actual ÷ Σ PVLib expected (GHI > 5). Full-width analytics chart.
 * Data: hourly_library_master.csv (expected_kwh, actual_kwh, ghi_wm2).
 */
import {
  rollingMedian7Trailing,
  rollingMedian7Centered,
  plotlyDarkTheme,
  nextPlotDomId,
  purgePlotlyInContainer,
  runAfterTabLayout,
  measureChartBox,
  PLOTLY_STATIC,
  linearRegression,
} from "./utils.js";

function daylightDailySums(hourlyFiltered) {
  const byDay = new Map();
  for (const r of hourlyFiltered) {
    const ghi = r.ghi;
    if (!(ghi > 5)) continue;
    const d = r.day;
    if (!byDay.has(d)) byDay.set(d, { act: 0, pred: 0, ghiSum: 0, ghiN: 0 });
    const o = byDay.get(d);
    o.act += r.actual;
    o.pred += r.expected;
    o.ghiSum += ghi;
    o.ghiN += 1;
  }
  const days = [...byDay.keys()].sort();
  const act = [];
  const pred = [];
  const ghiMean = [];
  for (const d of days) {
    const o = byDay.get(d);
    act.push(o.act);
    pred.push(o.pred);
    ghiMean.push(o.ghiN ? o.ghiSum / o.ghiN : NaN);
  }
  return { days, act, pred, ghiMean };
}

/** Daylight (GHI > 5) daily sums for health ratio: actual and PVLib expected. */
function daylightDailyForHealth(hourlyFiltered) {
  const byDay = new Map();
  for (const r of hourlyFiltered) {
    if (!(r.ghi > 5)) continue;
    const d = r.day;
    if (!byDay.has(d)) byDay.set(d, { act: 0, sim: 0 });
    const o = byDay.get(d);
    if (Number.isFinite(r.actual)) o.act += r.actual;
    if (Number.isFinite(r.expected)) o.sim += r.expected;
  }
  const days = [...byDay.keys()].sort();
  const act = [];
  const sim = [];
  for (const d of days) {
    const o = byDay.get(d);
    act.push(o.act);
    sim.push(o.sim);
  }
  return { days, act, sim };
}

function ceilToHourLocal(d) {
  const x = new Date(d.getFullYear(), d.getMonth(), d.getDate(), d.getHours(), 0, 0, 0);
  if (d.getTime() > x.getTime()) x.setHours(x.getHours() + 1);
  return x;
}

function buildForecastSlice(hourly, horizonDays) {
  if (!hourly.length) return null;
  let lastTs = null;
  for (let i = hourly.length - 1; i >= 0; i--) {
    if (Number.isFinite(hourly[i].actual)) {
      lastTs = hourly[i].ts;
      break;
    }
  }
  if (!lastTs) lastTs = hourly[hourly.length - 1].ts;
  const startTs = ceilToHourLocal(lastTs);
  const endMs = startTs.getTime() + horizonDays * 86400000;
  const slice = hourly.filter((r) => r.ts.getTime() >= startTs.getTime() && r.ts.getTime() < endMs);
  if (!slice.length) return null;
  const hasPred = slice.some(
    (r) => Number.isFinite(r.expected) || Number.isFinite(r.legacy)
  );
  if (!hasPred) return null;
  return { slice, startTs, endMs };
}

function forecastDayShapes(slice) {
  const shapes = [];
  let i = 0;
  while (i < slice.length) {
    const day = slice[i].ghi > 5;
    let j = i;
    while (j < slice.length && (slice[j].ghi > 5) === day) j++;
    if (day && j > i) {
      const x0 = slice[i].tsStr.replace(" ", "T");
      const x1 = slice[j - 1].tsStr.replace(" ", "T");
      shapes.push({
        type: "rect",
        xref: "x",
        yref: "paper",
        x0,
        x1,
        y0: 0,
        y1: 1,
        fillcolor: "#38bdf8",
        opacity: 0.06,
        line: { width: 0 },
        layer: "below",
      });
    }
    i = j;
  }
  return shapes;
}

let _lastDiffRender = null;
/** Per-chart analytics ranges (yyyy-mm-dd), persisted across re-renders; reset when hourly dataset changes. */
let _anaDateState = null;
let _hourlyKeyForAna = "";

function dayBounds(hourly) {
  if (!hourly.length) return { min: "", max: "" };
  let min = hourly[0].day;
  let max = hourly[0].day;
  for (const r of hourly) {
    if (r.day < min) min = r.day;
    if (r.day > max) max = r.day;
  }
  return { min, max };
}

function clampDayRange(from, to, minD, maxD) {
  if (!from || !to || !minD || !maxD) return [minD, maxD];
  let a = from < minD ? minD : from;
  let b = to > maxD ? maxD : to;
  if (a > b) [a, b] = [b, a];
  return [a, b];
}

function readAnaStateFromDom(container) {
  const from = container.querySelector("#ana-h-from")?.value;
  const to = container.querySelector("#ana-h-to")?.value;
  if (from && to) return { from, to };
  return null;
}

/** Plotly date axis: explicit range + tick step so multi-year spans show calendar ticks (not a squeezed few months). */
function plotlyXaxisDayRange(fromDay, toDay, themeX) {
  const t0 = new Date(`${fromDay}T00:00:00`).getTime();
  const t1 = new Date(`${toDay}T23:59:59`).getTime();
  const spanDays = Math.max(1, Math.round((t1 - t0) / 86400000));
  let tickformat = "%b %Y";
  let dtick;
  if (spanDays <= 45) {
    dtick = 7 * 86400000;
    tickformat = "%d %b";
  } else if (spanDays <= 120) {
    dtick = 14 * 86400000;
    tickformat = "%d %b";
  } else if (spanDays <= 400) {
    dtick = "M1";
    tickformat = "%b '%y";
  } else if (spanDays <= 800) {
    dtick = "M2";
    tickformat = "%b %Y";
  } else {
    dtick = "M3";
    tickformat = "%b %Y";
  }
  return {
    ...themeX,
    title: "Date",
    type: "date",
    autorange: false,
    range: [`${fromDay}T00:00:00`, `${toDay}T23:59:59`],
    tickformat,
    ...(dtick ? { dtick } : {}),
  };
}

export function renderDifference(container, hourly, dateFrom, dateTo) {
  const bounds = dayBounds(hourly);
  const hourlyKey = hourly.length ? `${hourly[0].day}|${hourly[hourly.length - 1].day}|${hourly.length}` : "";

  if (!_anaDateState || _hourlyKeyForAna !== hourlyKey) {
    _anaDateState = {
      hFrom: bounds.min,
      hTo: bounds.max,
    };
    _hourlyKeyForAna = hourlyKey;
  } else if (container && container.querySelector("#ana-h-from")) {
    const dom = readAnaStateFromDom(container);
    if (dom) {
      _anaDateState.hFrom = dom.from;
      _anaDateState.hTo = dom.to;
    }
  }
  if (!_anaDateState.hFrom) _anaDateState.hFrom = bounds.min;
  if (!_anaDateState.hTo) _anaDateState.hTo = bounds.max;

  const [hFrom, hTo] = clampDayRange(_anaDateState.hFrom, _anaDateState.hTo, bounds.min, bounds.max);
  Object.assign(_anaDateState, { hFrom, hTo });

  _lastDiffRender = { container, hourly, dateFrom, dateTo };

  const filtered = hourly.filter((r) => r.day >= dateFrom && r.day <= dateTo);
  const { days, act, pred, ghiMean } = daylightDailySums(filtered);
  const diff = pred.map((p, i) => p - act[i]);
  const med7 = rollingMedian7Trailing(diff, 3);

  let slope = NaN,
    intercept = NaN,
    trendY = [];
  if (days.length >= 10) {
    const t0 = new Date(days[0]).getTime();
    const x = days.map((d) => (new Date(d).getTime() - t0) / 86400000);
    const reg = linearRegression(x, diff);
    slope = reg.slope;
    intercept = reg.intercept;
    trendY = x.map((xi) => intercept + slope * xi);
  }

  const totalDiff = diff.reduce((s, v) => s + (Number.isFinite(v) ? v : 0), 0);

  const hFiltered = hourly.filter((r) => r.day >= hFrom && r.day <= hTo);

  const anaH = daylightDailyForHealth(hFiltered);
  const H = anaH.sim.map((s, i) => (s > 0 ? anaH.act[i] / s : NaN));
  const H7 = rollingMedian7Trailing(H, 3);
  const hasLegacy = hourly.some((r) => Number.isFinite(r.legacy));

  const horizonSelId = "diff-fc-horizon";
  let horizonDays = 7;
  const existingSel = container.querySelector(`#${horizonSelId}`);
  if (existingSel) horizonDays = Math.min(14, Math.max(3, parseInt(existingSel.value, 10) || 7));

  const fc = buildForecastSlice(hourly, horizonDays);

  purgePlotlyInContainer(container);
  const ids = {
    ghi: nextPlotDomId("plot-diff-ghi"),
    bars: nextPlotDomId("plot-diff-bars"),
    h: nextPlotDomId("plot-ana-h"),
    fc1: nextPlotDomId("plot-fc-daily"),
    fc2: nextPlotDomId("plot-fc-hourly"),
    fc3: nextPlotDomId("plot-fc-cum"),
  };

  container.innerHTML = `
    <h2>Difference (PVLib expected − actual)</h2>
    <p class="note">Daylight-only daily sums (hours with GHI &gt; 5 W/m²). Slope: <strong>${
      Number.isFinite(slope) ? slope.toFixed(2) : "—"
    }</strong> kWh/day · ΣDiff: <strong>${totalDiff.toFixed(0)}</strong> kWh</p>
    <div id="${ids.ghi}" class="chart-box"></div>
    <div id="${ids.bars}" class="chart-box"></div>

    <h3>Analytics — health ratio (H)</h3>
    <p class="note"><strong>H</strong> = Σ Actual ÷ Σ PVLib expected over daylight hours (GHI &gt; 5 W/m²), per day. Chart spans the full panel width.</p>
    <div class="analytics-health-row">
      <div class="analytics-range analytics-range--inline">
        <label>From <input type="date" id="ana-h-from" min="${bounds.min}" max="${bounds.max}" value="${hFrom}" /></label>
        <label>To <input type="date" id="ana-h-to" min="${bounds.min}" max="${bounds.max}" value="${hTo}" /></label>
      </div>
      <div id="${ids.h}" class="chart-box chart-box--health-wide"></div>
    </div>

    <h3>Next few days — forecast (model: Real vs Simulated)</h3>
    <p class="note controls-inline">
      <label>Forecast horizon (days)
        <select id="${horizonSelId}">
          ${[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            .map(
              (n) =>
                `<option value="${n}" ${n === horizonDays ? "selected" : ""}>${n}</option>`
            )
            .join("")}
        </select>
      </label>
    </p>
    <div id="diff-fc-caption" class="note"></div>
    <div class="chart-grid-3">
      <div id="${ids.fc1}" class="chart-box chart-box--compact"></div>
      <div id="${ids.fc2}" class="chart-box chart-box--compact"></div>
      <div id="${ids.fc3}" class="chart-box chart-box--compact"></div>
    </div>
  `;

  const sel = container.querySelector(`#${horizonSelId}`);
  if (sel && !sel.dataset.bound) {
    sel.dataset.bound = "1";
    sel.addEventListener("change", () => {
      if (_lastDiffRender) {
        renderDifference(
          _lastDiffRender.container,
          _lastDiffRender.hourly,
          _lastDiffRender.dateFrom,
          _lastDiffRender.dateTo
        );
      }
    });
  }

  for (const id of ["ana-h-from", "ana-h-to"]) {
    const el = container.querySelector(`#${id}`);
    if (el && !el.dataset.bound) {
      el.dataset.bound = "1";
      el.addEventListener("change", () => {
        if (_lastDiffRender) {
          renderDifference(
            _lastDiffRender.container,
            _lastDiffRender.hourly,
            _lastDiffRender.dateFrom,
            _lastDiffRender.dateTo
          );
        }
      });
    }
  }

  const barPos = diff.map((v) => (v > 0 ? v : 0));
  const barNeg = diff.map((v) => (v < 0 ? v : 0));
  const marginDiff = { t: 36, r: 20, b: 56, l: 56 };
  const marginSmall = { t: 32, r: 16, b: 48, l: 48 };

  runAfterTabLayout(container, () => {
    const themeG = plotlyDarkTheme();
    const elG = document.getElementById(ids.ghi);
    const elB = document.getElementById(ids.bars);
    if (!elG || !elB) return;
    const szG = measureChartBox(container, elG);
    const szB = measureChartBox(container, elB);

    Plotly.newPlot(
      elG,
      [
        {
          x: days,
          y: ghiMean,
          name: "Mean GHI (daylight hrs)",
          type: "scatter",
          mode: "lines",
          line: { color: "#38bdf8", width: 1.5 },
        },
      ],
      {
        ...themeG,
        margin: marginDiff,
        showlegend: true,
        hovermode: "x unified",
        autosize: false,
        width: szG.width,
        height: szG.height,
        title: "Global horizontal irradiance — daily mean (daylight hours)",
        xaxis: days.length ? plotlyXaxisDayRange(dateFrom, dateTo, themeG.xaxis) : themeG.xaxis,
        yaxis: { ...themeG.yaxis, title: "W/m²" },
      },
      PLOTLY_STATIC
    );

    const themeB = plotlyDarkTheme();
    Plotly.newPlot(
      elB,
      [
        {
          x: days,
          y: barPos,
          name: "Pred − Actual (>0)",
          type: "bar",
          marker: { color: "#e74c3c" },
        },
        {
          x: days,
          y: barNeg,
          name: "Pred − Actual (<0)",
          type: "bar",
          marker: { color: "#3498db" },
        },
        ...(trendY.length
          ? [
              {
                x: days,
                y: trendY,
                name: "Trend",
                type: "scatter",
                mode: "lines",
                line: { color: "#f8fafc", dash: "dash", width: 2 },
              },
            ]
          : []),
        {
          x: days,
          y: med7,
          name: "7-day median",
          type: "scatter",
          mode: "lines",
          line: { color: "#cbd5e1", width: 2 },
        },
      ],
      {
        ...themeB,
        margin: marginDiff,
        showlegend: true,
        hovermode: "x unified",
        autosize: false,
        width: szB.width,
        height: szB.height,
        title: "Daily difference (kWh) — PVLib expected minus actual",
        barmode: "relative",
        shapes: [
          {
            type: "line",
            x0: days[0],
            x1: days[days.length - 1],
            y0: 0,
            y1: 0,
            line: { color: "#64748b", width: 1, dash: "dash" },
          },
        ],
        xaxis: days.length ? plotlyXaxisDayRange(dateFrom, dateTo, themeB.xaxis) : themeB.xaxis,
        yaxis: { ...themeB.yaxis, title: "kWh" },
      },
      PLOTLY_STATIC
    );

    const themeA = plotlyDarkTheme();
    const elH = document.getElementById(ids.h);
    const marginHealth = { t: 40, r: 28, b: 56, l: 56 };
    const healthWidth = Math.max(320, Math.floor((container.clientWidth || 960) - 32));
    const healthHeight = 400;

    const xH0 = hFrom;
    const xH1 = hTo;
    if (elH) {
      if (!anaH.days.length) {
        Plotly.newPlot(
          elH,
          [],
          {
            ...themeA,
            margin: marginHealth,
            title: { text: "Health ratio — Actual ÷ PVLib expected", x: 0, xanchor: "left", y: 0.98, yanchor: "top" },
            annotations: [
              {
                text: "No data in this date range",
                xref: "paper",
                yref: "paper",
                x: 0.5,
                y: 0.5,
                showarrow: false,
                font: { color: "#94a3b8", size: 14 },
              },
            ],
            xaxis: plotlyXaxisDayRange(hFrom, hTo, themeA.xaxis),
            yaxis: { ...themeA.yaxis, title: "H", range: [0, 2] },
            autosize: false,
            width: healthWidth,
            height: healthHeight,
          },
          PLOTLY_STATIC
        );
      } else {
        Plotly.newPlot(
          elH,
          [
            {
              x: anaH.days,
              y: H,
              name: "Daily H (energy-based)",
              type: "scatter",
              mode: "markers",
              marker: { color: "#3b82f6", size: 6, opacity: 0.65 },
            },
            {
              x: anaH.days,
              y: H7,
              name: "7-day median",
              type: "scatter",
              mode: "lines",
              line: { color: "#fb923c", width: 2 },
              connectgaps: false,
            },
          ],
          {
            ...themeA,
            margin: marginHealth,
            title: { text: "Health ratio — Actual ÷ PVLib expected", x: 0, xanchor: "left", y: 0.98, yanchor: "top" },
            xaxis: plotlyXaxisDayRange(hFrom, hTo, themeA.xaxis),
            yaxis: { ...themeA.yaxis, title: "H", range: [0, 2] },
            shapes: [
              {
                type: "line",
                x0: xH0,
                x1: xH1,
                y0: 1,
                y1: 1,
                line: { dash: "dash", color: "#94a3b8" },
              },
              {
                type: "line",
                x0: xH0,
                x1: xH1,
                y0: 0.9,
                y1: 0.9,
                line: { dash: "dot", color: "#64748b" },
              },
            ],
            showlegend: true,
            legend: { x: 1, xanchor: "right", y: 1, bgcolor: "rgba(15,23,42,0.7)" },
            hovermode: "x unified",
            autosize: false,
            width: healthWidth,
            height: healthHeight,
          },
          PLOTLY_STATIC
        );
      }
    }

    const capEl = container.querySelector("#diff-fc-caption");
    const elFc1 = document.getElementById(ids.fc1);
    const elFc2 = document.getElementById(ids.fc2);
    const elFc3 = document.getElementById(ids.fc3);

    if (!fc || !hasLegacy || !elFc1 || !elFc2 || !elFc3) {
      if (capEl) {
        capEl.innerHTML = !hasLegacy
          ? "Forecast charts need a Real baseline: <code>legacy_expected_kwh</code> in the hourly CSV and/or a loadable <code>data_pvlib/expected_power_pvlib_cleaned_v2.csv</code> (same origin as this page)."
          : !fc
            ? "No hourly rows in the selected forecast window after the last actual reading."
            : "";
      }
      return;
    }

    const s = fc.slice;
    const gapH = s.map((r) =>
      Number.isFinite(r.legacy) && Number.isFinite(r.expected)
        ? r.expected - r.legacy
        : NaN
    );
    const gapMed7h = rollingMedian7Centered(gapH, 1);
    let cum = 0;
    const cumGap = gapH.map((g) => {
      cum += Number.isFinite(g) ? g : 0;
      return cum;
    });

    const byD = new Map();
    for (const r of s) {
      const d = r.day;
      if (!byD.has(d)) byD.set(d, { real: 0, sim: 0 });
      const o = byD.get(d);
      if (Number.isFinite(r.legacy)) o.real += r.legacy;
      if (Number.isFinite(r.expected)) o.sim += r.expected;
    }
    const fcDays = [...byD.keys()].sort();
    const realD = fcDays.map((d) => byD.get(d).real);
    const simD = fcDays.map((d) => byD.get(d).sim);
    const pctDiff = fcDays.map((_, i) =>
      simD[i] > 0 ? (100 * (realD[i] - simD[i])) / simD[i] : NaN
    );

    const weeklyReal = realD.reduce((a, b) => a + b, 0);
    const weeklySim = simD.reduce((a, b) => a + b, 0);
    const weeklyDiff = weeklySim - weeklyReal;
    const weeklyPcnt = weeklyReal ? (100 * weeklyDiff) / weeklyReal : NaN;
    const endTs = new Date(fc.endMs - 3600000);

    if (capEl) {
      capEl.innerHTML = `Forecast window: <strong>${fc.startTs.getFullYear()}-${String(fc.startTs.getMonth() + 1).padStart(2, "0")}-${String(fc.startTs.getDate()).padStart(2, "0")} ${String(fc.startTs.getHours()).padStart(2, "0")}:00</strong> → <strong>${endTs.getFullYear()}-${String(endTs.getMonth() + 1).padStart(2, "0")}-${String(endTs.getDate()).padStart(2, "0")} ${String(endTs.getHours()).padStart(2, "0")}:00</strong><br/>
        Totals: Real = <strong>${weeklyReal.toFixed(1)}</strong> kWh, Simulated = <strong>${weeklySim.toFixed(1)}</strong> kWh
        (Δ = <strong>${weeklyDiff >= 0 ? "+" : ""}${weeklyDiff.toFixed(1)}</strong> kWh, <strong>${weeklyPcnt >= 0 ? "+" : ""}${weeklyPcnt.toFixed(1)}%</strong>).`;
    }

    const xBar = fcDays.map((d) => {
      const [y, m, day] = d.split("-");
      const mo = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][parseInt(m, 10) - 1];
      return `${mo} ${parseInt(day, 10)}`;
    });

    const barTraces = [
      {
        x: xBar,
        y: realD,
        name: "Model (Real)",
        type: "bar",
        marker: { color: "#3b82f6" },
      },
      {
        x: xBar,
        y: simD,
        name: "Model (Simulated)",
        type: "bar",
        marker: { color: "#fb923c" },
      },
    ];
    const annotations = [];
    for (let i = 0; i < fcDays.length; i++) {
      const p = pctDiff[i];
      if (!Number.isFinite(p)) continue;
      const hi = Math.max(realD[i], simD[i]);
      annotations.push({
        x: xBar[i],
        y: hi * 1.02 + (hi > 0 ? 0 : 0.02),
        text: `${p >= 0 ? "+" : ""}${p.toFixed(0)}%`,
        showarrow: false,
        font: { color: "#e2e8f0", size: 11 },
      });
    }

    Plotly.newPlot(
      elFc1,
      barTraces,
      {
        ...themeA,
        margin: marginSmall,
        barmode: "group",
        title: "Daily energy — next days",
        yaxis: { ...themeA.yaxis, title: "Daily energy (kWh)" },
        xaxis: { ...themeA.xaxis, title: "Date" },
        annotations,
        showlegend: true,
        autosize: false,
        width: measureChartBox(container, elFc1).width,
        height: 320,
      },
      PLOTLY_STATIC
    );

    const xHour = s.map((r) => r.tsStr.replace(" ", "T"));
    const fcShapes = forecastDayShapes(s);

    Plotly.newPlot(
      elFc2,
      [
        {
          x: xHour,
          y: gapH,
          name: "Hourly gap (Sim − Real)",
          type: "scatter",
          mode: "markers",
          marker: { color: "#3b82f6", size: 5, opacity: 0.55 },
        },
        {
          x: xHour,
          y: gapMed7h,
          name: "7-hour median",
          type: "scatter",
          mode: "lines",
          line: { color: "#fb923c", width: 2 },
        },
      ],
      {
        ...themeA,
        margin: marginSmall,
        shapes: [
          ...fcShapes,
          {
            type: "line",
            x0: xHour[0],
            x1: xHour[xHour.length - 1],
            y0: 0,
            y1: 0,
            line: { dash: "dash", color: "#64748b" },
          },
        ],
        title: "Hourly gap (Simulated − Real)",
        yaxis: { ...themeA.yaxis, title: "Gap (kWh)" },
        showlegend: true,
        hovermode: "x unified",
        autosize: false,
        width: measureChartBox(container, elFc2).width,
        height: 320,
      },
      PLOTLY_STATIC
    );

    Plotly.newPlot(
      elFc3,
      [
        {
          x: xHour,
          y: cumGap,
          name: "Cumulative gap",
          type: "scatter",
          mode: "lines",
          line: { color: "#fb923c", width: 2 },
        },
      ],
      {
        ...themeA,
        margin: marginSmall,
        shapes: [
          {
            type: "line",
            x0: xHour[0],
            x1: xHour[xHour.length - 1],
            y0: 0,
            y1: 0,
            line: { dash: "dash", color: "#64748b" },
          },
        ],
        title: "Cumulative difference — Simulated vs Real",
        yaxis: { ...themeA.yaxis, title: "Cumulative gap (kWh)" },
        showlegend: false,
        hovermode: "x unified",
        autosize: false,
        width: measureChartBox(container, elFc3).width,
        height: 320,
      },
      PLOTLY_STATIC
    );
  });
}
