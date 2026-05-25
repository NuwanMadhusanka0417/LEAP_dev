/**
 * Difference tab: PVLib daily diff + health ratio (H).
 * H = daytime Σ Actual ÷ Σ PVLib expected (GHI > 5). Full-width analytics chart.
 * Data: hourly_library_master.csv (expected_kwh, actual_kwh, ghi_wm2).
 */
import {
  rollingMedian7Trailing,
  plotlyDarkTheme,
  nextPlotDomId,
  purgePlotlyInContainer,
  runAfterTabLayout,
  measureChartBox,
  PLOTLY_STATIC,
  linearRegression,
} from "./utils.js";
import { siteHeading } from "./meters.js";

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

export function renderDifference(container, hourly, dateFrom, dateTo, site = {}) {
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

  purgePlotlyInContainer(container);
  const ids = {
    ghi: nextPlotDomId("plot-diff-ghi"),
    bars: nextPlotDomId("plot-diff-bars"),
    h: nextPlotDomId("plot-ana-h"),
  };

  container.innerHTML = `
    <h2>${siteHeading("Difference (PVLib expected − actual)", site)}</h2>
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
  `;

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
  });
}
