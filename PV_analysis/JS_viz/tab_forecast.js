/**
 * 7-day forecast tab: hourly + daily + gap from forecast_7d_combined_library.csv
 */
import {
  plotlyDarkTheme,
  nextPlotDomId,
  purgePlotlyInContainer,
  runAfterTabLayout,
  measureChartBox,
  PLOTLY_STATIC,
} from "./utils.js";

function rollingMedianTrailing(values, windowSize) {
  const w = Math.max(1, windowSize);
  const out = new Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - w + 1);
    const chunk = values.slice(start, i + 1).slice().sort((a, b) => a - b);
    const m = Math.floor(chunk.length / 2);
    out[i] =
      chunk.length % 2 ? chunk[m] : (chunk[m - 1] + chunk[m]) / 2;
  }
  return out;
}

function shortDayLabel(isoDay) {
  const d = new Date(isoDay + "T12:00:00");
  return d.toLocaleDateString("en-AU", { month: "short", day: "numeric" });
}

function aggregateByDay(rows) {
  const byDay = new Map();
  for (const r of rows) {
    if (!byDay.has(r.day)) {
      byDay.set(r.day, { exp: 0, pred: 0 });
    }
    const o = byDay.get(r.day);
    o.exp += r.expected;
    o.pred += r.predicted;
  }
  const days = [...byDay.keys()].sort();
  return {
    days,
    labels: days.map(shortDayLabel),
    expected: days.map((d) => byDay.get(d).exp),
    predicted: days.map((d) => byDay.get(d).pred),
  };
}

function formatSignedPct(n) {
  if (!Number.isFinite(n)) return "—";
  const s = n > 0 ? "+" : "";
  return s + Math.round(n) + "%";
}

/**
 * @param {HTMLElement} container
 * @param {Array<{ts: Date, tsStr: string, day: string, expected: number, predicted: number, gap: number}>} rows
 */
export function renderForecast(container, rows) {
  purgePlotlyInContainer(container);

  if (!rows || !rows.length) {
    container.innerHTML = `
      <h2>7-day forecast (PVLib vs XGBoost)</h2>
      <p class="note">
        No data. Generate
        <code>PV_analysis/data_for_viz/forecast_7d_combined_library.csv</code> with
        <code>python 4_forecast_7d_pvlib_xgboost.py --building-key library --azure-live …</code>
      </p>
    `;
    return;
  }

  const xHourly = rows.map((r) => r.tsStr);
  const expectedH = rows.map((r) => r.expected);
  const predictedH = rows.map((r) => r.predicted);
  const gapH = rows.map((r) => r.gap);
  const gapMedian = rollingMedianTrailing(gapH, 7);

  const { days, labels: dayLabels, expected: dailyExp, predicted: dailyPred } =
    aggregateByDay(rows);

  const sumExp = dailyExp.reduce((a, b) => a + b, 0);
  const sumPred = dailyPred.reduce((a, b) => a + b, 0);
  const deltaTot = sumExp - sumPred;
  const pctTot =
    sumPred !== 0 ? (deltaTot / sumPred) * 100 : NaN;

  const t0 = rows[0].tsStr;
  const t1 = rows[rows.length - 1].tsStr;

  const dailyPct = dailyExp.map((exp, i) => {
    const pred = dailyPred[i];
    if (exp === 0) return pred === 0 ? 0 : NaN;
    return ((pred - exp) / exp) * 100;
  });

  const annotationsDaily = dayLabels.map((xl, i) => ({
    x: xl,
    y: Math.max(dailyExp[i], dailyPred[i], 1) * 1.06,
    text: formatSignedPct(dailyPct[i]),
    showarrow: false,
    yanchor: "bottom",
    font: { size: 11, color: "#94a3b8" },
  }));

  const idHour = nextPlotDomId("plot-fc-hour");
  const idDaily = nextPlotDomId("plot-fc-daily");
  const idGap = nextPlotDomId("plot-fc-gap");

  container.innerHTML = `
    <h2>Next 7 days — forecast (PVLib vs XGBoost)</h2>
    <p class="note forecast-summary">
      Forecast window: <strong>${t0}</strong> → <strong>${t1}</strong>
      · Real (XGBoost): <strong>${sumPred.toFixed(1)}</strong> kWh
      · Simulated (PVLib): <strong>${sumExp.toFixed(1)}</strong> kWh
      · Δ (Sim − Real): <strong>${deltaTot.toFixed(1)}</strong> kWh
      ${Number.isFinite(pctTot) ? `, <strong>${pctTot.toFixed(1)}%</strong> (Δ / Real)` : ""}
    </p>
    <div id="${idHour}" class="chart-box chart-box--tall"></div>
    <div class="forecast-grid">
      <div id="${idDaily}" class="chart-box"></div>
      <div id="${idGap}" class="chart-box"></div>
    </div>
  `;

  const margin = { t: 48, r: 20, b: 56, l: 56 };

  runAfterTabLayout(container, () => {
    const elH = document.getElementById(idHour);
    const elD = document.getElementById(idDaily);
    const elG = document.getElementById(idGap);
    if (!elH || !elD || !elG) return;

    const szH = measureChartBox(container, elH, 960, 420);
    const szD = measureChartBox(container, elD, 520, 400);
    const szG = measureChartBox(container, elG, 520, 400);

    const theme = plotlyDarkTheme();

    Plotly.newPlot(
      elH,
      [
        {
          x: xHourly,
          y: predictedH,
          name: "XGBoost (Real)",
          type: "scatter",
          mode: "lines",
          line: { color: "rgb(31, 119, 180)", width: 2 },
        },
        {
          x: xHourly,
          y: expectedH,
          name: "PVLib (Simulated)",
          type: "scatter",
          mode: "lines",
          line: { color: "rgb(255, 127, 14)", width: 2 },
        },
      ],
      {
        ...theme,
        margin,
        title: "Hourly generation (kWh)",
        yaxis: { ...theme.yaxis, title: "kWh / h" },
        xaxis: { ...theme.xaxis, title: "Time" },
        legend: { orientation: "h", yanchor: "bottom", y: 1.02, x: 0 },
        hovermode: "x unified",
        autosize: false,
        width: szH.width,
        height: szH.height,
      },
      PLOTLY_STATIC
    );

    Plotly.newPlot(
      elD,
      [
        {
          x: dayLabels,
          y: dailyPred,
          name: "Model (Real)",
          type: "bar",
          marker: { color: "rgb(31, 119, 180)" },
        },
        {
          x: dayLabels,
          y: dailyExp,
          name: "Model (Simulated)",
          type: "bar",
          marker: { color: "rgb(255, 127, 14)" },
        },
      ],
      {
        ...theme,
        margin,
        title: "Daily energy — next days",
        yaxis: { ...theme.yaxis, title: "Daily energy (kWh)" },
        xaxis: { ...theme.xaxis, title: "" },
        barmode: "group",
        bargap: 0.15,
        bargroupgap: 0.08,
        annotations: annotationsDaily,
        legend: { orientation: "h", yanchor: "bottom", y: 1.02, x: 1, xanchor: "right" },
        hovermode: "x unified",
        autosize: false,
        width: szD.width,
        height: szD.height,
      },
      PLOTLY_STATIC
    );

    Plotly.newPlot(
      elG,
      [
        {
          x: xHourly,
          y: gapH,
          name: "Hourly gap (Sim − Real)",
          type: "scatter",
          mode: "markers",
          marker: { color: "rgb(31, 119, 180)", size: 5 },
        },
        {
          x: xHourly,
          y: gapMedian,
          name: "7-hour median",
          type: "scatter",
          mode: "lines",
          line: { color: "rgb(255, 127, 14)", width: 2 },
        },
      ],
      {
        ...theme,
        margin,
        title: "Hourly gap (Simulated − Real)",
        yaxis: { ...theme.yaxis, title: "Gap (kWh)" },
        xaxis: { ...theme.xaxis, title: "Time" },
        shapes: [
          {
            type: "line",
            xref: "paper",
            x0: 0,
            x1: 1,
            yref: "y",
            y0: 0,
            y1: 0,
            line: { color: "#64748b", width: 1, dash: "dash" },
          },
        ],
        legend: { orientation: "h", yanchor: "bottom", y: 1.02, x: 0 },
        hovermode: "x unified",
        autosize: false,
        width: szG.width,
        height: szG.height,
      },
      PLOTLY_STATIC
    );
  });
}
