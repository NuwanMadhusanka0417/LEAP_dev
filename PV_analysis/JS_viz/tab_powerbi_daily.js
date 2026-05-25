/**
 * Streamlit "PowerBI-style Daily" tab: GHI strip + filled daily energy (prediction vs actual).
 */
import {
  plotlyDarkTheme,
  nextPlotDomId,
  purgePlotlyInContainer,
  runAfterTabLayout,
  measureChartBox,
  PLOTLY_STATIC,
} from "./utils.js";
import { siteHeading } from "./meters.js";

function dailyAllHours(hourlyFiltered) {
  const byDay = new Map();
  for (const r of hourlyFiltered) {
    const d = r.day;
    if (!byDay.has(d)) byDay.set(d, { act: 0, pred: 0, ghiSum: 0, n: 0 });
    const o = byDay.get(d);
    o.act += r.actual;
    o.pred += r.expected;
    o.ghiSum += r.ghi;
    o.n += 1;
  }
  const days = [...byDay.keys()].sort();
  return {
    days,
    actual: days.map((d) => byDay.get(d).act),
    pred: days.map((d) => byDay.get(d).pred),
    ghiMean: days.map((d) => {
      const o = byDay.get(d);
      return o.n ? o.ghiSum / o.n : NaN;
    }),
  };
}

export function renderPowerBIDaily(container, hourly, dateFrom, dateTo, site = {}) {
  const filtered = hourly.filter(
    (r) => r.day >= dateFrom && r.day <= dateTo
  );
  const { days, actual, pred, ghiMean } = dailyAllHours(filtered);

  const simSum = pred.reduce((a, b) => a + b, 0);
  const actSum = actual.reduce((a, b) => a + b, 0);
  const degPct =
    simSum !== 0 ? ((simSum - actSum) / simSum) * 100 : NaN;
  const diffKwh = pred.reduce((s, p, i) => s + (p - actual[i]), 0);

  purgePlotlyInContainer(container);
  const idGhi = nextPlotDomId("plot-pbi-ghi");
  const idGen = nextPlotDomId("plot-pbi-gen");
  container.innerHTML = `
    <h2>${siteHeading("Daily overview (PVLib)", site)}</h2>
    <p class="note">Degradation % (vs PVLib expected): <strong>${
      Number.isFinite(degPct) ? degPct.toFixed(2) + "%" : "—"
    }</strong>
    · Σ(Pred − Actual): <strong>${diffKwh.toFixed(0)}</strong> kWh</p>
    <div id="${idGhi}" class="chart-box"></div>
    <div id="${idGen}" class="chart-box"></div>
  `;

  const marginPbi = { t: 36, r: 20, b: 56, l: 56 };

  runAfterTabLayout(container, () => {
    const elG = document.getElementById(idGhi);
    const elN = document.getElementById(idGen);
    if (!elG || !elN) return;
    const szG = measureChartBox(container, elG);
    const szN = measureChartBox(container, elN);

    const themeG = plotlyDarkTheme();
    Plotly.newPlot(
      elG,
      [
        {
          x: days,
          y: ghiMean,
          type: "scatter",
          mode: "lines",
          name: "GHI (daily mean)",
          line: { color: "#38bdf8", width: 1.6 },
          fill: "tozeroy",
          fillcolor: "rgba(56,189,248,0.15)",
        },
      ],
      {
        ...themeG,
        margin: marginPbi,
        hovermode: "x unified",
        autosize: false,
        width: szG.width,
        height: szG.height,
        title: "Global horizontal irradiance (W/m²)",
        yaxis: { ...themeG.yaxis, title: "W/m²" },
      },
      PLOTLY_STATIC
    );

    const themeN = plotlyDarkTheme();
    Plotly.newPlot(
      elN,
      [
        {
          x: days,
          y: pred,
          name: "PVLib expected (kWh)",
          type: "scatter",
          mode: "lines",
          line: { color: "rgb(31, 119, 180)", width: 2 },
          fill: "tozeroy",
          fillcolor: "rgba(31, 119, 180, 0.25)",
        },
        {
          x: days,
          y: actual,
          name: "Actual (kWh)",
          type: "scatter",
          mode: "lines",
          line: { color: "rgb(255, 127, 14)", width: 2 },
          fill: "tozeroy",
          fillcolor: "rgba(255, 127, 14, 0.25)",
        },
      ],
      {
        ...themeN,
        margin: marginPbi,
        hovermode: "x unified",
        autosize: false,
        width: szN.width,
        height: szN.height,
        title: "Solar generation (kWh) — daily totals",
        yaxis: { ...themeN.yaxis, title: "kWh" },
        legend: { orientation: "h", y: 1.12 },
      },
      PLOTLY_STATIC
    );
  });
}
