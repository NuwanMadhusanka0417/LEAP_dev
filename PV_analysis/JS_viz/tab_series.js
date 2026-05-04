/**
 * Streamlit "Series Viewer" style hourly multi-series (actual vs PVLib expected).
 */
import {
  plotlyDarkTheme,
  nextPlotDomId,
  purgePlotlyInContainer,
  runAfterTabLayout,
  measureChartBox,
  PLOTLY_STATIC,
} from "./utils.js";

export function renderSeries(container, hourly, dateFrom, dateTo) {
  const filtered = hourly.filter(
    (r) => r.day >= dateFrom && r.day <= dateTo
  );
  const t = filtered.map((r) => r.tsStr);
  const actual = filtered.map((r) => r.actual);
  const expected = filtered.map((r) => r.expected);
  const ghi = filtered.map((r) => r.ghi);

  purgePlotlyInContainer(container);
  const plotId = nextPlotDomId("plot-series-main");
  container.innerHTML = `
    <h2>Hourly series</h2>
    <div class="controls" style="margin-bottom:12px;">
      <label><input type="checkbox" id="ser-act" checked /> Actual</label>
      <label><input type="checkbox" id="ser-exp" checked /> PVLib expected</label>
      <label><input type="checkbox" id="ser-ghi" /> GHI (right axis)</label>
    </div>
    <div id="${plotId}" class="chart-box" style="min-height:420px;"></div>
  `;

  let layout = {};
  let sz = { width: 960, height: 400 };

  function buildLayout() {
    const el = document.getElementById(plotId);
    sz = measureChartBox(container, el);
    const t = plotlyDarkTheme();
    layout = {
      ...t,
      title: "Forecast comparison (hourly kWh)",
      margin: { t: 48, r: 56, b: 56, l: 56 },
      hovermode: "x unified",
      autosize: false,
      width: sz.width,
      height: sz.height,
      xaxis: { ...t.xaxis, title: "Time" },
      yaxis: { ...t.yaxis, title: "kWh" },
      yaxis2: {
        ...t.yaxis,
        title: "GHI W/m²",
        overlaying: "y",
        side: "right",
        showgrid: false,
      },
      legend: { orientation: "h", y: 1.12 },
    };
    return layout;
  }

  function redraw() {
    buildLayout();
    const showA = document.getElementById("ser-act").checked;
    const showE = document.getElementById("ser-exp").checked;
    const showG = document.getElementById("ser-ghi").checked;
    const next = [];
    if (showA)
      next.push({
        x: t,
        y: actual,
        name: "Actual",
        type: "scatter",
        mode: "lines",
        line: { color: "#22c55e", width: 1.2 },
        yaxis: "y",
      });
    if (showE)
      next.push({
        x: t,
        y: expected,
        name: "PVLib expected",
        type: "scatter",
        mode: "lines",
        line: { color: "#3b82f6", width: 1.2 },
        yaxis: "y",
      });
    if (showG)
      next.push({
        x: t,
        y: ghi,
        name: "GHI",
        type: "scatter",
        mode: "lines",
        line: { color: "#94a3b8", width: 1 },
        yaxis: "y2",
      });
    const gd = document.getElementById(plotId);
    if (!gd) return;
    if (Array.isArray(gd.data) && gd.data.length) {
      Plotly.react(gd, next, layout, PLOTLY_STATIC);
    } else {
      Plotly.newPlot(gd, next, layout, PLOTLY_STATIC);
    }
  }

  runAfterTabLayout(container, () => {
    ["ser-act", "ser-exp", "ser-ghi"].forEach((id) => {
      document.getElementById(id).addEventListener("change", redraw);
    });
    redraw();
  });
}
