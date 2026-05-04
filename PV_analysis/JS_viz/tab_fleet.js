/**
 * Streamlit "Fleet Trend" style: slope of daily (pred − actual) per site.
 * With only library rows in data_for_viz, this shows one bar + KPI table.
 */
import {
  linearRegression,
  plotlyDarkTheme,
  nextPlotDomId,
  purgePlotlyInContainer,
  runAfterTabLayout,
  measureChartBox,
  PLOTLY_STATIC,
} from "./utils.js";

function dailyDiffSeries(hourlyFiltered) {
  const byDay = new Map();
  for (const r of hourlyFiltered) {
    const d = r.day;
    if (!byDay.has(d)) byDay.set(d, { act: 0, pred: 0 });
    const o = byDay.get(d);
    o.act += r.actual;
    o.pred += r.expected;
  }
  const days = [...byDay.keys()].sort();
  const diff = days.map((d) => {
    const o = byDay.get(d);
    return o.pred - o.act;
  });
  return { days, diff };
}

export function renderFleet(container, hourly, kpiRows, dateFrom, dateTo) {
  const filtered = hourly.filter(
    (r) => r.day >= dateFrom && r.day <= dateTo
  );
  const { days, diff } = dailyDiffSeries(filtered);

  let slope = NaN;
  if (days.length >= 10) {
    const t0 = new Date(days[0]).getTime();
    const x = days.map((d) => (new Date(d).getTime() - t0) / 86400000);
    slope = linearRegression(x, diff).slope;
  }
  const totalDiff = diff.reduce((s, v) => s + v, 0);

  const label =
    kpiRows.length && kpiRows[0].building_name
      ? kpiRows[0].building_name
      : "Library site";

  purgePlotlyInContainer(container);
  const idBar = nextPlotDomId("plot-fleet-bar");
  container.innerHTML = `
    <h2>Fleet / site trend</h2>
    <p class="note">Slope of daily (PVLib − actual) in kWh/day. Data in <code>data_for_viz</code> is currently single-site (library pipeline).</p>
    <div id="${idBar}" class="chart-box" style="min-height:280px;"></div>
    <h3 style="font-size:0.9rem;color:#94a3b8;margin-top:20px;">KPI summary (CSV)</h3>
    <div id="fleet-table-wrap"></div>
  `;

  const slopePlot = Number.isFinite(slope) ? slope : 0;
  runAfterTabLayout(container, () => {
    const el = document.getElementById(idBar);
    if (!el) return;
    const sz = measureChartBox(container, el);
    const theme = plotlyDarkTheme();
    Plotly.newPlot(
      el,
      [
        {
          type: "bar",
          orientation: "h",
          y: [label],
          x: [slopePlot],
          marker: { color: "#3b82f6" },
          name: "Slope (kWh/day)",
        },
      ],
      {
        ...theme,
        autosize: false,
        width: sz.width,
        height: sz.height,
        title:
          "Slope of degradation trend (prediction − actual)" +
          (Number.isFinite(slope) ? "" : " — need ≥10 days in range for slope"),
        margin: { t: 40, r: 24, b: 48, l: 160 },
        xaxis: { ...theme.xaxis, title: "kWh/day" },
      },
      PLOTLY_STATIC
    );
  });

  let body = "";
  if (kpiRows.length) {
    for (const k of kpiRows) {
      const isFirst = body === "";
      body += `<tr><td>${k.building_name ?? k.meter_id ?? "—"}</td>`;
      body += `<td>${isFirst && Number.isFinite(slope) ? slope.toFixed(3) : "—"}</td>`;
      body += `<td>${isFirst ? totalDiff.toFixed(0) : "—"}</td>`;
      body += `<td>${isFirst ? days.length : "—"}</td>`;
      body += `<td>${k.actual_over_expected_ratio ?? "—"}</td>`;
      body += `<td>${k.correlation_actual_vs_pvlib ?? "—"}</td></tr>`;
    }
  } else {
    body = `<tr><td>${label}</td><td>${
      Number.isFinite(slope) ? slope.toFixed(3) : "—"
    }</td><td>${totalDiff.toFixed(0)}</td><td>${days.length}</td><td>—</td><td>—</td></tr>`;
  }

  document.getElementById("fleet-table-wrap").innerHTML =
    "<table class='data-table'><thead><tr><th>Building</th><th>Slope kWh/d</th><th>Σ Diff kWh</th><th>Days</th><th>Actual/expected</th><th>r</th></tr></thead><tbody>" +
    body +
    "</tbody></table>";
}
