/**
 * Season Performance tab (6 · Season Analysis).
 * H = Actual ÷ PVLib expected per daylight day; linear season-year x-axis charts.
 */
import { nextPlotDomId, purgePlotlyInContainer, PLOTLY_STATIC }
  from "./utils.js";
import {
  SEASONS,
  computeDailyH,
  groupSeasons,
  seasonCellFrameStyle,
  seasonChartPlotWidth,
} from "./season_analysis_common.js";
import { buildSeasonLinearPlot } from "./season_linear_plot.js";
import { seasonChartFooter } from "./season_chart_footer.js";

export { formatTrendPct, seasonChartFooter } from "./season_chart_footer.js";

function summaryTable(results) {
  const rows = SEASONS.map((s) => {
    const r = results.get(s.id);
    if (!r) return `<tr><td style="color:${s.color}">${s.label}</td>
      <td colspan="4" style="color:#475569">No data</td></tr>`;
    const { overall, degradPct, ymeds } = r;
    let tc = "#94a3b8", ta = "≈";
    if (Number.isFinite(degradPct)) {
      if (degradPct < -0.5) { tc = "#f87171"; ta = "▼"; }
      else if (degradPct > 0.5) { tc = "#4ade80"; ta = "▲"; }
    }
    return `<tr>
      <td style="color:${s.color}">${s.label}</td>
      <td>${Number.isFinite(overall) ? overall.toFixed(3) : "N/A"}</td>
      <td style="color:${tc}">${ta} ${Number.isFinite(degradPct) ? degradPct.toFixed(2)+"%" : "N/A"}</td>
      <td style="font-size:0.78rem;color:#64748b">
        ${ymeds.map((d) => `${d.yr}: ${d.med.toFixed(2)}`).join(" · ") || "—"}
      </td>
    </tr>`;
  }).join("");
  return `<table class="data-table" style="width:100%;margin-top:8px">
    <thead><tr>
      <th>Season</th><th>Median H</th>
      <th>Degradation (%/yr)</th><th>Annual medians</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function buildHChart(s, pts, w, h) {
  return buildSeasonLinearPlot(s, pts, w, h, {
    yTitle: "H",
    yRange: [0, 1.5],
    dailyLineColor: s.color,
    dailyOpacity: 0.85,
    seasonMedianLineColor: "#2dd4bf",
    seasonMedianLabel: "Season median H",
    referencePaperLines: [
      { y: 1.0, color: "#94a3b8", dash: "dash" },
      { y: 0.9, color: "#475569", dash: "dot" },
    ],
    hoverValueLabel: "H",
    hoverDecimals: 3,
    showLegendGhost: true,
    ghostLegendName: "Daily H",
  });
}

/**
 * @param {HTMLElement} container  – the panel element
 * @param {Array}       hourly     – normalised rows from app.js (full dataset)
 */
export function renderSeasonPerf(container, hourly) {
  purgePlotlyInContainer(container);

  if (!hourly || !hourly.length) {
    container.innerHTML = `<p style="color:#94a3b8;padding:2rem">No hourly data loaded.</p>`;
    return;
  }

  const dailyH = computeDailyH(hourly);
  if (!dailyH.length) {
    container.innerHTML = `<p style="color:#94a3b8;padding:2rem">
      No daylight data (GHI &gt; 5 W/m²) found in the dataset.</p>`;
    return;
  }

  const groups = groupSeasons(dailyH);

  const pids = {};
  const dids = {};
  for (const s of SEASONS) {
    pids[s.id] = nextPlotDomId(`sp-${s.id}`);
    dids[s.id] = nextPlotDomId(`sd-${s.id}`);
  }
  const sumId = nextPlotDomId("sp-summary");

  const counts = SEASONS.map((s) => `${s.label.split(" ")[0]}: ${(groups.get(s.id)||[]).length}d`).join(" · ");
  const plotH = 280;

  container.innerHTML = `<div style="padding:16px">
    <p style="font-size:0.78rem;color:#64748b;margin:0 0 14px">
      H = Σ Actual ÷ Σ PVLib expected · daylight hours (GHI &gt; 5 W/m²) ·
      <strong style="color:#e2e8f0">${dailyH.length.toLocaleString()}</strong> day-records ·
      ${counts}
    </p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px 18px;align-items:start">
    ${SEASONS.map((s) => `
      <div style="${seasonCellFrameStyle(s.color)}">
        <h3 style="color:${s.color};margin:0 0 6px;font-size:0.95rem">${s.label}</h3>
        <div id="${pids[s.id]}" class="chart-box" style="height:${plotH}px;min-height:${plotH}px;margin-bottom:0;width:100%"></div>
        <div id="${dids[s.id]}"></div>
      </div>
    `).join("")}
    </div>
    <h3 style="margin:22px 0 8px;font-size:0.9rem;color:#cbd5e1;
               border-top:1px solid #334155;padding-top:12px">Summary</h3>
    <div id="${sumId}"></div>
  </div>`;

  void container.offsetWidth;

  const w = seasonChartPlotWidth(container.clientWidth);
  const h = plotH;
  const results = new Map();

  for (const s of SEASONS) {
    const pts = groups.get(s.id) || [];
    const plotDiv = document.getElementById(pids[s.id]);
    const degDiv = document.getElementById(dids[s.id]);

    if (!plotDiv) continue;

    if (!pts.length) {
      if (degDiv) degDiv.innerHTML =
        `<p style="color:#475569;font-size:0.8rem;margin:4px 0 20px">
          No records for this season in the current dataset.</p>`;
      continue;
    }

    try {
      const res = buildHChart(s, pts, w, h);
      results.set(s.id, res);
      Plotly.newPlot(plotDiv, res.traces, res.layout, PLOTLY_STATIC);
      if (degDiv) degDiv.innerHTML = seasonChartFooter(res);
    } catch (err) {
      console.error("[SeasonPerf]", s.label, err);
      if (plotDiv)
        plotDiv.innerHTML =
          `<p style="color:#f87171;padding:1rem;font-size:0.8rem">
            ${s.label}: ${err.message}</p>`;
    }
  }

  const sumEl = document.getElementById(sumId);
  if (sumEl) sumEl.innerHTML = summaryTable(results);
}
