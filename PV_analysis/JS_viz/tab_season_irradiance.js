/**
 * Season Analysis + irradiance (tab 7): same seasonal split as tab 6, with daily GHI
 * energy (kWh/m²·day, summed hourly daylight GHI > 5 W/m²) above each H chart.
 */
import { nextPlotDomId, purgePlotlyInContainer, PLOTLY_STATIC } from "./utils.js";
import {
  SEASONS,
  computeDailyH,
  computeDailyGhiKwh,
  groupSeasons,
  lightenSeasonColor,
  seasonCellFrameStyle,
  seasonChartPlotWidth,
} from "./season_analysis_common.js";
import { buildSeasonLinearPlot } from "./season_linear_plot.js";
import { seasonChartFooter } from "./season_chart_footer.js";

const PLOT_H_GHI = 200;
const PLOT_H_H = 258;

function seasonShortLabel(s) {
  return s.label.split("(")[0].trim();
}

function buildGhiChart(s, pts, w, h) {
  const light = lightenSeasonColor(s.color, 0.52);
  return buildSeasonLinearPlot(s, pts, w, h, {
    chartTitle: `${seasonShortLabel(s)} · GHI`,
    yTitle: "kWh/m²·day",
    yRange: undefined,
    dailyLineColor: light,
    dailyOpacity: 0.9,
    seasonMedianLineColor: "#5eead4",
    seasonMedianLabel: "Season median GHI",
    referencePaperLines: [],
    hoverValueLabel: "GHI",
    hoverDecimals: 2,
    showLegendGhost: true,
    ghostLegendName: "Daily GHI",
  });
}

function buildHChartPanel(s, pts, w, h) {
  return buildSeasonLinearPlot(s, pts, w, h, {
    chartTitle: `${seasonShortLabel(s)} · H`,
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

function dualSummary(resultsGhi, resultsH) {
  const rows = SEASONS.map((s) => {
    const rg = resultsGhi.get(s.id);
    const rh = resultsH.get(s.id);
    if (!rg && !rh)
      return `<tr><td style="color:${s.color}">${s.label}</td>
        <td colspan="5" style="color:#475569">No data</td></tr>`;

    const fmtMed = (r, dec) =>
      r && Number.isFinite(r.overall) ? r.overall.toFixed(dec) : "—";
    const fmtDeg = (r) =>
      r && Number.isFinite(r.degradPct) ? `${r.degradPct.toFixed(2)}%` : "—";

    return `<tr>
      <td style="color:${s.color}">${s.label}</td>
      <td>${fmtMed(rg, 2)}</td>
      <td style="font-size:0.78rem;color:#64748b">${fmtDeg(rg)}</td>
      <td>${fmtMed(rh, 3)}</td>
      <td style="font-size:0.78rem;color:#64748b">${fmtDeg(rh)}</td>
    </tr>`;
  }).join("");

  return `<table class="data-table" style="width:100%;margin-top:8px">
    <thead><tr>
      <th>Season</th>
      <th>Median daily GHI<br/><span style="font-weight:400;color:#64748b">(kWh/m²·day)</span></th>
      <th>GHI degrad.<br/><span style="font-weight:400;color:#64748b">(%/yr)</span></th>
      <th>Median H</th>
      <th>H degrad.<br/><span style="font-weight:400;color:#64748b">(%/yr)</span></th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

/**
 * @param {HTMLElement} container
 * @param {Array} hourly
 */
export function renderSeasonIrradiance(container, hourly, site = {}) {
  if (!container) {
    console.error("[SeasonIrr] Missing panel container (#panel-seasonirr).");
    return;
  }
  purgePlotlyInContainer(container);

  if (!hourly || !hourly.length) {
    container.innerHTML = `<p style="color:#94a3b8;padding:2rem">No hourly data loaded.</p>`;
    return;
  }

  const dailyH = computeDailyH(hourly);
  const dailyGhi = computeDailyGhiKwh(hourly);

  if (!dailyH.length && !dailyGhi.length) {
    container.innerHTML = `<p style="color:#94a3b8;padding:2rem">
      No daylight rows (GHI &gt; 5 W/m²) found in the dataset.</p>`;
    return;
  }

  const groupsH = groupSeasons(dailyH);
  const groupsGhi = groupSeasons(dailyGhi);

  const ghiIds = {};
  const hIds = {};
  const footGhi = {};
  const footH = {};
  for (const s of SEASONS) {
    ghiIds[s.id] = nextPlotDomId(`sir-g-${s.id}`);
    hIds[s.id] = nextPlotDomId(`sir-h-${s.id}`);
    footGhi[s.id] = nextPlotDomId(`sir-fg-${s.id}`);
    footH[s.id] = nextPlotDomId(`sir-fh-${s.id}`);
  }
  const sumId = nextPlotDomId("sir-sum");

  const counts = SEASONS.map((s) => {
    const nH = (groupsH.get(s.id) || []).length;
    const nG = (groupsGhi.get(s.id) || []).length;
    return `${s.label.split(" ")[0]}: ${nG}d GHI · ${nH}d H`;
  }).join(" · ");

  const siteLine = site?.label
    ? `<strong style="color:#e2e8f0">${site.label}</strong> · `
    : "";
  container.innerHTML = `<div style="padding:16px">
    <p style="font-size:0.78rem;color:#64748b;margin:0 0 14px">
      ${siteLine}Same Australian seasons as tab 6. <strong style="color:#e2e8f0">Upper:</strong> summed hourly GHI on daylight hours (GHI &gt; 5 W/m²), kWh/m² per calendar day.
      <strong style="color:#e2e8f0">Lower:</strong> health ratio H. ${counts}
    </p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px 18px;align-items:start">
    ${SEASONS.map((s) => `
      <div style="${seasonCellFrameStyle(s.color)}">
        <h3 style="color:${s.color};margin:0 0 8px;font-size:0.95rem">${s.label}</h3>
        <div id="${ghiIds[s.id]}" class="chart-box"
          style="height:${PLOT_H_GHI}px;min-height:${PLOT_H_GHI}px;margin-bottom:0;width:100%"></div>
        <div id="${footGhi[s.id]}"></div>
        <div id="${hIds[s.id]}" class="chart-box"
          style="height:${PLOT_H_H}px;min-height:${PLOT_H_H}px;margin-bottom:0;width:100%;margin-top:6px"></div>
        <div id="${footH[s.id]}"></div>
      </div>
    `).join("")}
    </div>
    <h3 style="margin:22px 0 8px;font-size:0.9rem;color:#cbd5e1;
               border-top:1px solid #334155;padding-top:12px">Summary</h3>
    <div id="${sumId}"></div>
  </div>`;

  /** Plot after the panel is visible + laid out (avoids 0×0 Plotly gl sometimes seen on tab switch). */
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      if (!container.isConnected || !container.classList.contains("active")) return;

      void container.offsetWidth;

      const w = seasonChartPlotWidth(container.clientWidth);

      const resultsGhi = new Map();
      const resultsH = new Map();

      for (const s of SEASONS) {
        const ptsG = groupsGhi.get(s.id) || [];
        const ptsH = groupsH.get(s.id) || [];
        const elG = document.getElementById(ghiIds[s.id]);
        const elH = document.getElementById(hIds[s.id]);
        const fg = document.getElementById(footGhi[s.id]);
        const fh = document.getElementById(footH[s.id]);

        if (!ptsG.length && !ptsH.length) {
          const msg =
            `<p style="color:#475569;font-size:0.8rem;margin:4px 0 12px">
          No records for this season.</p>`;
          if (fg) fg.innerHTML = "";
          if (fh) fh.innerHTML = msg;
          if (elG) elG.innerHTML = "";
          if (elH) elH.innerHTML = "";
          continue;
        }

        try {
          if (ptsG.length && elG) {
            const resG = buildGhiChart(s, ptsG, w, PLOT_H_GHI);
            resultsGhi.set(s.id, resG);
            Plotly.newPlot(elG, resG.traces, resG.layout, PLOTLY_STATIC);
            if (fg)
              fg.innerHTML = seasonChartFooter(resG, {
                medianLabel: "Median daily GHI",
                medianDecimals: 2,
                medianSuffix: " kWh/m²",
                detailMedDecimals: 2,
              });
          } else {
            if (elG) elG.innerHTML =
              `<p style="color:#475569;font-size:0.75rem;padding:0.5rem">No GHI days.</p>`;
            if (fg) fg.innerHTML = "";
          }

          if (ptsH.length && elH) {
            const resH = buildHChartPanel(s, ptsH, w, PLOT_H_H);
            resultsH.set(s.id, resH);
            Plotly.newPlot(elH, resH.traces, resH.layout, PLOTLY_STATIC);
            if (fh) fh.innerHTML = seasonChartFooter(resH);
          } else {
            if (elH) elH.innerHTML =
              `<p style="color:#475569;font-size:0.75rem;padding:0.5rem">No H days.</p>`;
            if (fh) fh.innerHTML = "";
          }
        } catch (err) {
          console.error("[SeasonIrr]", s.label, err);
          const errHtml =
            `<p style="color:#f87171;padding:0.5rem;font-size:0.8rem">${s.label}: ${err.message}</p>`;
          if (elG) elG.innerHTML = errHtml;
          if (elH) elH.innerHTML = "";
        }
      }

      const sumEl = document.getElementById(sumId);
      if (sumEl) sumEl.innerHTML = dualSummary(resultsGhi, resultsH);
    });
  });
}
