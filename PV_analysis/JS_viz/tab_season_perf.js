/**
 * Season Performance tab.
 * Breaks the full H = Actual ÷ PVLib health-ratio dataset into four Australian
 * meteorological seasons and renders one chart per season, stacked vertically.
 * No deferred rendering — Plotly.newPlot is called directly after innerHTML,
 * using a forced reflow so dimensions are available immediately.
 *
 * Australian seasons:
 *   Summer  Dec–Feb  (Dec belongs to the *next* year's summer, e.g. Dec 2022 → 2023)
 *   Autumn  Mar–May
 *   Winter  Jun–Aug
 *   Spring  Sep–Nov
 */
import { plotlyDarkTheme, nextPlotDomId, purgePlotlyInContainer,
         PLOTLY_STATIC, linearRegression }
  from "./utils.js";

const SEASONS = [
  { id: "sum", label: "Summer (Dec–Feb)", color: "#f59e0b", midM: 1  },
  { id: "aut", label: "Autumn (Mar–May)", color: "#22c55e", midM: 4  },
  { id: "win", label: "Winter (Jun–Aug)", color: "#38bdf8", midM: 7  },
  { id: "spr", label: "Spring (Sep–Nov)", color: "#a78bfa", midM: 10 },
];

// ─── data ───────────────────────────────────────────────────────────────────

function computeDailyH(hourly) {
  const byDay = new Map();
  for (const r of hourly) {
    if (!(r.ghi > 5)) continue;
    let o = byDay.get(r.day);
    if (!o) { o = { act: 0, exp: 0 }; byDay.set(r.day, o); }
    if (Number.isFinite(r.actual))   o.act += r.actual;
    if (Number.isFinite(r.expected)) o.exp += r.expected;
  }
  const out = [];
  for (const [day, { act, exp }] of byDay)
    if (exp > 0) out.push({ day, H: act / exp });
  return out.sort((a, b) => (a.day < b.day ? -1 : 1));
}

function seasonOf(dayStr) {
  const yr = +dayStr.slice(0, 4);
  const mo = +dayStr.slice(5, 7);
  if (mo === 12 || mo === 1 || mo === 2)
    return { id: "sum", yr: mo === 12 ? yr + 1 : yr };
  if (mo <= 5) return { id: "aut", yr };
  if (mo <= 8) return { id: "win", yr };
  return { id: "spr", yr };
}

function groupSeasons(dailyH) {
  const map = new Map(SEASONS.map((s) => [s.id, []]));
  for (const { day, H } of dailyH) {
    const info = seasonOf(day);
    map.get(info.id)?.push({ day, H, yr: info.yr });
  }
  return map;
}

function medianOf(arr) {
  if (!arr.length) return NaN;
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function yearlyMedians(pts) {
  const m = new Map();
  for (const { H, yr } of pts) { if (!m.has(yr)) m.set(yr, []); m.get(yr).push(H); }
  const out = [];
  for (const [yr, vals] of m) {
    const med = medianOf(vals);
    out.push({ yr, med, n: vals.length });
  }
  return out.sort((a, b) => a.yr - b.yr);
}

// ─── chart ──────────────────────────────────────────────────────────────────

/** Linear x zones: first season-year centre at 61, spacing 46 (matches 2020→61 … 2025→291). */
const YEAR_CENTER_STEP = 46;
const YEAR_CENTER_FIRST = 61;
const SEASON_SPAN = 40;
const X_RANGE_PAD = 23;

function buildChart(s, pts, w, h) {
  const hvals = pts.map((p) => p.H);
  const ymeds = yearlyMedians(pts);
  const overall = medianOf(hvals.filter(Number.isFinite));
  const dayCount = pts.length;

  const byYear = new Map();
  for (const p of pts) {
    if (!byYear.has(p.yr)) byYear.set(p.yr, []);
    byYear.get(p.yr).push(p);
  }
  const sortedYears = [...byYear.keys()].sort((a, b) => a - b);
  const yearCenterX = {};
  sortedYears.forEach((yr, i) => {
    yearCenterX[yr] = YEAR_CENTER_FIRST + i * YEAR_CENTER_STEP;
  });

  const tickvals = sortedYears.map((yr) => yearCenterX[yr]);
  const ticktext = sortedYears.map(String);
  const firstC = tickvals[0];
  const lastC = tickvals[tickvals.length - 1];
  const xRange = [
    firstC - SEASON_SPAN / 2 - X_RANGE_PAD,
    lastC + SEASON_SPAN / 2 + X_RANGE_PAD,
  ];

  let slopeYear = NaN, interceptYear = NaN, degradPct = NaN;
  if (ymeds.length >= 2) {
    const xsYr = ymeds.map((d) => d.yr);
    const ysMed = ymeds.map((d) => d.med);
    ({ slope: slopeYear, intercept: interceptYear } = linearRegression(xsYr, ysMed));
    if (Number.isFinite(slopeYear) && Number.isFinite(interceptYear) && ymeds[0].med > 0)
      degradPct = (slopeYear / ymeds[0].med) * 100;
  }

  const traces = [
    {
      x: [null],
      y: [null],
      type: "scatter",
      mode: "markers",
      name: "Daily H",
      marker: { color: s.color, size: 6, opacity: 0.75 },
      showlegend: true,
      hoverinfo: "skip",
    },
  ];

  for (const yr of sortedYears) {
    const yearPts = [...byYear.get(yr)].sort((a, b) => a.day.localeCompare(b.day));
    const totalDaysInThatYear = Math.max(yearPts.length, 1);
    const xc = yearCenterX[yr];
    const xs = yearPts.map((p, dayOfSeason) =>
      xc -
      SEASON_SPAN / 2 +
      (dayOfSeason / totalDaysInThatYear) * SEASON_SPAN);
    const ys = yearPts.map((p) => p.H);
    const text = yearPts.map((p) => p.day);
    traces.push({
      x: xs,
      y: ys,
      type: "scatter",
      mode: "lines+markers",
      name: `${yr}`,
      line: { color: s.color, width: 1.2 },
      marker: {
        color: s.color,
        size: 5,
        opacity: 0.75,
        line: { width: 0 },
      },
      opacity: 0.85,
      showlegend: false,
      hovertemplate: "%{text}<br>H = %{y:.3f}<extra></extra>",
      text,
    });
  }

  if (Number.isFinite(overall)) {
    traces.push({
      x: [xRange[0], xRange[1]],
      y: [overall, overall],
      type: "scatter",
      mode: "lines",
      name: "Season median H",
      line: { color: "#2dd4bf", width: 1.5, dash: "dot" },
      hovertemplate: `Season median H = ${overall.toFixed(3)}<extra></extra>`,
      showlegend: true,
    });
  }

  const medX = ymeds.map((d) => yearCenterX[d.yr]);
  const medY = ymeds.map((d) => d.med);
  if (medX.length) {
    traces.push({
      x: medX,
      y: medY,
      type: "scatter",
      mode: "lines+markers",
      name: "Annual median",
      line: { color: "#fb923c", width: 1.8 },
      marker: {
        symbol: "diamond",
        size: 8,
        color: "white",
        line: { color: "#64748b", width: 1 },
      },
      text: ymeds.map((d) => `${d.yr}: H=${d.med.toFixed(2)}`),
      hovertemplate: "%{text}<extra></extra>",
      showlegend: true,
    });
  }

  if (
    ymeds.length >= 2 &&
    Number.isFinite(slopeYear) &&
    Number.isFinite(interceptYear)
  ) {
    const tx = ymeds.map((d) => yearCenterX[d.yr]);
    const ty = ymeds.map((d) => slopeYear * d.yr + interceptYear);
    traces.push({
      x: tx,
      y: ty,
      type: "scatter",
      mode: "lines",
      name: "Trend (annual medians)",
      line: { color: "#f87171", width: 1.4, dash: "dash" },
      showlegend: true,
      hoverinfo: "skip",
    });
  }

  const shapes = [
    {
      type: "line",
      xref: "paper",
      yref: "y",
      x0: 0,
      x1: 1,
      y0: 1.0,
      y1: 1.0,
      line: { color: "#94a3b8", width: 1, dash: "dash" },
    },
    {
      type: "line",
      xref: "paper",
      yref: "y",
      x0: 0,
      x1: 1,
      y0: 0.9,
      y1: 0.9,
      line: { color: "#475569", width: 1, dash: "dot" },
    },
  ];
  for (let i = 0; i < sortedYears.length - 1; i++) {
    const xb =
      (yearCenterX[sortedYears[i]] + yearCenterX[sortedYears[i + 1]]) / 2;
    shapes.push({
      type: "line",
      xref: "x",
      yref: "paper",
      x0: xb,
      x1: xb,
      y0: 0,
      y1: 1,
      line: { color: "#1e293b", width: 0.5 },
    });
  }

  const theme = plotlyDarkTheme();
  const layout = {
    ...theme,
    title: {
      text: s.label,
      font: { color: s.color, size: 14 },
      x: 0,
      xanchor: "left",
      y: 0.98,
      yanchor: "top",
    },
    xaxis: {
      ...theme.xaxis,
      type: "linear",
      autorange: false,
      range: xRange,
      tickmode: "array",
      tickvals,
      ticktext,
      showgrid: false,
      gridcolor: "#1e293b",
      linecolor: "#334155",
    },
    yaxis: { ...theme.yaxis, title: "H", range: [0, 1.5], autorange: false },
    shapes,
    legend: {
      x: 1,
      xanchor: "right",
      y: 1,
      yanchor: "top",
      bgcolor: "rgba(15,23,42,0.82)",
      bordercolor: "#334155",
      borderwidth: 1,
      font: { color: "#e2e8f0", size: 9 },
    },
    margin: { t: 32, r: 12, b: 40, l: 48 },
    hovermode: "closest",
    showlegend: true,
    autosize: false,
    width: w,
    height: h,
  };

  return { traces, layout, overall, degradPct, ymeds, dayCount };
}

function formatTrendPct(pct) {
  if (!Number.isFinite(pct)) return "N/A";
  const abs = Math.abs(pct).toFixed(2);
  return pct < 0 ? `\u2212${abs}` : `+${abs}`;
}

/** Compact stats block below each season chart: summary row + per-year medians. */
function seasonChartFooter(res) {
  const { overall, degradPct, dayCount, ymeds } = res;
  const medStr = Number.isFinite(overall) ? overall.toFixed(3) : "N/A";

  let badgeStyle =
    "background:#1e293b;color:#94a3b8";
  let trendStr = "Degradation N/A";
  if (Number.isFinite(degradPct)) {
    const pctPart = `${formatTrendPct(degradPct)}/yr`;
    if (degradPct < -0.5) {
      badgeStyle = "background:#fef2f2;color:#b91c1c";
      trendStr = `Degradation \u25bc ${pctPart}`;
    } else if (degradPct > 0.5) {
      badgeStyle = "background:#f0fdf4;color:#15803d";
      trendStr = `Degradation \u25b2 ${pctPart}`;
    } else {
      trendStr = `Degradation \u2248 ${pctPart}`;
    }
  }

  const detail =
    ymeds && ymeds.length
      ? ymeds.map((d) => `${d.yr}: ${d.med.toFixed(2)} (n=${d.n})`).join(" · ")
      : "";

  return `<div style="margin-top:2px">
    <div style="display:flex;gap:12px 14px;align-items:center;flex-wrap:wrap;font-size:12px;
                color:#94a3b8;line-height:1.35">
      <span>Median H: <strong style="color:#e2e8f0">${medStr}</strong></span>
      <span style="${badgeStyle};font-size:11px;padding:2px 8px;border-radius:4px;
                  font-weight:500;">${trendStr}</span>
      <span>${dayCount} days</span>
    </div>
    ${detail ? `<div style="margin-top:4px;padding:6px 8px;background:#1e293b;
      border:1px solid #334155;border-radius:6px;font-size:11px;color:#94a3b8;
      line-height:1.4">${detail}</div>` : ""}
  </div>`;
}

// ─── summary table ───────────────────────────────────────────────────────────

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

// ─── public ──────────────────────────────────────────────────────────────────

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

  // Unique IDs per render via nextPlotDomId — same as every working tab.
  const pids = {};
  const dids = {};
  for (const s of SEASONS) {
    pids[s.id] = nextPlotDomId(`sp-${s.id}`);
    dids[s.id] = nextPlotDomId(`sd-${s.id}`);
  }
  const sumId = nextPlotDomId("sp-summary");

  const counts = SEASONS.map((s) => `${s.label.split(" ")[0]}: ${(groups.get(s.id)||[]).length}d`).join(" · ");
  /** Matches Plotly layout height; overrides global `.chart-box` 400px + margin-bottom. */
  const plotH = 280;

  container.innerHTML = `<div style="padding:16px">
    <p style="font-size:0.78rem;color:#64748b;margin:0 0 14px">
      H = Σ Actual ÷ Σ PVLib expected · daylight hours (GHI &gt; 5 W/m²) ·
      <strong style="color:#e2e8f0">${dailyH.length.toLocaleString()}</strong> day-records ·
      ${counts}
    </p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px 18px;align-items:start">
    ${SEASONS.map((s) => `
      <div style="min-width:0">
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

  // Force a synchronous layout reflow so clientWidth is accurate immediately —
  // avoids requestAnimationFrame timing issues that plagued earlier versions.
  void container.offsetWidth;

  const padOuter = 32;
  const gridGap = 18;
  const cw = Math.max(480, container.clientWidth || 960);
  const inner = cw - padOuter;
  const w = Math.max(260, Math.floor((inner - gridGap) / 2));
  const h = plotH;
  const results = new Map();

  for (const s of SEASONS) {
    const pts     = groups.get(s.id) || [];
    const plotDiv = document.getElementById(pids[s.id]);
    const degDiv  = document.getElementById(dids[s.id]);

    if (!plotDiv) continue;

    if (!pts.length) {
      if (degDiv) degDiv.innerHTML =
        `<p style="color:#475569;font-size:0.8rem;margin:4px 0 20px">
          No records for this season in the current dataset.</p>`;
      continue;
    }

    try {
      const res = buildChart(s, pts, w, h);
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
