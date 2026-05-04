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
         PLOTLY_STATIC, rollingMedian7Trailing, linearRegression }
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

function xRange(minDay, maxDay, theme) {
  const span = Math.max(1, Math.round(
    (new Date(maxDay + "T23:59:59") - new Date(minDay + "T00:00:00")) / 86400000));
  let dtick, tickformat = "%b %Y";
  if (span <= 45)        { dtick = 7 * 86400000;  tickformat = "%d %b"; }
  else if (span <= 120)  { dtick = 14 * 86400000; tickformat = "%d %b"; }
  else if (span <= 400)  { dtick = "M1";           tickformat = "%b '%y"; }
  else if (span <= 800)  { dtick = "M2"; }
  else                   { dtick = "M3"; }
  return { ...theme, type: "date", autorange: false,
    range: [minDay + "T00:00:00", maxDay + "T23:59:59"],
    tickformat, ...(dtick ? { dtick } : {}) };
}

function buildChart(s, pts, w, h) {
  const days  = pts.map((p) => p.day);
  const hvals = pts.map((p) => p.H);
  const med7  = rollingMedian7Trailing(hvals);
  const ymeds = yearlyMedians(pts);
  const overall = medianOf(hvals.filter(Number.isFinite));

  let slope = NaN, intercept = NaN, degradPct = NaN;
  let rDates = [], rY = [];
  if (ymeds.length >= 2) {
    const xs = ymeds.map((_, i) => i);
    const ys = ymeds.map((d) => d.med);
    ({ slope, intercept } = linearRegression(xs, ys));
    if (Number.isFinite(slope) && Number.isFinite(intercept)) {
      if (ymeds[0].med > 0) degradPct = (slope / ymeds[0].med) * 100;
      rDates = xs.map((x) => `${ymeds[0].yr + x}-${String(s.midM).padStart(2,"0")}-15T12:00:00`);
      rY = xs.map((x) => slope * x + intercept);
    }
  }

  const mDates = ymeds.map((d) => `${d.yr}-${String(s.midM).padStart(2,"0")}-15T12:00:00`);
  const mVals  = ymeds.map((d) => d.med);

  const traces = [
    { x: days, y: hvals, mode: "markers", type: "scatter", name: "Daily H",
      marker: { color: s.color, size: 5, opacity: 0.55 } },
    { x: days, y: med7, mode: "lines", type: "scatter", name: "7-day median",
      line: { color: "#fb923c", width: 2 }, connectgaps: false },
  ];
  if (mDates.length) traces.push({
    x: mDates, y: mVals, mode: "markers+text", type: "scatter", name: "Annual median",
    marker: { color: "#ffffff", size: 9, symbol: "diamond",
               line: { color: "#334155", width: 1.5 } },
    text: mVals.map((v) => v.toFixed(2)), textposition: "top center",
    textfont: { color: "#e2e8f0", size: 10 },
  });
  if (rDates.length >= 2) traces.push({
    x: rDates, y: rY, mode: "lines", type: "scatter", name: "Trend",
    line: { color: "#f87171", width: 2, dash: "dash" },
  });

  const theme = plotlyDarkTheme();
  const layout = {
    ...theme,
    title: { text: s.label, font: { color: s.color, size: 14 },
             x: 0, xanchor: "left", y: 0.98, yanchor: "top" },
    xaxis: xRange(days[0], days[days.length - 1], theme.xaxis),
    yaxis: { ...theme.yaxis, title: "H", range: [0, 1.5], autorange: false },
    shapes: [
      { type:"line", xref:"paper", yref:"y", x0:0, x1:1, y0:1.0, y1:1.0,
        line:{ color:"#94a3b8", width:1, dash:"dash" } },
      { type:"line", xref:"paper", yref:"y", x0:0, x1:1, y0:0.9, y1:0.9,
        line:{ color:"#475569", width:1, dash:"dot"  } },
    ],
    annotations: [
      { xref:"paper", yref:"y", x:1, y:1.0, xanchor:"right", yanchor:"bottom",
        text:"H=1.0", showarrow:false, font:{ color:"#94a3b8", size:9 } },
      { xref:"paper", yref:"y", x:1, y:0.9, xanchor:"right", yanchor:"bottom",
        text:"H=0.9", showarrow:false, font:{ color:"#475569", size:9 } },
    ],
    legend: { x:1, xanchor:"right", y:1, bgcolor:"rgba(15,23,42,0.7)",
              font:{ color:"#e2e8f0", size:10 } },
    margin: { t:40, r:28, b:56, l:56 },
    hovermode: "x unified", showlegend: true,
    autosize: false, width: w, height: h,
  };

  return { traces, layout, overall, degradPct, ymeds };
}

// ─── degradation stat bar ────────────────────────────────────────────────────

function degradBar(res) {
  const { overall, degradPct, ymeds } = res;
  const medStr = Number.isFinite(overall) ? overall.toFixed(3) : "N/A";
  let dc = "#94a3b8", arrow = "≈", dStr = "N/A";
  if (Number.isFinite(degradPct)) {
    const sign = degradPct >= 0 ? "+" : "";
    dStr = `${sign}${degradPct.toFixed(2)}%/yr`;
    if (degradPct < -0.5) { dc = "#f87171"; arrow = "▼"; }
    else if (degradPct > 0.5) { dc = "#4ade80"; arrow = "▲"; }
  }
  const detail = ymeds.map((d) => `${d.yr}: ${d.med.toFixed(2)} (n=${d.n})`).join("  ·  ");
  return `<div style="background:#1e293b;border:1px solid #334155;border-radius:8px;
    padding:10px 16px;margin:4px 0 28px;font-size:0.82rem;
    display:flex;flex-wrap:wrap;gap:8px 24px;align-items:center">
    <span style="color:#94a3b8">Median H: <strong style="color:#e2e8f0">${medStr}</strong></span>
    <span style="color:#94a3b8">Degradation:
      <strong style="color:${dc};font-size:1rem"> ${arrow} ${dStr}</strong></span>
    <span style="color:#64748b;font-size:0.78rem">${detail}</span>
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

  container.innerHTML = `<div style="padding:16px">
    <p style="font-size:0.78rem;color:#64748b;margin:0 0 16px">
      H = Σ Actual ÷ Σ PVLib expected · daylight hours (GHI &gt; 5 W/m²) ·
      <strong style="color:#e2e8f0">${dailyH.length.toLocaleString()}</strong> day-records ·
      ${counts}
    </p>
    ${SEASONS.map((s) => `
      <h3 style="color:${s.color};margin:24px 0 6px;font-size:1rem">${s.label}</h3>
      <div id="${pids[s.id]}" class="chart-box"></div>
      <div id="${dids[s.id]}"></div>
    `).join("")}
    <h3 style="margin:28px 0 8px;font-size:0.9rem;color:#cbd5e1;
               border-top:1px solid #334155;padding-top:12px">Summary</h3>
    <div id="${sumId}"></div>
  </div>`;

  // Force a synchronous layout reflow so clientWidth is accurate immediately —
  // avoids requestAnimationFrame timing issues that plagued earlier versions.
  void container.offsetWidth;

  const w = Math.max(320, (container.clientWidth || 960) - 32);
  const h = 400;
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
      if (degDiv) degDiv.innerHTML = degradBar(res);
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
