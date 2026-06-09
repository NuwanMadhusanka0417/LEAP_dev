/**
 * All meters — horizontal bar comparison of degradation (%/yr) from monthly and annual H trends.
 */
import {
  parseCSV,
  plotlyDarkTheme,
  nextPlotDomId,
  purgePlotlyInContainer,
  runAfterTabLayout,
  PLOTLY_STATIC,
} from "./utils.js";
import {
  loadMeterCatalog,
  fetchHourlyMasterText,
  hourlyMasterFilename,
} from "./meters.js";
import {
  computeDegradationMetrics,
  formatPctPerYear,
  MIN_POINTS_MONTHLY,
  MIN_POINTS_YEARLY,
  normalizeHourlyRows,
} from "./degradation_common.js";
import {
  seasonCellFrameStyle,
  seasonChartPlotWidth,
} from "./season_analysis_common.js";

const FRAME_YEARLY = "#34d399";
const FRAME_MONTHLY = "#a78bfa";

function barColor(pct) {
  if (!Number.isFinite(pct)) return "#64748b";
  if (pct < -0.5) return "#f87171";
  if (pct > 0.5) return "#34d399";
  return "#94a3b8";
}

function chartHeight(nBars) {
  return Math.max(160, Math.min(280, nBars * 30 + 64));
}

function xRangeWithLabelRoom(values) {
  const finite = values.filter(Number.isFinite);
  let minV = finite.length ? Math.min(...finite, 0) : -1;
  let maxV = finite.length ? Math.max(...finite, 0) : 1;
  const span = Math.max(maxV - minV, 0.5);
  const barPad = span * 0.06;
  const labelRoom = span * 0.42;
  if (maxV > 0) maxV += barPad + labelRoom;
  else maxV += barPad;
  if (minV < 0) minV -= barPad + labelRoom;
  else minV -= barPad;
  return [minV, maxV];
}

function buildHorizontalBarLayout(theme, sz, xTitle, values) {
  const [xMin, xMax] = xRangeWithLabelRoom(values);

  return {
    ...theme,
    autosize: false,
    width: sz.width,
    height: sz.height,
    margin: { t: 8, r: 4, b: 40, l: 8 },
    xaxis: {
      ...theme.xaxis,
      title: { text: xTitle, standoff: 6 },
      zeroline: true,
      zerolinecolor: "#475569",
      zerolinewidth: 1.5,
      range: [xMin, xMax],
      fixedrange: true,
    },
    yaxis: {
      ...theme.yaxis,
      automargin: true,
      tickfont: { size: 10 },
      fixedrange: true,
    },
    bargap: 0.62,
  };
}

const BAR_VALUE_FONT = {
  size: 13,
  color: "#f8fafc",
  family: "Segoe UI Semibold, Segoe UI, system-ui, sans-serif",
};

function plotHorizontalBars(el, labels, values, xTitle, plotWidth) {
  const h = chartHeight(labels.length);
  const sz = { width: plotWidth, height: h };
  const colors = values.map(barColor);
  const theme = plotlyDarkTheme();

  el.style.height = `${h}px`;
  el.style.minHeight = `${h}px`;
  el.style.maxWidth = "100%";
  el.style.overflow = "hidden";

  Plotly.newPlot(
    el,
    [
      {
        type: "bar",
        orientation: "h",
        y: labels,
        x: values,
        marker: { color: colors, line: { color: "#0f172a", width: 0.5 } },
        text: values.map((v) => formatPctPerYear(v)),
        textposition: "outside",
        textfont: BAR_VALUE_FONT,
        outsidetextfont: BAR_VALUE_FONT,
        cliponaxis: false,
        hovertemplate: "%{y}<br>~%{x:.2f} %/yr<extra></extra>",
      },
    ],
    buildHorizontalBarLayout(theme, sz, xTitle, values),
    PLOTLY_STATIC,
  );
}

/**
 * @param {HTMLElement} container
 * @param {string} dateFrom
 * @param {string} dateTo
 */
export async function renderAllMetersDegradation(container, dateFrom, dateTo) {
  purgePlotlyInContainer(container);

  const idYearly = nextPlotDomId("alldeg-yearly");
  const idMonthly = nextPlotDomId("alldeg-monthly");
  const idTable = nextPlotDomId("alldeg-table");

  container.innerHTML = `
    <div style="padding:16px">
    <h2>All meters — degradation comparison</h2>
    <p class="note">
      Compares every site in <code>sites_kpis_summary.csv</code> on the selected date window.
      <strong>H = Σ actual ÷ Σ PVLib expected</strong> (daylight, GHI &gt; 5 W/m²).
      Bars show the linear-trend proxy (~%/yr): <strong>annual</strong> regression on calendar-year H,
      and <strong>monthly</strong> regression on monthly H (slope annualised).
      Red = H trending down; green = trending up.
    </p>
    <p style="font-size:0.78rem;color:#64748b;margin:0 0 16px">
      Window: <strong>${dateFrom}</strong> → <strong>${dateTo}</strong>
    </p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px 18px;align-items:start">
      <div style="${seasonCellFrameStyle(FRAME_YEARLY)}">
        <h3 style="color:${FRAME_YEARLY};margin:0 0 8px;font-size:0.95rem">
          Year-based degradation
          <span style="font-weight:400;color:#64748b;font-size:0.82rem"> (~%/yr)</span>
        </h3>
        <div id="${idYearly}" class="chart-box chart-box--compact"
          style="height:200px;min-height:160px;margin-bottom:0;width:100%;overflow:hidden"></div>
      </div>
      <div style="${seasonCellFrameStyle(FRAME_MONTHLY)}">
        <h3 style="color:${FRAME_MONTHLY};margin:0 0 8px;font-size:0.95rem">
          Monthly-based degradation
          <span style="font-weight:400;color:#64748b;font-size:0.82rem"> (~%/yr)</span>
        </h3>
        <div id="${idMonthly}" class="chart-box chart-box--compact"
          style="height:200px;min-height:160px;margin-bottom:0;width:100%;overflow:hidden"></div>
      </div>
    </div>
    <h3 style="font-size:0.92rem;color:#cbd5e1;margin:20px 0 8px;border-top:1px solid #334155;padding-top:12px">Detail</h3>
    <div id="${idTable}"><p style="color:#94a3b8">Loading all meters…</p></div>
    </div>
  `;

  const meters = await loadMeterCatalog();
  const results = [];

  await Promise.all(
    meters.map(async (m) => {
      try {
        const { text } = await fetchHourlyMasterText(m.key);
        const { rows } = parseCSV(text);
        const hourly = normalizeHourlyRows(rows);
        const metrics = computeDegradationMetrics(hourly, dateFrom, dateTo);
        results.push({ ...m, ...metrics, ok: true });
      } catch (e) {
        results.push({
          ...m,
          ok: false,
          error: e.message,
          pctYearly: NaN,
          pctMonthly: NaN,
        });
      }
    }),
  );

  results.sort((a, b) => a.label.localeCompare(b.label));

  const tableEl = document.getElementById(idTable);
  if (tableEl) {
    tableEl.innerHTML = `
      <div style="overflow-x:auto">
        <table class="data-table">
          <thead>
            <tr>
              <th>Meter</th>
              <th>Years (H)</th>
              <th>Mean annual H</th>
              <th>Yearly ~%/yr</th>
              <th>Months (H)</th>
              <th>Mean monthly H</th>
              <th>Monthly ~%/yr</th>
            </tr>
          </thead>
          <tbody>
            ${results
              .map((r) => {
                if (!r.ok) {
                  return `<tr><td>${r.label}</td><td colspan="6" style="color:#f87171">
                    Failed to load ${hourlyMasterFilename(r.key)}: ${r.error}</td></tr>`;
                }
                const fmtH = (v) =>
                  Number.isFinite(v) ? v.toFixed(4) : "—";
                const fmtPct = (p, n, min) =>
                  n >= min ? formatPctPerYear(p) : `— <span style="color:#64748b">(need ≥${min})</span>`;
                return `<tr>
                  <td><strong>${r.label}</strong></td>
                  <td>${r.yearCount}</td>
                  <td>${fmtH(r.meanYearly)}</td>
                  <td>${fmtPct(r.pctYearly, r.yearCount, MIN_POINTS_YEARLY)}</td>
                  <td>${r.monthCount}</td>
                  <td>${fmtH(r.meanMonthly)}</td>
                  <td>${fmtPct(r.pctMonthly, r.monthCount, MIN_POINTS_MONTHLY)}</td>
                </tr>`;
              })
              .join("")}
          </tbody>
        </table>
      </div>`;
  }

  runAfterTabLayout(container, () => {
    void container.offsetWidth;
    const plotW = seasonChartPlotWidth(container.clientWidth);

    const okRows = results.filter((r) => r.ok);
    const labels = okRows.map((r) => r.label);
    const yearlyVals = okRows.map((r) => r.pctYearly);
    const monthlyVals = okRows.map((r) => r.pctMonthly);

    const elY = document.getElementById(idYearly);
    if (elY) {
      if (!labels.length) {
        elY.innerHTML =
          `<p style="color:#64748b;padding:1rem;font-size:0.85rem">No meter data loaded.</p>`;
      } else {
        plotHorizontalBars(elY, labels, yearlyVals, "Trend (%/yr)", plotW);
      }
    }

    const elM = document.getElementById(idMonthly);
    if (elM) {
      if (!labels.length) {
        elM.innerHTML =
          `<p style="color:#64748b;padding:1rem;font-size:0.85rem">No meter data loaded.</p>`;
      } else {
        plotHorizontalBars(elM, labels, monthlyVals, "Trend (%/yr)", plotW);
      }
    }
  });
}
