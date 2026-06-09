/**
 * Meter degradation — **calendar-year** performance ratios and **linear regression on annual** points;
 * **monthly H** for medium-term detail (slope vs elapsed calendar days).
 *
 * **H = Σ actual ÷ Σ PVLib expected** (daylight hours only, **GHI > 5 W/m²**).
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
import {
  yearlyPerformanceRatio,
  monthlyPerformanceRatio,
  regressOnCalendarYears,
  regressOnElapsedDays,
  computeDegradationMetrics,
  formatPctPerYear,
  MIN_POINTS_MONTHLY,
  MIN_POINTS_YEARLY,
} from "./degradation_common.js";

function buildMonthlyHChartLayout(theme, title, yTitle, sz, annotations, nMonths) {
  const dtick =
    nMonths <= 10 ? "M1" : nMonths <= 24 ? "M2" : nMonths <= 48 ? "M3" : "M6";
  return {
    ...theme,
    autosize: false,
    width: sz.width,
    height: Math.max(280, sz.height),
    title: { text: title, font: { color: "#e2e8f0", size: 14 } },
    margin: { t: 52, r: 24, b: 68, l: 56 },
    hovermode: "x unified",
    xaxis: {
      ...theme.xaxis,
      type: "date",
      title: "Month",
      tickformat: "%b %Y",
      dtick,
      tickangle: -40,
      ticklabelposition: "outside bottom",
      automargin: true,
    },
    yaxis: { ...theme.yaxis, title: yTitle },
    legend: {
      orientation: "h",
      y: 1.05,
      x: 0,
      bgcolor: "rgba(15,23,42,0.7)",
      font: { size: 11 },
    },
    annotations: annotations || [],
  };
}

function buildYearlyHChartLayout(theme, title, sz, annotations) {
  return {
    ...theme,
    autosize: false,
    width: sz.width,
    height: Math.max(300, sz.height),
    title: { text: title, font: { color: "#e2e8f0", size: 14 } },
    margin: { t: 52, r: 24, b: 52, l: 56 },
    hovermode: "closest",
    xaxis: {
      ...theme.xaxis,
      type: "linear",
      title: "Calendar year",
      dtick: 1,
      tickformat: "d",
    },
    yaxis: { ...theme.yaxis, title: "Annual H" },
    legend: {
      orientation: "h",
      y: 1.05,
      x: 0,
      bgcolor: "rgba(15,23,42,0.7)",
      font: { size: 11 },
    },
    annotations: annotations || [],
  };
}

export function renderMeterDegradation(container, hourly, dateFrom, dateTo, site = {}) {
  purgePlotlyInContainer(container);

  if (!hourly || !hourly.length) {
    container.innerHTML =
      `<p style="color:#94a3b8;padding:2rem">No hourly data loaded.</p>`;
    return;
  }

  const filtered = hourly.filter(
    (r) => r.day >= dateFrom && r.day <= dateTo,
  );

  const { years: yearStrs, ratio: yearlyHRaw } = yearlyPerformanceRatio(filtered);
  const yearlyClean = yearStrs
    .map((yy, i) => ({ year: yy, h: yearlyHRaw[i] }))
    .filter((o) => Number.isFinite(o.h));
  const yearLabels = yearlyClean.map((o) => o.year);
  const yearY = yearlyClean.map((o) => o.h);
  const yearNums = yearLabels.map((y) => parseInt(y, 10));

  const { monthStarts, ratio: monthlyH } = monthlyPerformanceRatio(filtered);
  const monthlyClean = monthStarts
    .map((d, i) => ({
      day: d,
      h: monthlyH[i],
    }))
    .filter((o) => Number.isFinite(o.h));
  const monthDays = monthlyClean.map((o) => o.day);
  const monthY = monthlyClean.map((o) => o.h);

  const idYearly = nextPlotDomId("deg-yearly");
  const idMonthly = nextPlotDomId("deg-monthly");

  const yearlyReg =
    yearLabels.length >= MIN_POINTS_YEARLY
      ? regressOnCalendarYears(yearLabels, yearY)
      : { slope: NaN, intercept: NaN, fitY: [] };
  const monthlyReg =
    monthDays.length >= MIN_POINTS_MONTHLY
      ? regressOnElapsedDays(monthDays, monthY)
      : { slope: NaN, intercept: NaN, fitY: [] };

  const metrics = computeDegradationMetrics(hourly, dateFrom, dateTo);
  const { meanYearly, meanMonthly, pctYearly: pctY, pctMonthly: pctM } = metrics;

  const fmtSlopeDay = (s) =>
    Number.isFinite(s) ? `${s.toFixed(8)} / day` : "—";
  const fmtSlopeYear = (s) =>
    Number.isFinite(s) ? `${s.toFixed(6)} / yr` : "—";

  container.innerHTML = `
    <h2>${siteHeading("Meter degradation (performance vs PVLib)", site)}</h2>
    <p class="note">
      <strong>Annual aggregation</strong> smooths most weather and seasonal swing.
      <strong>Linear regression on annual H</strong> summarises long-run trend;
      <strong>H = Σ actual ÷ Σ PVLib expected</strong> (daylight, <strong>GHI &gt; 5 W/m²</strong>) keeps each year
      weather-normalised vs raw kWh-only trends. Partial calendar years in your date filter still count as one
      “year” bucket — use full Jan–Dec spans when you need comparable years.
    </p>
    <h3 style="font-size:0.92rem;color:#cbd5e1;margin:16px 0 8px">
      Annual H — primary degradation trend
      <span style="font-weight:400;color:#64748b">(linear regression vs calendar year)</span>
    </h3>
    <div id="${idYearly}" class="chart-box" style="min-height:300px"></div>

    <h3 style="font-size:0.92rem;color:#cbd5e1;margin:20px 0 8px">
      Monthly H <span style="font-weight:400;color:#64748b">(medium-term detail)</span>
    </h3>
    <div id="${idMonthly}" class="chart-box" style="min-height:280px"></div>

    <h3 style="font-size:0.92rem;color:#cbd5e1;margin-top:22px">Regression summary</h3>
    <p style="font-size:0.76rem;color:#64748b;margin:0 0 8px">
      Yearly <strong>slope</strong> is ΔH per <em>calendar year</em>. Monthly slope is ΔH per <em>calendar day</em>
      (annualised in the last column).
    </p>
    <div style="overflow-x:auto">
      <table class="data-table">
        <thead>
          <tr>
            <th>Series</th>
            <th>Points</th>
            <th>Mean H</th>
            <th>Trend slope</th>
            <th>~%/yr proxy</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Yearly</strong></td>
            <td>${yearLabels.length}</td>
            <td>${Number.isFinite(meanYearly) ? meanYearly.toFixed(4) : "—"}</td>
            <td>${yearLabels.length >= MIN_POINTS_YEARLY ? fmtSlopeYear(yearlyReg.slope) : `— <span style="color:#64748b">(need ≥${MIN_POINTS_YEARLY} yrs)</span>`}</td>
            <td>${formatPctPerYear(pctY)}</td>
          </tr>
          <tr>
            <td>Monthly</td>
            <td>${monthDays.length}</td>
            <td>${Number.isFinite(meanMonthly) ? meanMonthly.toFixed(4) : "—"}</td>
            <td>${monthDays.length >= MIN_POINTS_MONTHLY ? fmtSlopeDay(monthlyReg.slope) : `— <span style="color:#64748b">(need ≥${MIN_POINTS_MONTHLY} mo)</span>`}</td>
            <td>${formatPctPerYear(pctM)}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p style="font-size:0.76rem;color:#64748b;margin-top:10px">
      Window: <strong>${dateFrom}</strong> → <strong>${dateTo}</strong> ·
      Negative slope / %/yr ⇒ H trending down (possible degradation); cross-check with O&amp;M and multi-year span.
    </p>
  `;

  if (!monthDays.length && !yearLabels.length) {
    container.innerHTML = `
      <h2>${siteHeading("Meter degradation", site)}</h2>
      <p style="color:#94a3b8;padding:2rem">No daylight rows with PVLib expected in this date range.</p>`;
    return;
  }

  runAfterTabLayout(container, () => {
    const theme = plotlyDarkTheme();

    const elY = document.getElementById(idYearly);
    if (elY) {
      if (!yearLabels.length) {
        elY.innerHTML =
          `<p style="color:#64748b;padding:1rem;font-size:0.85rem">No annual H (need daylight rows with Σ expected &gt; 0).</p>`;
      } else {
        const sz = measureChartBox(container, elY);
        const tracesY = [
          {
            x: yearNums,
            y: yearY,
            type: "scatter",
            mode: "lines+markers",
            name: "Annual H",
            line: { color: "#34d399", width: 2 },
            marker: { size: 11, color: "#34d399" },
            hovertemplate: "Year %{x}<br>H = %{y:.4f}<extra></extra>",
          },
        ];
        if (
          yearlyReg.fitY.length === yearY.length &&
          yearLabels.length >= MIN_POINTS_YEARLY
        ) {
          tracesY.push({
            x: yearNums,
            y: yearlyReg.fitY,
            type: "scatter",
            mode: "lines",
            name: `OLS fit · ${fmtSlopeYear(yearlyReg.slope)}`,
            line: { color: "#fb923c", width: 2.4, dash: "dash" },
          });
        }
        const annY =
          Number.isFinite(yearlyReg.slope) &&
          yearLabels.length >= MIN_POINTS_YEARLY
            ? [
                {
                  xref: "paper",
                  yref: "paper",
                  x: 0.02,
                  y: 0.98,
                  xanchor: "left",
                  yanchor: "top",
                  text: `<b>Annual trend</b> ~${formatPctPerYear(pctY)}`,
                  showarrow: false,
                  font: { color: "#e2e8f0", size: 11 },
                  bgcolor: "rgba(15,23,42,0.75)",
                  bordercolor: "#334155",
                  borderwidth: 1,
                  borderpad: 5,
                },
              ]
            : [];
        Plotly.newPlot(
          elY,
          tracesY,
          buildYearlyHChartLayout(
            theme,
            "Annual performance ratio H — regression vs calendar year",
            sz,
            annY,
          ),
          PLOTLY_STATIC,
        );
      }
    }

    const elM = document.getElementById(idMonthly);
    if (elM) {
      if (!monthDays.length) {
        elM.innerHTML =
          `<p style="color:#64748b;padding:1rem;font-size:0.85rem">No monthly H points in this range.</p>`;
      } else {
        const sz = measureChartBox(container, elM);
        const tracesM = [
          {
            x: monthDays,
            y: monthY,
            type: "scatter",
            mode: "lines+markers",
            name: "Monthly H",
            line: { color: "#a78bfa", width: 2 },
            marker: { size: 8, color: "#a78bfa" },
            hovertemplate: "%{x}<br>H = %{y:.4f}<extra></extra>",
          },
        ];
        if (
          monthlyReg.fitY.length === monthY.length &&
          monthDays.length >= MIN_POINTS_MONTHLY
        ) {
          tracesM.push({
            x: monthDays,
            y: monthlyReg.fitY,
            type: "scatter",
            mode: "lines",
            name: `Fit · ${fmtSlopeDay(monthlyReg.slope)}`,
            line: { color: "#fb923c", width: 2.4, dash: "dash" },
          });
        }
        const annM =
          Number.isFinite(monthlyReg.slope) &&
          monthDays.length >= MIN_POINTS_MONTHLY
            ? [
                {
                  xref: "paper",
                  yref: "paper",
                  x: 0.02,
                  y: 0.98,
                  xanchor: "left",
                  yanchor: "top",
                  text: `<b>Monthly</b> ~${formatPctPerYear(pctM)}`,
                  showarrow: false,
                  font: { color: "#e2e8f0", size: 11 },
                  bgcolor: "rgba(15,23,42,0.75)",
                  bordercolor: "#334155",
                  borderwidth: 1,
                  borderpad: 5,
                },
              ]
            : [];
        Plotly.newPlot(
          elM,
          tracesM,
          buildMonthlyHChartLayout(
            theme,
            "Monthly H — medium-term detail",
            "H",
            sz,
            annM,
            monthDays.length,
          ),
          PLOTLY_STATIC,
        );
      }
    }
  });
}
