/**
 * Linear x-axis season-year charts (daily traces per year + medians + trend).
 */
import { plotlyDarkTheme, linearRegression } from "./utils.js";
import { yearlyMedians, medianOf } from "./season_analysis_common.js";

const YEAR_CENTER_STEP = 46;
const YEAR_CENTER_FIRST = 61;
const SEASON_SPAN = 40;
const X_RANGE_PAD = 23;

/**
 * @param {{ id: string, label: string, color: string }} s  season meta
 * @param {{ day: string, yr: number, value: number }[]} pts
 * @param {object} opts
 * @param {string} opts.yTitle
 * @param {[number, number] | undefined} opts.yRange  omit for autorange
 * @param {string} opts.dailyLineColor
 * @param {number} [opts.dailyOpacity]
 * @param {string} opts.seasonMedianLineColor
 * @param {string} opts.seasonMedianLabel  legend + hover
 * @param {{ y: number, color: string, dash: string }[]} [opts.referencePaperLines]
 * @param {string} opts.hoverValueLabel  e.g. "H" or "GHI"
 * @param {number} [opts.hoverDecimals]
 * @param {boolean} [opts.showLegendGhost]
 * @param {string} [opts.ghostLegendName]
 * @param {string} [opts.chartTitle]  override subplot title (default s.label)
 */
export function buildSeasonLinearPlot(s, pts, w, h, opts) {
  const {
    chartTitle,
    yTitle,
    yRange,
    dailyLineColor,
    dailyOpacity = 0.85,
    seasonMedianLineColor,
    seasonMedianLabel,
    referencePaperLines = [],
    hoverValueLabel,
    hoverDecimals = 3,
    showLegendGhost = true,
    ghostLegendName = "Daily",
  } = opts;

  const vals = pts.map((p) => p.value);
  const ymeds = yearlyMedians(pts);
  const overall = medianOf(vals.filter(Number.isFinite));
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
  const xRangeAxis = [
    firstC - SEASON_SPAN / 2 - X_RANGE_PAD,
    lastC + SEASON_SPAN / 2 + X_RANGE_PAD,
  ];

  let slopeYear = NaN,
    interceptYear = NaN,
    degradPct = NaN;
  if (ymeds.length >= 2) {
    const xsYr = ymeds.map((d) => d.yr);
    const ysMed = ymeds.map((d) => d.med);
    ({ slope: slopeYear, intercept: interceptYear } = linearRegression(
      xsYr,
      ysMed,
    ));
    if (
      Number.isFinite(slopeYear) &&
      Number.isFinite(interceptYear) &&
      ymeds[0].med > 0
    )
      degradPct = (slopeYear / ymeds[0].med) * 100;
  }

  const traces = [];
  if (showLegendGhost) {
    traces.push({
      x: [null],
      y: [null],
      type: "scatter",
      mode: "markers",
      name: ghostLegendName,
      marker: { color: dailyLineColor, size: 6, opacity: dailyOpacity },
      showlegend: true,
      hoverinfo: "skip",
    });
  }

  const ht =
    hoverDecimals === 3
      ? `%{text}<br>${hoverValueLabel} = %{y:.3f}<extra></extra>`
      : `%{text}<br>${hoverValueLabel} = %{y:.2f}<extra></extra>`;

  for (const yr of sortedYears) {
    const yearPts = [...byYear.get(yr)].sort((a, b) =>
      a.day.localeCompare(b.day),
    );
    const totalDaysInThatYear = Math.max(yearPts.length, 1);
    const xc = yearCenterX[yr];
    const xs = yearPts.map(
      (_, dayOfSeason) =>
        xc -
        SEASON_SPAN / 2 +
        (dayOfSeason / totalDaysInThatYear) * SEASON_SPAN,
    );
    const ys = yearPts.map((p) => p.value);
    const text = yearPts.map((p) => p.day);
    traces.push({
      x: xs,
      y: ys,
      type: "scatter",
      mode: "lines+markers",
      name: `${yr}`,
      line: { color: dailyLineColor, width: 1.2 },
      marker: {
        color: dailyLineColor,
        size: 5,
        opacity: dailyOpacity,
        line: { width: 0 },
      },
      opacity: dailyOpacity,
      showlegend: false,
      hovertemplate: ht,
      text,
    });
  }

  if (Number.isFinite(overall)) {
    traces.push({
      x: [xRangeAxis[0], xRangeAxis[1]],
      y: [overall, overall],
      type: "scatter",
      mode: "lines",
      name: seasonMedianLabel,
      line: { color: seasonMedianLineColor, width: 1.5, dash: "dot" },
      hovertemplate: `${seasonMedianLabel} = ${overall.toFixed(hoverDecimals)}<extra></extra>`,
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
      text: ymeds.map((d) => `${d.yr}: ${hoverValueLabel}=${d.med.toFixed(hoverDecimals)}`),
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

  const shapes = [];
  for (const ref of referencePaperLines) {
    shapes.push({
      type: "line",
      xref: "paper",
      yref: "y",
      x0: 0,
      x1: 1,
      y0: ref.y,
      y1: ref.y,
      line: { color: ref.color, width: 1, dash: ref.dash },
    });
  }
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
      text: chartTitle ?? s.label,
      font: { color: s.color, size: 13 },
      x: 0,
      xanchor: "left",
      y: 0.98,
      yanchor: "top",
    },
    xaxis: {
      ...theme.xaxis,
      type: "linear",
      autorange: false,
      range: xRangeAxis,
      tickmode: "array",
      tickvals,
      ticktext,
      showgrid: false,
      gridcolor: "#1e293b",
      linecolor: "#334155",
    },
    yaxis: {
      ...theme.yaxis,
      title: yTitle,
      ...(yRange
        ? { range: yRange, autorange: false }
        : { autorange: true }),
    },
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
    margin: { t: 30, r: 12, b: 36, l: 48 },
    hovermode: "closest",
    showlegend: true,
    autosize: false,
    width: w,
    height: h,
  };

  return { traces, layout, overall, degradPct, ymeds, dayCount };
}
