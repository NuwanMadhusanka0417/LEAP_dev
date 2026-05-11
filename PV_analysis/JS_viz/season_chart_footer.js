/** Shared HTML footer under season linear charts (median, degradation badge, detail line). */

export function formatTrendPct(pct) {
  if (!Number.isFinite(pct)) return "N/A";
  const abs = Math.abs(pct).toFixed(2);
  return pct < 0 ? `\u2212${abs}` : `+${abs}`;
}

/**
 * @param {object} opt
 * @param {string} [opt.medianLabel]
 * @param {number} [opt.medianDecimals]
 * @param {string} [opt.medianSuffix]
 * @param {number} [opt.detailMedDecimals]
 */
export function seasonChartFooter(res, opt = {}) {
  const {
    medianLabel = "Median H",
    medianDecimals = 3,
    medianSuffix = "",
    detailMedDecimals = 2,
  } = opt;
  const { overall, degradPct, dayCount, ymeds } = res;
  const medStr = Number.isFinite(overall)
    ? overall.toFixed(medianDecimals) + medianSuffix
    : "N/A";

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
      ? ymeds
          .map((d) => `${d.yr}: ${d.med.toFixed(detailMedDecimals)} (n=${d.n})`)
          .join(" · ")
      : "";

  return `<div style="margin-top:2px">
    <div style="display:flex;gap:12px 14px;align-items:center;flex-wrap:wrap;font-size:12px;
                color:#94a3b8;line-height:1.35">
      <span>${medianLabel}: <strong style="color:#e2e8f0">${medStr}</strong></span>
      <span style="${badgeStyle};font-size:11px;padding:2px 8px;border-radius:4px;
                  font-weight:500;">${trendStr}</span>
      <span>${dayCount} days</span>
    </div>
    ${detail ? `<div style="margin-top:4px;padding:6px 8px;background:#1e293b;
      border:1px solid #334155;border-radius:6px;font-size:11px;color:#94a3b8;
      line-height:1.4">${detail}</div>` : ""}
  </div>`;
}
