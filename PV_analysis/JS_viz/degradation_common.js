/**
 * Shared performance-ratio (H) aggregation and degradation trend helpers.
 */
import { linearRegression, parseHour } from "./utils.js";

export const MIN_POINTS_MONTHLY = 4;
export const MIN_POINTS_YEARLY = 2;

/** Normalize hourly_master CSV rows for degradation / season tabs. */
export function normalizeHourlyRows(rows) {
  const out = [];
  for (const r of rows) {
    const ts = parseHour(r.timestamp);
    if (!ts) continue;
    const day = r.timestamp.trim().slice(0, 10);
    out.push({
      ts,
      tsStr: r.timestamp.trim(),
      day,
      actual: Number(r.actual_kwh),
      expected: Number(r.expected_kwh),
      ghi: Number(r.ghi_wm2),
    });
  }
  return out.sort((a, b) => a.ts - b.ts);
}

export function regressOnElapsedDays(datesIso, y) {
  const n = Math.min(datesIso.length, y.length);
  if (n < 2) return { slope: NaN, intercept: NaN, fitY: [] };
  const t0 = new Date(datesIso[0] + "T12:00:00").getTime();
  const x = datesIso.map((d) =>
    (new Date(d + "T12:00:00").getTime() - t0) / 86400000,
  );
  const reg = linearRegression(x.slice(0, n), y.slice(0, n));
  if (!Number.isFinite(reg.slope))
    return { slope: NaN, intercept: NaN, fitY: [] };
  const fitY = x.slice(0, n).map((xi) => reg.slope * xi + reg.intercept);
  return { slope: reg.slope, intercept: reg.intercept, fitY };
}

export function regressOnCalendarYears(yearStrings, y) {
  const n = Math.min(yearStrings.length, y.length);
  if (n < 2) return { slope: NaN, intercept: NaN, fitY: [] };
  const y0 = parseInt(yearStrings[0], 10);
  if (!Number.isFinite(y0)) return { slope: NaN, intercept: NaN, fitY: [] };
  const x = yearStrings.map((ys) => parseInt(ys, 10) - y0);
  const reg = linearRegression(x.slice(0, n), y.slice(0, n));
  if (!Number.isFinite(reg.slope))
    return { slope: NaN, intercept: NaN, fitY: [] };
  const fitY = x.slice(0, n).map((xi) => reg.slope * xi + reg.intercept);
  return { slope: reg.slope, intercept: reg.intercept, fitY };
}

export function yearlyPerformanceRatio(hourlyFiltered) {
  const byY = new Map();
  for (const r of hourlyFiltered) {
    if (!(r.ghi > 5)) continue;
    const key = r.day.slice(0, 4);
    let o = byY.get(key);
    if (!o) {
      o = { act: 0, exp: 0 };
      byY.set(key, o);
    }
    if (Number.isFinite(r.actual)) o.act += r.actual;
    if (Number.isFinite(r.expected)) o.exp += r.expected;
  }
  const years = [...byY.keys()].sort();
  const ratio = years.map((yy) => {
    const { act, exp } = byY.get(yy);
    return exp > 0 ? act / exp : NaN;
  });
  return { years, ratio };
}

export function monthlyPerformanceRatio(hourlyFiltered) {
  const byM = new Map();
  for (const r of hourlyFiltered) {
    if (!(r.ghi > 5)) continue;
    const key = r.day.slice(0, 7);
    let o = byM.get(key);
    if (!o) {
      o = { act: 0, exp: 0 };
      byM.set(key, o);
    }
    if (Number.isFinite(r.actual)) o.act += r.actual;
    if (Number.isFinite(r.expected)) o.exp += r.expected;
  }
  const months = [...byM.keys()].sort();
  const ratio = months.map((m) => {
    const { act, exp } = byM.get(m);
    return exp > 0 ? act / exp : NaN;
  });
  const monthStarts = months.map((m) => `${m}-01`);
  return { monthStarts, ratio };
}

export function pctPerYearFromDailySlope(slope, meanH) {
  if (!Number.isFinite(slope) || !Number.isFinite(meanH) || meanH < 1e-9)
    return NaN;
  return (slope * 365 * 100) / meanH;
}

export function pctPerYearFromAnnualSlope(slope, meanH) {
  if (!Number.isFinite(slope) || !Number.isFinite(meanH) || meanH < 1e-9)
    return NaN;
  return (slope * 100) / meanH;
}

/**
 * Degradation metrics for one meter over a date window.
 * @returns {{ pctYearly: number, pctMonthly: number, yearCount: number, monthCount: number, meanYearly: number, meanMonthly: number }}
 */
export function computeDegradationMetrics(hourly, dateFrom, dateTo) {
  const filtered = hourly.filter((r) => r.day >= dateFrom && r.day <= dateTo);

  const { years: yearStrs, ratio: yearlyHRaw } = yearlyPerformanceRatio(filtered);
  const yearlyClean = yearStrs
    .map((yy, i) => ({ year: yy, h: yearlyHRaw[i] }))
    .filter((o) => Number.isFinite(o.h));
  const yearLabels = yearlyClean.map((o) => o.year);
  const yearY = yearlyClean.map((o) => o.h);

  const { monthStarts, ratio: monthlyH } = monthlyPerformanceRatio(filtered);
  const monthlyClean = monthStarts
    .map((d, i) => ({ day: d, h: monthlyH[i] }))
    .filter((o) => Number.isFinite(o.h));
  const monthDays = monthlyClean.map((o) => o.day);
  const monthY = monthlyClean.map((o) => o.h);

  const yearlyReg =
    yearLabels.length >= MIN_POINTS_YEARLY
      ? regressOnCalendarYears(yearLabels, yearY)
      : { slope: NaN };
  const monthlyReg =
    monthDays.length >= MIN_POINTS_MONTHLY
      ? regressOnElapsedDays(monthDays, monthY)
      : { slope: NaN };

  const meanYearly = yearY.length
    ? yearY.reduce((a, b) => a + b, 0) / yearY.length
    : NaN;
  const meanMonthly = monthY.length
    ? monthY.reduce((a, b) => a + b, 0) / monthY.length
    : NaN;

  return {
    pctYearly: pctPerYearFromAnnualSlope(yearlyReg.slope, meanYearly),
    pctMonthly: pctPerYearFromDailySlope(monthlyReg.slope, meanMonthly),
    yearCount: yearLabels.length,
    monthCount: monthDays.length,
    meanYearly,
    meanMonthly,
  };
}

export function formatPctPerYear(p) {
  if (!Number.isFinite(p)) return "—";
  return `${p >= 0 ? "+" : ""}${p.toFixed(2)} %/yr`;
}
