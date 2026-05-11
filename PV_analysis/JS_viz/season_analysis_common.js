/**
 * Shared Australian meteorological seasons + daily aggregates for season tabs.
 */

export const SEASONS = [
  { id: "sum", label: "Summer (Dec–Feb)", color: "#f59e0b", midM: 1 },
  { id: "aut", label: "Autumn (Mar–May)", color: "#22c55e", midM: 4 },
  { id: "win", label: "Winter (Jun–Aug)", color: "#38bdf8", midM: 7 },
  { id: "spr", label: "Spring (Sep–Nov)", color: "#a78bfa", midM: 10 },
];

export function seasonOf(dayStr) {
  const yr = +dayStr.slice(0, 4);
  const mo = +dayStr.slice(5, 7);
  if (mo === 12 || mo === 1 || mo === 2)
    return { id: "sum", yr: mo === 12 ? yr + 1 : yr };
  if (mo <= 5) return { id: "aut", yr };
  if (mo <= 8) return { id: "win", yr };
  return { id: "spr", yr };
}

export function computeDailyH(hourly) {
  const byDay = new Map();
  for (const r of hourly) {
    if (!(r.ghi > 5)) continue;
    let o = byDay.get(r.day);
    if (!o) {
      o = { act: 0, exp: 0 };
      byDay.set(r.day, o);
    }
    if (Number.isFinite(r.actual)) o.act += r.actual;
    if (Number.isFinite(r.expected)) o.exp += r.expected;
  }
  const out = [];
  for (const [day, { act, exp }] of byDay)
    if (exp > 0) out.push({ day, value: act / exp });
  return out.sort((a, b) => (a.day < b.day ? -1 : 1));
}

/** Sum hourly GHI (W·m⁻²) for daylight rows (GHI > 5); convert to daily energy kWh·m⁻² (1 h steps). */
export function computeDailyGhiKwh(hourly) {
  const byDay = new Map();
  for (const r of hourly) {
    if (!(r.ghi > 5)) continue;
    const g = Number(r.ghi);
    if (!Number.isFinite(g)) continue;
    byDay.set(r.day, (byDay.get(r.day) ?? 0) + g);
  }
  const out = [];
  for (const [day, sumWhPerM2] of byDay)
    out.push({ day, value: sumWhPerM2 / 1000 });
  return out.sort((a, b) => (a.day < b.day ? -1 : 1));
}

export function groupSeasons(daily) {
  const map = new Map(SEASONS.map((s) => [s.id, []]));
  for (const row of daily) {
    const info = seasonOf(row.day);
    map.get(info.id)?.push({ day: row.day, yr: info.yr, value: row.value });
  }
  return map;
}

export function medianOf(arr) {
  if (!arr.length) return NaN;
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function yearlyMedians(pts) {
  const m = new Map();
  for (const { value, yr } of pts) {
    if (!m.has(yr)) m.set(yr, []);
    m.get(yr).push(value);
  }
  const out = [];
  for (const [yr, vals] of m) {
    const med = medianOf(vals);
    out.push({ yr, med, n: vals.length });
  }
  return out.sort((a, b) => a.yr - b.yr);
}

/** Convert #RRGGBB to rgba(); falls back to slate if invalid. */
export function hexToRgba(hex, alpha = 1) {
  const n = hex.replace("#", "").trim();
  if (n.length !== 6 || !/^[0-9a-fA-F]{6}$/.test(n))
    return `rgba(148,163,184,${alpha})`;
  const r = parseInt(n.slice(0, 2), 16);
  const g = parseInt(n.slice(2, 4), 16);
  const b = parseInt(n.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

/** Light frame around each season block (tabs 6 & 7). */
export function seasonCellFrameStyle(hex) {
  return [
    "min-width:0",
    "box-sizing:border-box",
    "padding:12px 14px",
    "border-radius:12px",
    `border:1px solid ${hexToRgba(hex, 0.38)}`,
    `background:${hexToRgba(hex, 0.075)}`,
    "box-shadow:inset 0 1px 0 rgba(255,255,255,0.04)",
  ].join(";");
}

/**
 * Horizontal padding (each side) on season cards — keep in sync with `seasonCellFrameStyle`.
 * @type {number}
 */
export const SEASON_CELL_PAD_X = 14;

/** 1px border on left + right inside the framed cell. */
export const SEASON_CELL_BORDER_X = 2;

/**
 * Plotly layout width for one chart in the 2×2 season grid (tabs 6 & 7).
 * Subtracts outer panel padding, grid gap, cell padding, and border so plots stay inside the frame.
 *
 * @param {number} panelClientWidth  typically `container.clientWidth`
 * @param {object} [opt]
 * @param {number} [opt.padOuter=32]   outer content padding (16px × 2)
 * @param {number} [opt.gridGap=18]    gap between the two columns
 * @param {number} [opt.minW=200]      floor width
 */
export function seasonChartPlotWidth(panelClientWidth, opt = {}) {
  const padOuter = opt.padOuter ?? 32;
  const gridGap = opt.gridGap ?? 18;
  const minW = opt.minW ?? 200;
  const cw = Math.max(480, panelClientWidth || 960);
  const inner = cw - padOuter;
  const colW = Math.floor((inner - gridGap) / 2);
  const cellInset = SEASON_CELL_PAD_X * 2 + SEASON_CELL_BORDER_X;
  return Math.max(minW, colW - cellInset);
}

/** Blend hex colour toward white for lighter irradiance traces (matches season hue). */
export function lightenSeasonColor(hex, t = 0.5) {
  const n = hex.replace("#", "");
  if (n.length !== 6) return hex;
  const r = parseInt(n.slice(0, 2), 16);
  const g = parseInt(n.slice(2, 4), 16);
  const b = parseInt(n.slice(4, 6), 16);
  const R = Math.round(r + (255 - r) * t);
  const G = Math.round(g + (255 - g) * t);
  const B = Math.round(b + (255 - b) * t);
  return `rgb(${R},${G},${B})`;
}
