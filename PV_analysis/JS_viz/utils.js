/**
 * CSV (comma-separated, no embedded commas in fields), date helpers, stats.
 */

let _plotDomSeq = 0;

/** Fresh DOM id each render so Plotly never reuses a stale graph div. */
export function nextPlotDomId(prefix) {
  return `${prefix}--${_plotDomSeq++}`;
}

/** Call before replacing panel innerHTML so Plotly releases listeners and WebGL state. */
export function purgePlotlyInContainer(container) {
  if (!container || typeof Plotly === "undefined") return;
  container.querySelectorAll(".js-plotly-plot").forEach((gd) => {
    try {
      Plotly.purge(gd);
    } catch (_) {
      /* ignore */
    }
  });
}

/** Purge every Plotly graph under a root (e.g. main). Stops hidden-tab graphs from reacting to resize/layout. */
export function purgeAllPlotlyInRoot(root) {
  if (!root || typeof Plotly === "undefined") return;
  root.querySelectorAll(".js-plotly-plot").forEach((gd) => {
    try {
      Plotly.purge(gd);
    } catch (_) {
      /* ignore */
    }
  });
}

/**
 * Run fn after the next two animation frames (layout stable).
 * `panelEl` must be the tab panel element (e.g. #panel-difference). Each new render
 * bumps a generation counter so an older scheduled callback is skipped — otherwise
 * Plotly can run on replaced DOM and corrupt axes/traces after switching tabs.
 */
export function runAfterTabLayout(panelEl, fn, opts = {}) {
  if (!panelEl || typeof fn !== "function") return;
  const gen = (panelEl.__plotLayoutGen = (panelEl.__plotLayoutGen || 0) + 1);
  const requireActive = opts.requireActive !== false;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      if (panelEl.__plotLayoutGen !== gen) return;
      if (!panelEl.isConnected) return;
      if (requireActive && !panelEl.classList.contains("active")) return;
      fn();
    });
  });
}

export function measureChartBox(panelRoot, el, fallbackW = 960, fallbackH = 400) {
  const box = el?.closest?.(".chart-box");
  const h =
    box && box.clientHeight > 0 ? box.clientHeight : fallbackH;
  const w = Math.max(
    panelRoot?.clientWidth || 0,
    el?.clientWidth || 0,
    box?.clientWidth || 0,
    fallbackW
  );
  return { width: Math.floor(w), height: Math.floor(h) };
}

/** Plotly responsive mode breaks with tab panels (display:none); use explicit sizes. */
export const PLOTLY_STATIC = { responsive: false, displaylogo: false };

/**
 * Candidate base URLs for data_for_viz (tried in order until a file loads).
 * Uses the /JS_viz/ segment in the path so /PV_analysis/JS_viz/ → /PV_analysis/data_for_viz/
 * even when ../ resolution would otherwise fail (missing trailing slash, etc.).
 *
 * Override: ?data=/PV_analysis/data_for_viz/
 */
export function getCandidateDataBases() {
  const bases = [];
  const seen = new Set();
  const add = (u) => {
    const key = u.href.replace(/\/+$/, "");
    if (seen.has(key)) return;
    seen.add(key);
    bases.push(u);
  };

  const qs = new URLSearchParams(window.location.search).get("data");
  if (qs?.trim()) {
    const p = qs.trim().endsWith("/") ? qs.trim() : `${qs.trim()}/`;
    add(new URL(p, window.location.origin));
  }

  const path = window.location.pathname;
  // Match /PV_analysis/JS_viz/... or /JS_viz (with or without trailing slash)
  const m = path.match(/^(.*?)\/JS_viz(?:\/|$)/);
  if (m) {
    const prefix = m[1];
    const pathPart = `${prefix}/data_for_viz/`.replace(/\/{2,}/g, "/");
    add(new URL(pathPart, window.location.origin));
  }

  add(new URL("../data_for_viz/", window.location.href));
  return bases;
}

/** @returns {{ text: string, url: string }} */
export async function fetchDataVizFile(relPath, { cacheBust = false } = {}) {
  const name = relPath.replace(/^\//, "");
  const bases = getCandidateDataBases();
  const tried = [];
  for (const base of bases) {
    const url = new URL(name, base);
    if (cacheBust) {
      url.searchParams.set("_", String(Date.now()));
    }
    try {
      const resp = await fetch(url.href, cacheBust ? { cache: "no-store" } : {});
      if (resp.ok) return { text: await resp.text(), url: url.href };
      tried.push(`${url.href} (${resp.status})`);
    } catch (e) {
      tried.push(`${url.href} (${e.message})`);
    }
  }
  throw new Error(
    `Could not load ${name}. Tried: ${tried.join("; ")}. ` +
      `Use a URL containing /PV_analysis/JS_viz/ or set ?data=/PV_analysis/data_for_viz/`
  );
}

export async function tryFetchDataVizFile(relPath) {
  try {
    return await fetchDataVizFile(relPath);
  } catch {
    return null;
  }
}

/**
 * Bases for PV_analysis/data_pvlib (precomputed hourly PVLib), parallel to getCandidateDataBases().
 */
export function getCandidatePvlibBases() {
  const bases = [];
  const seen = new Set();
  const add = (u) => {
    const key = u.href.replace(/\/+$/, "");
    if (seen.has(key)) return;
    seen.add(key);
    bases.push(u);
  };
  const path = window.location.pathname;
  const m = path.match(/^(.*?)\/JS_viz(?:\/|$)/);
  if (m) {
    const prefix = m[1] || "";
    const pathPart = `${prefix}/data_pvlib/`.replace(/\/{2,}/g, "/");
    add(new URL(pathPart, window.location.origin));
  }
  add(new URL("../data_pvlib/", window.location.href));
  return bases;
}

/** Try to load a file from data_pvlib (e.g. expected_power_pvlib_cleaned_v2.csv). */
export async function tryFetchPvlibFile(relPath) {
  const name = relPath.replace(/^\//, "");
  for (const base of getCandidatePvlibBases()) {
    const url = new URL(name, base);
    try {
      const resp = await fetch(url.href);
      if (resp.ok) return { text: await resp.text(), url: url.href };
    } catch {
      /* try next base */
    }
  }
  return null;
}

/** @deprecated use getCandidateDataBases()[0] or fetchDataVizFile */
export function getDataBase() {
  const bases = getCandidateDataBases();
  return bases[0] || new URL("../data_for_viz/", window.location.href);
}

export function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (!lines.length) return { headers: [], rows: [] };
  const headers = lines[0].split(",").map((h) => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    const parts = lines[i].split(",");
    if (parts.length < headers.length) continue;
    const o = {};
    headers.forEach((h, j) => {
      o[h] = (parts[j] ?? "").trim();
    });
    rows.push(o);
  }
  return { headers, rows };
}

export function parseHour(ts) {
  const d = new Date(ts.replace(" ", "T"));
  return Number.isNaN(d.getTime()) ? null : d;
}

/** Inclusive start, inclusive end (date string yyyy-mm-dd). */
export function inDateRange(isoDay, fromStr, toStr) {
  return isoDay >= fromStr && isoDay <= toStr;
}

export function linearRegression(x, y) {
  const n = Math.min(x.length, y.length);
  if (n < 2) return { slope: NaN, intercept: NaN };
  let sx = 0,
    sy = 0,
    sxy = 0,
    sxx = 0;
  for (let i = 0; i < n; i++) {
    sx += x[i];
    sy += y[i];
    sxy += x[i] * y[i];
    sxx += x[i] * x[i];
  }
  const den = n * sxx - sx * sx;
  if (Math.abs(den) < 1e-12) return { slope: NaN, intercept: NaN };
  const slope = (n * sxy - sx * sy) / den;
  const intercept = (sy - slope * sx) / n;
  return { slope, intercept };
}

export function rollingMedian(arr, window, minPeriods = 3) {
  const out = new Array(arr.length).fill(NaN);
  const half = Math.floor(window / 2);
  for (let i = 0; i < arr.length; i++) {
    const lo = Math.max(0, i - half);
    const hi = Math.min(arr.length, i + half + 1);
    const slice = [];
    for (let j = lo; j < hi; j++) if (Number.isFinite(arr[j])) slice.push(arr[j]);
    if (slice.length < minPeriods) continue;
    slice.sort((a, b) => a - b);
    const m = Math.floor(slice.length / 2);
    out[i] =
      slice.length % 2 ? slice[m] : (slice[m - 1] + slice[m]) / 2;
  }
  return out;
}

/** 7-day trailing median aligned to end of window (like pandas rolling(7).median()). */
export function rollingMedian7Trailing(values, minPeriods = 3) {
  const n = values.length;
  const out = new Array(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    const start = Math.max(0, i - 6);
    const slice = [];
    for (let j = start; j <= i; j++) if (Number.isFinite(values[j])) slice.push(values[j]);
    if (slice.length < minPeriods) continue;
    slice.sort((a, b) => a - b);
    const m = Math.floor(slice.length / 2);
    out[i] =
      slice.length % 2 ? slice[m] : (slice[m - 1] + slice[m]) / 2;
  }
  return out;
}

/** 7-point centered rolling median (pandas rolling(7, center=True).median()). */
export function rollingMedian7Centered(values, minPeriods = 1) {
  const n = values.length;
  const out = new Array(n).fill(NaN);
  const half = 3;
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - half);
    const hi = Math.min(n, i + half + 1);
    const slice = [];
    for (let j = lo; j < hi; j++) if (Number.isFinite(values[j])) slice.push(values[j]);
    if (slice.length < minPeriods) continue;
    slice.sort((a, b) => a - b);
    const m = Math.floor(slice.length / 2);
    out[i] =
      slice.length % 2 ? slice[m] : (slice[m - 1] + slice[m]) / 2;
  }
  return out;
}

export function nanPercentile(arr, p) {
  const v = arr.filter(Number.isFinite);
  if (!v.length) return NaN;
  v.sort((a, b) => a - b);
  if (v.length === 1) return v[0];
  const x = (p / 100) * (v.length - 1);
  const lo = Math.floor(x);
  const hi = Math.ceil(x);
  if (lo === hi) return v[lo];
  return v[lo] + (v[hi] - v[lo]) * (x - lo);
}

/**
 * Return a new theme object for every Plotly.newPlot/react call.
 * Plotly mutates layout (including axis objects) in place; sharing `PLOTLY_DARK.yaxis`
 * across tabs corrupts axes after some Plotly layouts mutate nested axis objects.
 */
export function plotlyDarkTheme() {
  return {
    paper_bgcolor: "#0f172a",
    plot_bgcolor: "#0f172a",
    font: { color: "#e2e8f0" },
    xaxis: { gridcolor: "#1e293b", linecolor: "#334155" },
    yaxis: { gridcolor: "#1e293b", linecolor: "#334155" },
  };
}
