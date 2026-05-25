/**
 * Meter list and data_for_viz filenames (aligned with Python config._BUILDING_PVLIB_GEOMETRY).
 */
import { fetchDataVizFile, tryFetchDataVizFile, parseCSV } from "./utils.js";

/** Fallback when sites_kpis_summary.csv is missing — keep in sync with config.py. */
export const DEFAULT_METERS = [
  { key: "library", label: "Library (L)" },
  { key: "dmw", label: "David Myers (DMW)" },
  { key: "dw", label: "Donald Whitehead (DW)" },
];

export function hourlyMasterFilename(meterKey) {
  return `hourly_${meterKey.trim().toLowerCase()}_master.csv`;
}

export function forecastCombinedFilename(meterKey) {
  return `forecast_7d_combined_${meterKey.trim().toLowerCase()}.csv`;
}

/**
 * Load meter catalog from sites_kpis_summary.csv or DEFAULT_METERS.
 * @returns {Promise<Array<{key: string, label: string}>>}
 */
export async function loadMeterCatalog() {
  const res = await tryFetchDataVizFile("sites_kpis_summary.csv");
  if (!res) return [...DEFAULT_METERS];
  try {
    const { rows } = parseCSV(res.text);
    const out = rows
      .map((r) => ({
        key: String(r.building_key ?? "").trim().toLowerCase(),
        label: String(r.building_name || r.building_key || "").trim(),
      }))
      .filter((m) => m.key && m.label);
    return out.length ? out : [...DEFAULT_METERS];
  } catch {
    return [...DEFAULT_METERS];
  }
}

export async function fetchHourlyMasterText(meterKey) {
  const key = meterKey.trim().toLowerCase();
  return fetchDataVizFile(hourlyMasterFilename(key));
}

export async function tryFetchForecastCombinedText(meterKey) {
  const key = meterKey.trim().toLowerCase();
  return tryFetchDataVizFile(forecastCombinedFilename(key));
}

/** Page heading with site name when provided. */
export function siteHeading(baseTitle, site) {
  if (site?.label) return `${baseTitle} — ${site.label}`;
  return baseTitle;
}
