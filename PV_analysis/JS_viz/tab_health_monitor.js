/**
 * Health monitor — EWMA control chart on seasonal H residuals (predictive maintenance).
 */
import {
  plotlyDarkTheme,
  nextPlotDomId,
  purgePlotlyInContainer,
  PLOTLY_STATIC,
  linearRegression,
  rollingMedian,
  parseCSV,
  tryFetchCleanedFile,
} from "./utils.js";
import { siteHeading, fetchHourlyMasterText } from "./meters.js";
import { seasonOf } from "./season_analysis_common.js";

// ── Tunable constants ───────────────────────────────────────────────────────
const REFERENCE_MONTHS = 24;
const EWMA_LAMBDA = 0.1;
const EWMA_L = 3;
const RAIN_CLEAN_THRESHOLD = 0.5; // mm/day — rain cleaning threshold (EWMA + soiling)
const RECOVERY_WINDOW_DAYS = 14;
const ALARM_BRIDGE_DAYS = 3;
const STATUS_LOOKBACK_DAYS = 30;
// CSV column names (hourly master may use temp_cell_c instead of temp_c)
const RAIN_FIELD = "rain_mm";
const TEMP_FIELD = "temp_c";
const WIND_FIELD = "wind_ms"; // wind speed (m/s)

// Environmental soiling analysis
const WIND_CLEAN_THRESHOLD = 6; // m/s daily max — wind cleaning candidate
const CLEAN_JUMP_THRESHOLD = 0.08; // min upward jump in smoothed H to count as recovery
const SOILING_MIN_SEGMENT_DAYS = 5; // min dry-spell length to fit a soiling slope
const SMOOTH_WINDOW_DAYS = 3; // centered smoothing window for H before jump detection
const SIGNIFICANT_JUMP_THRESHOLD = 0.12; // min |ΔH| shown on charts / key-events table
const MIN_DAYS_BETWEEN_EVENTS = 7; // debounce nearby recovery markers
const EVENT_POPUP_RADIUS_DAYS = 3; // ±3 days around marker → 7-day window in hover popup
const EVENT_POPUP_CHART_WIDTH = 340;
const MAX_CHART_RECOVERY_MARKERS = 35;
const MAX_SOILING_LINES_ON_CHART = 8;

const CLASS = {
  unclassified: {
    label: "Unclassified (no weather data)",
    color: "#94a3b8",
    action: "Review manually — weather not available",
  },
  soiling: {
    label: "Soiling (rain-recovered)",
    color: "#3b82f6",
    action: "No action needed",
  },
  thermal: {
    label: "Thermal (heat-related)",
    color: "#fbbf24",
    action: "Expected — monitor",
  },
  fault: {
    label: "Candidate fault",
    color: "#f87171",
    action: "Inspect system",
  },
};

const CLASS_PILL = {
  soiling: { bg: "#1e3a5f", fg: "#93c5fd" },
  thermal: { bg: "#422006", fg: "#fbbf24" },
  fault: { bg: "#450a0a", fg: "#f87171" },
  unclassified: { bg: "#1e293b", fg: "#94a3b8" },
};

function classPillHtml(kind, label) {
  const s = CLASS_PILL[kind] || CLASS_PILL.unclassified;
  return `<span style="display:inline-block;font-size:11px;padding:2px 8px;border-radius:4px;font-weight:500;background:${s.bg};color:${s.fg}">${label}</span>`;
}

function classificationLegendHtml() {
  const entries = [
    {
      kind: "soiling",
      pill: "Soiling (rain-recovered)",
      text: "— dust/pollen buildup; cleared by rain. No action.",
    },
    {
      kind: "thermal",
      pill: "Thermal (heat-related)",
      text: "— hot-weather efficiency loss. Recovers when cool.",
    },
    {
      kind: "fault",
      pill: "Candidate fault",
      text: "— unexplained, no recovery. Inspect system.",
    },
    {
      kind: "unclassified",
      pill: "Unclassified",
      text: "— no weather data to classify.",
    },
  ];
  return `<div style="display:flex;flex-wrap:wrap;gap:16px;font-size:12px;color:#94a3b8;margin:0 0 10px">
    ${entries
      .map(
        (e) =>
          `<span style="display:inline-flex;align-items:center;gap:6px">${classPillHtml(e.kind, e.pill)} ${e.text}</span>`,
      )
      .join("")}
  </div>`;
}

function howItWorksHtml() {
  return `<details style="margin:12px 0;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:0 14px">
    <summary style="cursor:pointer;padding:12px 0;font-size:0.9rem;color:#cbd5e1;font-weight:500">How this works — chart, alarms &amp; classification</summary>
    <div style="padding:0 0 14px 0;font-size:0.82rem;color:#94a3b8;line-height:1.6">
      <p style="margin:0 0 4px;font-size:0.78rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em">Chart elements</p>
      <p style="margin:0 0 10px"><strong>Y-axis — Residual (H − seasonal baseline)</strong><br>
        H is the daily health ratio: total actual generation divided by total PVLib-expected
        generation over daylight hours (GHI &gt; 5 W/m²). The seasonal baseline is the median H
        for that season during the healthy reference window (first ${REFERENCE_MONTHS} months).
        Residual = today's H minus that baseline. Zero means the system matches its healthy
        seasonal norm; negative means it is underperforming.</p>
      <hr style="border:none;border-top:1px solid #334155;margin:10px 0">
      <p style="margin:0 0 10px"><strong>X-axis — Date</strong><br>
        One point per day, in calendar order across the full dataset.</p>
      <hr style="border:none;border-top:1px solid #334155;margin:10px 0">
      <p style="margin:0 0 10px"><strong>Gray dots — Daily residual</strong><br>
        The raw daily residual. Noisy by nature, so individual days are not alarmed on.</p>
      <hr style="border:none;border-top:1px solid #334155;margin:10px 0">
      <p style="margin:0 0 10px"><strong>Blue line — EWMA</strong><br>
        Exponentially-weighted moving average of the residual:
        EWMA(today) = λ·residual(today) + (1−λ)·EWMA(yesterday), with λ = ${EWMA_LAMBDA}.
        It smooths daily noise but bends down when a drop is sustained, which is what a real
        fault looks like.</p>
      <hr style="border:none;border-top:1px solid #334155;margin:10px 0">
      <p style="margin:0 0 10px"><strong>Gray dotted line — Center (μ)</strong><br>
        Mean residual during the reference window. The EWMA sits near here when healthy.</p>
      <hr style="border:none;border-top:1px solid #334155;margin:10px 0">
      <p style="margin:0 0 10px"><strong>Red dashed line — Lower control limit (LCL)</strong><br>
        LCL = μ − L·σ·√(λ / (2−λ)), with L = ${EWMA_L} and σ = standard deviation of residuals
        in the reference window. When the EWMA drops below this line the underperformance is
        too large and too sustained to be normal variation — an alarm.</p>
      <hr style="border:none;border-top:1px solid #334155;margin:10px 0">
      <p style="margin:0 0 10px"><strong>Red shaded bands — Alarm events</strong><br>
        Periods where the EWMA stayed below the LCL. Each band is one alarm event in the log
        below. Only downward breaches are flagged; rises above the baseline are not a maintenance concern.</p>

      <hr style="border:none;border-top:1px solid #475569;margin:16px 0">
      <p style="margin:0 0 4px;font-size:0.78rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em">How alarms are captured</p>
      <p style="margin:0 0 10px">
        <strong>Step 1 — Daily health signal.</strong> Hourly meter data is rolled up to one row
        per day (daylight hours only, GHI &gt; 5 W/m²): actual kWh, expected kWh, daily H, and
        optional daily rainfall total and mean temperature (from <code>${RAIN_FIELD}</code> /
        <code>${TEMP_FIELD}</code> or equivalent columns in the hourly CSV).
      </p>
      <p style="margin:0 0 10px">
        <strong>Step 2 — Seasonal baseline &amp; residual.</strong> For each day, subtract the
        median H for that calendar season from the first ${REFERENCE_MONTHS} months of data.
        That residual removes normal summer/winter swing so we only watch deviation from the
        system's own healthy seasonal norm.
      </p>
      <p style="margin:0 0 10px">
        <strong>Step 3 — EWMA alarm days.</strong> Any day whose EWMA (λ = ${EWMA_LAMBDA}) falls
        below the LCL is marked as an alarm day. Single noisy days rarely trigger this; a
        sustained drop does.
      </p>
      <p style="margin:0 0 10px">
        <strong>Step 4 — Group into events.</strong> Consecutive alarm days are merged into one
        event. Gaps of up to ${ALARM_BRIDGE_DAYS} days without an alarm are bridged so brief
        recoveries do not split one real incident into many rows.
      </p>
      <p style="margin:0 0 10px">
        <strong>Step 5 — Classify each event.</strong> After the event ends, a
        ${RECOVERY_WINDOW_DAYS}-day recovery window is checked. Rainfall and temperature in that
        window (plus the event itself) decide whether the drop is soiling, thermal, a candidate
        fault, or unclassified. Results appear in the alarm log with start/end dates, duration,
        minimum EWMA during the event, classification pill, and recommended action.
      </p>

      <hr style="border:none;border-top:1px solid #475569;margin:16px 0">
      <p style="margin:0 0 10px;font-size:0.78rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em">Alarm types — detection, data used &amp; what to do</p>

      <div style="margin:0 0 12px;padding:10px 12px;background:#0f172a;border-radius:6px;border:1px solid #334155">
        <div style="margin-bottom:6px">${classPillHtml("soiling", CLASS.soiling.label)}</div>
        <p style="margin:0 0 6px"><strong>How detected:</strong> EWMA stayed below LCL during
        the event, then returned to normal (EWMA ≥ LCL) within ${RECOVERY_WINDOW_DAYS} days
        <em>after</em> the event ended, and at least one day in the event + recovery window had
        daily rainfall ≥ ${RAIN_CLEAN_THRESHOLD} mm (typical panel wash-off).</p>
        <p style="margin:0 0 6px"><strong>Data used:</strong> Daily H residual &amp; EWMA; daily
        rainfall total (<code>${RAIN_FIELD}</code>); recovery timing vs LCL.</p>
        <p style="margin:0"><strong>What to do:</strong> ${CLASS.soiling.action}. Dust or pollen
        buildup that rain cleared — not a hardware fault. Log for records; no O&amp;M dispatch
        unless the pattern repeats without rain recovery.</p>
      </div>

      <div style="margin:0 0 12px;padding:10px 12px;background:#0f172a;border-radius:6px;border:1px solid #334155">
        <div style="margin-bottom:6px">${classPillHtml("thermal", CLASS.thermal.label)}</div>
        <p style="margin:0 0 6px"><strong>How detected:</strong> EWMA below LCL during the event;
        mean temperature during the event was in the hottest ~20% of all days in the dataset
        (≥ 80th percentile); EWMA recovered to ≥ LCL within ${RECOVERY_WINDOW_DAYS} days after
        the event ended, and recovery coincided with cooler days (temperature dropped below the
        event-end temperature while EWMA rose back above LCL).</p>
        <p style="margin:0 0 6px"><strong>Data used:</strong> Daily H residual &amp; EWMA; daily
        mean temperature (<code>${TEMP_FIELD}</code> or cell temperature); historical temperature
        distribution; recovery timing vs LCL.</p>
        <p style="margin:0"><strong>What to do:</strong> ${CLASS.thermal.action}. Hot-weather
        efficiency loss is expected for PV. Watch for repeated summer dips; investigate only if
        performance stays low after temperatures normalize.</p>
      </div>

      <div style="margin:0 0 12px;padding:10px 12px;background:#0f172a;border-radius:6px;border:1px solid #334155">
        <div style="margin-bottom:6px">${classPillHtml("fault", CLASS.fault.label)}</div>
        <p style="margin:0 0 6px"><strong>How detected:</strong> EWMA stayed below LCL for a
        grouped alarm event, but the drop is <em>not</em> explained by the soiling rule (rain +
        recovery) or the thermal rule (hot event + cooler recovery). This includes sustained
        underperformance with no recovery, or recovery without a matching weather explanation.</p>
        <p style="margin:0 0 6px"><strong>Data used:</strong> Daily H residual &amp; EWMA; rainfall
        and temperature checks above; anything that fails both classification paths defaults here.</p>
        <p style="margin:0"><strong>What to do:</strong> ${CLASS.fault.action}. Treat as a possible
        inverter fault, string outage, communication loss, or other hardware issue. Cross-check
        with site visits, SCADA alerts, and the Difference / degradation tabs before closing.</p>
      </div>

      <div style="margin:0;padding:10px 12px;background:#0f172a;border-radius:6px;border:1px solid #334155">
        <div style="margin-bottom:6px">${classPillHtml("unclassified", CLASS.unclassified.label)}</div>
        <p style="margin:0 0 6px"><strong>How detected:</strong> An EWMA alarm event was captured
        (Steps 1–4 above), but hourly data has no usable rainfall or temperature fields, so
        automatic soiling/thermal rules cannot run.</p>
        <p style="margin:0 0 6px"><strong>Data used:</strong> H, EWMA, and LCL only — weather
        columns missing or empty in <code>hourly_*_master.csv</code>.</p>
        <p style="margin:0"><strong>What to do:</strong> ${CLASS.unclassified.action}. Review the
        dates and min EWMA in the table; add rainfall/temperature to the pipeline if available,
        or classify manually using external weather records and field inspection.</p>
      </div>
    </div>
  </details>`;
}

function soilingHowItWorksHtml() {
  return `<details style="margin:8px 0 12px;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:0 14px">
    <summary style="cursor:pointer;padding:12px 0;font-size:0.9rem;color:#cbd5e1;font-weight:500">How the weather &amp; performance analysis works</summary>
    <div style="padding:0 0 14px 0;font-size:0.82rem;color:#94a3b8;line-height:1.6">
      <p style="margin:0 0 10px"><strong>Health ratio (H)</strong> is actual ÷ expected generation on daylight hours (GHI &gt; 5 W/m²).
        The top chart shows daily H (faint green) and a ${SMOOTH_WINDOW_DAYS}-day smoothed trend (white). Orange lines are soiling
        fits during long dry spells — only the largest spells are drawn so the chart stays readable.</p>
      <p style="margin:0 0 10px"><strong>Significant changes</strong> (markers on the H chart) are days where smoothed H moves by at least
        ${SIGNIFICANT_JUMP_THRESHOLD}, debounced to one marker every ${MIN_DAYS_BETWEEN_EVENTS} days. Each is classified using
        <em>all available weather</em>, not rain/wind alone: rainfall ≥ ${RAIN_CLEAN_THRESHOLD} mm, wind ≥ ${WIND_CLEAN_THRESHOLD} m/s,
        temperature shift, daily GHI (cloud/resource), and dry-spell soiling trend.</p>
      <p style="margin:0 0 10px">The <strong>rain, wind, temperature, and GHI charts</strong> below share one time axis — pan/zoom any panel
        to line up a recovery or drop in H with weather on that date.</p>
      <p style="margin:0">The <strong>driver correlations</strong> table summarises how daily weather relates to day-to-day H movement
        across the full dataset. Energy lost to soiling is estimated from fitted decline slopes during dry spells.</p>
    </div>
  </details>`;
}

const DRIVER_PILL = {
  Rain: { bg: "#1e3a5f", fg: "#93c5fd" },
  Wind: { bg: "#0e7490", fg: "#a5f3fc" },
  "Clearer sky (GHI)": { bg: "#422006", fg: "#fde68a" },
  "Cooler weather": { bg: "#312e81", fg: "#c4b5fd" },
  "Heat stress": { bg: "#450a0a", fg: "#fca5a5" },
  "Cloud / low GHI": { bg: "#374151", fg: "#d1d5db" },
  "Soiling build-up": { bg: "#431407", fg: "#fdba74" },
  "Other / unknown": { bg: "#1e293b", fg: "#94a3b8" },
};

function driverPillHtml(cause) {
  const s = DRIVER_PILL[cause] || DRIVER_PILL["Other / unknown"];
  return `<span style="display:inline-block;font-size:11px;padding:2px 8px;border-radius:4px;font-weight:500;background:${s.bg};color:${s.fg}">${cause}</span>`;
}

function pearsonR(xs, ys) {
  const pairs = [];
  const n = Math.min(xs.length, ys.length);
  for (let i = 0; i < n; i++) {
    if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) pairs.push([xs[i], ys[i]]);
  }
  if (pairs.length < 2) return NaN;
  const px = pairs.map((p) => p[0]);
  const py = pairs.map((p) => p[1]);
  const mx = mean(px);
  const my = mean(py);
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < pairs.length; i++) {
    const a = px[i] - mx;
    const b = py[i] - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  const den = Math.sqrt(dx * dy);
  return den > 1e-12 ? num / den : NaN;
}

function readRainSoiling(r) {
  const raw = r[RAIN_FIELD] ?? r.precipitation_mm ?? r.precipitation_rate;
  const v = Number(raw);
  return Number.isFinite(v) ? v : 0;
}

function numField(r, ...keys) {
  for (const k of keys) {
    if (r[k] === undefined || r[k] === null || r[k] === "") continue;
    const v = Number(r[k]);
    if (Number.isFinite(v)) return v;
  }
  return NaN;
}

function readWind(r) {
  return numField(r, WIND_FIELD, "wind_speed_10m", "wind_speed_100m", "wind_ms");
}

function hasWindData(hourly) {
  return hourly.some((r) => Number.isFinite(readWind(r)));
}

function readRain(r) {
  const raw = r[RAIN_FIELD] ?? r.precipitation_mm ?? r.precipitation_rate;
  const v = Number(raw);
  return Number.isFinite(v) ? v : NaN;
}

function readTemp(r) {
  const raw = r[TEMP_FIELD] ?? r.temp_cell_c ?? r.air_temp;
  const v = Number(raw);
  return Number.isFinite(v) ? v : NaN;
}

function hasWeatherFields(hourly) {
  return hourly.some((r) => Number.isFinite(readRain(r)) || Number.isFinite(readTemp(r)));
}

function addMonthsIso(iso, n) {
  const d = new Date(iso + "T12:00:00");
  d.setMonth(d.getMonth() + n);
  return d.toISOString().slice(0, 10);
}

function addDaysIso(iso, n) {
  const d = new Date(iso + "T12:00:00");
  d.setDate(d.getDate() + n);
  return d.toISOString().slice(0, 10);
}

function seasonId(day) {
  return seasonOf(day).id;
}

function mean(arr) {
  if (!arr.length) return NaN;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function stddev(arr) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  const v = arr.reduce((s, x) => s + (x - m) ** 2, 0) / (arr.length - 1);
  return Math.sqrt(v);
}

function median(arr) {
  if (!arr.length) return NaN;
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Stage 1 — daily H + optional weather aggregates. */
function computeDailySeries(hourly) {
  const byDay = new Map();
  for (const r of hourly) {
    if (!(r.ghi > 5)) continue;
    let o = byDay.get(r.day);
    if (!o) {
      o = { act: 0, exp: 0, rain: 0, rainN: 0, temp: 0, tempN: 0 };
      byDay.set(r.day, o);
    }
    if (Number.isFinite(r.actual)) o.act += r.actual;
    if (Number.isFinite(r.expected)) o.exp += r.expected;
    const rain = readRain(r);
    if (Number.isFinite(rain)) {
      o.rain += rain;
      o.rainN += 1;
    }
    const temp = readTemp(r);
    if (Number.isFinite(temp)) {
      o.temp += temp;
      o.tempN += 1;
    }
  }
  const out = [];
  for (const [day, o] of byDay) {
    if (o.exp <= 0) continue;
    out.push({
      day,
      H: o.act / o.exp,
      rainTotal: o.rainN ? o.rain : NaN,
      tempMean: o.tempN ? o.temp / o.tempN : NaN,
    });
  }
  return out.sort((a, b) => (a.day < b.day ? -1 : 1));
}

function buildSeasonMedians(daily, refStart, refEnd) {
  const buckets = { sum: [], aut: [], win: [], spr: [] };
  const fallback = { sum: [], aut: [], win: [], spr: [] };
  for (const d of daily) {
    const sid = seasonId(d.day);
    fallback[sid]?.push(d.H);
    if (d.day >= refStart && d.day <= refEnd) buckets[sid]?.push(d.H);
  }
  const med = {};
  for (const k of Object.keys(buckets)) {
    const vals = buckets[k].length ? buckets[k] : fallback[k];
    med[k] = vals?.length ? median(vals) : NaN;
  }
  return med;
}

function computeEwmaLimitFactor() {
  return Math.sqrt(EWMA_LAMBDA / (2 - EWMA_LAMBDA));
}

function computeAnalysis(daily, weatherAvailable) {
  if (daily.length < 2) return null;

  const refStart = daily[0].day;
  const refEnd = addMonthsIso(refStart, REFERENCE_MONTHS);
  const seasonMed = buildSeasonMedians(daily, refStart, refEnd);

  const series = daily.map((d) => {
    const sid = seasonId(d.day);
    const expectedH = seasonMed[sid];
    const residual = Number.isFinite(expectedH) ? d.H - expectedH : NaN;
    return { ...d, expectedH, residual, inReference: d.day >= refStart && d.day <= refEnd };
  });

  const refResiduals = series
    .filter((d) => d.inReference && Number.isFinite(d.residual))
    .map((d) => d.residual);
  const mu = mean(refResiduals);
  const sigma = stddev(refResiduals);
  const limitFactor = computeEwmaLimitFactor();
  const LCL = mu - EWMA_L * sigma * limitFactor;
  const UCL = mu + EWMA_L * sigma * limitFactor;

  let ewmaPrev = null;
  for (let i = 0; i < series.length; i++) {
    const r = series[i].residual;
    if (!Number.isFinite(r)) {
      series[i].ewma = NaN;
      series[i].alarm = false;
      continue;
    }
    if (ewmaPrev === null) {
      ewmaPrev = r;
      series[i].ewma = r;
    } else {
      ewmaPrev = EWMA_LAMBDA * r + (1 - EWMA_LAMBDA) * ewmaPrev;
      series[i].ewma = ewmaPrev;
    }
    series[i].alarm =
      Number.isFinite(LCL) &&
      Number.isFinite(series[i].ewma) &&
      series[i].ewma < LCL;
  }

  const alarmDays = series.filter((d) => d.alarm);
  const events = groupAlarmEvents(alarmDays, series, LCL, weatherAvailable);

  return {
    series,
    refStart,
    refEnd,
    seasonMed,
    mu,
    sigma,
    LCL,
    UCL,
    events,
    weatherAvailable,
  };
}

function groupAlarmEvents(alarmDays, series, lcl, weatherAvailable) {
  if (!alarmDays.length) return [];
  const dayIndex = new Map(series.map((d, i) => [d.day, i]));
  const groups = [];
  let cur = [alarmDays[0].day];

  for (let i = 1; i < alarmDays.length; i++) {
    const prev = alarmDays[i - 1].day;
    const d = alarmDays[i].day;
    const gap =
      (new Date(d + "T12:00:00").getTime() -
        new Date(prev + "T12:00:00").getTime()) /
      86400000;
    if (gap <= ALARM_BRIDGE_DAYS) cur.push(d);
    else {
      groups.push(cur);
      cur = [d];
    }
  }
  groups.push(cur);

  return groups.map((days) => {
    const startDay = days[0];
    const endDay = days[days.length - 1];
    const ewmas = days
      .map((day) => series[dayIndex.get(day)]?.ewma)
      .filter(Number.isFinite);
    const minEWMA = ewmas.length ? Math.min(...ewmas) : NaN;
    const durationDays =
      Math.round(
        (new Date(endDay + "T12:00:00").getTime() -
          new Date(startDay + "T12:00:00").getTime()) /
          86400000,
      ) + 1;

    const cls = classifyEvent(
      startDay,
      endDay,
      series,
      lcl,
      weatherAvailable,
    );
    return {
      startDay,
      endDay,
      durationDays,
      minEWMA,
      ...cls,
    };
  });
}

function classifyEvent(startDay, endDay, series, lcl, weatherAvailable) {
  if (!weatherAvailable) {
    return { kind: "unclassified", ...CLASS.unclassified };
  }

  const recoveryEnd = addDaysIso(endDay, RECOVERY_WINDOW_DAYS);
  const window = series.filter((d) => d.day >= startDay && d.day <= recoveryEnd);

  const recovered = window.some(
    (d) => d.day > endDay && Number.isFinite(d.ewma) && d.ewma >= lcl,
  );

  const eventDays = series.filter((d) => d.day >= startDay && d.day <= endDay);
  const eventTempMean = mean(
    eventDays.map((d) => d.tempMean).filter(Number.isFinite),
  );

  const allTemps = series.map((d) => d.tempMean).filter(Number.isFinite);
  allTemps.sort((a, b) => a - b);
  const p80 =
    allTemps.length >= 5
      ? allTemps[Math.floor(allTemps.length * 0.8)]
      : NaN;

  const rainInWindow = window.some(
    (d) => Number.isFinite(d.rainTotal) && d.rainTotal >= RAIN_CLEAN_THRESHOLD,
  );

  if (recovered && rainInWindow) {
    return { kind: "soiling", ...CLASS.soiling };
  }

  const hotEvent =
    Number.isFinite(eventTempMean) &&
    Number.isFinite(p80) &&
    eventTempMean >= p80;

  if (hotEvent && recovered) {
    const after = window.filter((d) => d.day > endDay && Number.isFinite(d.tempMean));
    const eventEndTemp = eventDays.length
      ? eventDays[eventDays.length - 1].tempMean
      : NaN;
    const coolerRecovery = after.some(
      (d) =>
        Number.isFinite(eventEndTemp) &&
        d.tempMean < eventEndTemp &&
        Number.isFinite(d.ewma) &&
        d.ewma >= lcl,
    );
    if (coolerRecovery) {
      return { kind: "thermal", ...CLASS.thermal };
    }
  }

  return { kind: "fault", ...CLASS.fault };
}

function recentStatus(events, series, lastDay) {
  const cut = addDaysIso(lastDay, -STATUS_LOOKBACK_DAYS);
  const recent = events.filter((e) => e.endDay >= cut);
  if (!recent.length) {
    return {
      level: "healthy",
      color: "#22c55e",
      title: "System healthy",
      summary: "No alarms in the last 30 days.",
    };
  }
  const last = recent[recent.length - 1];
  const hasFault = recent.some((e) => e.kind === "fault");
  const allExplained = recent.every(
    (e) => e.kind === "soiling" || e.kind === "thermal",
  );
  if (hasFault) {
    return {
      level: "fault",
      color: "#f87171",
      title: "Inspection recommended",
      summary: `Last alarm: ${last.endDay} · ${last.label} · ${last.durationDays} days`,
    };
  }
  if (allExplained) {
    return {
      level: "explained",
      color: "#fbbf24",
      title: "Explained underperformance (soiling/thermal)",
      summary: `Last alarm: ${last.endDay} · ${last.label} · ${last.durationDays} days`,
    };
  }
  return {
    level: "watch",
    color: "#fbbf24",
    title: "Monitor — mixed or unclassified alarms",
    summary: `Last alarm: ${last.endDay} · ${last.label} · ${last.durationDays} days`,
  };
}

function measurePlotWidth(el, panel) {
  const parent = el?.parentElement;
  const rect = parent?.getBoundingClientRect?.();
  if (rect && rect.width > 80) return Math.floor(rect.width);
  return Math.max(480, Math.floor(panel?.clientWidth || 960) - 32);
}

/** Daily H + full weather aggregates for performance analysis. */
function computeWeatherDaily(hourly) {
  const byDay = new Map();
  for (const r of hourly) {
    let o = byDay.get(r.day);
    if (!o) {
      o = {
        act: 0,
        exp: 0,
        rain: 0,
        windMax: NaN,
        tempN: 0,
        tempSum: 0,
        ghiSum: 0,
        cloudN: 0,
        cloudSum: 0,
      };
      byDay.set(r.day, o);
    }
    o.rain += readRainSoiling(r);
    const w = readWind(r);
    if (Number.isFinite(w)) {
      o.windMax = Number.isFinite(o.windMax) ? Math.max(o.windMax, w) : w;
    }
    const cloud = numField(r, "cloud_opacity");
    if (Number.isFinite(cloud)) {
      o.cloudSum += cloud;
      o.cloudN += 1;
    }
    if (!(r.ghi > 5)) continue;
    if (Number.isFinite(r.actual)) o.act += r.actual;
    if (Number.isFinite(r.expected)) o.exp += r.expected;
    if (Number.isFinite(r.ghi)) o.ghiSum += r.ghi;
    const t = readTemp(r);
    if (Number.isFinite(t)) {
      o.tempSum += t;
      o.tempN += 1;
    }
  }
  const out = [];
  for (const [day, o] of byDay) {
    if (o.exp <= 0) continue;
    out.push({
      day,
      H: o.act / o.exp,
      rain: o.rain,
      windMax: o.windMax,
      tempMean: o.tempN ? o.tempSum / o.tempN : NaN,
      ghiSum: o.ghiSum,
      cloudMean: o.cloudN ? o.cloudSum / o.cloudN : NaN,
      expKwh: o.exp,
    });
  }
  out.sort((a, b) => a.day.localeCompare(b.day));
  const Hvals = out.map((d) => d.H);
  const Hsmooth = rollingMedian(Hvals, SMOOTH_WINDOW_DAYS, 1);
  out.forEach((d, i) => {
    d.Hsmooth = Hsmooth[i];
    d.dH =
      i > 0 && Number.isFinite(Hsmooth[i]) && Number.isFinite(Hsmooth[i - 1])
        ? Hsmooth[i] - Hsmooth[i - 1]
        : NaN;
  });
  return out;
}

function driverColor(cause) {
  const map = {
    Rain: "#3b82f6",
    Wind: "#22d3ee",
    "Clearer sky (GHI)": "#fbbf24",
    "Cooler weather": "#a78bfa",
    "Heat stress": "#f87171",
    "Cloud / low GHI": "#94a3b8",
    "Soiling build-up": "#fb923c",
    "Other / unknown": "#64748b",
  };
  return map[cause] || map["Other / unknown"];
}

function attributeChange(daily, idx, delta) {
  const isUp = delta > 0;
  const window3 = daily.slice(Math.max(0, idx - 2), idx + 1);
  const prev3 = daily.slice(Math.max(0, idx - 3), idx);
  const d = daily[idx];
  const rainAmt = Math.max(...window3.map((x) => x.rain), 0);
  const windAmt = Number.isFinite(d.windMax) ? d.windMax : NaN;
  const tempPrior = mean(prev3.map((x) => x.tempMean).filter(Number.isFinite));
  const ghiPrior = mean(prev3.map((x) => x.ghiSum).filter(Number.isFinite));
  const dryDays = prev3.filter((x) => x.rain < RAIN_CLEAN_THRESHOLD).length;
  const scores = [];

  if (isUp) {
    if (window3.some((x) => x.rain >= RAIN_CLEAN_THRESHOLD)) {
      scores.push({ cause: "Rain", score: rainAmt + 10 });
    }
    if (Number.isFinite(windAmt) && windAmt >= WIND_CLEAN_THRESHOLD) {
      scores.push({ cause: "Wind", score: windAmt });
    }
    if (
      Number.isFinite(d.tempMean) &&
      Number.isFinite(tempPrior) &&
      d.tempMean < tempPrior - 1.5
    ) {
      scores.push({ cause: "Cooler weather", score: tempPrior - d.tempMean });
    }
    if (
      Number.isFinite(d.ghiSum) &&
      Number.isFinite(ghiPrior) &&
      ghiPrior > 0 &&
      d.ghiSum > ghiPrior * 1.12
    ) {
      scores.push({ cause: "Clearer sky (GHI)", score: (d.ghiSum / ghiPrior - 1) * 100 });
    }
  } else {
    if (
      Number.isFinite(d.tempMean) &&
      Number.isFinite(tempPrior) &&
      d.tempMean > tempPrior + 1.5
    ) {
      scores.push({ cause: "Heat stress", score: d.tempMean - tempPrior });
    }
    if (
      Number.isFinite(d.ghiSum) &&
      Number.isFinite(ghiPrior) &&
      ghiPrior > 0 &&
      d.ghiSum < ghiPrior * 0.88
    ) {
      scores.push({ cause: "Cloud / low GHI", score: (1 - d.ghiSum / ghiPrior) * 100 });
    }
    if (dryDays >= 2 && delta > -SIGNIFICANT_JUMP_THRESHOLD) {
      scores.push({ cause: "Soiling build-up", score: dryDays * 2 });
    }
  }

  scores.sort((a, b) => b.score - a.score);
  const cause = scores.length ? scores[0].cause : "Other / unknown";
  return {
    cause,
    color: driverColor(cause),
    rainAmt,
    windAmt,
    tempMean: d.tempMean,
    ghiSum: d.ghiSum,
    cloudMean: d.cloudMean,
  };
}

function selectSignificantEvents(daily) {
  const raw = [];
  for (let i = 1; i < daily.length; i++) {
    const prev = daily[i - 1].Hsmooth;
    const cur = daily[i].Hsmooth;
    if (!Number.isFinite(prev) || !Number.isFinite(cur)) continue;
    const jump = cur - prev;
    if (Math.abs(jump) < CLEAN_JUMP_THRESHOLD) continue;
    const attr = attributeChange(daily, i, jump);
    raw.push({
      day: daily[i].day,
      idx: i,
      jump,
      direction: jump >= 0 ? "Recovery" : "Drop",
      ...attr,
    });
  }
  raw.sort((a, b) => Math.abs(b.jump) - Math.abs(a.jump));
  const picked = [];
  for (const ev of raw) {
    if (Math.abs(ev.jump) < SIGNIFICANT_JUMP_THRESHOLD) continue;
    const tooClose = picked.some(
      (p) =>
        Math.abs(
          (new Date(ev.day + "T12:00:00").getTime() -
            new Date(p.day + "T12:00:00").getTime()) /
            86400000,
        ) < MIN_DAYS_BETWEEN_EVENTS,
    );
    if (tooClose) continue;
    picked.push(ev);
    if (picked.length >= MAX_CHART_RECOVERY_MARKERS) break;
  }
  return picked.sort((a, b) => a.day.localeCompare(b.day));
}

/** Weather impact analysis — multi-factor drivers, debounced events, soiling segments. */
function computeWeatherImpactAnalysis(weatherDaily) {
  const emptySummary = {
    meanSoilingPct: NaN,
    byCause: {},
    meanRainJump: NaN,
    meanWindJump: NaN,
    pearsonR: NaN,
    totalEnergyLostKwh: 0,
    pctEnergyLost: NaN,
    hasWind: false,
    totalEvents: 0,
    correlations: [],
  };
  if (!weatherDaily.length) {
    return {
      daily: weatherDaily,
      significantEvents: [],
      segments: [],
      chartSegments: [],
      summary: emptySummary,
    };
  }

  const significantEvents = selectSignificantEvents(weatherDaily);
  const recoveryDays = significantEvents
    .filter((e) => e.direction === "Recovery")
    .map((e) => e.day);
  const dayToIdx = new Map(weatherDaily.map((d, i) => [d.day, i]));
  const segments = [];

  for (let k = 0; k <= recoveryDays.length; k++) {
    let startIdx;
    let endIdx;
    if (k === 0) {
      startIdx = 0;
      endIdx =
        recoveryDays.length > 0
          ? dayToIdx.get(recoveryDays[0]) - 1
          : weatherDaily.length - 1;
    } else if (k === recoveryDays.length) {
      startIdx = dayToIdx.get(recoveryDays[k - 1]) + 1;
      endIdx = weatherDaily.length - 1;
    } else {
      startIdx = dayToIdx.get(recoveryDays[k - 1]) + 1;
      endIdx = dayToIdx.get(recoveryDays[k]) - 1;
    }
    if (startIdx > endIdx) continue;
    const length = endIdx - startIdx + 1;
    if (length < SOILING_MIN_SEGMENT_DAYS) continue;

    const slice = weatherDaily.slice(startIdx, endIdx + 1);
    const xs = slice.map((_, j) => j);
    const ys = slice.map((d) => d.H);
    const { slope, intercept } = linearRegression(xs, ys);
    if (!Number.isFinite(slope)) continue;

    const startH = slice[0].H;
    const pctPerDay =
      Number.isFinite(startH) && startH !== 0 ? (slope / startH) * 100 : NaN;
    let hLost = 0;
    let energyLostKwh = 0;
    for (let j = 0; j < slice.length; j++) {
      const fitted = intercept + slope * j;
      const deficit = Math.max(0, startH - fitted);
      hLost += deficit;
      energyLostKwh += deficit * slice[j].expKwh;
    }

    segments.push({
      startDay: slice[0].day,
      endDay: slice[slice.length - 1].day,
      days: length,
      slopePerDay: slope,
      pctPerDay,
      startH,
      intercept,
      slope,
      slice,
      hLost,
      energyLostKwh,
    });
  }

  const chartSegments = [...segments]
    .sort((a, b) => b.days - a.days)
    .slice(0, MAX_SOILING_LINES_ON_CHART);
  const validPct = segments.map((s) => s.pctPerDay).filter(Number.isFinite);
  const meanSoilingPct = validPct.length ? mean(validPct) : NaN;
  const byCause = {};
  for (const e of significantEvents) byCause[e.cause] = (byCause[e.cause] || 0) + 1;

  const rainEv = significantEvents.filter(
    (e) => e.cause === "Rain" && e.direction === "Recovery",
  );
  const windEv = significantEvents.filter(
    (e) => e.cause === "Wind" && e.direction === "Recovery",
  );
  const meanRainJump = rainEv.length ? mean(rainEv.map((e) => e.jump)) : NaN;
  const meanWindJump = windEv.length ? mean(windEv.map((e) => e.jump)) : NaN;
  const rainPairs = rainEv.filter((e) => Number.isFinite(e.rainAmt));
  const rRain = pearsonR(
    rainPairs.map((e) => e.rainAmt),
    rainPairs.map((e) => e.jump),
  );
  const totalEnergyLostKwh = segments.reduce((s, seg) => s + seg.energyLostKwh, 0);
  const totalExpected = weatherDaily.reduce((s, d) => s + d.expKwh, 0);
  const pctEnergyLost =
    totalExpected > 0 ? (totalEnergyLostKwh / totalExpected) * 100 : NaN;
  const hasWind = weatherDaily.some((d) => Number.isFinite(d.windMax));
  const correlations = [
    {
      factor: "Rain (mm/day)",
      r: pearsonR(weatherDaily.map((d) => d.rain), weatherDaily.map((d) => d.dH)),
    },
    {
      factor: "Wind max (m/s)",
      r: pearsonR(weatherDaily.map((d) => d.windMax), weatherDaily.map((d) => d.dH)),
    },
    {
      factor: "Temperature (°C)",
      r: pearsonR(weatherDaily.map((d) => d.tempMean), weatherDaily.map((d) => d.dH)),
    },
    {
      factor: "Daily GHI sum",
      r: pearsonR(weatherDaily.map((d) => d.ghiSum), weatherDaily.map((d) => d.dH)),
    },
    {
      factor: "Cloud opacity",
      r: pearsonR(weatherDaily.map((d) => d.cloudMean), weatherDaily.map((d) => d.dH)),
    },
  ].filter((c) => Number.isFinite(c.r));

  return {
    daily: weatherDaily,
    significantEvents,
    segments,
    chartSegments,
    summary: {
      meanSoilingPct,
      byCause,
      meanRainJump,
      meanWindJump,
      pearsonR: rRain,
      totalEnergyLostKwh,
      pctEnergyLost,
      hasWind,
      totalEvents: significantEvents.length,
      correlations,
    },
  };
}

function buildDailyWeatherFromRaw(rawRows) {
  const byDay = new Map();
  for (const r of rawRows) {
    const ts = String(r.timestamp || "").trim();
    if (!ts) continue;
    const day = ts.slice(0, 10);
    let o = byDay.get(day);
    if (!o) {
      o = { rain: 0, windMax: NaN, cloudN: 0, cloudSum: 0, tempN: 0, tempSum: 0, ghiSum: 0 };
      byDay.set(day, o);
    }
    o.rain += readRainSoiling(r);
    const w = numField(r, "wind_speed_10m", "wind_speed_100m", WIND_FIELD, "wind_ms");
    if (Number.isFinite(w)) {
      o.windMax = Number.isFinite(o.windMax) ? Math.max(o.windMax, w) : w;
    }
    const c = numField(r, "cloud_opacity");
    if (Number.isFinite(c)) {
      o.cloudSum += c;
      o.cloudN += 1;
    }
    const t = numField(r, "air_temp", "temp_c", "temp_cell_c");
    const ghi = numField(r, "ghi", "ghi_wm2");
    if (Number.isFinite(t)) {
      o.tempSum += t;
      o.tempN += 1;
    }
    if (Number.isFinite(ghi) && ghi > 5) o.ghiSum += ghi;
  }
  return byDay;
}

const SOLCAST_CLEANED_CSV = "solcast_df_cleaned_2020_2025.csv";

function csvRowsHaveWeatherColumns(rows) {
  if (!rows?.length) return false;
  const r = rows[0];
  return (
    r.precipitation_rate !== undefined ||
    r.rain_mm !== undefined ||
    r.wind_speed_10m !== undefined ||
    r.cloud_opacity !== undefined
  );
}

async function fetchWeatherEnrichmentRows(site) {
  if (site?.key) {
    try {
      const { text } = await fetchHourlyMasterText(site.key);
      const { rows } = parseCSV(text);
      if (csvRowsHaveWeatherColumns(rows)) return rows;
    } catch (fetchErr) {
      console.warn("[HealthMonitor] hourly master weather columns missing", fetchErr);
    }
  }
  const cleaned = await tryFetchCleanedFile(SOLCAST_CLEANED_CSV);
  if (cleaned) {
    const { rows } = parseCSV(cleaned.text);
    return rows;
  }
  return [];
}

function enrichDailyWithRawWeather(daily, rawRows) {
  if (!rawRows?.length) return;
  const raw = buildDailyWeatherFromRaw(rawRows);
  for (const d of daily) {
    const w = raw.get(d.day);
    if (!w) continue;
    d.rain = w.rain;
    if (Number.isFinite(w.windMax)) d.windMax = w.windMax;
    if (w.cloudN) d.cloudMean = w.cloudSum / w.cloudN;
    if (w.tempN && !Number.isFinite(d.tempMean)) d.tempMean = w.tempSum / w.tempN;
    if (w.ghiSum > 0 && (!Number.isFinite(d.ghiSum) || d.ghiSum <= 0)) d.ghiSum = w.ghiSum;
  }
}

function wireLinkedXAxes(chartEls) {
  const applyXRange = (range, skipEl) => {
    const update = range
      ? { "xaxis.range": range, "xaxis.autorange": false }
      : { "xaxis.autorange": true };
    for (const el of chartEls) {
      if (el && el !== skipEl && el.data) Plotly.relayout(el, update);
    }
  };

  for (const el of chartEls) {
    if (!el) continue;
    el.on("plotly_relayout", (ev) => {
      if (ev["xaxis.range[0]"] !== undefined && ev["xaxis.range[1]"] !== undefined) {
        applyXRange([ev["xaxis.range[0]"], ev["xaxis.range[1]"]], el);
      } else if (ev["xaxis.autorange"] === true) {
        applyXRange(null, el);
      }
    });
  }
}

function dailyWindowAround(daily, centerDay, radiusDays = EVENT_POPUP_RADIUS_DAYS) {
  const start = addDaysIso(centerDay, -radiusDays);
  const end = addDaysIso(centerDay, radiusDays);
  return daily.filter((d) => d.day >= start && d.day <= end);
}

function eventDayMarkerShape(eventDay, color) {
  return {
    type: "line",
    x0: eventDay,
    x1: eventDay,
    y0: 0,
    y1: 1,
    yref: "paper",
    line: { color, width: 2, dash: "dot" },
  };
}

function eventPopupMetaHtml(event) {
  const sign = event.jump >= 0 ? "+" : "";
  const accent = event.direction === "Recovery" ? "#22c55e" : "#f87171";
  const thirdLabel = event.direction === "Recovery" ? "Wind max" : "Temp";
  const thirdValue =
    event.direction === "Recovery"
      ? Number.isFinite(event.windAmt)
        ? `${event.windAmt.toFixed(2)} m/s`
        : "—"
      : Number.isFinite(event.tempMean)
        ? `${event.tempMean.toFixed(1)} °C`
        : "—";
  return `
    <div class="hm-event-popup__title" style="border-left-color:${accent}">
      <span class="hm-event-popup__dir">${event.direction}</span>
      <span class="hm-event-popup__cause">${event.cause}</span>
    </div>
    <div class="hm-event-popup__meta">
      <div><span class="hm-event-popup__label">Date</span>${event.day}</div>
      <div><span class="hm-event-popup__label">ΔH</span>${sign}${event.jump.toFixed(3)}</div>
      <div><span class="hm-event-popup__label">Rain (3-day)</span>${event.rainAmt.toFixed(2)} mm</div>
      <div><span class="hm-event-popup__label">${thirdLabel}</span>${thirdValue}</div>
    </div>
    <div class="hm-event-popup__window">7-day window · ${addDaysIso(event.day, -EVENT_POPUP_RADIUS_DAYS)} → ${addDaysIso(event.day, EVENT_POPUP_RADIUS_DAYS)}</div>`;
}

function buildMiniHChartSpec(windowDaily, event, width) {
  const days = windowDaily.map((d) => d.day);
  const theme = plotlyDarkTheme();
  const markerColor = event.direction === "Recovery" ? "#22c55e" : "#f87171";
  const eventPoint = windowDaily.find((d) => d.day === event.day);
  const traces = [
    {
      x: days,
      y: windowDaily.map((d) => d.H),
      type: "scatter",
      mode: "lines",
      name: "Daily H",
      line: { color: "#22c55e", width: 1 },
      opacity: 0.45,
      hovertemplate: "%{x}<br>H = %{y:.3f}<extra></extra>",
    },
    {
      x: days,
      y: windowDaily.map((d) => d.Hsmooth),
      type: "scatter",
      mode: "lines+markers",
      name: "Smoothed H",
      line: { color: "#e2e8f0", width: 2 },
      marker: { size: 5, color: "#e2e8f0" },
      hovertemplate: "%{x}<br>Smoothed H = %{y:.3f}<extra></extra>",
    },
  ];
  if (eventPoint && Number.isFinite(eventPoint.Hsmooth)) {
    traces.push({
      x: [event.day],
      y: [eventPoint.Hsmooth],
      type: "scatter",
      mode: "markers",
      name: event.direction,
      marker: {
        size: 11,
        symbol: event.direction === "Recovery" ? "triangle-up" : "triangle-down",
        color: markerColor,
        line: { width: 1, color: "#0f172a" },
      },
      hoverinfo: "skip",
    });
  }
  return {
    traces,
    layout: {
      ...theme,
      autosize: false,
      width,
      height: 130,
      margin: { t: 28, r: 12, b: 32, l: 44 },
      title: {
        text: "Health ratio (H)",
        font: { color: "#cbd5e1", size: 11 },
      },
      xaxis: { ...theme.xaxis, type: "date", tickformat: "%d %b" },
      yaxis: { ...theme.yaxis, title: { text: "H", font: { size: 10 } } },
      shapes: [eventDayMarkerShape(event.day, markerColor)],
      showlegend: false,
      hovermode: "x unified",
    },
  };
}

function buildMiniRainChartSpec(windowDaily, event, width) {
  const days = windowDaily.map((d) => d.day);
  const theme = plotlyDarkTheme();
  const markerColor = event.direction === "Recovery" ? "#22c55e" : "#f87171";
  return {
    traces: [
      {
        x: days,
        y: windowDaily.map((d) => d.rain),
        type: "bar",
        name: "Rain",
        marker: { color: "#3b82f6" },
        hovertemplate: "%{x}<br>Rain = %{y:.2f} mm<extra></extra>",
      },
      {
        x: [days[0], days[days.length - 1]],
        y: [RAIN_CLEAN_THRESHOLD, RAIN_CLEAN_THRESHOLD],
        type: "scatter",
        mode: "lines",
        line: { color: "#f87171", width: 1, dash: "dash" },
        hoverinfo: "skip",
      },
    ],
    layout: {
      ...theme,
      autosize: false,
      width,
      height: 120,
      margin: { t: 28, r: 12, b: 32, l: 44 },
      title: {
        text: "Daily rainfall",
        font: { color: "#cbd5e1", size: 11 },
      },
      xaxis: { ...theme.xaxis, type: "date", tickformat: "%d %b" },
      yaxis: { ...theme.yaxis, title: { text: "mm", font: { size: 10 } } },
      shapes: [eventDayMarkerShape(event.day, markerColor)],
      bargap: 0.2,
      showlegend: false,
      hovermode: "x unified",
    },
  };
}

let _eventPopupEl = null;
let _eventPopupHideTimer = null;
let _eventPopupPinned = false;
let _eventPopupPlotEl = null;

function hoverDayFromPlotlyEvent(ev) {
  const pt = ev.points?.[0];
  if (!pt) return null;
  const raw = pt.x;
  if (typeof raw === "string") return raw.slice(0, 10);
  try {
    return new Date(raw).toISOString().slice(0, 10);
  } catch {
    return null;
  }
}

function setPlotlyNativeHoverVisible(plotEl, visible) {
  if (!plotEl) return;
  plotEl.querySelectorAll(".hoverlayer").forEach((el) => {
    el.style.visibility = visible ? "visible" : "hidden";
  });
}

function pointerFromPlotlyEvent(ev) {
  const mouse = ev?.event || {};
  const clientX = Number.isFinite(mouse.clientX)
    ? mouse.clientX
    : Number.isFinite(mouse.pageX)
      ? mouse.pageX
      : window.innerWidth / 2;
  const clientY = Number.isFinite(mouse.clientY)
    ? mouse.clientY
    : Number.isFinite(mouse.pageY)
      ? mouse.pageY
      : window.innerHeight / 2;
  return { clientX, clientY };
}

function removeEventHoverPopup() {
  if (_eventPopupHideTimer) {
    clearTimeout(_eventPopupHideTimer);
    _eventPopupHideTimer = null;
  }
  _eventPopupPinned = false;
  if (_eventPopupPlotEl) {
    setPlotlyNativeHoverVisible(_eventPopupPlotEl, true);
    _eventPopupPlotEl = null;
  }
  if (!_eventPopupEl) return;
  const h = _eventPopupEl.querySelector(".hm-event-popup__plot--h");
  const rain = _eventPopupEl.querySelector(".hm-event-popup__plot--rain");
  if (h && h.data) Plotly.purge(h);
  if (rain && rain.data) Plotly.purge(rain);
  _eventPopupEl.remove();
  _eventPopupEl = null;
}

function getOrCreateEventHoverPopup() {
  if (_eventPopupEl) return _eventPopupEl;
  const el = document.createElement("div");
  el.id = "hm-event-hover-popup";
  el.className = "hm-event-popup";
  el.innerHTML = `
    <button type="button" class="hm-event-popup__close" aria-label="Close">×</button>
    <div class="hm-event-popup__header"></div>
    <div class="hm-event-popup__plot hm-event-popup__plot--h"></div>
    <div class="hm-event-popup__plot hm-event-popup__plot--rain"></div>`;
  el.style.display = "none";
  document.body.appendChild(el);

  el.querySelector(".hm-event-popup__close")?.addEventListener("click", () => {
    _eventPopupPinned = false;
    el.style.display = "none";
    if (_eventPopupPlotEl) setPlotlyNativeHoverVisible(_eventPopupPlotEl, true);
  });

  el.addEventListener("mouseenter", () => {
    if (_eventPopupHideTimer) {
      clearTimeout(_eventPopupHideTimer);
      _eventPopupHideTimer = null;
    }
    _eventPopupPinned = true;
  });
  el.addEventListener("mouseleave", () => {
    _eventPopupPinned = false;
    _eventPopupHideTimer = setTimeout(() => {
      if (!_eventPopupPinned) el.style.display = "none";
      if (_eventPopupPlotEl) setPlotlyNativeHoverVisible(_eventPopupPlotEl, true);
    }, 280);
  });

  _eventPopupEl = el;
  return el;
}

function positionEventHoverPopup(popup, clientX, clientY) {
  popup.style.display = "block";
  popup.style.visibility = "hidden";
  const rect = popup.getBoundingClientRect();
  const pad = 14;
  let left = clientX + pad;
  let top = clientY + pad;
  if (left + rect.width > window.innerWidth - pad) {
    left = Math.max(pad, clientX - rect.width - pad);
  }
  if (top + rect.height > window.innerHeight - pad) {
    top = Math.max(pad, clientY - rect.height - pad);
  }
  popup.style.left = `${left}px`;
  popup.style.top = `${top}px`;
  popup.style.visibility = "visible";
}

async function showEventHoverPopup(popup, event, daily, clientX, clientY, plotEl) {
  if (_eventPopupHideTimer) {
    clearTimeout(_eventPopupHideTimer);
    _eventPopupHideTimer = null;
  }

  const header = popup.querySelector(".hm-event-popup__header");
  const elH = popup.querySelector(".hm-event-popup__plot--h");
  const elRain = popup.querySelector(".hm-event-popup__plot--rain");
  if (!header || !elH || !elRain) return;

  _eventPopupPlotEl = plotEl || _eventPopupPlotEl;
  if (_eventPopupPlotEl) setPlotlyNativeHoverVisible(_eventPopupPlotEl, false);

  header.innerHTML = eventPopupMetaHtml(event);
  popup.style.display = "block";
  popup.style.visibility = "visible";
  popup.style.left = "0px";
  popup.style.top = "0px";

  const windowDaily = dailyWindowAround(daily, event.day);
  const w = EVENT_POPUP_CHART_WIDTH;
  const hSpec = buildMiniHChartSpec(windowDaily, event, w);
  const rainSpec = buildMiniRainChartSpec(windowDaily, event, w);

  try {
    if (elH.data) {
      await Plotly.react(elH, hSpec.traces, hSpec.layout, PLOTLY_STATIC);
      await Plotly.react(elRain, rainSpec.traces, rainSpec.layout, PLOTLY_STATIC);
    } else {
      await Plotly.newPlot(elH, hSpec.traces, hSpec.layout, PLOTLY_STATIC);
      await Plotly.newPlot(elRain, rainSpec.traces, rainSpec.layout, PLOTLY_STATIC);
    }
  } catch (err) {
    console.error("[HealthMonitor] event popup chart error", err);
  }

  positionEventHoverPopup(popup, clientX, clientY);
}

async function openEventDetailPopup(plotEl, weather, event, ev) {
  if (!event) return;
  const popup = getOrCreateEventHoverPopup();
  const { clientX, clientY } = pointerFromPlotlyEvent(ev);
  _eventPopupPinned = true;
  await showEventHoverPopup(popup, event, weather.daily, clientX, clientY, plotEl);
}

function scheduleHideEventHoverPopup() {
  if (_eventPopupPinned) return;
  if (_eventPopupHideTimer) clearTimeout(_eventPopupHideTimer);
  _eventPopupHideTimer = setTimeout(() => {
    if (_eventPopupPinned || !_eventPopupEl) return;
    if (_eventPopupEl.matches(":hover")) return;
    _eventPopupEl.style.display = "none";
    if (_eventPopupPlotEl) setPlotlyNativeHoverVisible(_eventPopupPlotEl, true);
  }, 280);
}

function eventFromPlotlyPoint(ev, weather) {
  const markerPt = ev.points?.find(
    (pt) => pt.data?.name === "Recovery" || pt.data?.name === "Drop",
  );
  if (markerPt) {
    const day = String(markerPt.x).slice(0, 10);
    return weather.significantEvents.find(
      (e) => e.day === day && e.direction === markerPt.data.name,
    );
  }
  const day = hoverDayFromPlotlyEvent(ev);
  if (!day) return null;
  return weather.significantEvents.find((e) => e.day === day) || null;
}

function wireSignificantEventHover(plotEl, weather) {
  if (!plotEl || !weather?.significantEvents?.length) return;

  plotEl.on("plotly_hover", (ev) => {
    const event = eventFromPlotlyPoint(ev, weather);
    if (!event) {
      if (!_eventPopupPinned) {
        setPlotlyNativeHoverVisible(plotEl, true);
        scheduleHideEventHoverPopup();
      }
      return;
    }
    void openEventDetailPopup(plotEl, weather, event, ev);
  });

  plotEl.on("plotly_unhover", () => {
    if (!_eventPopupPinned) scheduleHideEventHoverPopup();
  });

  plotEl.on("plotly_click", (ev) => {
    const pt = ev.points?.[0];
    const name = pt?.data?.name;
    let event = null;
    if (name === "Recovery" || name === "Drop") {
      const day = String(pt.x).slice(0, 10);
      event = weather.significantEvents.find(
        (e) => e.day === day && e.direction === name,
      );
    } else {
      event = eventFromPlotlyPoint(ev, weather);
    }
    if (!event) return;
    void openEventDetailPopup(plotEl, weather, event, ev);
  });
}

function buildSoilingHChart(weather, width) {
  const { daily, significantEvents, chartSegments } = weather;
  const days = daily.map((d) => d.day);
  const theme = plotlyDarkTheme();
  const hVals = daily.map((d) => d.H).filter(Number.isFinite);
  const hMin = hVals.length ? Math.min(...hVals) : 0;
  const hMax = hVals.length ? Math.max(...hVals) : 1;
  const yPad = (hMax - hMin) * 0.08 || 0.05;

  const traces = [
    {
      x: days,
      y: daily.map((d) => d.H),
      type: "scatter",
      mode: "lines",
      name: "Daily H",
      line: { color: "#22c55e", width: 1 },
      opacity: 0.35,
      hovertemplate: "%{x}<br>H = %{y:.3f}<extra></extra>",
    },
    {
      x: days,
      y: daily.map((d) => d.Hsmooth),
      type: "scatter",
      mode: "lines",
      name: "Smoothed H",
      line: { color: "#e2e8f0", width: 2 },
      hovertemplate: "%{x}<br>Smoothed H = %{y:.3f}<extra></extra>",
    },
  ];

  for (const seg of chartSegments) {
    traces.push({
      x: seg.slice.map((d) => d.day),
      y: seg.slice.map((_, j) => seg.intercept + seg.slope * j),
      type: "scatter",
      mode: "lines",
      name: "Soiling trend",
      line: { color: "#fb923c", width: 2, dash: "dot" },
      showlegend: false,
      hoverinfo: "skip",
    });
  }

  if (significantEvents.length) {
    const recoveries = significantEvents.filter((e) => e.direction === "Recovery");
    const drops = significantEvents.filter((e) => e.direction === "Drop");
    if (recoveries.length) {
      traces.push({
        x: recoveries.map((e) => e.day),
        y: recoveries.map(() => hMax + yPad * 0.5),
        type: "scatter",
        mode: "markers",
        name: "Recovery",
        marker: {
          size: 11,
          symbol: "triangle-up",
          color: recoveries.map((e) => e.color),
          line: { width: 1, color: "#0f172a" },
        },
        hoverinfo: "none",
      });
    }
    if (drops.length) {
      traces.push({
        x: drops.map((e) => e.day),
        y: drops.map(() => hMin - yPad * 0.5),
        type: "scatter",
        mode: "markers",
        name: "Drop",
        marker: {
          size: 11,
          symbol: "triangle-down",
          color: drops.map((e) => e.color),
          line: { width: 1, color: "#0f172a" },
        },
        hoverinfo: "none",
      });
    }
  }

  return {
    traces,
    layout: {
      ...theme,
      autosize: false,
      width,
      height: 340,
      title: {
        text: "Health ratio (H) — smoothed trend & significant changes",
        font: { color: "#e2e8f0", size: 14 },
      },
      margin: { t: 48, r: 20, b: 44, l: 56 },
      xaxis: { ...theme.xaxis, type: "date", title: "Date" },
      yaxis: {
        ...theme.yaxis,
        title: "H",
        range: [hMin - yPad, hMax + yPad * 1.2],
      },
      hovermode: "closest",
      legend: { orientation: "h", y: 1.12, x: 0, font: { size: 10 } },
    },
  };
}

function buildRainChart(weather, width) {
  const { daily } = weather;
  const days = daily.map((d) => d.day);
  const theme = plotlyDarkTheme();

  return {
    traces: [
      {
        x: days,
        y: daily.map((d) => d.rain),
        type: "bar",
        name: "Rain (mm/day)",
        marker: { color: "#3b82f6" },
        hovertemplate: "%{x}<br>Rain = %{y:.2f} mm<extra></extra>",
      },
      {
        x: [days[0], days[days.length - 1]],
        y: [RAIN_CLEAN_THRESHOLD, RAIN_CLEAN_THRESHOLD],
        type: "scatter",
        mode: "lines",
        name: `Clean threshold (${RAIN_CLEAN_THRESHOLD} mm)`,
        line: { color: "#f87171", width: 1.5, dash: "dash" },
        hoverinfo: "skip",
      },
    ],
    layout: {
      ...theme,
      autosize: false,
      width,
      height: 170,
      title: { text: "Daily rainfall", font: { color: "#e2e8f0", size: 12 } },
      margin: { t: 32, r: 16, b: 36, l: 52 },
      xaxis: { ...theme.xaxis, type: "date", showticklabels: false },
      yaxis: { ...theme.yaxis, title: "mm/day" },
      bargap: 0.15,
      hovermode: "x unified",
      showlegend: false,
    },
  };
}

function buildWindChart(weather, width) {
  const { daily } = weather;
  const days = daily.map((d) => d.day);
  const theme = plotlyDarkTheme();

  return {
    traces: [
      {
        x: days,
        y: daily.map((d) => d.windMax),
        type: "scatter",
        mode: "lines",
        name: "Wind max (m/s)",
        line: { color: "#22d3ee", width: 1.2 },
        connectgaps: false,
        hovertemplate: "%{x}<br>Wind max = %{y:.2f} m/s<extra></extra>",
      },
      {
        x: [days[0], days[days.length - 1]],
        y: [WIND_CLEAN_THRESHOLD, WIND_CLEAN_THRESHOLD],
        type: "scatter",
        mode: "lines",
        name: `Threshold (${WIND_CLEAN_THRESHOLD} m/s)`,
        line: { color: "#f87171", width: 1.5, dash: "dash" },
        hoverinfo: "skip",
      },
    ],
    layout: {
      ...theme,
      autosize: false,
      width,
      height: 170,
      title: { text: "Daily max wind speed", font: { color: "#e2e8f0", size: 12 } },
      margin: { t: 32, r: 16, b: 36, l: 52 },
      xaxis: { ...theme.xaxis, type: "date", showticklabels: false },
      yaxis: { ...theme.yaxis, title: "m/s" },
      hovermode: "x unified",
      showlegend: false,
    },
  };
}

function buildTempChart(weather, width) {
  const { daily } = weather;
  const days = daily.map((d) => d.day);
  const theme = plotlyDarkTheme();

  return {
    traces: [
      {
        x: days,
        y: daily.map((d) => d.tempMean),
        type: "scatter",
        mode: "lines",
        name: "Mean temp (°C)",
        line: { color: "#f97316", width: 1.2 },
        connectgaps: false,
        hovertemplate: "%{x}<br>Temp = %{y:.1f} °C<extra></extra>",
      },
    ],
    layout: {
      ...theme,
      autosize: false,
      width,
      height: 170,
      title: { text: "Daylight mean temperature", font: { color: "#e2e8f0", size: 12 } },
      margin: { t: 32, r: 16, b: 36, l: 52 },
      xaxis: { ...theme.xaxis, type: "date", showticklabels: false },
      yaxis: { ...theme.yaxis, title: "°C" },
      hovermode: "x unified",
      showlegend: false,
    },
  };
}

function buildGhiChart(weather, width) {
  const { daily } = weather;
  const days = daily.map((d) => d.day);
  const theme = plotlyDarkTheme();

  return {
    traces: [
      {
        x: days,
        y: daily.map((d) => d.ghiSum),
        type: "scatter",
        mode: "lines",
        name: "GHI sum",
        line: { color: "#fbbf24", width: 1.2 },
        fill: "tozeroy",
        fillcolor: "rgba(251,191,36,0.08)",
        hovertemplate: "%{x}<br>GHI sum = %{y:.0f} W/m²·h<extra></extra>",
      },
    ],
    layout: {
      ...theme,
      autosize: false,
      width,
      height: 170,
      title: { text: "Daylight GHI (solar resource)", font: { color: "#e2e8f0", size: 12 } },
      margin: { t: 32, r: 16, b: 44, l: 52 },
      xaxis: { ...theme.xaxis, type: "date", title: "Date" },
      yaxis: { ...theme.yaxis, title: "W/m²·h sum" },
      hovermode: "x unified",
      showlegend: false,
    },
  };
}

function plotWeatherPanels(weather, ids, width, windAvailable) {
  const elH = document.getElementById(ids.h);
  const elRain = document.getElementById(ids.rain);
  const elWind = windAvailable ? document.getElementById(ids.wind) : null;
  const elTemp = document.getElementById(ids.temp);
  const elGhi = document.getElementById(ids.ghi);
  if (!elH || !elRain || !elTemp || !elGhi) return [];

  const hSpec = buildSoilingHChart(weather, width);
  Plotly.newPlot(elH, hSpec.traces, hSpec.layout, PLOTLY_STATIC);
  wireSignificantEventHover(elH, weather);
  const rainSpec = buildRainChart(weather, width);
  Plotly.newPlot(elRain, rainSpec.traces, rainSpec.layout, PLOTLY_STATIC);
  const tempSpec = buildTempChart(weather, width);
  Plotly.newPlot(elTemp, tempSpec.traces, tempSpec.layout, PLOTLY_STATIC);
  const ghiSpec = buildGhiChart(weather, width);
  Plotly.newPlot(elGhi, ghiSpec.traces, ghiSpec.layout, PLOTLY_STATIC);

  const linked = [elH, elRain, elTemp, elGhi];
  if (elWind) {
    const windSpec = buildWindChart(weather, width);
    Plotly.newPlot(elWind, windSpec.traces, windSpec.layout, PLOTLY_STATIC);
    linked.splice(1, 0, elWind);
  }
  wireLinkedXAxes(linked);
  return linked;
}

function statCard(label, value) {
  return `<div style="flex:1 1 180px;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px 16px;min-width:0">
    <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px">${label}</div>
    <div style="font-size:1rem;color:#e2e8f0;font-weight:600">${value}</div>
  </div>`;
}

function soilingStatsHtml(summary) {
  const meanPct = Number.isFinite(summary.meanSoilingPct)
    ? `${summary.meanSoilingPct.toFixed(3)} %/day`
    : "—";
  const causeParts = Object.entries(summary.byCause || {})
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${v} ${k.split(" ")[0].toLowerCase()}`)
    .join(" · ");
  const eventsLine = summary.totalEvents
    ? `${summary.totalEvents} significant (${causeParts || "see table"})`
    : "None detected";
  const energyKwh = Math.round(summary.totalEnergyLostKwh).toLocaleString();
  const energyPct = Number.isFinite(summary.pctEnergyLost)
    ? summary.pctEnergyLost.toFixed(2)
    : "—";
  const topCorr = (summary.correlations || [])
    .slice()
    .sort((a, b) => Math.abs(b.r) - Math.abs(a.r))[0];
  const corrLine = topCorr
    ? `${topCorr.factor} (r = ${topCorr.r.toFixed(2)})`
    : "—";

  return `<div style="display:flex;flex-wrap:wrap;gap:10px;margin:16px 0">
    ${statCard("Mean soiling rate", meanPct)}
    ${statCard("Significant H changes", eventsLine)}
    ${statCard("Strongest daily driver", corrLine)}
    ${statCard("Energy lost to soiling", `${energyKwh} kWh (${energyPct}%)`)}
  </div>`;
}

function driverCorrelationsTableHtml(correlations) {
  if (!correlations?.length) {
    return `<tr><td colspan="3" style="color:#94a3b8;text-align:center">Not enough weather variation to estimate correlations.</td></tr>`;
  }
  return correlations
    .slice()
    .sort((a, b) => Math.abs(b.r) - Math.abs(a.r))
    .map((c) => {
      const abs = Math.abs(c.r);
      let meaning =
        abs < 0.15
          ? "Weak link to day-to-day H moves"
          : c.r > 0
            ? "Higher values tend to coincide with H increases"
            : "Higher values tend to coincide with H decreases";
      return `<tr>
        <td>${c.factor}</td>
        <td>${c.r.toFixed(2)}</td>
        <td style="color:#94a3b8">${meaning}</td>
      </tr>`;
    })
    .join("");
}

function significantEventsTableHtml(events) {
  if (!events.length) {
    return `<tr><td colspan="7" style="color:#94a3b8;text-align:center">No significant H changes detected at current thresholds.</td></tr>`;
  }
  return events
    .map(
      (e) => `<tr>
        <td>${e.day}</td>
        <td>${e.direction}</td>
        <td>${driverPillHtml(e.cause)}</td>
        <td>${e.jump >= 0 ? "+" : ""}${e.jump.toFixed(3)}</td>
        <td>${e.rainAmt.toFixed(2)}</td>
        <td>${Number.isFinite(e.windAmt) ? e.windAmt.toFixed(2) : "—"}</td>
        <td>${Number.isFinite(e.tempMean) ? e.tempMean.toFixed(1) : "—"}</td>
      </tr>`,
    )
    .join("");
}

function soilingSegmentsTableHtml(segments) {
  if (!segments.length) {
    return `<tr><td colspan="5" style="color:#94a3b8;text-align:center">No dry spells long enough to fit a soiling slope.</td></tr>`;
  }
  return segments
    .map(
      (s) => `<tr>
        <td>${s.startDay}</td>
        <td>${s.endDay}</td>
        <td>${s.days}</td>
        <td>${Number.isFinite(s.pctPerDay) ? s.pctPerDay.toFixed(3) : "—"}</td>
        <td>${s.hLost.toFixed(3)}</td>
      </tr>`,
    )
    .join("");
}

function weatherInsightNote(summary) {
  const { pearsonR: r, byCause, hasWind } = summary;
  const parts = [];
  if ((summary.correlations || []).length) {
    const top = summary.correlations
      .slice()
      .sort((a, b) => Math.abs(b.r) - Math.abs(a.r))[0];
    parts.push(
      `Day-to-day H moves correlate most with <strong>${top.factor}</strong> (r = ${top.r.toFixed(2)}).`,
    );
  }
  if ((byCause.Rain || 0) + (byCause.Wind || 0) > 0) {
    parts.push(
      `Significant recoveries attributed to rain (${byCause.Rain || 0}) and wind (${byCause.Wind || 0}) where weather supported it.`,
    );
  }
  if ((byCause["Heat stress"] || 0) + (byCause["Cloud / low GHI"] || 0) > 0) {
    parts.push(
      `Drops also track heat (${byCause["Heat stress"] || 0}) and cloud/low-GHI days (${byCause["Cloud / low GHI"] || 0}) — not only soiling.`,
    );
  }
  if (!hasWind) {
    parts.push(
      "Wind/cloud columns are read from the hourly master CSV (Solcast: <code>wind_speed_10m</code>, <code>cloud_opacity</code>) when present.",
    );
  }
  if (Number.isFinite(r) && (byCause.Rain || 0) >= 2) {
    parts.push(`Rain amount vs recovery size: Pearson r = ${r.toFixed(2)}.`);
  }
  if (!parts.length) {
    return `<p style="font-size:0.82rem;color:#94a3b8;margin:8px 0 0">Weather drivers: insufficient variation for a summary at current thresholds.</p>`;
  }
  return `<p style="font-size:0.82rem;color:#94a3b8;margin:8px 0 0">${parts.join(" ")}</p>`;
}

function buildEwmaChart(analysis, width) {
  const { series, mu, LCL, events } = analysis;
  const days = series.map((d) => d.day);
  const theme = plotlyDarkTheme();

  const shapes = events.map((ev) => ({
    type: "rect",
    xref: "x",
    yref: "paper",
    x0: ev.startDay,
    x1: addDaysIso(ev.endDay, 1),
    y0: 0,
    y1: 1,
    fillcolor: "rgba(248,113,113,0.12)",
    line: { width: 0 },
  }));

  return {
    traces: [
      {
        x: days,
        y: series.map((d) => d.residual),
        type: "scatter",
        mode: "markers",
        name: "Daily residual",
        marker: { color: "#64748b", size: 4, opacity: 0.5 },
        hovertemplate: "%{x}<br>Residual = %{y:.3f}<extra></extra>",
      },
      {
        x: days,
        y: series.map((d) => d.ewma),
        type: "scatter",
        mode: "lines",
        name: "EWMA",
        line: { color: "#3b82f6", width: 2 },
        hovertemplate: "%{x}<br>EWMA = %{y:.3f}<extra></extra>",
      },
      {
        x: [days[0], days[days.length - 1]],
        y: [LCL, LCL],
        type: "scatter",
        mode: "lines",
        name: "Control limit (LCL)",
        line: { color: "#f87171", width: 1.5, dash: "dash" },
        hoverinfo: "skip",
      },
      {
        x: [days[0], days[days.length - 1]],
        y: [mu, mu],
        type: "scatter",
        mode: "lines",
        name: "Center (μ)",
        line: { color: "#94a3b8", width: 1, dash: "dot" },
        hoverinfo: "skip",
      },
    ],
    layout: {
      ...theme,
      autosize: false,
      width,
      height: 380,
      title: {
        text: "EWMA control chart — H residual (alarms = sustained drop below limit)",
        font: { color: "#e2e8f0", size: 14 },
      },
      margin: { t: 48, r: 20, b: 52, l: 56 },
      xaxis: { ...theme.xaxis, type: "date", title: "Date" },
      yaxis: { ...theme.yaxis, title: "Residual (H − seasonal baseline)" },
      shapes,
      hovermode: "x unified",
      legend: { orientation: "h", y: 1.08, x: 0, font: { size: 10 } },
    },
  };
}

function alarmTableHtml(events) {
  if (!events.length) {
    return `<tr><td colspan="6" style="color:#94a3b8;text-align:center">
      No alarms detected — system within control limits.</td></tr>`;
  }
  return events
    .map(
      (ev) => `<tr>
        <td>${ev.startDay}</td>
        <td>${ev.endDay}</td>
        <td>${ev.durationDays}</td>
        <td>${Number.isFinite(ev.minEWMA) ? ev.minEWMA.toFixed(3) : "—"}</td>
        <td>${classPillHtml(ev.kind, ev.label)}</td>
        <td>${ev.action}</td>
      </tr>`,
    )
    .join("");
}

/**
 * @param {HTMLElement} container
 * @param {Array} hourly
 * @param {object} [site]
 */
export function renderHealthMonitor(container, hourly, site = {}) {
  purgePlotlyInContainer(container);
  removeEventHoverPopup();

  if (!hourly?.length) {
    container.innerHTML =
      `<p style="color:#94a3b8;padding:2rem">No hourly data loaded.</p>`;
    return;
  }

  const daily = computeDailySeries(hourly);
  if (daily.length < 10) {
    container.innerHTML =
      `<p style="color:#94a3b8;padding:2rem">Not enough daylight days for health monitoring.</p>`;
    return;
  }

  const weatherAvailable = hasWeatherFields(hourly);
  const analysis = computeAnalysis(daily, weatherAvailable);
  if (!analysis) {
    container.innerHTML =
      `<p style="color:#94a3b8;padding:2rem">Unable to compute health analysis.</p>`;
    return;
  }

  const lastDay = daily[daily.length - 1].day;
  const status = recentStatus(analysis.events, analysis.series, lastDay);

  const weatherDaily = computeWeatherDaily(hourly);
  const weather = computeWeatherImpactAnalysis(weatherDaily);

  const idBanner = nextPlotDomId("hm-banner");
  const idEwma = nextPlotDomId("hm-ewma");
  const idSoilH = nextPlotDomId("hm-soil-h");
  const idSoilRain = nextPlotDomId("hm-soil-rain");
  const idSoilWind = nextPlotDomId("hm-soil-wind");
  const idSoilTemp = nextPlotDomId("hm-soil-temp");
  const idSoilGhi = nextPlotDomId("hm-soil-ghi");
  const idWeatherStats = nextPlotDomId("hm-wx-stats");
  const idWeatherTables = nextPlotDomId("hm-wx-tables");
  const gen = (container.__hmGen = (container.__hmGen || 0) + 1);

  container.innerHTML = `
    <div style="padding:16px">
      <h2>${siteHeading("Health monitor (EWMA)", site)}</h2>
      <p class="note">
        Detects abnormal sustained underperformance after removing seasonal H patterns.
        Alarms trigger when the EWMA of the residual drops below the lower control limit (LCL).
      </p>
      <div id="${idBanner}" style="background:#1e293b;border:1px solid ${status.color};
        border-radius:10px;padding:14px 18px;margin:12px 0 0">
        <div style="color:${status.color};font-weight:700;font-size:1.05rem">${status.title}</div>
        <div style="color:#94a3b8;font-size:0.85rem;margin-top:4px">${status.summary}</div>
      </div>
      ${howItWorksHtml()}
      <div id="${idEwma}" class="chart-box" style="height:380px;min-height:380px;margin-bottom:16px;width:100%;overflow:hidden"></div>

      <h3 style="font-size:0.92rem;color:#cbd5e1;margin:24px 0 0">Weather &amp; performance analysis</h3>
      ${soilingHowItWorksHtml()}
      <p class="note" style="margin:8px 0 12px">Linked panels — pan/zoom any chart to align H changes with rain, wind, temperature, and solar resource.
        Hover or click a ▲/▼ marker on the H chart for a 7-day rain + H detail popup.</p>
      <div id="${idSoilH}" class="chart-box" style="height:340px;min-height:340px;margin-bottom:4px;width:100%;overflow:hidden"></div>
      <div id="${idSoilRain}" class="chart-box" style="height:170px;min-height:170px;margin-bottom:4px;width:100%;overflow:hidden"></div>
      <div id="${idSoilWind}" class="chart-box" style="height:170px;min-height:170px;margin-bottom:4px;width:100%;overflow:hidden"></div>
      <div id="${idSoilTemp}" class="chart-box" style="height:170px;min-height:170px;margin-bottom:4px;width:100%;overflow:hidden"></div>
      <div id="${idSoilGhi}" class="chart-box" style="height:170px;min-height:170px;margin-bottom:8px;width:100%;overflow:hidden"></div>
      <div id="${idWeatherStats}">${soilingStatsHtml(weather.summary)}</div>
      <div id="${idWeatherTables}">
        <h4 style="font-size:0.88rem;color:#cbd5e1;margin:8px 0 8px">What drives day-to-day H change</h4>
        <div style="overflow-x:auto;margin-bottom:16px">
          <table class="data-table">
            <thead><tr><th>Weather factor</th><th>Pearson r vs ΔH</th><th>Interpretation</th></tr></thead>
            <tbody>${driverCorrelationsTableHtml(weather.summary.correlations)}</tbody>
          </table>
        </div>
        <h4 style="font-size:0.88rem;color:#cbd5e1;margin:8px 0 8px">Significant H changes</h4>
        <div style="overflow-x:auto;margin-bottom:16px">
          <table class="data-table">
            <thead>
              <tr>
                <th>Date</th><th>Type</th><th>Likely driver</th><th>ΔH</th>
                <th>Rain (mm)</th><th>Wind max</th><th>Temp (°C)</th>
              </tr>
            </thead>
            <tbody>${significantEventsTableHtml(weather.significantEvents)}</tbody>
          </table>
        </div>
        <h4 style="font-size:0.88rem;color:#cbd5e1;margin:8px 0 8px">Soiling segments (dry spells)</h4>
        <div style="overflow-x:auto;margin-bottom:8px">
          <table class="data-table">
            <thead>
              <tr>
                <th>Start</th><th>End</th><th>Days</th><th>Soiling rate (%/day)</th><th>H lost over segment</th>
              </tr>
            </thead>
            <tbody>${soilingSegmentsTableHtml(weather.segments)}</tbody>
          </table>
        </div>
        ${weatherInsightNote(weather.summary)}
      </div>

      <h3 style="font-size:0.92rem;color:#cbd5e1;margin:24px 0 8px">Alarm log</h3>
      ${classificationLegendHtml()}
      <div style="overflow-x:auto">
        <table class="data-table">
          <thead>
            <tr>
              <th>Start</th><th>End</th><th>Duration (days)</th>
              <th>Min EWMA</th><th>Classification</th><th>Recommended action</th>
            </tr>
          </thead>
          <tbody>${alarmTableHtml(analysis.events)}</tbody>
        </table>
      </div>
      <p class="note" style="margin-top:12px">
        Baseline: first <strong>${REFERENCE_MONTHS}</strong> months per season (median H);
        EWMA λ=<strong>${EWMA_LAMBDA}</strong>, control multiplier L=<strong>${EWMA_L}</strong>.
        Only <strong>downward</strong> EWMA breaches (below LCL) are flagged as alarms.
        ${weatherAvailable ? "" : " Rain/temperature columns not found in hourly data — events are unclassified."}
      </p>
    </div>`;

  setTimeout(async () => {
    if (!container.isConnected || container.__hmGen !== gen) return;

    const elEwma = document.getElementById(idEwma);
    if (!elEwma) return;

    void container.offsetWidth;
    const w = Math.max(480, measurePlotWidth(elEwma, container));

    try {
      const ewma = buildEwmaChart(analysis, w);
      Plotly.newPlot(elEwma, ewma.traces, ewma.layout, PLOTLY_STATIC);

      let wxDaily = computeWeatherDaily(hourly);
      try {
        const rows = await fetchWeatherEnrichmentRows(site);
        enrichDailyWithRawWeather(wxDaily, rows);
      } catch (fetchErr) {
        console.warn("[HealthMonitor] weather enrichment not loaded", fetchErr);
      }

      const wx = computeWeatherImpactAnalysis(wxDaily);
      const windOk = wx.summary.hasWind;

      const statsEl = document.getElementById(idWeatherStats);
      const tablesEl = document.getElementById(idWeatherTables);
      if (statsEl) statsEl.innerHTML = soilingStatsHtml(wx.summary);
      if (tablesEl) {
        tablesEl.innerHTML = `
          <h4 style="font-size:0.88rem;color:#cbd5e1;margin:8px 0 8px">What drives day-to-day H change</h4>
          <div style="overflow-x:auto;margin-bottom:16px">
            <table class="data-table">
              <thead><tr><th>Weather factor</th><th>Pearson r vs ΔH</th><th>Interpretation</th></tr></thead>
              <tbody>${driverCorrelationsTableHtml(wx.summary.correlations)}</tbody>
            </table>
          </div>
          <h4 style="font-size:0.88rem;color:#cbd5e1;margin:8px 0 8px">Significant H changes</h4>
          <div style="overflow-x:auto;margin-bottom:16px">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Date</th><th>Type</th><th>Likely driver</th><th>ΔH</th>
                  <th>Rain (mm)</th><th>Wind max</th><th>Temp (°C)</th>
                </tr>
              </thead>
              <tbody>${significantEventsTableHtml(wx.significantEvents)}</tbody>
            </table>
          </div>
          <h4 style="font-size:0.88rem;color:#cbd5e1;margin:8px 0 8px">Soiling segments (dry spells)</h4>
          <div style="overflow-x:auto;margin-bottom:8px">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Start</th><th>End</th><th>Days</th><th>Soiling rate (%/day)</th><th>H lost over segment</th>
                </tr>
              </thead>
              <tbody>${soilingSegmentsTableHtml(wx.segments)}</tbody>
            </table>
          </div>
          ${weatherInsightNote(wx.summary)}`;
      }

      const elWind = document.getElementById(idSoilWind);
      if (elWind && !windOk) {
        elWind.innerHTML = `<p style="color:#94a3b8;padding:1.2rem;font-size:0.85rem;text-align:center">No wind column in hourly master CSV (expected <code>wind_speed_10m</code> from Solcast).</p>`;
        elWind.style.height = "80px";
        elWind.style.minHeight = "80px";
      }

      plotWeatherPanels(
        wx,
        { h: idSoilH, rain: idSoilRain, wind: idSoilWind, temp: idSoilTemp, ghi: idSoilGhi },
        w,
        windOk,
      );
    } catch (err) {
      console.error("[HealthMonitor] plot error", err);
      elEwma.innerHTML = `<p style="color:#f87171;padding:1rem;font-size:0.85rem">${err.message}</p>`;
    }
  }, 150);
}
