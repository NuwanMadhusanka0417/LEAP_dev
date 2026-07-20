/**
 * Health monitor - EWMA control chart on seasonal H residuals (predictive maintenance).
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
import { siteHeading, fetchHourlyMasterText, tryFetchHsuSoilingText } from "./meters.js";
import { seasonOf } from "./season_analysis_common.js";

const HM_PLOTLY_CONFIG = { ...PLOTLY_STATIC, scrollZoom: false };
// -- Tunable constants --
const REFERENCE_MONTHS = 24;
const EWMA_LAMBDA = 0.1;
const EWMA_L = 3;
const EWMA_RAIN_CLEAN_THRESHOLD = 0.5; // mm/day  - EWMA alarm soiling classification
const RECOVERY_WINDOW_DAYS = 14;
const ALARM_BRIDGE_DAYS = 3;
const STATUS_LOOKBACK_DAYS = 30;
const RAIN_FIELD = "rain_mm";
const TEMP_FIELD = "temp_c";
// Kimber soiling model (pvlib-style, partial rain cleaning, calibrated to confirmed clean)
const KNOWN_CLEANINGS = ["2025-11-28"];
const SOILING_RATE_PCT_PER_DAY = 0.0; // 0 = auto-calibrate from longest dry spell (%/day)
const RAIN_FULL_CLEAN_MM = 5.0; // mm/day for full clean — tune using confirmed-clean H step
const RAIN_MIN_CLEAN_MM = 0.3; // below this, rain has no cleaning effect
const PARTIAL_CLEAN = true;
const WIND_CLEAN_THRESHOLD = 10; // m/s daily max — secondary cleaning
const WIND_CLEAN_FRACTION = 0.3; // partial recovery toward SR=1 on wind clean
const GRACE_PERIOD_DAYS = 3;
const ENERGY_TARIFF = 0.25; // $/kWh
const CLEANING_COST = 300; // $ per manual clean
const PROJECTION_DAYS = [7, 14, 30];
const KIMBER_RAIN_KEYS = ["precipitation_rate", "rain_mm", "precipitation_mm"];
const KIMBER_WIND_KEYS = ["wind_speed_10m", "wind_ms", "wind_speed_100m"];
const KIMBER_TEMP_KEYS = ["air_temp", "temp_c", "temp_cell_c"];
// pvlib HSU soiling (Bundoora library — precomputed in 0_download_data.py)
const HSU_CLEANING_THRESHOLD_MM = 0.5;
// Weather & performance analysis
const WIND_FIELD = "wind_ms";
const WX_RAIN_CLEAN_THRESHOLD = 0.5;
const WX_WIND_CLEAN_THRESHOLD = 6;
const WX_CLEAN_JUMP_THRESHOLD = 0.08;
const WX_SMOOTH_WINDOW_DAYS = 3;
const WX_SIGNIFICANT_JUMP_THRESHOLD = 0.12;
const WX_MIN_DAYS_BETWEEN_EVENTS = 7;
const WX_SOILING_MIN_SEGMENT_DAYS = 5;
const EVENT_POPUP_RADIUS_DAYS = 3;
const EVENT_POPUP_CHART_WIDTH = 340;
const MAX_CHART_RECOVERY_MARKERS = 35;
const MAX_SOILING_LINES_ON_CHART = 8;
const CLASS = {
  unclassified: {
    label: "Unclassified (no weather data)",
    color: "#94a3b8",
    action: "Review manually  - weather not available",
  },
  soiling: {
    label: "Soiling (rain-recovered)",
    color: "#3b82f6",
    action: "No action needed",
  },
  thermal: {
    label: "Thermal (heat-related)",
    color: "#fbbf24",
    action: "Expected  - monitor",
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
      text: " - dust/pollen buildup; cleared by rain. No action.",
    },
    {
      kind: "thermal",
      pill: "Thermal (heat-related)",
      text: " - hot-weather efficiency loss. Recovers when cool.",
    },
    {
      kind: "fault",
      pill: "Candidate fault",
      text: " - unexplained, no recovery. Inspect system.",
    },
    {
      kind: "unclassified",
      pill: "Unclassified",
      text: " - no weather data to classify.",
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
    <summary style="cursor:pointer;padding:12px 0;font-size:0.9rem;color:#cbd5e1;font-weight:500">How this works  - chart, alarms &amp; classification</summary>
    <div style="padding:0 0 14px 0;font-size:0.82rem;color:#94a3b8;line-height:1.6">
      <p style="margin:0 0 4px;font-size:0.78rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em">Chart elements</p>
      <p style="margin:0 0 10px"><strong>Y-axis  - Residual (H - seasonal baseline)</strong><br>
        H is the daily health ratio: total actual generation divided by total PVLib-expected
        generation over daylight hours (GHI &gt; 5 W/m2). The seasonal baseline is the median H
        for that season during the healthy reference window (first ${REFERENCE_MONTHS} months).
        Residual = today's H minus that baseline. Zero means the system matches its healthy
        seasonal norm; negative means it is underperforming.</p>
      <hr style="border:none;border-top:1px solid #334155;margin:10px 0">
      <p style="margin:0 0 10px"><strong>X-axis  - Date</strong><br>
        One point per day, in calendar order across the full dataset.</p>
      <hr style="border:none;border-top:1px solid #334155;margin:10px 0">
      <p style="margin:0 0 10px"><strong>Gray dots  - Daily residual</strong><br>
        The raw daily residual. Noisy by nature, so individual days are not alarmed on.</p>
      <hr style="border:none;border-top:1px solid #334155;margin:10px 0">
      <p style="margin:0 0 10px"><strong>Blue line  - EWMA</strong><br>
        Exponentially-weighted moving average of the residual:
        EWMA(today) = lambda * residual(today) + (1-lambda) * EWMA(yesterday), with lambda = ${EWMA_LAMBDA}.
        It smooths daily noise but bends down when a drop is sustained, which is what a real
        fault looks like.</p>
      <hr style="border:none;border-top:1px solid #334155;margin:10px 0">
      <p style="margin:0 0 10px"><strong>Gray dotted line  - Center (mu)</strong><br>
        Mean residual during the reference window. The EWMA sits near here when healthy.</p>
      <hr style="border:none;border-top:1px solid #334155;margin:10px 0">
      <p style="margin:0 0 10px"><strong>Red dashed line  - Lower control limit (LCL)</strong><br>
        LCL = mu - L * sigma * sqrt(lambda / (2-lambda)), with L = ${EWMA_L} and sigma = standard deviation of residuals
        in the reference window. When the EWMA drops below this line the underperformance is
        too large and too sustained to be normal variation  - an alarm.</p>
      <hr style="border:none;border-top:1px solid #334155;margin:10px 0">
      <p style="margin:0 0 10px"><strong>Red shaded bands  - Alarm events</strong><br>
        Periods where the EWMA stayed below the LCL. Each band is one alarm event in the log
        below. Only downward breaches are flagged; rises above the baseline are not a maintenance concern.</p>
      <hr style="border:none;border-top:1px solid #475569;margin:16px 0">
      <p style="margin:0 0 4px;font-size:0.78rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em">How alarms are captured</p>
      <p style="margin:0 0 10px">
        <strong>Step 1  - Daily health signal.</strong> Hourly meter data is rolled up to one row
        per day (daylight hours only, GHI &gt; 5 W/m2): actual kWh, expected kWh, daily H, and
        optional daily rainfall total and mean temperature (from <code>${RAIN_FIELD}</code> /
        <code>${TEMP_FIELD}</code> or equivalent columns in the hourly CSV).
      </p>
      <p style="margin:0 0 10px">
        <strong>Step 2  - Seasonal baseline &amp; residual.</strong> For each day, subtract the
        median H for that calendar season from the first ${REFERENCE_MONTHS} months of data.
        That residual removes normal summer/winter swing so we only watch deviation from the
        system's own healthy seasonal norm.
      </p>
      <p style="margin:0 0 10px">
        <strong>Step 3  - EWMA alarm days.</strong> Any day whose EWMA (lambda = ${EWMA_LAMBDA}) falls
        below the LCL is marked as an alarm day. Single noisy days rarely trigger this; a
        sustained drop does.
      </p>
      <p style="margin:0 0 10px">
        <strong>Step 4  - Group into events.</strong> Consecutive alarm days are merged into one
        event. Gaps of up to ${ALARM_BRIDGE_DAYS} days without an alarm are bridged so brief
        recoveries do not split one real incident into many rows.
      </p>
      <p style="margin:0 0 10px">
        <strong>Step 5  - Classify each event.</strong> After the event ends, a
        ${RECOVERY_WINDOW_DAYS}-day recovery window is checked. Rainfall and temperature in that
        window (plus the event itself) decide whether the drop is soiling, thermal, a candidate
        fault, or unclassified. Results appear in the alarm log with start/end dates, duration,
        minimum EWMA during the event, classification pill, and recommended action.
      </p>
      <hr style="border:none;border-top:1px solid #475569;margin:16px 0">
      <p style="margin:0 0 10px;font-size:0.78rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em">Alarm types  - detection, data used &amp; what to do</p>
      <div style="margin:0 0 12px;padding:10px 12px;background:#0f172a;border-radius:6px;border:1px solid #334155">
        <div style="margin-bottom:6px">${classPillHtml("soiling", CLASS.soiling.label)}</div>
        <p style="margin:0 0 6px"><strong>How detected:</strong> EWMA stayed below LCL during
        the event, then returned to normal (EWMA &gt;= LCL) within ${RECOVERY_WINDOW_DAYS} days
        <em>after</em> the event ended, and at least one day in the event + recovery window had
        daily rainfall &gt;= ${EWMA_RAIN_CLEAN_THRESHOLD} mm (typical panel wash-off).</p>
        <p style="margin:0 0 6px"><strong>Data used:</strong> Daily H residual &amp; EWMA; daily
        rainfall total (<code>${RAIN_FIELD}</code>); recovery timing vs LCL.</p>
        <p style="margin:0"><strong>What to do:</strong> ${CLASS.soiling.action}. Dust or pollen
        buildup that rain cleared  - not a hardware fault. Log for records; no O&amp;M dispatch
        unless the pattern repeats without rain recovery.</p>
      </div>
      <div style="margin:0 0 12px;padding:10px 12px;background:#0f172a;border-radius:6px;border:1px solid #334155">
        <div style="margin-bottom:6px">${classPillHtml("thermal", CLASS.thermal.label)}</div>
        <p style="margin:0 0 6px"><strong>How detected:</strong> EWMA below LCL during the event;
        mean temperature during the event was in the hottest ~20% of all days in the dataset
        (&gt;= 80th percentile); EWMA recovered to &gt;= LCL within ${RECOVERY_WINDOW_DAYS} days after
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
        (Steps 1-4 above), but hourly data has no usable rainfall or temperature fields, so
        automatic soiling/thermal rules cannot run.</p>
        <p style="margin:0 0 6px"><strong>Data used:</strong> H, EWMA, and LCL only  - weather
        columns missing or empty in <code>hourly_*_master.csv</code>.</p>
        <p style="margin:0"><strong>What to do:</strong> ${CLASS.unclassified.action}. Review the
        dates and min EWMA in the table; add rainfall/temperature to the pipeline if available,
        or classify manually using external weather records and field inspection.</p>
      </div>
    </div>
  </details>`;
}


function hsuHowItWorksHtml() {
  return `<details style="margin:8px 0 12px;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:0 14px">
    <summary style="cursor:pointer;padding:12px 0;font-size:0.9rem;color:#cbd5e1;font-weight:500">How HSU soiling analysis works</summary>
    <div style="padding:0 0 14px 0;font-size:0.82rem;color:#94a3b8;line-height:1.6">
      <p style="margin:0 0 10px">Soiling ratio comes from <strong>pvlib.soiling.hsu</strong>, computed in
        <code>0_download_data.py</code> using Bundoora Solcast rain (BUNDOORA campus), Open-Meteo CAMS PM2.5/PM10,
        and library array tilt (10&deg;). Hourly CSV:
        <code>data_raw/hsu_soiling_output.csv</code> &rarr; dashboard copy
        <code>data_for_viz/hsu_soiling_bundoora.csv</code>.</p>
      <p style="margin:0 0 10px">Daily loss uses the worst hourly SR each day. The dashed trend is a least-squares fit;
        its slope is the soiling rate (%/day on SR). Rain bars show daily accumulated rainfall; days with rain &ge;
        ${HSU_CLEANING_THRESHOLD_MM} mm count as HSU cleaning events.</p>
      <p style="margin:0">Cumulative energy loss = expected PVLib energy &times; HSU loss factor
        (<code>1 &minus; SR</code>) for each day, reset after rain-cleaning days.</p>
    </div>
  </details>`;
}


function kimberHowItWorksHtml() {
  const rateText =
    SOILING_RATE_PCT_PER_DAY > 0
      ? `${SOILING_RATE_PCT_PER_DAY.toFixed(3)} %/day (fixed)`
      : "auto-calibrated from the longest dry spell";
  return `<details style="margin:8px 0 12px;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:0 14px">
    <summary style="cursor:pointer;padding:12px 0;font-size:0.9rem;color:#cbd5e1;font-weight:500">How the soiling analysis works</summary>
    <div style="padding:0 0 14px 0;font-size:0.82rem;color:#94a3b8;line-height:1.6">
      <p style="margin:0 0 10px">The <strong>Kimber model</strong> resets soiling ratio (SR) to 1.0 (clean) at each cleaning event and
        decays linearly between events at ${rateText}. SR is anchored to physical clean state using the confirmed manual clean on
        ${KNOWN_CLEANINGS[0]}, not a statistical percentile.</p>
      <p style="margin:0 0 10px"><strong>Rain cleaning:</strong> rain &gt;= ${RAIN_FULL_CLEAN_MM} mm/day fully cleans; rain between
        ${RAIN_MIN_CLEAN_MM} and ${RAIN_FULL_CLEAN_MM} mm cleans proportionally${PARTIAL_CLEAN ? "" : " (partial cleaning disabled)"};
        rain below ${RAIN_MIN_CLEAN_MM} mm has no cleaning effect (typical temperate-climate threshold).</p>
      <p style="margin:0 0 10px"><strong>Wind cleaning:</strong> daily max wind &gt;= ${WIND_CLEAN_THRESHOLD} m/s gives a partial clean
        (30% recovery toward SR=1). A ${GRACE_PERIOD_DAYS}-day grace period after any clean applies no soiling accumulation.</p>
      <p style="margin:0 0 10px"><strong>Restorable power</strong> is the energy recoverable by cleaning now, projected forward over
        ${PROJECTION_DAYS.join("/")} days. Cleaning is recommended once restorable value exceeds the $${CLEANING_COST} cleaning cost
        within a projection horizon.</p>
      <p style="margin:0">Natural rain only partially cleans panels in temperate climates; the model reflects that partial recovery rather
        than treating all rain as a full wash.</p>
    </div>
  </details>`;
}



const KIMBER_CAUSE_STYLE = {
  "full-rain": { bg: "#1e3a5f", fg: "#93c5fd", line: "#3b82f6" },
  "partial-rain": { bg: "#1e3a5f", fg: "#bfdbfe", line: "#60a5fa" },
  wind: { bg: "#0e7490", fg: "#a5f3fc", line: "#22d3ee" },
  manual: { bg: "#064e3b", fg: "#6ee7b7", line: "#10b981" },
};



function kimberCausePillHtml(cause) {
  const s = KIMBER_CAUSE_STYLE[cause] || KIMBER_CAUSE_STYLE.manual;
  const label = kimberCauseLabel(cause);
  return `<span style="display:inline-block;font-size:11px;padding:2px 8px;border-radius:4px;font-weight:500;background:${s.bg};color:${s.fg}">${label}</span>`;
}

function kimberCauseLabel(cause) {
  if (cause === "full-rain") return "Full rain clean";
  if (cause === "partial-rain") return "Partial rain clean";
  if (cause === "wind") return "Wind clean";
  return "Manual clean";
}

function kimberCleaningExplanation(cleaning) {
  if (cleaning.cause === "full-rain") {
    return `Rain &gt;= ${RAIN_FULL_CLEAN_MM} mm/day fully reset soiling ratio to clean.`;
  }
  if (cleaning.cause === "partial-rain") {
    return `Rain between ${RAIN_MIN_CLEAN_MM} and ${RAIN_FULL_CLEAN_MM} mm/day partially restored SR (Kimber partial-clean model).`;
  }
  if (cleaning.cause === "wind") {
    return `Daily max wind &gt;= ${WIND_CLEAN_THRESHOLD} m/s triggered a partial clean (${(WIND_CLEAN_FRACTION * 100).toFixed(0)}% recovery toward SR=1).`;
  }
  return "Confirmed manual panel clean (calibration anchor).";
}



function meanHWindow(daily, centerIdx, before, after) {
  const vals = [];
  for (let j = centerIdx - before; j <= centerIdx + after; j++) {
    if (j < 0 || j >= daily.length || j === centerIdx) continue;
    if (Number.isFinite(daily[j].H)) vals.push(daily[j].H);
  }
  return vals.length ? mean(vals) : NaN;
}



function measureConfirmedCleanStep(daily, knownDay) {
  const idx = daily.findIndex((d) => d.day === knownDay);
  if (idx < 0) return { hBefore: NaN, hAfter: NaN, hStep: NaN };
  const hBefore = meanHWindow(daily, idx, 7, 0);
  const hAfter = meanHWindow(daily, idx, 0, 7);
  const hStep =
    Number.isFinite(hBefore) && Number.isFinite(hAfter) ? hAfter - hBefore : NaN;
  return { hBefore, hAfter, hStep };
}



function computeHcleanBaseline(daily) {
  const refH = [];
  const dayToIdx = new Map(daily.map((d, i) => [d.day, i]));
  for (const knownDay of KNOWN_CLEANINGS) {
    const idx = dayToIdx.get(knownDay);
    if (idx === undefined) continue;
    for (let j = idx; j <= Math.min(daily.length - 1, idx + GRACE_PERIOD_DAYS); j++) {
      if (Number.isFinite(daily[j].H) && daily[j].H > 0) refH.push(daily[j].H);
    }
  }
  for (let i = 0; i < daily.length; i++) {
    if (daily[i].rain >= RAIN_FULL_CLEAN_MM && Number.isFinite(daily[i].H) && daily[i].H > 0) {
      refH.push(daily[i].H);
    }
  }
  if (refH.length >= 3) return median(refH);
  const allH = daily.map((d) => d.H).filter((h) => Number.isFinite(h) && h > 0);
  return allH.length ? percentile(allH, 90) : NaN;
}



function findLongestDrySpell(daily) {
  let best = { start: 0, end: -1, len: 0 };
  let curStart = 0;
  for (let i = 0; i < daily.length; i++) {
    const wet = (daily[i].rain ?? 0) > RAIN_MIN_CLEAN_MM;
    if (!wet) continue;
    const len = i - curStart;
    if (len > best.len) best = { start: curStart, end: i - 1, len };
    curStart = i + 1;
  }
  const tailLen = daily.length - curStart;
  if (tailLen > best.len) best = { start: curStart, end: daily.length - 1, len: tailLen };
  return best.len >= 5 ? best : null;
}



function autoCalibrateSoilingRate(daily, rawSR) {
  const spell = findLongestDrySpell(daily);
  if (!spell) return 0.002;
  const xs = [];
  const ys = [];
  for (let j = spell.start; j <= spell.end; j++) {
    if (!Number.isFinite(rawSR[j])) continue;
    xs.push(j - spell.start);
    ys.push(rawSR[j]);
  }
  if (xs.length < 5) return 0.002;
  const { slope } = linearRegression(xs, ys);
  if (!Number.isFinite(slope)) return 0.002;
  return Math.max(0.0001, Math.abs(slope));
}



function applyPartialClean(srPrev, fraction) {
  return srPrev + fraction * (1 - srPrev);
}



function computeKimberAnalysis(dailyIn) {
  const daily = dailyIn.map((d) => ({ ...d }));
  const emptySummary = {
    soilingRatePerDay: NaN,
    soilingRatePctPerDay: NaN,
    soilingRateAuto: true,
    currentSR: NaN,
    currentSRpctClean: NaN,
    restorableKwh: NaN,
    restorableDollar: NaN,
    paybackHorizonDays: null,
    paybackRestorableDollar: NaN,
    cleaningRecommended: false,
    totalLossKwh: 0,
    totalLossPct: NaN,
    totalLossDollar: 0,
    lossSinceLastCleanKwh: 0,
    lastResetDay: daily[0]?.day ?? "",
    hasWind: false,
    hclean: NaN,
    confirmedCleanStep: { hBefore: NaN, hAfter: NaN, hStep: NaN },
    projections: { days: [], noRain: [], avgClean: [], breakEvenKwh: 0 },
    restorableByHorizon: {},
  };
  const empty = {
    daily,
    cleanings: [],
    validations: [],
    summary: emptySummary,
  };
  if (daily.length < 14) return empty;



  const confirmedCleanStep = measureConfirmedCleanStep(daily, KNOWN_CLEANINGS[0]);
  const hclean = computeHcleanBaseline(daily);
  emptySummary.hclean = hclean;
  emptySummary.confirmedCleanStep = confirmedCleanStep;



  const rawSR = daily.map((d) =>
    Number.isFinite(hclean) && hclean > 0 && Number.isFinite(d.H)
      ? Math.min(1.05, Math.max(0, d.H / hclean))
      : NaN,
  );
  daily.forEach((d, i) => {
    d.rawSR = rawSR[i];
  });



  let soilingRatePerDay;
  let soilingRateAuto = false;
  if (SOILING_RATE_PCT_PER_DAY > 0) {
    soilingRatePerDay = SOILING_RATE_PCT_PER_DAY / 100;
  } else {
    soilingRatePerDay = autoCalibrateSoilingRate(daily, rawSR);
    soilingRateAuto = true;
  }



  const cleanings = [];
  let srPrev = 1.0;
  let graceRemaining = GRACE_PERIOD_DAYS;
  let lossSinceLastCleanKwh = 0;
  let lastResetDay = daily[0].day;
  let totalLossKwh = 0;
  let totalExpKwh = 0;
  const knownSet = new Set(KNOWN_CLEANINGS);



  for (let i = 0; i < daily.length; i++) {
    const d = daily[i];
    const rain = d.rain ?? 0;
    const wind = d.windMax;
    const srBefore = srPrev;
    let sr = srPrev;
    let cause = null;
    let fullReset = false;



    if (knownSet.has(d.day)) {
      sr = 1.0;
      graceRemaining = GRACE_PERIOD_DAYS;
      cause = "manual";
      fullReset = true;
    } else if (rain >= RAIN_FULL_CLEAN_MM) {
      sr = 1.0;
      graceRemaining = GRACE_PERIOD_DAYS;
      cause = "full-rain";
      fullReset = true;
    } else if (PARTIAL_CLEAN && rain > RAIN_MIN_CLEAN_MM) {
      const span = RAIN_FULL_CLEAN_MM - RAIN_MIN_CLEAN_MM;
      const fraction = span > 0 ? (rain - RAIN_MIN_CLEAN_MM) / span : 0;
      const srNew = applyPartialClean(srPrev, Math.min(1, fraction));
      if (srNew - srPrev > 0.005) {
        sr = srNew;
        cause = "partial-rain";
      }
    } else if (Number.isFinite(wind) && wind >= WIND_CLEAN_THRESHOLD) {
      const srNew = applyPartialClean(srPrev, WIND_CLEAN_FRACTION);
      if (srNew - srPrev > 0.005) {
        sr = srNew;
        cause = "wind";
      }
    }



    if (!fullReset && !cause) {
      if (graceRemaining > 0) {
        sr = srPrev;
        graceRemaining -= 1;
      } else {
        sr = srPrev - soilingRatePerDay;
      }
    } else if (fullReset) {
      /* grace already set */
    } else if (cause && !fullReset) {
      /* partial clean day - still count down grace? spec: grace after full clean only */
    }



    sr = Math.min(1, Math.max(0, sr));
    d.SR = sr;
    d.lossKwh = Math.max(0, d.expKwh * (1 - sr));



    if (fullReset) {
      if (cause) {
        cleanings.push({
          day: d.day,
          cause,
          rain,
          windMax: wind,
          srBefore,
          srAfter: sr,
        });
      }
      lossSinceLastCleanKwh = d.lossKwh;
      lastResetDay = d.day;
    } else if (cause) {
      cleanings.push({
        day: d.day,
        cause,
        rain,
        windMax: wind,
        srBefore,
        srAfter: sr,
      });
      lossSinceLastCleanKwh += d.lossKwh;
    } else {
      lossSinceLastCleanKwh += d.lossKwh;
    }



    d.cumLossKwh = lossSinceLastCleanKwh;
    totalLossKwh += d.lossKwh;
    totalExpKwh += d.expKwh;
    srPrev = sr;
  }



  const last = daily[daily.length - 1];
  const currentSR = last?.SR ?? NaN;
  const avgDailyExp =
    daily.length >= 30
      ? mean(daily.slice(-30).map((d) => d.expKwh))
      : mean(daily.map((d) => d.expKwh));



  const cleaningIntervals = [];
  for (let c = 1; c < cleanings.length; c++) {
    cleaningIntervals.push(
      daysBetween(cleanings[c - 1].day, cleanings[c].day),
    );
  }
  const avgCleanInterval =
    cleaningIntervals.length > 0
      ? mean(cleaningIntervals)
      : daily.length / Math.max(1, cleanings.length + 1);



  const maxProj = Math.max(...PROJECTION_DAYS);
  const projDays = [];
  const noRainCum = [];
  const avgCleanCum = [];
  let srNoRain = currentSR;
  let srAvg = currentSR;
  let graceNo = 0;
  let graceAvg = 0;
  let daysSinceClean = 0;
  let cumNoRain = 0;
  let cumAvg = 0;
  const breakEvenKwh = CLEANING_COST / ENERGY_TARIFF;



  for (let h = 1; h <= maxProj; h++) {
    projDays.push(h);
    if (graceNo > 0) {
      graceNo -= 1;
    } else {
      srNoRain = Math.max(0, srNoRain - soilingRatePerDay);
    }
    cumNoRain += avgDailyExp * (1 - srNoRain);
    noRainCum.push(cumNoRain);



    daysSinceClean += 1;
    if (daysSinceClean >= avgCleanInterval) {
      srAvg = 1.0;
      graceAvg = GRACE_PERIOD_DAYS;
      daysSinceClean = 0;
    } else if (graceAvg > 0) {
      graceAvg -= 1;
    } else {
      srAvg = Math.max(0, srAvg - soilingRatePerDay);
    }
    cumAvg += avgDailyExp * (1 - srAvg);
    avgCleanCum.push(cumAvg);
  }



  const restorableByHorizon = {};
  let paybackHorizonDays = null;
  let paybackRestorableDollar = NaN;
  for (const horizon of PROJECTION_DAYS) {
    const idx = horizon - 1;
    const kwh = noRainCum[idx] ?? NaN;
    const dollars = Math.round(kwh * ENERGY_TARIFF);
    restorableByHorizon[horizon] = { kwh, dollars };
    if (
      paybackHorizonDays === null &&
      Number.isFinite(dollars) &&
      dollars >= CLEANING_COST
    ) {
      paybackHorizonDays = horizon;
      paybackRestorableDollar = dollars;
    }
  }



  const restorableKwh = noRainCum[PROJECTION_DAYS[0] - 1] ?? NaN;
  const restorableDollar = Math.round(
    (Number.isFinite(restorableKwh) ? restorableKwh : 0) * ENERGY_TARIFF,
  );



  const dayToIdx = new Map(daily.map((d, i) => [d.day, i]));
  const validations = KNOWN_CLEANINGS.map((knownDay) => {
    const idx = dayToIdx.get(knownDay);
    const srBefore = idx > 0 ? daily[idx - 1]?.SR : NaN;
    const srOn = idx !== undefined ? daily[idx]?.SR : NaN;
    const srAfter =
      idx !== undefined && idx < daily.length - 1 ? daily[idx + 1]?.SR : NaN;
    const manualEvent = cleanings.find(
      (c) => c.day === knownDay && c.cause === "manual",
    );
    return {
      knownDay,
      detected: Boolean(manualEvent),
      srBefore,
      srOn,
      srAfter,
      hStep: confirmedCleanStep.hStep,
      hBefore: confirmedCleanStep.hBefore,
      hAfter: confirmedCleanStep.hAfter,
    };
  });



  return {
    daily,
    cleanings,
    validations,
    summary: {
      soilingRatePerDay,
      soilingRatePctPerDay: soilingRatePerDay * 100,
      soilingRateAuto,
      currentSR,
      currentSRpctClean: Number.isFinite(currentSR) ? currentSR * 100 : NaN,
      restorableKwh,
      restorableDollar,
      paybackHorizonDays,
      paybackRestorableDollar,
      cleaningRecommended: paybackHorizonDays !== null,
      totalLossKwh,
      totalLossPct: totalExpKwh > 0 ? (totalLossKwh / totalExpKwh) * 100 : NaN,
      totalLossDollar: Math.round(totalLossKwh * ENERGY_TARIFF),
      lossSinceLastCleanKwh,
      lastResetDay,
      hasWind: daily.some((d) => Number.isFinite(d.windMax)),
      hclean,
      confirmedCleanStep,
      projections: {
        days: projDays,
        noRain: noRainCum,
        avgClean: avgCleanCum,
        breakEvenKwh,
      },
      restorableByHorizon,
    },
  };
}



function kimberValidationCardHtml(validations) {
  if (!validations?.length) return "";
  return validations
    .map((v) => {
      const srRecovered =
        Number.isFinite(v.srBefore) &&
        Number.isFinite(v.srAfter) &&
        v.srAfter > v.srBefore;
      const border = srRecovered && v.srAfter >= 0.95 ? "#10b981" : "#fbbf24";
      const pctBefore = Number.isFinite(v.srBefore)
        ? `${((1 - v.srBefore) * 100).toFixed(1)}% soiled (SR = ${v.srBefore.toFixed(3)})`
        : "-";
      const recovery =
        Number.isFinite(v.srBefore) && Number.isFinite(v.srAfter)
          ? `SR ${v.srBefore.toFixed(3)} -> ${v.srAfter.toFixed(3)}`
          : "-";
      const hStepLine =
        Number.isFinite(v.hStep) && Number.isFinite(v.hBefore) && Number.isFinite(v.hAfter)
          ? `H step across clean: ${v.hBefore.toFixed(3)} -> ${v.hAfter.toFixed(3)} (delta ${v.hStep >= 0 ? "+" : ""}${v.hStep.toFixed(3)}). Use this to tune RAIN_FULL_CLEAN_MM if needed.`
          : "H step across confirmed clean: insufficient data.";
      return `<div style="margin:12px 0;padding:14px 16px;background:#0f172a;border:2px solid ${border};border-radius:10px">
        <div style="font-weight:700;color:#e2e8f0;font-size:0.95rem;margin-bottom:8px">Validation against confirmed clean (${v.knownDay})</div>
        <div style="font-size:0.84rem;color:#cbd5e1;line-height:1.65">
          <div><strong>Manual reset applied:</strong> ${v.detected ? "YES (SR forced to 1.0 on confirmed date)" : "NO (date not in dataset)"}</div>
          <div><strong>Soiling before clean:</strong> ${pctBefore}</div>
          <div><strong>SR after clean:</strong> ${recovery} ${srRecovered ? "(recovered toward clean)" : "(check thresholds if SR stays low)"}</div>
          <div><strong>Observed H shift:</strong> ${hStepLine}</div>
        </div>
      </div>`;
    })
    .join("");
}



function kimberDecisionBannerHtml(summary) {
  if (summary.cleaningRecommended && summary.paybackHorizonDays) {
    const amt = Math.round(summary.paybackRestorableDollar);
    return `<div style="margin:12px 0;padding:14px 18px;background:#450a0a;border:1px solid #f87171;border-radius:10px">
      <div style="color:#f87171;font-weight:700;font-size:1rem">Cleaning recommended</div>
      <div style="color:#fca5a5;font-size:0.85rem;margin-top:4px">Pays back in ${summary.paybackHorizonDays} days ($${amt.toLocaleString()} restorable &gt; $${CLEANING_COST} cost, no-rain projection).</div>
    </div>`;
  }
  return `<div style="margin:12px 0;padding:14px 18px;background:#052e16;border:1px solid #22c55e;border-radius:10px">
    <div style="color:#22c55e;font-weight:700;font-size:1rem">Cleaning not yet justified</div>
    <div style="color:#86efac;font-size:0.85rem;margin-top:4px">Would take &gt;${Math.max(...PROJECTION_DAYS)} days to pay back at $${ENERGY_TARIFF}/kWh (cost $${CLEANING_COST}). Current SR = ${Number.isFinite(summary.currentSR) ? (summary.currentSR * 100).toFixed(1) : "-"}% clean.</div>
  </div>`;
}



function kimberStatCard(label, value) {
  return `<div style="flex:1 1 180px;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px 16px;min-width:0">
    <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px">${label}</div>
    <div style="font-size:1rem;color:#e2e8f0;font-weight:600">${value}</div>
  </div>`;
}



function kimberStatsHtml(summary) {
  const rateStr = summary.soilingRateAuto
    ? `${summary.soilingRatePctPerDay.toFixed(3)} %/day (auto)`
    : `${summary.soilingRatePctPerDay.toFixed(3)} %/day`;
  const srStr = Number.isFinite(summary.currentSR)
    ? `${(summary.currentSR * 100).toFixed(1)}% clean (SR=${summary.currentSR.toFixed(3)})`
    : "-";
  const restKwh = Number.isFinite(summary.restorableKwh)
    ? Math.round(summary.restorableKwh).toLocaleString()
    : "-";
  const rest$ = Number.isFinite(summary.restorableDollar)
    ? `$${summary.restorableDollar.toLocaleString()}`
    : "-";
  const payback = summary.paybackHorizonDays
    ? `${summary.paybackHorizonDays} days`
    : `&gt;${Math.max(...PROJECTION_DAYS)} days`;
  const totalKwh = Math.round(summary.totalLossKwh).toLocaleString();
  const totalPct = Number.isFinite(summary.totalLossPct)
    ? summary.totalLossPct.toFixed(2)
    : "-";
  return `<div style="display:flex;flex-wrap:wrap;gap:10px;margin:16px 0">
    ${kimberStatCard("Soiling rate", rateStr)}
    ${kimberStatCard("Current SR", srStr)}
    ${kimberStatCard("Restorable (7d)", `${restKwh} kWh (${rest$})`)}
    ${kimberStatCard("Payback horizon", payback)}
    ${kimberStatCard("Total energy lost", `${totalKwh} kWh (${totalPct}%)`)}
  </div>`;
}



function kimberCleaningTableHtml(cleanings) {
  if (!cleanings.length) {
    return `<tr><td colspan="5" style="color:#94a3b8;text-align:center">No cleaning events at current rain/wind thresholds.</td></tr>`;
  }
  return cleanings
    .map(
      (c) => `<tr>
        <td>${c.day}</td>
        <td>${kimberCausePillHtml(c.cause)}</td>
        <td>${c.rain.toFixed(2)}</td>
        <td>${Number.isFinite(c.windMax) ? c.windMax.toFixed(2) : "-"}</td>
        <td>${c.srBefore.toFixed(3)} &rarr; ${c.srAfter.toFixed(3)}</td>
      </tr>`,
    )
    .join("");
}



function kimberTablesHtml(kimber) {
  const n = kimber.cleanings.length;
  return `
    <details style="margin:8px 0 12px;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:0 14px">
      <summary style="cursor:pointer;padding:12px 0;font-size:0.9rem;color:#cbd5e1;font-weight:500">Cleaning events log (${n} event${n === 1 ? "" : "s"})</summary>
      <div style="padding-bottom:12px">
        <div style="overflow-x:auto">
          <table class="data-table">
            <thead><tr><th>Date</th><th>Cause</th><th>Rain (mm)</th><th>Wind max (m/s)</th><th>SR before &rarr; after</th></tr></thead>
            <tbody>${kimberCleaningTableHtml(kimber.cleanings)}</tbody>
          </table>
        </div>
      </div>
    </details>`;
}



/** Hourly HSU CSV -> daily mean SR + summed rain (matches Soiling Project dashboard). */
function parseHsuHourlyToDaily(text) {
  const { headers, rows } = parseCSV(text);
  if (!rows.length) return [];
  const norm = headers.map((h) => String(h || "").trim().toLowerCase());
  const iSR = norm.indexOf("soiling_ratio");
  const iR = norm.indexOf("rainfall");
  if (iSR < 0) throw new Error("HSU CSV needs a soiling_ratio column.");
  const tsHeader = headers[0];
  const byDay = new Map();
  for (const r of rows) {
    const tsRaw = String(r.timestamp ?? r[tsHeader] ?? Object.values(r)[0] ?? "").trim();
    if (!tsRaw) continue;
    const t = new Date(tsRaw.replace(" ", "T"));
    if (Number.isNaN(t.getTime())) continue;
    const sr = Number(r[headers[iSR]] ?? r.soiling_ratio);
    if (!Number.isFinite(sr)) continue;
    const rain = iR >= 0 ? Number(r[headers[iR]]) || 0 : 0;
    const day = t.toISOString().slice(0, 10);
    let o = byDay.get(day);
    if (!o) o = { day, srSum: 0, n: 0, rain: 0 };
    o.srSum += sr;
    o.n += 1;
    o.rain += rain;
    byDay.set(day, o);
  }
  return [...byDay.values()]
    .sort((a, b) => a.day.localeCompare(b.day))
    .map((o) => ({
      day: o.day,
      sr: o.srSum / o.n,
      rain: o.rain,
      loss: 1 - o.srSum / o.n,
    }));
}



function hsuSlopePerDay(rows) {
  if (rows.length < 2) {
    return { slope: 0, intercept: rows[0]?.sr ?? 1, xs: [], days: rows.map((r) => r.day) };
  }
  const t0 = new Date(rows[0].day + "T12:00:00").getTime();
  const xs = rows.map(
    (r) => (new Date(r.day + "T12:00:00").getTime() - t0) / 86400000,
  );
  const ys = rows.map((r) => r.sr);
  const { slope, intercept } = linearRegression(xs, ys);
  return { slope, intercept, xs, days: rows.map((r) => r.day) };
}



function computeHsuEnergyAnalysis(hsuDaily, kimberDaily) {
  const expMap = new Map(kimberDaily.map((d) => [d.day, d.expKwh]));
  const daily = [];
  const cleanings = [];
  let lossSinceClean = 0;
  let totalLoss = 0;
  let totalExp = 0;

  for (const h of hsuDaily) {
    const expKwh = expMap.get(h.day) ?? NaN;
    const lossKwh =
      Number.isFinite(expKwh) && Number.isFinite(h.loss) ? Math.max(0, expKwh * h.loss) : 0;
    if (Number.isFinite(expKwh)) totalExp += expKwh;
    totalLoss += lossKwh;

    if (h.rain >= HSU_CLEANING_THRESHOLD_MM) {
      cleanings.push({ day: h.day, rain: h.rain, sr: h.sr, lossKwh });
      lossSinceClean = lossKwh;
    } else {
      lossSinceClean += lossKwh;
    }
    daily.push({ ...h, expKwh, lossKwh, cumLossKwh: lossSinceClean });
  }

  const fit = hsuSlopePerDay(hsuDaily);
  const meanSR = mean(hsuDaily.map((d) => d.sr));
  const peakLoss = Math.max(0, ...hsuDaily.map((d) => d.loss));
  const peakDay = hsuDaily.find((d) => d.loss === peakLoss)?.day ?? "";

  return {
    daily,
    cleanings,
    summary: {
      ratePctPerDay: fit.slope * 100,
      meanSR,
      peakLossPct: peakLoss * 100,
      peakDay,
      cleanCount: cleanings.length,
      totalLossKwh: totalLoss,
      totalLossPct: totalExp > 0 ? (totalLoss / totalExp) * 100 : NaN,
      currentSR: hsuDaily[hsuDaily.length - 1]?.sr ?? NaN,
    },
  };
}



function hsuDataBannerHtml(sourceUrl) {
  const src = sourceUrl
    ? `<code>${sourceUrl.split("/").slice(-2).join("/")}</code>`
    : "Bundoora HSU series";
  return `<div style="margin:12px 0;padding:12px 16px;background:#0f172a;border:1px solid #334155;border-radius:10px">
    <div style="color:#93c5fd;font-weight:600;font-size:0.92rem">pvlib HSU soiling (Bundoora library)</div>
    <div style="color:#94a3b8;font-size:0.82rem;margin-top:4px">Loaded from ${src}. Regenerate with
      <code>python 0_download_data.py --soiling-only</code>.</div>
  </div>`;
}



function hsuStatsHtml(summary) {
  const rate = Number.isFinite(summary.ratePctPerDay)
    ? `${summary.ratePctPerDay >= 0 ? "+" : ""}${summary.ratePctPerDay.toFixed(3)} %/day`
    : "-";
  const rateColor = summary.ratePctPerDay < 0 ? "#f59e0b" : "#22c55e";
  const srStr = Number.isFinite(summary.currentSR)
    ? `${(summary.currentSR * 100).toFixed(2)}% clean (SR=${summary.currentSR.toFixed(4)})`
    : "-";
  const totalKwh = Math.round(summary.totalLossKwh || 0).toLocaleString();
  const totalPct = Number.isFinite(summary.totalLossPct) ? summary.totalLossPct.toFixed(2) : "-";
  return `<div style="display:flex;flex-wrap:wrap;gap:10px;margin:16px 0">
    ${kimberStatCard("Soiling rate (HSU fit)", `<span style="color:${rateColor}">${rate}</span>`)}
    ${kimberStatCard("Mean soiling ratio", Number.isFinite(summary.meanSR) ? summary.meanSR.toFixed(4) : "-")}
    ${kimberStatCard("Peak loss", `${summary.peakLossPct.toFixed(2)}%${summary.peakDay ? ` on ${summary.peakDay}` : ""}`)}
    ${kimberStatCard("Rain cleaning days", String(summary.cleanCount))}
    ${kimberStatCard("Total energy lost", `${totalKwh} kWh (${totalPct}%)`)}
    ${kimberStatCard("Current SR", srStr)}
  </div>`;
}



function hsuCleaningTableHtml(cleanings) {
  if (!cleanings.length) {
    return `<tr><td colspan="4" style="color:#94a3b8;text-align:center">No days with rain &ge; ${HSU_CLEANING_THRESHOLD_MM} mm in HSU series.</td></tr>`;
  }
  return cleanings
    .map(
      (c) => `<tr>
        <td>${c.day}</td>
        <td>${c.rain.toFixed(2)}</td>
        <td>${c.sr.toFixed(4)}</td>
        <td>${Number.isFinite(c.lossKwh) ? c.lossKwh.toFixed(1) : "-"}</td>
      </tr>`,
    )
    .join("");
}



function hsuTablesHtml(hsu) {
  const n = hsu.cleanings.length;
  return `
    <details style="margin:8px 0 12px;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:0 14px">
      <summary style="cursor:pointer;padding:12px 0;font-size:0.9rem;color:#cbd5e1;font-weight:500">HSU rain-cleaning days (${n})</summary>
      <div style="padding-bottom:12px">
        <div style="overflow-x:auto">
          <table class="data-table">
            <thead><tr><th>Date</th><th>Rain (mm)</th><th>SR (min)</th><th>Daily loss (kWh)</th></tr></thead>
            <tbody>${hsuCleaningTableHtml(hsu.cleanings)}</tbody>
          </table>
        </div>
      </div>
    </details>`;
}



function computeHsuMonitorStats(rows, threshold = HSU_CLEANING_THRESHOLD_MM) {
  const fit = hsuSlopePerDay(rows);
  const meanSR = rows.length ? mean(rows.map((d) => d.sr)) : NaN;
  const peakLoss = rows.length ? Math.max(0, ...rows.map((d) => d.loss)) : 0;
  const peakDay = rows.find((d) => d.loss === peakLoss)?.day ?? "";
  const cleanCount = rows.filter((d) => d.rain >= threshold).length;
  const ratePctPerDay = fit.slope * 100;
  return {
    fit,
    meanSR,
    peakLossPct: peakLoss * 100,
    peakDay,
    cleanCount,
    ratePctPerDay,
    rateSub: ratePctPerDay < 0 ? "soiling (ratio falling)" : "net recovery",
  };
}



function hsuMonitorStatsHtml(stats) {
  const rate = Number.isFinite(stats.ratePctPerDay)
    ? `${stats.ratePctPerDay >= 0 ? "+" : ""}${stats.ratePctPerDay.toFixed(3)} %/day`
    : "-";
  const rateColor = stats.ratePctPerDay < 0 ? "#f59e0b" : "#22c55e";
  return `<div style="display:flex;flex-wrap:wrap;gap:10px;margin:12px 0">
    ${kimberStatCard("Soiling rate (gradient)", `<span style="color:${rateColor}">${rate}</span><div style="font-size:0.72rem;color:#64748b;font-weight:400;margin-top:2px">${stats.rateSub || "slope over window"}</div>`)}
    ${kimberStatCard("Mean soiling ratio", Number.isFinite(stats.meanSR) ? stats.meanSR.toFixed(4) : "-")}
    ${kimberStatCard("Peak loss", `${stats.peakLossPct.toFixed(2)}%${stats.peakDay ? `<div style="font-size:0.72rem;color:#64748b;font-weight:400;margin-top:2px">on ${stats.peakDay}</div>` : ""}`)}
    ${kimberStatCard("Cleaning events", String(stats.cleanCount))}
  </div>`;
}



function isoDayFromPlotlyAxis(v) {
  const s = String(v);
  if (/^\d{4}-\d{2}-\d{2}/.test(s)) return s.slice(0, 10);
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? s.slice(0, 10) : d.toISOString().slice(0, 10);
}



function clampHsuDayRange(hsuDaily, startDay, endDay) {
  if (!hsuDaily.length) return { start: startDay, end: endDay };
  const lo = hsuDaily[0].day;
  const hi = hsuDaily[hsuDaily.length - 1].day;
  let start = startDay && startDay > lo ? startDay : lo;
  let end = endDay && endDay < hi ? endDay : hi;
  if (startDay && startDay > hi) start = hi;
  if (endDay && endDay < lo) end = lo;
  if (start > end) end = start;
  return { start, end };
}



function refreshHsuMonitorChart(elChart, statsId, startDay, endDay, width) {
  const hsuDaily = elChart.__hsuDaily;
  if (!hsuDaily?.length) return;
  const { start, end } = clampHsuDayRange(hsuDaily, startDay, endDay);
  const rows = filterHsuDailyByRange(hsuDaily, start, end);
  const elStats = document.getElementById(statsId);
  if (!rows.length) {
    if (elStats) elStats.innerHTML = `<p style="color:#94a3b8">No HSU data in selected range.</p>`;
    return;
  }
  const stats = computeHsuMonitorStats(rows, HSU_CLEANING_THRESHOLD_MM);
  if (elStats) elStats.innerHTML = hsuMonitorStatsHtml(stats);
  const w = width || elChart.__hsuWidth || 960;
  const spec = buildHsuMonitorChart(rows, stats, "ratio", w);
  spec.layout.xaxis = {
    ...spec.layout.xaxis,
    range: [start, end],
    autorange: false,
  };
  elChart.__hsuSyncing = true;
  const plotFn = elChart.data ? Plotly.react : Plotly.newPlot;
  const done = plotFn(elChart, spec.traces, spec.layout, HM_PLOTLY_CONFIG);
  if (done?.finally) {
    done.finally(() => {
      elChart.__hsuSyncing = false;
    });
  } else {
    elChart.__hsuSyncing = false;
  }
}



function initHsuMonitorPanel(hsuDaily, { statsId, chartId, width, dateFrom, dateTo }) {
  const elChart = document.getElementById(chartId);
  const elStats = document.getElementById(statsId);
  if (!elChart || !elStats || !hsuDaily.length) return null;

  elChart.__hsuDaily = hsuDaily;
  elChart.__hsuWidth = width;
  elChart.__hsuStatsId = statsId;

  const initial = clampHsuDayRange(
    hsuDaily,
    dateFrom || hsuDaily[0].day,
    dateTo || hsuDaily[hsuDaily.length - 1].day,
  );
  refreshHsuMonitorChart(elChart, statsId, initial.start, initial.end, width);

  elChart.on("plotly_relayout", (ev) => {
    if (elChart.__hsuSyncing) return;
    const range = xRangeFromRelayout(ev);
    if (range === undefined) return;
    if (range === null) {
      refreshHsuMonitorChart(
        elChart,
        statsId,
        hsuDaily[0].day,
        hsuDaily[hsuDaily.length - 1].day,
        width,
      );
      return;
    }
    refreshHsuMonitorChart(
      elChart,
      statsId,
      isoDayFromPlotlyAxis(range[0]),
      isoDayFromPlotlyAxis(range[1]),
      width,
    );
  });

  return elChart;
}



function applyHealthMonitorDateRange(chartEls, dateFrom, dateTo) {
  if (!dateFrom || !dateTo || !chartEls?.length) return;
  const range = [dateFrom, dateTo];
  const els = chartEls.filter((el) => el?.data);
  if (!els.length) return;
  Plotly.relayout(els[0], { "xaxis.range": range, "xaxis.autorange": false });
}



function renderHsuMonitorPanel(hsuDaily, { statsId, chartId, width, dateFrom, dateTo }) {
  return initHsuMonitorPanel(hsuDaily, { statsId, chartId, width, dateFrom, dateTo });
}


function filterHsuDailyByRange(hsuDaily, startDay, endDay) {
  return hsuDaily.filter((d) => d.day >= startDay && d.day <= endDay);
}



function defaultHsuMonitorRange(hsuDaily, months = 3) {
  if (!hsuDaily.length) return { start: "", end: "" };
  const end = hsuDaily[hsuDaily.length - 1].day;
  let start = addMonthsIso(end, -months);
  if (start < hsuDaily[0].day) start = hsuDaily[0].day;
  return { start, end };
}



function buildHsuMonitorChart(rows, stats, viewMode, width) {
  const isLoss = viewMode === "loss";
  const days = rows.map((d) => d.day);
  const yMain = rows.map((d) => (isLoss ? d.loss : d.sr));
  const rain = rows.map((d) => d.rain);
  const fitY = rows.map((_, i) => {
    const sr = stats.fit.intercept + stats.fit.slope * stats.fit.xs[i];
    return isLoss ? Math.max(0, 1 - sr) : sr;
  });
  const rainMax = Math.max(...rain, 1);
  const theme = plotlyDarkTheme();
  const srMin = rows.length ? Math.min(...rows.map((d) => d.sr)) : 0.9;

  return {
    traces: [
      {
        x: days,
        y: yMain,
        type: "scatter",
        mode: "lines",
        name: isLoss ? "Loss factor" : "Soiling ratio",
        line: { color: "#d97706", width: 1.8, shape: "linear" },
        yaxis: "y",
        hovertemplate: isLoss
          ? "%{x}<br>loss = %{y:.2%}<extra></extra>"
          : "%{x}<br>SR = %{y:.4f}<extra></extra>",
      },
      {
        x: days,
        y: fitY,
        type: "scatter",
        mode: "lines",
        name: "Fit (slope = soiling rate)",
        line: { color: "#e2e8f0", width: 1.4, dash: "dash", shape: "linear" },
        yaxis: "y",
        hovertemplate: isLoss
          ? "%{x}<br>fit = %{y:.2%}<extra></extra>"
          : "%{x}<br>fit SR = %{y:.4f}<extra></extra>",
      },
      {
        x: days,
        y: rain,
        type: "bar",
        name: "Rainfall (mm)",
        marker: { color: "rgba(37,99,235,0.6)" },
        yaxis: "y2",
        hovertemplate: "%{x}<br>rain = %{y:.2f} mm<extra></extra>",
      },
    ],
    layout: {
      ...theme,
      autosize: false,
      width,
      height: 430,
      title: {
        text: "Soiling & cleaning monitor (pvlib HSU, Bundoora)",
        font: { color: "#e2e8f0", size: 12 },
      },
      margin: { t: 36, r: 52, b: 44, l: 56 },
      xaxis: { ...theme.xaxis, type: "date" },
      yaxis: isLoss
        ? { ...theme.yaxis, title: "Loss factor", tickformat: ".2%", rangemode: "tozero" }
        : {
            ...theme.yaxis,
            title: "Soiling ratio (1 = clean)",
            range: [Math.max(0.85, srMin - 0.008), 1.002],
          },
      yaxis2: {
        title: "Rainfall (mm)",
        overlaying: "y",
        side: "right",
        range: [0, rainMax * 3],
        showgrid: false,
        zeroline: false,
      },
      showlegend: true,
      legend: { orientation: "h", y: 1.1, x: 0, font: { size: 10 } },
      hovermode: "x unified",
      barmode: "overlay",
    },
  };
}



function kimberSrToLoss(sr) {
  return Number.isFinite(sr) ? 1 - sr : null;
}

function buildKimberSrChart(kimber, width) {
  const { daily, cleanings } = kimber;
  const days = daily.map((d) => d.day);
  const theme = plotlyDarkTheme();
  const traces = [
    {
      x: days,
      y: daily.map((d) => kimberSrToLoss(d.rawSR)),
      type: "scatter",
      mode: "markers",
      name: "Observed soiling loss",
      marker: { color: "#64748b", size: 4, opacity: 0.35 },
      hovertemplate: "%{x}<br>loss = %{y:.1%}<extra></extra>",
    },
    {
      x: days,
      y: daily.map((d) => kimberSrToLoss(d.SR)),
      type: "scatter",
      mode: "lines",
      name: "Soiling loss (Kimber)",
      line: { color: "#e2e8f0", width: 2 },
      hovertemplate: "%{x}<br>Kimber loss = %{y:.1%}<extra></extra>",
    },
    {
      x: [days[0], days[days.length - 1]],
      y: [0, 0],
      type: "scatter",
      mode: "lines",
      name: "Clean (no loss)",
      line: { color: "#94a3b8", width: 1, dash: "dash" },
      hoverinfo: "skip",
    },
  ];



  const shapes = [];
  for (const c of cleanings) {
    const col = KIMBER_CAUSE_STYLE[c.cause]?.line || "#94a3b8";
    const dash = c.cause === "manual" ? "solid" : "dot";
    const w = c.cause === "manual" ? 2 : 1.5;
    shapes.push({
      type: "line",
      x0: c.day,
      x1: c.day,
      y0: 0,
      y1: 1,
      yref: "paper",
      line: { color: col, width: w, dash },
    });
  }

  if (cleanings.length) {
    traces.push({
      x: cleanings.map((c) => c.day),
      y: cleanings.map(() => 0),
      type: "scatter",
      mode: "markers",
      name: "Cleaning event",
      showlegend: false,
      marker: {
        symbol: "line-ns",
        size: 58,
        color: "rgba(0,0,0,0)",
        line: { width: 12, color: "rgba(0,0,0,0)" },
      },
      hovertemplate: "<extra></extra>",
    });
  }



  return {
    traces,
    layout: {
      ...theme,
      autosize: false,
      width,
      height: 340,
      title: {
        text: "Soiling loss (Kimber model, calibrated to confirmed clean)",
        font: { color: "#e2e8f0", size: 14 },
      },
      margin: { t: 48, r: 20, b: 44, l: 56 },
      xaxis: { ...theme.xaxis, type: "date", title: "Date" },
      yaxis: {
        ...theme.yaxis,
        title: "Soiling loss (0 = clean)",
        tickformat: ".0%",
        range: [0, 0.32],
      },
      shapes,
      hovermode: "closest",
      legend: { orientation: "h", y: 1.1, x: 0, font: { size: 10 } },
    },
  };
}



function buildKimberCumLossChart(kimber, width) {
  const { daily } = kimber;
  const days = daily.map((d) => d.day);
  const theme = plotlyDarkTheme();
  return {
    traces: [
      {
        x: days,
        y: daily.map((d) => d.cumLossKwh),
        type: "scatter",
        mode: "lines",
        name: "Cumulative loss",
        line: { color: "#f59e0b", width: 2 },
        hovertemplate: "%{x}<br>Cumulative = %{y:.0f} kWh<extra></extra>",
      },
    ],
    layout: {
      ...theme,
      autosize: false,
      width,
      height: 200,
      title: {
        text: "Cumulative energy lost since last clean",
        font: { color: "#e2e8f0", size: 12 },
      },
      margin: { t: 32, r: 16, b: 36, l: 52 },
      xaxis: { ...theme.xaxis, type: "date", showticklabels: false },
      yaxis: { ...theme.yaxis, title: "Cumulative kWh lost" },
      showlegend: false,
      hovermode: "x unified",
    },
  };
}



function buildKimberProjChart(kimber, width) {
  const { projections } = kimber.summary;
  const theme = plotlyDarkTheme();
  const today = kimber.daily[kimber.daily.length - 1]?.day;
  const xDays = projections.days.map((h) => addDaysIso(today, h));
  return {
    traces: [
      {
        x: xDays,
        y: projections.noRain,
        type: "scatter",
        mode: "lines",
        name: "No rain (worst case)",
        line: { color: "#f87171", width: 2 },
        hovertemplate: "Day +%{text}<br>Restorable = %{y:.0f} kWh<extra></extra>",
        text: projections.days,
      },
      {
        x: xDays,
        y: projections.avgClean,
        type: "scatter",
        mode: "lines",
        name: "Avg cleaning frequency",
        line: { color: "#fb923c", width: 2 },
        hovertemplate: "Day +%{text}<br>Restorable = %{y:.0f} kWh<extra></extra>",
        text: projections.days,
      },
      {
        x: [xDays[0], xDays[xDays.length - 1]],
        y: [projections.breakEvenKwh, projections.breakEvenKwh],
        type: "scatter",
        mode: "lines",
        name: "Break-even energy",
        line: { color: "#e2e8f0", width: 1.5, dash: "dash" },
        hoverinfo: "skip",
      },
    ],
    layout: {
      ...theme,
      autosize: false,
      width,
      height: 220,
      title: {
        text: "Projected recoverable energy vs clean-now break-even",
        font: { color: "#e2e8f0", size: 12 },
      },
      margin: { t: 36, r: 16, b: 36, l: 52 },
      xaxis: { ...theme.xaxis, type: "date", showticklabels: false },
      yaxis: { ...theme.yaxis, title: "Cumulative restorable kWh" },
      hovermode: "x unified",
      legend: { orientation: "h", y: 1.15, x: 0, font: { size: 9 } },
    },
  };
}



function buildKimberRainChart(kimber, width) {
  const { daily } = kimber;
  const days = daily.map((d) => d.day);
  const theme = plotlyDarkTheme();
  return {
    traces: [
      {
        x: days,
        y: daily.map((d) => d.rain),
        type: "bar",
        name: "Rain",
        marker: { color: "#3b82f6" },
        hovertemplate: "%{x}<br>Rain = %{y:.2f} mm<extra></extra>",
      },
      {
        x: [days[0], days[days.length - 1]],
        y: [RAIN_MIN_CLEAN_MM, RAIN_MIN_CLEAN_MM],
        type: "scatter",
        mode: "lines",
        line: { color: "#94a3b8", width: 1, dash: "dash" },
        hoverinfo: "skip",
      },
      {
        x: [days[0], days[days.length - 1]],
        y: [RAIN_FULL_CLEAN_MM, RAIN_FULL_CLEAN_MM],
        type: "scatter",
        mode: "lines",
        line: { color: "#f87171", width: 1.5, dash: "dash" },
        hoverinfo: "skip",
      },
    ],
    layout: {
      ...theme,
      autosize: false,
      width,
      height: 180,
      title: { text: "Daily rainfall (cleaning thresholds)", font: { color: "#e2e8f0", size: 12 } },
      margin: { t: 32, r: 16, b: 36, l: 52 },
      xaxis: { ...theme.xaxis, type: "date", showticklabels: false },
      yaxis: { ...theme.yaxis, title: "mm/day" },
      bargap: 0.15,
      showlegend: false,
      hovermode: "x unified",
    },
  };
}



function buildKimberWindChart(kimber, width) {
  const { daily } = kimber;
  const days = daily.map((d) => d.day);
  const theme = plotlyDarkTheme();
  return {
    traces: [
      {
        x: days,
        y: daily.map((d) => d.windMax),
        type: "scatter",
        mode: "lines",
        name: "Wind max",
        line: { color: "#22d3ee", width: 1.2 },
        connectgaps: false,
        hovertemplate: "%{x}<br>Wind = %{y:.2f} m/s<extra></extra>",
      },
      {
        x: [days[0], days[days.length - 1]],
        y: [WIND_CLEAN_THRESHOLD, WIND_CLEAN_THRESHOLD],
        type: "scatter",
        mode: "lines",
        line: { color: "#f87171", width: 1.5, dash: "dash" },
        hoverinfo: "skip",
      },
    ],
    layout: {
      ...theme,
      autosize: false,
      width,
      height: 160,
      title: { text: "Daily max wind speed", font: { color: "#e2e8f0", size: 12 } },
      margin: { t: 32, r: 16, b: 44, l: 52 },
      xaxis: { ...theme.xaxis, type: "date", title: "Date" },
      yaxis: { ...theme.yaxis, title: "m/s" },
      showlegend: false,
      hovermode: "x unified",
    },
  };
}



function xRangeFromRelayout(ev) {
  if (ev["xaxis.range[0]"] !== undefined && ev["xaxis.range[1]"] !== undefined) {
    return [ev["xaxis.range[0]"], ev["xaxis.range[1]"]];
  }
  if (Array.isArray(ev["xaxis.range"]) && ev["xaxis.range"].length === 2) {
    return ev["xaxis.range"];
  }
  if (ev["xaxis.autorange"] === true) return null;
  return undefined;
}

function normalizeRangeMs(range) {
  if (!range || range.length !== 2) return null;
  const t0 = new Date(range[0]).getTime();
  const t1 = new Date(range[1]).getTime();
  if (!Number.isFinite(t0) || !Number.isFinite(t1)) return null;
  return t0 <= t1 ? [t0, t1] : [t1, t0];
}

function rangesEqualMs(a, b) {
  const am = normalizeRangeMs(a);
  const bm = normalizeRangeMs(b);
  if (!am || !bm) return false;
  return am[0] === bm[0] && am[1] === bm[1];
}

function getElXRange(el) {
  const r = el._fullLayout?.xaxis?.range;
  return r && r.length === 2 ? r : null;
}

function wireHealthMonitorLinkedAxes(chartEls) {
  const els = chartEls.filter((el) => el && el.data);
  if (els.length < 2) return;

  let syncing = false;
  let pendingRange = undefined;
  let pendingSkipEl = null;
  let rafId = null;

  const applyXRangeNow = (range, skipEl) => {
    syncing = true;
    try {
      if (range === null) {
        for (const el of els) {
          if (el === skipEl) continue;
          if (el._fullLayout?.xaxis?.autorange) continue;
          Plotly.relayout(el, { "xaxis.autorange": true });
        }
        return;
      }

      const update = { "xaxis.range": range, "xaxis.autorange": false };
      for (const el of els) {
        if (el === skipEl) continue;
        const cur = getElXRange(el);
        if (cur && rangesEqualMs(cur, range)) continue;
        Plotly.relayout(el, update);
      }
    } finally {
      syncing = false;
    }
  };

  const flushPending = () => {
    rafId = null;
    if (pendingRange === undefined) return;
    const range = pendingRange;
    const skipEl = pendingSkipEl;
    pendingRange = undefined;
    pendingSkipEl = null;
    applyXRangeNow(range, skipEl);
  };

  const queueSync = (range, skipEl) => {
    if (syncing) return;

    const others = els.filter((el) => el !== skipEl);
    if (range === null) {
      if (others.every((el) => el._fullLayout?.xaxis?.autorange)) return;
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      pendingRange = undefined;
      pendingSkipEl = null;
      applyXRangeNow(null, skipEl);
      return;
    }

    if (others.every((el) => rangesEqualMs(getElXRange(el), range))) return;

    pendingRange = range;
    pendingSkipEl = skipEl;
    if (!rafId) rafId = requestAnimationFrame(flushPending);
  };

  for (const el of els) {
    el.on("plotly_relayout", (ev) => {
      if (syncing) return;
      const range = xRangeFromRelayout(ev);
      if (range === undefined) return;
      queueSync(range, el);
    });
  }
}



function wireKimberLinkedAxes(chartEls) {
  wireHealthMonitorLinkedAxes(chartEls);
}



function plotKimberCharts(kimber, ids, width) {
  const elSr = document.getElementById(ids.sr);
  const elCum = document.getElementById(ids.cum);
  const elProj = document.getElementById(ids.proj);
  if (!elSr || !elCum || !elProj) return [];

  const srSpec = buildKimberSrChart(kimber, width);
  Plotly.newPlot(elSr, srSpec.traces, srSpec.layout, HM_PLOTLY_CONFIG);
  wireKimberCleaningHover(elSr, kimber);
  const cumSpec = buildKimberCumLossChart(kimber, width);
  Plotly.newPlot(elCum, cumSpec.traces, cumSpec.layout, HM_PLOTLY_CONFIG);
  const projSpec = buildKimberProjChart(kimber, width);
  Plotly.newPlot(elProj, projSpec.traces, projSpec.layout, HM_PLOTLY_CONFIG);

  return [elSr, elCum];
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
function numField(r, ...keys) {
  for (const k of keys) {
    if (r[k] === undefined || r[k] === null || r[k] === "") continue;
    const v = Number(r[k]);
    if (Number.isFinite(v)) return v;
  }
  return NaN;
}
function readKimberRain(r) {
  for (const k of KIMBER_RAIN_KEYS) {
    const v = Number(r[k]);
    if (Number.isFinite(v)) return v;
  }
  return 0;
}
function readKimberWind(r) {
  return numField(r, ...KIMBER_WIND_KEYS);
}
function readKimberTemp(r) {
  return numField(r, ...KIMBER_TEMP_KEYS);
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
function daysBetween(a, b) {
  return Math.round(
    (new Date(b + "T12:00:00").getTime() - new Date(a + "T12:00:00").getTime()) / 86400000,
  );
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
function percentile(arr, p) {
  const s = arr.filter(Number.isFinite).sort((a, b) => a - b);
  if (!s.length) return NaN;
  const idx = (p / 100) * (s.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return s[lo];
  return s[lo] + (s[hi] - s[lo]) * (idx - lo);
}
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
  return out.sort((a, b) => a.day.localeCompare(b.day));
}
/** Stage A  - Kimber daily aggregates. */
function computeKimberDaily(hourly) {
  const byDay = new Map();
  for (const r of hourly) {
    let o = byDay.get(r.day);
    if (!o) {
      o = { act: 0, exp: 0, rain: 0, windMax: NaN, tempN: 0, tempSum: 0 };
      byDay.set(r.day, o);
    }
    o.rain += readKimberRain(r);
    const w = readKimberWind(r);
    if (Number.isFinite(w)) {
      o.windMax = Number.isFinite(o.windMax) ? Math.max(o.windMax, w) : w;
    }
    if (!(r.ghi > 5)) continue;
    if (Number.isFinite(r.actual)) o.act += r.actual;
    if (Number.isFinite(r.expected)) o.exp += r.expected;
    const t = readKimberTemp(r);
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
      expKwh: o.exp,
      rain: o.rain,
      windMax: o.windMax,
      tempMean: o.tempN ? o.tempSum / o.tempN : NaN,
    });
  }
  return out.sort((a, b) => a.day.localeCompare(b.day));
}
function mergeKimberWeatherFromRaw(daily, rawRows) {
  if (!rawRows?.length) return;
  const byDay = new Map();
  for (const r of rawRows) {
    const ts = String(r.timestamp || "").trim();
    if (!ts) continue;
    const day = ts.slice(0, 10);
    let o = byDay.get(day);
    if (!o) o = { rain: 0, windMax: NaN, tempN: 0, tempSum: 0 };
    o.rain += readKimberRain(r);
    const w = readKimberWind(r);
    if (Number.isFinite(w)) {
      o.windMax = Number.isFinite(o.windMax) ? Math.max(o.windMax, w) : w;
    }
    const t = readKimberTemp(r);
    if (Number.isFinite(t)) {
      o.tempSum += t;
      o.tempN += 1;
    }
    byDay.set(day, o);
  }
  for (const d of daily) {
    const w = byDay.get(d.day);
    if (!w) continue;
    if (w.rain > 0 || d.rain === 0) d.rain = w.rain;
    if (Number.isFinite(w.windMax)) d.windMax = w.windMax;
    if (w.tempN && !Number.isFinite(d.tempMean)) d.tempMean = w.tempSum / w.tempN;
  }
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
    } catch {
      /* fall through to Solcast */
    }
  }
  const cleaned = await tryFetchCleanedFile(SOLCAST_CLEANED_CSV);
  return cleaned ? parseCSV(cleaned.text).rows : [];
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
      Number.isFinite(LCL) && Number.isFinite(series[i].ewma) && series[i].ewma < LCL;
  }
  const alarmDays = series.filter((d) => d.alarm);
  const events = groupAlarmEvents(alarmDays, series, LCL, weatherAvailable);
  return { series, refStart, refEnd, seasonMed, mu, sigma, LCL, UCL, events, weatherAvailable };
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
      (new Date(d + "T12:00:00").getTime() - new Date(prev + "T12:00:00").getTime()) /
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
    const ewmas = days.map((day) => series[dayIndex.get(day)]?.ewma).filter(Number.isFinite);
    const minEWMA = ewmas.length ? Math.min(...ewmas) : NaN;
    const durationDays =
      Math.round(
        (new Date(endDay + "T12:00:00").getTime() -
          new Date(startDay + "T12:00:00").getTime()) /
          86400000,
      ) + 1;
    const cls = classifyEvent(startDay, endDay, series, lcl, weatherAvailable);
    return { startDay, endDay, durationDays, minEWMA, ...cls };
  });
}
function classifyEvent(startDay, endDay, series, lcl, weatherAvailable) {
  if (!weatherAvailable) return { kind: "unclassified", ...CLASS.unclassified };
  const recoveryEnd = addDaysIso(endDay, RECOVERY_WINDOW_DAYS);
  const window = series.filter((d) => d.day >= startDay && d.day <= recoveryEnd);
  const recovered = window.some(
    (d) => d.day > endDay && Number.isFinite(d.ewma) && d.ewma >= lcl,
  );
  const eventDays = series.filter((d) => d.day >= startDay && d.day <= endDay);
  const eventTempMean = mean(eventDays.map((d) => d.tempMean).filter(Number.isFinite));
  const allTemps = series.map((d) => d.tempMean).filter(Number.isFinite);
  allTemps.sort((a, b) => a - b);
  const p80 = allTemps.length >= 5 ? allTemps[Math.floor(allTemps.length * 0.8)] : NaN;
  const rainInWindow = window.some(
    (d) => Number.isFinite(d.rainTotal) && d.rainTotal >= EWMA_RAIN_CLEAN_THRESHOLD,
  );
  if (recovered && rainInWindow) return { kind: "soiling", ...CLASS.soiling };
  const hotEvent =
    Number.isFinite(eventTempMean) && Number.isFinite(p80) && eventTempMean >= p80;
  if (hotEvent && recovered) {
    const after = window.filter((d) => d.day > endDay && Number.isFinite(d.tempMean));
    const eventEndTemp = eventDays.length ? eventDays[eventDays.length - 1].tempMean : NaN;
    const coolerRecovery = after.some(
      (d) =>
        Number.isFinite(eventEndTemp) &&
        d.tempMean < eventEndTemp &&
        Number.isFinite(d.ewma) &&
        d.ewma >= lcl,
    );
    if (coolerRecovery) return { kind: "thermal", ...CLASS.thermal };
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
  const allExplained = recent.every((e) => e.kind === "soiling" || e.kind === "thermal");
  if (hasFault) {
    return {
      level: "fault",
      color: "#f87171",
      title: "Inspection recommended",
      summary: `Last alarm: ${last.endDay}  *  ${last.label}  *  ${last.durationDays} days`,
    };
  }
  if (allExplained) {
    return {
      level: "explained",
      color: "#fbbf24",
      title: "Explained underperformance (soiling/thermal)",
      summary: `Last alarm: ${last.endDay}  *  ${last.label}  *  ${last.durationDays} days`,
    };
  }
  return {
    level: "watch",
    color: "#fbbf24",
    title: "Monitor  - mixed or unclassified alarms",
    summary: `Last alarm: ${last.endDay}  *  ${last.label}  *  ${last.durationDays} days`,
  };
}
function measurePlotWidth(el, panel) {
  const parent = el?.parentElement;
  const rect = parent?.getBoundingClientRect?.();
  if (rect && rect.width > 80) return Math.floor(rect.width);
  return Math.max(480, Math.floor(panel?.clientWidth || 960) - 32);
}
// --- Weather & performance analysis ---
function soilingHowItWorksHtml() {
  return `<details style="margin:8px 0 12px;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:0 14px">
    <summary style="cursor:pointer;padding:12px 0;font-size:0.9rem;color:#cbd5e1;font-weight:500">How the weather &amp; performance analysis works</summary>
    <div style="padding:0 0 14px 0;font-size:0.82rem;color:#94a3b8;line-height:1.6">
      <p style="margin:0 0 10px"><strong>Health ratio (H)</strong> is actual divided by expected generation on daylight hours (GHI &gt; 5 W/m2).
        The top chart shows daily H (faint green) and a ${WX_SMOOTH_WINDOW_DAYS}-day smoothed trend (white). Orange lines are soiling
        fits during long dry spells  - only the largest spells are drawn so the chart stays readable.</p>
      <p style="margin:0 0 10px"><strong>Significant changes</strong> (markers on the H chart) are days where smoothed H moves by at least
        ${WX_SIGNIFICANT_JUMP_THRESHOLD}, debounced to one marker every ${WX_MIN_DAYS_BETWEEN_EVENTS} days. Each is classified using
        <em>all available weather</em>, not rain/wind alone: rainfall &gt;= ${WX_RAIN_CLEAN_THRESHOLD} mm, wind &gt;= ${WX_WIND_CLEAN_THRESHOLD} m/s,
        temperature shift, daily GHI (cloud/resource), and dry-spell soiling trend.</p>
      <p style="margin:0 0 10px">The <strong>rain, wind, temperature, and GHI charts</strong> below share one time axis  - pan/zoom any panel
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
function readRainSoiling(r) {
  const raw = r[RAIN_FIELD] ?? r.precipitation_mm ?? r.precipitation_rate;
  const v = Number(raw);
  return Number.isFinite(v) ? v : 0;
}
function readWind(r) {
  return numField(r, WIND_FIELD, "wind_speed_10m", "wind_speed_100m", "wind_ms");
}
function hasWindData(hourly) {
  return hourly.some((r) => Number.isFinite(readWind(r)));
}
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
  const Hsmooth = rollingMedian(Hvals, WX_SMOOTH_WINDOW_DAYS, 1);
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
  const dryDays = prev3.filter((x) => x.rain < WX_RAIN_CLEAN_THRESHOLD).length;
  const scores = [];
  if (isUp) {
    if (window3.some((x) => x.rain >= WX_RAIN_CLEAN_THRESHOLD)) {
      scores.push({ cause: "Rain", score: rainAmt + 10 });
    }
    if (Number.isFinite(windAmt) && windAmt >= WX_WIND_CLEAN_THRESHOLD) {
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
    if (dryDays >= 2 && delta > -WX_SIGNIFICANT_JUMP_THRESHOLD) {
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
    if (Math.abs(jump) < WX_CLEAN_JUMP_THRESHOLD) continue;
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
    if (Math.abs(ev.jump) < WX_SIGNIFICANT_JUMP_THRESHOLD) continue;
    const tooClose = picked.some(
      (p) =>
        Math.abs(
          (new Date(ev.day + "T12:00:00").getTime() -
            new Date(p.day + "T12:00:00").getTime()) /
            86400000,
        ) < WX_MIN_DAYS_BETWEEN_EVENTS,
    );
    if (tooClose) continue;
    picked.push(ev);
    if (picked.length >= MAX_CHART_RECOVERY_MARKERS) break;
  }
  return picked.sort((a, b) => a.day.localeCompare(b.day));
}
/** Weather impact analysis  - multi-factor drivers, debounced events, soiling segments. */
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
    if (length < WX_SOILING_MIN_SEGMENT_DAYS) continue;
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
      factor: "Temperature ( C)",
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
  wireHealthMonitorLinkedAxes(chartEls);
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
        : "-"
      : Number.isFinite(event.tempMean)
        ? `${event.tempMean.toFixed(1)} C`
        : "-";
  return `
    <div class="hm-event-popup__title" style="border-left-color:${accent}">
      <span class="hm-event-popup__dir">${event.direction}</span>
      <span class="hm-event-popup__cause">${event.cause}</span>
    </div>
    <div class="hm-event-popup__meta">
      <div><span class="hm-event-popup__label">Date</span>${event.day}</div>
      <div><span class="hm-event-popup__label">dH</span>${sign}${event.jump.toFixed(3)}</div>
      <div><span class="hm-event-popup__label">Rain (3-day)</span>${event.rainAmt.toFixed(2)} mm</div>
      <div><span class="hm-event-popup__label">${thirdLabel}</span>${thirdValue}</div>
    </div>
    <div class="hm-event-popup__window">7-day window &middot; ${addDaysIso(event.day, -EVENT_POPUP_RADIUS_DAYS)} - ${addDaysIso(event.day, EVENT_POPUP_RADIUS_DAYS)}</div>`;
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
        y: [WX_RAIN_CLEAN_THRESHOLD, WX_RAIN_CLEAN_THRESHOLD],
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
function kimberCleaningPopupMetaHtml(cleaning) {
  const style = KIMBER_CAUSE_STYLE[cleaning.cause] || KIMBER_CAUSE_STYLE.manual;
  const lossBefore = kimberSrToLoss(cleaning.srBefore);
  const lossAfter = kimberSrToLoss(cleaning.srAfter);
  const lossRecoveredPct = (lossBefore - lossAfter) * 100;
  const windStr = Number.isFinite(cleaning.windMax)
    ? `${cleaning.windMax.toFixed(2)} m/s`
    : "-";
  return `
    <div class="hm-event-popup__title" style="border-left-color:${style.line}">
      <span class="hm-event-popup__dir">Cleaning</span>
      <span class="hm-event-popup__cause">${kimberCauseLabel(cleaning.cause)}</span>
    </div>
    <div class="hm-event-popup__meta">
      <div><span class="hm-event-popup__label">Date</span>${cleaning.day}</div>
      <div><span class="hm-event-popup__label">Rain</span>${cleaning.rain.toFixed(2)} mm</div>
      <div><span class="hm-event-popup__label">Wind max</span>${windStr}</div>
      <div><span class="hm-event-popup__label">SR</span>${cleaning.srBefore.toFixed(3)} &rarr; ${cleaning.srAfter.toFixed(3)}</div>
      <div><span class="hm-event-popup__label">Loss recovered</span>${lossRecoveredPct.toFixed(1)}%</div>
    </div>
    <div class="hm-event-popup__window">${kimberCleaningExplanation(cleaning)}</div>
    <div class="hm-event-popup__window">7-day window &middot; ${addDaysIso(cleaning.day, -EVENT_POPUP_RADIUS_DAYS)} - ${addDaysIso(cleaning.day, EVENT_POPUP_RADIUS_DAYS)}</div>`;
}
function buildMiniKimberLossChartSpec(windowDaily, cleaning, width) {
  const days = windowDaily.map((d) => d.day);
  const theme = plotlyDarkTheme();
  const accent = KIMBER_CAUSE_STYLE[cleaning.cause]?.line || "#10b981";
  const eventPoint = windowDaily.find((d) => d.day === cleaning.day);
  const traces = [
    {
      x: days,
      y: windowDaily.map((d) => kimberSrToLoss(d.rawSR)),
      type: "scatter",
      mode: "markers",
      name: "Observed loss",
      marker: { color: "#64748b", size: 4, opacity: 0.4 },
      hovertemplate: "%{x}<br>observed = %{y:.1%}<extra></extra>",
    },
    {
      x: days,
      y: windowDaily.map((d) => kimberSrToLoss(d.SR)),
      type: "scatter",
      mode: "lines+markers",
      name: "Kimber loss",
      line: { color: "#e2e8f0", width: 2 },
      marker: { size: 5, color: "#e2e8f0" },
      hovertemplate: "%{x}<br>Kimber = %{y:.1%}<extra></extra>",
    },
  ];
  if (eventPoint && Number.isFinite(eventPoint.SR)) {
    traces.push({
      x: [cleaning.day],
      y: [kimberSrToLoss(cleaning.srAfter)],
      type: "scatter",
      mode: "markers",
      name: "Clean point",
      marker: {
        size: 11,
        symbol: "diamond",
        color: accent,
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
        text: "Soiling loss (Kimber)",
        font: { color: "#cbd5e1", size: 11 },
      },
      xaxis: { ...theme.xaxis, type: "date", tickformat: "%d %b" },
      yaxis: {
        ...theme.yaxis,
        tickformat: ".0%",
        title: { text: "loss", font: { size: 10 } },
      },
      shapes: [eventDayMarkerShape(cleaning.day, accent)],
      showlegend: false,
      hovermode: "x unified",
    },
  };
}
function buildMiniKimberRainChartSpec(windowDaily, cleaning, width) {
  const days = windowDaily.map((d) => d.day);
  const theme = plotlyDarkTheme();
  const accent = KIMBER_CAUSE_STYLE[cleaning.cause]?.line || "#10b981";
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
        y: [RAIN_MIN_CLEAN_MM, RAIN_MIN_CLEAN_MM],
        type: "scatter",
        mode: "lines",
        line: { color: "#94a3b8", width: 1, dash: "dot" },
        hoverinfo: "skip",
      },
      {
        x: [days[0], days[days.length - 1]],
        y: [RAIN_FULL_CLEAN_MM, RAIN_FULL_CLEAN_MM],
        type: "scatter",
        mode: "lines",
        line: { color: "#22c55e", width: 1, dash: "dash" },
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
        text: "Daily rainfall (Kimber thresholds)",
        font: { color: "#cbd5e1", size: 11 },
      },
      xaxis: { ...theme.xaxis, type: "date", tickformat: "%d %b" },
      yaxis: { ...theme.yaxis, title: { text: "mm", font: { size: 10 } } },
      shapes: [eventDayMarkerShape(cleaning.day, accent)],
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
function plotlyXToDay(raw) {
  if (raw == null) return null;
  if (typeof raw === "string") return raw.slice(0, 10);
  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return null;
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function hoverDayFromPlotlyEvent(ev) {
  const pt = ev.points?.[0];
  if (!pt) return null;
  return plotlyXToDay(pt.x);
}

const KIMBER_CLEANING_HIT_PX = 18;

function cleaningAtMousePixel(plotEl, cleanings, clientX, clientY) {
  const layout = plotEl?._fullLayout;
  if (!layout?.xaxis || !layout._size?.w) return null;

  const xa = layout.xaxis;
  const size = layout._size;
  const rect = plotEl.getBoundingClientRect();
  const xPx = clientX - rect.left;
  const yPx = clientY - rect.top;
  const plotLeft = size.l;
  const plotTop = size.t;
  const plotRight = plotLeft + size.w;
  const plotBottom = plotTop + size.h;
  if (
    xPx < plotLeft - KIMBER_CLEANING_HIT_PX ||
    xPx > plotRight + KIMBER_CLEANING_HIT_PX ||
    yPx < plotTop ||
    yPx > plotBottom
  ) {
    return null;
  }

  const xRange = xa.range || xa._rl;
  const r0 = new Date(plotlyXToDay(xRange?.[0]) + "T12:00:00").getTime();
  const r1 = new Date(plotlyXToDay(xRange?.[1]) + "T12:00:00").getTime();
  if (!Number.isFinite(r0) || !Number.isFinite(r1) || r1 <= r0) return null;

  let best = null;
  let bestDist = KIMBER_CLEANING_HIT_PX + 1;
  for (const c of cleanings) {
    const t = new Date(c.day + "T12:00:00").getTime();
    const frac = (t - r0) / (r1 - r0);
    const linePx = plotLeft + frac * size.w;
    const dist = Math.abs(xPx - linePx);
    if (dist <= KIMBER_CLEANING_HIT_PX && dist < bestDist) {
      bestDist = dist;
      best = c;
    }
  }
  return best;
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
    <button type="button" class="hm-event-popup__close" aria-label="Close">&times;</button>
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
async function showKimberCleaningPopup(popup, kimber, cleaning, clientX, clientY, plotEl) {
  if (_eventPopupHideTimer) {
    clearTimeout(_eventPopupHideTimer);
    _eventPopupHideTimer = null;
  }
  const header = popup.querySelector(".hm-event-popup__header");
  const elLoss = popup.querySelector(".hm-event-popup__plot--h");
  const elRain = popup.querySelector(".hm-event-popup__plot--rain");
  if (!header || !elLoss || !elRain) return;
  _eventPopupPlotEl = plotEl || _eventPopupPlotEl;
  if (_eventPopupPlotEl) setPlotlyNativeHoverVisible(_eventPopupPlotEl, false);
  header.innerHTML = kimberCleaningPopupMetaHtml(cleaning);
  popup.style.display = "block";
  popup.style.visibility = "visible";
  popup.style.left = "0px";
  popup.style.top = "0px";
  const windowDaily = dailyWindowAround(kimber.daily, cleaning.day);
  const w = EVENT_POPUP_CHART_WIDTH;
  const lossSpec = buildMiniKimberLossChartSpec(windowDaily, cleaning, w);
  const rainSpec = buildMiniKimberRainChartSpec(windowDaily, cleaning, w);
  try {
    if (elLoss.data) {
      await Plotly.react(elLoss, lossSpec.traces, lossSpec.layout, PLOTLY_STATIC);
      await Plotly.react(elRain, rainSpec.traces, rainSpec.layout, PLOTLY_STATIC);
    } else {
      await Plotly.newPlot(elLoss, lossSpec.traces, lossSpec.layout, PLOTLY_STATIC);
      await Plotly.newPlot(elRain, rainSpec.traces, rainSpec.layout, PLOTLY_STATIC);
    }
  } catch (err) {
    console.error("[HealthMonitor] Kimber cleaning popup chart error", err);
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
function wireKimberCleaningHover(plotEl, kimber) {
  if (!plotEl || !kimber?.cleanings?.length) return;
  if (plotEl.__hmCleaningHoverWired) return;
  plotEl.__hmCleaningHoverWired = true;

  let moveRaf = null;
  let lastCleaningDay = null;

  const showCleaningAtPointer = (clientX, clientY, pin = false) => {
    const cleaning = cleaningAtMousePixel(plotEl, kimber.cleanings, clientX, clientY);
    if (!cleaning) {
      lastCleaningDay = null;
      if (!_eventPopupPinned) {
        setPlotlyNativeHoverVisible(plotEl, true);
        scheduleHideEventHoverPopup();
      }
      return;
    }
    if (!pin && cleaning.day === lastCleaningDay && _eventPopupEl?.style.display === "block") {
      return;
    }
    lastCleaningDay = cleaning.day;
    if (pin) _eventPopupPinned = true;
    const popup = getOrCreateEventHoverPopup();
    void showKimberCleaningPopup(popup, kimber, cleaning, clientX, clientY, plotEl);
  };

  plotEl.addEventListener("mousemove", (e) => {
    if (_eventPopupPinned) return;
    if (moveRaf) cancelAnimationFrame(moveRaf);
    moveRaf = requestAnimationFrame(() => {
      moveRaf = null;
      showCleaningAtPointer(e.clientX, e.clientY, false);
    });
  });

  plotEl.addEventListener("mouseleave", () => {
    lastCleaningDay = null;
    if (!_eventPopupPinned) scheduleHideEventHoverPopup();
  });

  plotEl.on("plotly_click", (ev) => {
    const { clientX, clientY } = pointerFromPlotlyEvent(ev);
    const cleaning = cleaningAtMousePixel(plotEl, kimber.cleanings, clientX, clientY);
    if (!cleaning) return;
    showCleaningAtPointer(clientX, clientY, true);
  });
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
        text: "Health ratio (H)  - smoothed trend & significant changes",
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
        y: [WX_RAIN_CLEAN_THRESHOLD, WX_RAIN_CLEAN_THRESHOLD],
        type: "scatter",
        mode: "lines",
        name: `Clean threshold (${WX_RAIN_CLEAN_THRESHOLD} mm)`,
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
        y: [WX_WIND_CLEAN_THRESHOLD, WX_WIND_CLEAN_THRESHOLD],
        type: "scatter",
        mode: "lines",
        name: `Threshold (${WX_WIND_CLEAN_THRESHOLD} m/s)`,
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
        name: "Mean temp ( C)",
        line: { color: "#f97316", width: 1.2 },
        connectgaps: false,
        hovertemplate: "%{x}<br>Temp = %{y:.1f}  C<extra></extra>",
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
      yaxis: { ...theme.yaxis, title: " C" },
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
        hovertemplate: "%{x}<br>GHI sum = %{y:.0f}  W/m2*h<extra></extra>",
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
      yaxis: { ...theme.yaxis, title: " W/m2*h sum" },
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
  Plotly.newPlot(elH, hSpec.traces, hSpec.layout, HM_PLOTLY_CONFIG);
  wireSignificantEventHover(elH, weather);
  const rainSpec = buildRainChart(weather, width);
  Plotly.newPlot(elRain, rainSpec.traces, rainSpec.layout, HM_PLOTLY_CONFIG);
  const tempSpec = buildTempChart(weather, width);
  Plotly.newPlot(elTemp, tempSpec.traces, tempSpec.layout, HM_PLOTLY_CONFIG);
  const ghiSpec = buildGhiChart(weather, width);
  Plotly.newPlot(elGhi, ghiSpec.traces, ghiSpec.layout, HM_PLOTLY_CONFIG);
  const linked = [elH, elRain, elTemp, elGhi];
  if (elWind) {
    const windSpec = buildWindChart(weather, width);
    Plotly.newPlot(elWind, windSpec.traces, windSpec.layout, HM_PLOTLY_CONFIG);
    linked.splice(1, 0, elWind);
  }
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
    : " -";
  const causeParts = Object.entries(summary.byCause || {})
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${v} ${k.split(" ")[0].toLowerCase()}`)
    .join(" * ");
  const eventsLine = summary.totalEvents
    ? `${summary.totalEvents} significant (${causeParts || "see table"})`
    : "None detected";
  const energyKwh = Math.round(summary.totalEnergyLostKwh).toLocaleString();
  const energyPct = Number.isFinite(summary.pctEnergyLost)
    ? summary.pctEnergyLost.toFixed(2)
    : " -";
  const topCorr = (summary.correlations || [])
    .slice()
    .sort((a, b) => Math.abs(b.r) - Math.abs(a.r))[0];
  const corrLine = topCorr
    ? `${topCorr.factor} (r = ${topCorr.r.toFixed(2)})`
    : " -";
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
        <td>${Number.isFinite(e.windAmt) ? e.windAmt.toFixed(2) : " -"}</td>
        <td>${Number.isFinite(e.tempMean) ? e.tempMean.toFixed(1) : " -"}</td>
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
        <td>${Number.isFinite(s.pctPerDay) ? s.pctPerDay.toFixed(3) : " -"}</td>
        <td>${s.hLost.toFixed(3)}</td>
      </tr>`,
    )
    .join("");
}

function weatherTablesHtml(weather) {
  const nEvents = weather.significantEvents.length;
  const nSegments = weather.segments.length;
  return `
    <details style="margin:8px 0 12px;background:#1e293b;border:1px solid #334155;border-radius:8px;padding:0 14px">
      <summary style="cursor:pointer;padding:12px 0;font-size:0.9rem;color:#cbd5e1;font-weight:500">Weather analysis tables (${nEvents} H change${nEvents === 1 ? "" : "s"}, ${nSegments} dry spell${nSegments === 1 ? "" : "s"})</summary>
      <div style="padding-bottom:12px">
        <h4 style="font-size:0.88rem;color:#cbd5e1;margin:8px 0 8px">What drives day-to-day H change</h4>
        <div style="overflow-x:auto;margin-bottom:16px">
          <table class="data-table">
            <thead><tr><th>Weather factor</th><th>Pearson r vs dH</th><th>Interpretation</th></tr></thead>
            <tbody>${driverCorrelationsTableHtml(weather.summary.correlations)}</tbody>
          </table>
        </div>
        <h4 style="font-size:0.88rem;color:#cbd5e1;margin:8px 0 8px">Significant H changes</h4>
        <div style="overflow-x:auto;margin-bottom:16px">
          <table class="data-table">
            <thead>
              <tr>
                <th>Date</th><th>Type</th><th>Likely driver</th><th>dH</th>
                <th>Rain (mm)</th><th>Wind max</th><th>Temp (C)</th>
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
    </details>`;
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
      `Drops also track heat (${byCause["Heat stress"] || 0}) and cloud/low-GHI days (${byCause["Cloud / low GHI"] || 0})  - not only soiling.`,
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
        name: "Center (mu)",
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
        text: "EWMA control chart  - H residual (alarms = sustained drop below limit)",
        font: { color: "#e2e8f0", size: 14 },
      },
      margin: { t: 48, r: 20, b: 52, l: 56 },
      xaxis: { ...theme.xaxis, type: "date", title: "Date" },
      yaxis: { ...theme.yaxis, title: "Residual (H - seasonal baseline)" },
      shapes,
      hovermode: "x unified",
      legend: { orientation: "h", y: 1.08, x: 0, font: { size: 10 } },
    },
  };
}
function alarmTableHtml(events) {
  if (!events.length) {
    return `<tr><td colspan="6" style="color:#94a3b8;text-align:center">
      No alarms detected  - system within control limits.</td></tr>`;
  }
  return events
    .map(
      (ev) => `<tr>
        <td>${ev.startDay}</td>
        <td>${ev.endDay}</td>
        <td>${ev.durationDays}</td>
        <td>${Number.isFinite(ev.minEWMA) ? ev.minEWMA.toFixed(3) : " -"}</td>
        <td>${classPillHtml(ev.kind, ev.label)}</td>
        <td>${ev.action}</td>
      </tr>`,
    )
    .join("");
}
/**
 * @param {HTMLElement} container
 * @param {Array} hourly
 * @param {object} [site] `{ key, label, dateFrom?, dateTo? }`
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
  const hsuLoading = `<p style="color:#94a3b8;font-size:0.85rem;margin:8px 0">Loading HSU soiling data…</p>`;
  const idBanner = nextPlotDomId("hm-banner");
  const idEwma = nextPlotDomId("hm-ewma");
  const idSoilH = nextPlotDomId("hm-soil-h");
  const idSoilRain = nextPlotDomId("hm-soil-rain");
  const idSoilWind = nextPlotDomId("hm-soil-wind");
  const idSoilTemp = nextPlotDomId("hm-soil-temp");
  const idSoilGhi = nextPlotDomId("hm-soil-ghi");
  const idWeatherStats = nextPlotDomId("hm-wx-stats");
  const idWeatherTables = nextPlotDomId("hm-wx-tables");
  const idHsuMonitorStats = nextPlotDomId("hm-hsu-mon-stats");
  const idHsuMonitorChart = nextPlotDomId("hm-hsu-mon-chart");
  const idKimberValid = nextPlotDomId("hm-kimber-valid");
  const idKimberDecision = nextPlotDomId("hm-kimber-decision");
  const idKimberStats = nextPlotDomId("hm-kimber-stats");
  const idKimberTables = nextPlotDomId("hm-kimber-tables");
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
      <h3 style="font-size:0.92rem;color:#cbd5e1;margin:16px 0 8px">EWMA control chart</h3>
      <div id="${idEwma}" class="chart-box" style="height:380px;min-height:380px;margin-bottom:16px;width:100%;overflow:hidden"></div>
      <h3 style="font-size:0.92rem;color:#cbd5e1;margin:24px 0 0">Weather &amp; performance analysis</h3>
      <div id="${idKimberDecision}">${hsuLoading}</div>
      <h3 style="font-size:0.92rem;color:#cbd5e1;margin:20px 0 4px">Soiling &amp; cleaning monitor (HSU)</h3>
      <div id="${idHsuMonitorStats}">${hsuLoading}</div>
      <div id="${idHsuMonitorChart}" class="chart-box" style="height:430px;min-height:430px;margin-bottom:16px;width:100%;overflow:hidden"></div>
      ${soilingHowItWorksHtml()}
      <p class="note" style="margin:8px 0 12px">All charts share the same date range  - pan or box-select on any panel to align EWMA, health ratio (H), HSU soiling, and weather views.
        Double-click a chart to reset the range on all panels.</p>
      <div id="${idSoilH}" class="chart-box" style="height:340px;min-height:340px;margin-bottom:4px;width:100%;overflow:hidden"></div>
      <div id="${idSoilRain}" class="chart-box" style="height:170px;min-height:170px;margin-bottom:4px;width:100%;overflow:hidden"></div>
      <div id="${idSoilWind}" class="chart-box" style="height:170px;min-height:170px;margin-bottom:4px;width:100%;overflow:hidden"></div>
      <div id="${idSoilTemp}" class="chart-box" style="height:170px;min-height:170px;margin-bottom:4px;width:100%;overflow:hidden"></div>
      <div id="${idSoilGhi}" class="chart-box" style="height:170px;min-height:170px;margin-bottom:12px;width:100%;overflow:hidden"></div>
      <div id="${idKimberValid}"></div>
      ${hsuHowItWorksHtml()}
      <div id="${idKimberStats}">${hsuLoading}</div>
      <div id="${idKimberTables}"></div>
      <div id="${idWeatherStats}">${soilingStatsHtml(weather.summary)}</div>
      <div id="${idWeatherTables}">${weatherTablesHtml(weather)}</div>
    </div>`;
  setTimeout(async () => {
    if (!container.isConnected || container.__hmGen !== gen) return;
    const elEwma = document.getElementById(idEwma);
    if (!elEwma) return;
    void container.offsetWidth;
    const w = Math.max(480, measurePlotWidth(elEwma, container));
    const linkedCharts = [];
    try {
      const ewma = buildEwmaChart(analysis, w);
      Plotly.newPlot(elEwma, ewma.traces, ewma.layout, HM_PLOTLY_CONFIG);
      linkedCharts.push(elEwma);
    } catch (err) {
      console.error("[HealthMonitor] EWMA plot error", err);
      elEwma.innerHTML = `<p style="color:#f87171;padding:1rem;font-size:0.85rem">${err.message}</p>`;
    }
    try {
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
      if (tablesEl) tablesEl.innerHTML = weatherTablesHtml(wx);
      const elWind = document.getElementById(idSoilWind);
      if (elWind && !windOk) {
        elWind.innerHTML = `<p style="color:#94a3b8;padding:1.2rem;font-size:0.85rem;text-align:center">No wind column in hourly master CSV (expected <code>wind_speed_10m</code> from Solcast).</p>`;
        elWind.style.height = "80px";
        elWind.style.minHeight = "80px";
      }
      linkedCharts.push(
        ...plotWeatherPanels(
          wx,
          { h: idSoilH, rain: idSoilRain, wind: idSoilWind, temp: idSoilTemp, ghi: idSoilGhi },
          w,
          windOk,
        ),
      );
    } catch (err) {
      console.error("[HealthMonitor] weather plot error", err);
      const elH = document.getElementById(idSoilH);
      if (elH) {
        elH.innerHTML = `<p style="color:#f87171;padding:1rem;font-size:0.85rem">${err.message}</p>`;
      }
    }
    try {
      let kd = computeKimberDaily(hourly);
      try {
        const rows = await fetchWeatherEnrichmentRows(site);
        mergeKimberWeatherFromRaw(kd, rows);
      } catch (fetchErr) {
        console.warn("[HealthMonitor] weather enrichment not loaded", fetchErr);
      }

      const hsuRes = await tryFetchHsuSoilingText();
      const decisionEl = document.getElementById(idKimberDecision);
      const statsEl = document.getElementById(idKimberStats);
      const tablesEl = document.getElementById(idKimberTables);
      const monChart = document.getElementById(idHsuMonitorChart);

      if (!hsuRes?.text) {
        const msg =
          `<p style="color:#f87171;padding:1rem;font-size:0.85rem">HSU soiling CSV not found. Run ` +
          `<code>python 0_download_data.py --soiling-only</code> then hard-refresh.</p>`;
        if (decisionEl) decisionEl.innerHTML = msg;
        if (statsEl) statsEl.innerHTML = "";
        if (tablesEl) tablesEl.innerHTML = "";
        const monStats = document.getElementById(idHsuMonitorStats);
        if (monStats) monStats.innerHTML = msg;
        if (monChart) monChart.innerHTML = msg;
      } else {
        const hsuDaily = parseHsuHourlyToDaily(hsuRes.text);
        if (!hsuDaily.length) {
          throw new Error("HSU CSV parsed to zero daily rows.");
        }
        const hsu = computeHsuEnergyAnalysis(hsuDaily, kd);
        if (decisionEl) decisionEl.innerHTML = hsuDataBannerHtml(hsuRes.url);
        if (statsEl) statsEl.innerHTML = hsuStatsHtml(hsu.summary);
        if (tablesEl) tablesEl.innerHTML = hsuTablesHtml(hsu);
        const monEl = initHsuMonitorPanel(hsuDaily, {
          statsId: idHsuMonitorStats,
          chartId: idHsuMonitorChart,
          width: w,
          dateFrom: site.dateFrom,
          dateTo: site.dateTo,
        });
        if (monEl) linkedCharts.push(monEl);
      }
    } catch (err) {
      console.error("[HealthMonitor] HSU soiling plot error", err);
      const monChart = document.getElementById(idHsuMonitorChart);
      if (monChart) {
        monChart.innerHTML = `<p style="color:#f87171;padding:1rem;font-size:0.85rem">${err.message}</p>`;
      }
    }
    wireHealthMonitorLinkedAxes(linkedCharts);
    applyHealthMonitorDateRange(linkedCharts, site.dateFrom, site.dateTo);
  }, 150);
}
