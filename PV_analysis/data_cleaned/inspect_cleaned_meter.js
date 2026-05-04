/**
 * QC plot for SolarMeterReadings1hour_cleaned.csv (same folder as this script).
 * Open inspect_cleaned.html via HTTP from the PV_analysis root (see serve_js_dashboard.py).
 */

const DATA_CSV = new URL("./SolarMeterReadings1hour_cleaned.csv", import.meta.url).href;

function parseCSV(text) {
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

function isoDay(ts) {
  const s = String(ts).replace(" ", "T");
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d.toISOString().slice(0, 10);
}

/** Same reading within relative tolerance (CSV floats), length >= minRun. */
function sameReading(a, b) {
  const tol = Math.max(1e-5, 1e-4 * Math.max(Math.abs(a), Math.abs(b), 1));
  return Math.abs(a - b) <= tol;
}

/** Runs of near-identical positive values with length >= minRun — flat plateaus in data or chart. */
function plateauIndices(values, minRun = 3) {
  const bad = new Set();
  const n = values.length;
  let i = 0;
  while (i < n) {
    const v = values[i];
    if (!(v > 0) || !Number.isFinite(v)) {
      i += 1;
      continue;
    }
    let j = i + 1;
    while (j < n && sameReading(values[j], v)) j += 1;
    const len = j - i;
    if (len >= minRun) {
      for (let k = i; k < j; k++) bad.add(k);
    }
    i = j;
  }
  return bad;
}

function drawPlot(rows, meter, dateFrom, dateTo, showSuspect) {
  const filtered = rows.filter((r) => {
    if (r.meter !== meter) return false;
    const day = isoDay(r.timestamp);
    if (!day) return false;
    return day >= dateFrom && day <= dateTo;
  });
  const t = filtered.map((r) => r.timestamp.replace(" ", "T"));
  const y = filtered.map((r) => Number(r.meter_reading));

  const traces = [
    {
      x: t,
      y,
      name: "meter_reading (kWh)",
      type: "scatter",
      mode: "lines",
      line: { color: "#38bdf8", width: 1.2 },
    },
  ];

  if (showSuspect) {
    const bad = plateauIndices(y, 3);
    const xSus = [];
    const ySus = [];
    for (let i = 0; i < y.length; i++) {
      if (bad.has(i)) {
        xSus.push(t[i]);
        ySus.push(y[i]);
      }
    }
    if (xSus.length) {
      traces.push({
        x: xSus,
        y: ySus,
        name: "Suspect: ≥3h flat (same kWh)",
        type: "scatter",
        mode: "markers",
        marker: { color: "#f97316", size: 8, symbol: "circle-open" },
      });
    }
  }

  if (filtered.length && filtered[0].outage_flag !== undefined) {
    const xo = [];
    const yo = [];
    for (let i = 0; i < filtered.length; i++) {
      const f = filtered[i];
      if (f.outage_flag === "True" || f.outage_flag === "true" || f.outage_flag === true) {
        xo.push(t[i]);
        yo.push(y[i]);
      }
    }
    if (xo.length) {
      traces.push({
        x: xo,
        y: yo,
        name: "outage_flag (V2)",
        type: "scatter",
        mode: "markers",
        marker: { color: "#a855f7", size: 6 },
      });
    }
  }

  const layout = {
    paper_bgcolor: "#0f172a",
    plot_bgcolor: "#0f172a",
    font: { color: "#e2e8f0" },
    title: `Cleaned meter · ${meter}`,
    xaxis: { title: "Time", gridcolor: "#1e293b" },
    yaxis: { title: "kWh (hourly)", gridcolor: "#1e293b" },
    legend: { orientation: "h", y: 1.12 },
    margin: { t: 48, r: 24, b: 48, l: 56 },
    hovermode: "x unified",
  };

  Plotly.newPlot("plot", traces, layout, { responsive: true, displaylogo: false });
}

async function main() {
  const status = document.getElementById("status");
  try {
    status.textContent = "Loading SolarMeterReadings1hour_cleaned.csv…";
    const resp = await fetch(DATA_CSV);
    if (!resp.ok) {
      status.textContent =
        `Could not load SolarMeterReadings1hour_cleaned.csv (${resp.status}). Run python serve_js_dashboard.py from PV_analysis and open /data_cleaned/inspect_cleaned.html`;
      return;
    }
    const text = await resp.text();
    const { rows } = parseCSV(text);
    const meters = [...new Set(rows.map((r) => r.meter))].sort();
    const sel = document.getElementById("meter");
    sel.innerHTML = meters.map((m) => `<option value="${m}">${m}</option>`).join("");

    const lib = meters.find((m) => /library/i.test(m));
    if (lib) sel.value = lib;

    let maxD = "";
    for (const r of rows) {
      const d = isoDay(r.timestamp);
      if (d && d > maxD) maxD = d;
    }
    let minD = maxD;
    for (const r of rows) {
      const d = isoDay(r.timestamp);
      if (d && d < minD) minD = d;
    }
    const d0 = document.getElementById("d0");
    const d1 = document.getElementById("d1");
    if (maxD) {
      const end = new Date(maxD + "T12:00:00");
      end.setMonth(end.getMonth() - 1);
      d0.value = end.toISOString().slice(0, 10);
      d1.value = maxD;
    }

    function redraw() {
      const m = sel.value;
      drawPlot(rows, m, d0.value, d1.value, document.getElementById("showSuspect").checked);
      status.textContent = `${rows.filter((r) => r.meter === m).length.toLocaleString()} rows · orange = ≥3h same reading (flat run)`;
    }

    document.getElementById("draw").addEventListener("click", redraw);
    document.getElementById("showSuspect").addEventListener("change", redraw);
    sel.addEventListener("change", redraw);
    status.textContent = `Loaded ${rows.length.toLocaleString()} rows from SolarMeterReadings1hour_cleaned.csv`;
    redraw();
  } catch (e) {
    status.textContent = "Error: " + e.message;
    console.error(e);
  }
}

main();
