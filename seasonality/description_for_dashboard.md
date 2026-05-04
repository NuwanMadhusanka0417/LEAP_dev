Here's a complete guide to every dashboard and graph produced by your analysis.

---

## 1. Per-Meter Main Dashboard (`dashboard_{meter}.png`)

A 6-panel (3x2) dashboard generated for each meter, showing all aspects of performance.

### Panel 1 — Power Generation Over Time (Row 1, Left)
**What it shows:** Monthly total power generation in kWh as a bar chart with a linear trend line (red dashed).

**Information you get:**
- Raw energy output for every month across 2021-2025
- The trend line slope tells you how many kWh/year the output is changing
- The r-value shows how consistent the trend is (r close to 1 or -1 = very consistent)

**Advantage:** This is the simplest, most intuitive graph to show a supervisor. No normalisation or complex metrics — just "how much electricity did this meter produce each month, and is it going up or down?" Anyone can understand it immediately.

---

### Panel 2 — Yearly Power Generation Comparison (Row 1, Right)
**What it shows:** Total kWh produced each year as side-by-side bars, with the total and number of valid days annotated on each bar. The x-axis shows the daily average kWh for the first and last year with a percentage change.

**Information you get:**
- Which years produced the most/least energy
- Whether differences are due to data gaps (fewer valid days) or actual performance changes
- The daily average kWh corrects for different numbers of valid days, giving a fair comparison

**Advantage:** Lets you quickly answer "did this meter produce more or less this year vs last year?" The per-day average handles the common problem of incomplete years, so you're comparing apples to apples.

---

### Panel 3 — Degradation Trend (Row 2, Left)
**What it shows:** Monthly median of GHI-normalised, temperature-corrected yield over time, with a linear trend line. The yield is normalised so 2021 = 1.0.

**Information you get:**
- The degradation rate in %/year (shown in the legend)
- Whether the panel efficiency is increasing or decreasing relative to the sunlight available
- The green dashed line at 1.0 represents the baseline — points above 1.0 mean the meter is performing better than 2021, below 1.0 means worse

**Advantage:** Unlike raw power, this metric removes the effect of weather (sunny vs cloudy years) and temperature (hot vs cool days). It isolates the panel's actual efficiency change. This is the industry-standard method (IEC 61724) used in PV research papers.

---

### Panel 4 — Seasonal Yield Distribution (Row 2, Right)
**What it shows:** Box plots of the normalised yield for each season (Summer, Autumn, Winter, Spring — southern hemisphere).

**Information you get:**
- Which seasons have the highest and lowest performance efficiency
- The spread (variability) of performance within each season
- Whether certain seasons have more outliers or inconsistent performance
- The green line at 1.0 shows the baseline reference

**Advantage:** Helps identify seasonal patterns. For example, if Winter consistently has lower normalised yield, it could indicate shading from nearby buildings/trees (low sun angle in winter). If Summer has high variability, it could point to overheating or inverter clipping issues.

---

### Panel 5 — Daily Yield & Sudden Events (Row 3, Left)
**What it shows:** Every valid day's normalised yield as gray dots, with a 14-day rolling median (blue line) overlaid. Red dots mark sudden drops, orange triangles mark sudden spikes or step changes.

**Information you get:**
- The day-to-day variability of the system
- When sudden performance drops occurred (red dots) — these could indicate equipment failures, tripped breakers, shading from new construction, etc.
- Whether the system recovered after a drop or if it's a permanent step change
- The smooth blue line reveals the underlying trend without daily noise

**Advantage:** This is your **predictive maintenance alarm view**. A cluster of red dots means something went wrong at that time. A step change (sustained shift in the blue line) means a permanent system change. You can correlate event dates with maintenance logs to identify root causes.

---

### Panel 6 — Soiling Events (Row 3, Right)
**What it shows:** Horizontal bars showing yield drop during each confirmed soiling spell. Blue bars = cleaned by rain afterward, purple = cleaned by wind, brown = other/unknown.

**Information you get:**
- How many soiling events were statistically confirmed
- The magnitude of yield loss from each soiling episode
- What cleaned the panels (rain vs wind)
- If "No soiling events detected" appears, soiling is not a significant issue for this meter

**Advantage:** Directly answers "should we invest in panel cleaning for this meter?" If soiling events are frequent with large yield drops and mostly rain-cleaned, a cleaning schedule could be beneficial. If no events are detected (common in Melbourne due to frequent rain), cleaning is not cost-effective.

---

## 2. Per-Meter Loss Attribution Dashboard (`loss_{meter}.png`)

A 4-panel (2x2) dashboard that breaks down WHY performance changed, attributing losses to specific physical causes.

### Panel 1 — Year-Season Heatmap (Top Left)
**What it shows:** A color-coded grid with years as rows and seasons as columns. Each cell shows the total yield change (%) relative to the 2021 baseline for that season.

**Information you get:**
- Which specific year-season combinations had the best/worst performance
- Green cells = performing above baseline, Red cells = below baseline
- Patterns like "every Winter is red" indicate a seasonal structural issue
- A single red cell (e.g., Autumn 2023) indicates a specific incident

**Advantage:** Gives you a complete at-a-glance picture of when problems occurred. You can quickly say "this meter dropped 15% in Winter 2023" and investigate what happened during that period.

---

### Panel 2 — Yearly Loss Breakdown (Top Right)
**What it shows:** Stacked bars for each year showing the contribution of each loss component:
- **Blue** = Degradation (the long-term trend)
- **Orange** = Soiling loss
- **Red** = Sudden event loss
- **Gray** = Other/residual (unexplained)
- **Black dots** = Net change (actual total)

**Information you get:**
- How much of the total change is explained by each cause
- Whether the "Other" category is dominant (suggesting unmeasured factors like system upgrades, meter changes, or data quality)
- The relative importance of degradation vs soiling vs events for this specific meter

**Advantage:** This is the key decision-making chart. If soiling dominates, invest in cleaning. If events dominate, investigate equipment reliability. If degradation is the main driver, consider panel replacement planning. If "Other" dominates, you need to investigate data quality or unmeasured system changes.

---

### Panel 3 — Seasonal Loss Components (Bottom Left)
**What it shows:** Grouped bars comparing degradation, soiling loss, event loss, and net change for each season, averaged across all post-baseline years.

**Information you get:**
- Which loss mechanism is most important in each season
- Whether soiling is worse in Summer (dry, dusty) vs Winter (rainy, clean)
- Whether sudden events cluster in particular seasons

**Advantage:** Guides seasonal maintenance planning. If soiling is worst in Summer, schedule cleaning before Summer. If events peak in Winter, check for storm damage or waterproofing. This is directly actionable for maintenance scheduling.

---

### Panel 4 — Waterfall Chart (Bottom Right)
**What it shows:** A waterfall showing how the yield went from 100% (baseline) to the actual value for the latest year (2025), with each loss component as a step:
Baseline (100%) → +Degradation → -Soiling → -Events → +/-Other → Actual

**Information you get:**
- A visual, intuitive breakdown of "where did the performance go?"
- The exact percentage contribution of each factor
- Whether the system ended above or below the baseline

**Advantage:** This is the **single best chart to present to management**. It tells a clear story: "We started at 100%, gained/lost X% from aging, lost Y% from soiling, lost Z% from equipment events, and ended at W%." Non-technical stakeholders can understand it immediately.

---

## 3. Fleet Overview Dashboard (`fleet_overview.png`)

A 4-panel (2x2) dashboard comparing all meters across the fleet.

### Panel 1 — Per-Meter Degradation Rate (Top Left)
**What it shows:** Horizontal bars showing the degradation rate (%/year) for each meter, sorted from worst to best. Green dashed lines mark the typical range (-0.5 to -1%/yr).

**Information you get:**
- Which meters are degrading fastest (needing attention)
- Which meters are performing above the fleet average
- How the fleet compares to industry-typical degradation rates

**Advantage:** Immediately identifies problem meters. A meter significantly below the fleet average deserves investigation — it could be partially shaded, have a faulty inverter, or need cleaning.

### Panel 2 — Seasonal Degradation Heatmap (Top Right)
**What it shows:** A heatmap with meters as rows and seasons as columns, showing the degradation rate for each meter in each season.

**Information you get:**
- Whether certain meters degrade faster in specific seasons
- Fleet-wide seasonal patterns (e.g., all meters degrade more in Autumn)
- Anomalous meter-season combinations worth investigating

**Advantage:** Reveals whether degradation is uniform or seasonal, helping diagnose the cause (soiling = worse in dry seasons, shading = worse in winter, thermal stress = worse in summer).

### Panel 3 — Monthly Yield All Meters (Bottom Left)
**What it shows:** All meters' monthly normalised yield overlaid on the same timeline.

**Information you get:**
- Whether the fleet moves together (fleet-wide issue) or individual meters diverge
- Which meters are consistently above/below the fleet
- The point in time when divergence started (indicating a specific event)

**Advantage:** Quickly separates site-wide issues (weather, grid curtailment) from meter-specific issues (equipment fault, shading).

### Panel 4 — Soiling Events: Rain vs Wind (Bottom Right)
**What it shows:** Per-meter counts of soiling spells cleaned by rain vs wind.

**Information you get:**
- Which meters have the most soiling issues
- Whether rain or wind is the dominant natural cleaning mechanism
- Meters with high soiling counts may need manual cleaning

**Advantage:** Identifies which specific meters would benefit most from a cleaning program.

---

## 4. Fleet Loss Attribution Dashboard (`fleet_loss_attribution.png`)

A 4-panel (2x2) fleet-wide loss attribution view.

### Panel 1 — Fleet Median Loss Components per Year (Top Left)
**What it shows:** Grouped bars of fleet-median degradation, soiling loss, and event loss per year, with net change overlaid.

**Information you get:**
- Fleet-wide trends in each loss component year-over-year
- Whether losses are growing (indicating worsening system health)
- Which year had the worst combination of losses

**Advantage:** Answers the fleet-level question: "Is our solar portfolio getting worse, and if so, why?"

### Panel 2 — Seasonal Yield Change Heatmap (Top Right)
**What it shows:** Fleet-median yield change (%) for each year-season combination across all meters.

**Information you get:**
- Site-wide seasonal performance patterns
- Whether certain year-season combinations were anomalous for the entire fleet
- Long-term seasonal trends (is Winter getting worse year after year?)

**Advantage:** Separates weather-driven fleet-wide effects from individual meter issues.

### Panel 3 — Soiling vs Event Loss Scatter (Bottom Left)
**What it shows:** Each meter plotted with average soiling loss on x-axis and average event loss on y-axis, colored by net yield change.

**Information you get:**
- Meters in the top-right corner have BOTH high soiling and high event losses — highest priority
- Meters near the origin have minimal losses — healthy systems
- The color tells you if these losses actually translate to net underperformance

**Advantage:** A single glance tells you which meters need the most attention and what type of intervention (cleaning vs repair). This is the **prioritisation chart** for maintenance resource allocation.

### Panel 4 — Per-Meter Loss Component Bars (Bottom Right)
**What it shows:** Horizontal stacked bars for each meter showing degradation, soiling loss, and event loss, sorted by total change.

**Information you get:**
- Complete ranking of all meters by performance
- The breakdown of WHY each meter is where it is
- Which loss component dominates for each meter

**Advantage:** The definitive "maintenance priority list." Start fixing meters at the bottom of this chart and work your way up. The color breakdown tells you what to fix (blue = aging/degradation, orange = clean the panels, red = repair equipment).

---

## 5. CSV Outputs

| File | What it contains | Use case |
|---|---|---|
| `degradation_summary.csv` | Per-meter overall rates, soiling/event counts | Quick reference table |
| `seasonal_degradation.csv` | Per-meter per-season trend analysis | Seasonal pattern research |
| `monthly_yield.csv` | Monthly median normalised yield per meter | Time-series analysis, custom plots |
| `soiling_events.csv` | Every dry spell with wind/rain context | Detailed soiling investigation |
| `sudden_events.csv` | Every anomaly with weather context | Equipment fault investigation |
| `loss_attribution.csv` | Per (meter, year, season) loss breakdown | The core data for custom analysis |