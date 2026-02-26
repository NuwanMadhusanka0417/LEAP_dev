Best day to start analytics.

Library bundoora - 2020-04-16 07:00.
Common best day t bundoora - 2020-07-02 08:00





# Data Cleaning Report: Solar Meter Readings

## Dataset

| Property | Details |
|----------|---------|
| **File** | `SolarMeterReadings1hour.csv` |
| **Size** | 1,443,200 rows, 28 solar meters |
| **Locations** | Bundoora, Mildura, Shepparton (Victoria, AU) |
| **Period** | January 2020 – November 2025 |
| **Pipelines** | V1 (imputation-based) · V2 (flag-based, no imputation) |

---

## Issues & Resolutions

### Issue 1 — Solcast Weather Data Time Misalignment

**Problem:** The Solcast weather dataset (GHI, zenith, temperature, etc.) may be offset by up to ±3 hours relative to meter timestamps, causing incorrect zenith/GHI values to be associated with each hour and corrupting all downstream daylight detection and imputation.

**Resolution:** `find_best_solcast_shift()` *(lines 114–139)* tries shifting Solcast by −3 to +3 hours and picks the shift that maximises the fraction of meter timestamps that get non-NaN GHI on merge. The best shift is applied before any cleaning begins.

---

### Issue 2 — Non-Zero Readings at Nighttime

**Problem:** 32,046 nighttime rows (hours when solar zenith ≥ 90°) had non-zero meter readings ranging from tiny noise (0.003 kWh) to absurd values (120 kWh at 3am). Solar panels physically cannot generate power at night. Across all 28 meters, **124,683 nighttime values** were contaminated.

**Resolution:** `definite_night_mask()` *(lines 186–200)* determines when the sun is definitely down using two methods in priority order:
1. Solcast zenith ≥ 90° (astronomically precise)
2. Fallback to hours 21:00–04:00 (Victoria AU) when zenith data is missing

All readings during definite night are forced to zero at the very start of cleaning *(lines 461–463 in V1, 633–635 in V2)*.

---

### Issue 3 — Post-Imputation Night Contamination

**Problem:** The XGBoost imputation model can predict non-zero values for nighttime hours because it learns from the relationship between weather features and power output, and may not perfectly distinguish day from night in its predictions.

**Resolution:** After all imputation completes, nighttime values are enforced to zero a second time *(line 522 in V1)*. This guarantees that no imputation method can introduce fake nighttime generation into the final output.

---

### Issue 4 — Stuck/Frozen Meter Values

**Problem:** Several meters reported the same non-zero value continuously for long periods (8 to 127 consecutive hours), spanning both day and night, indicating meter malfunction (frozen/stuck sensor). Across all meters, **3,316 hours of stuck readings** were detected in **330 separate blocks**.

Notable examples:
- `bun_bg` — 0.003944 repeating for 127 hours
- `bun_cs1` — 1.565974 repeating for 92 hours
- `bun_carm2` — 0.000592 repeating for 52 hours

**Resolution:** `detect_stuck_meter()` *(lines 203–224)* scans the time series for runs of identical non-zero values lasting ≥ 8 consecutive hours (`STUCK_MIN_HOURS`). All values in detected stuck blocks are set to NaN (V1, line 468) or zero (V2, line 640), then treated as missing data for imputation.

---

### Issue 5 — Missing-as-Zero During Daylight (Unreported Outages)

**Problem:** During sunny daylight hours (zenith < 90°, GHI > 50), some meters report exactly 0.0 or values ≤ 1e-6 kWh. These represent meter communication failures or unreported outages — not real zero-generation hours — and would deflate performance metrics if left as-is.

**Resolution:** `flag_missing_as_zero()` *(lines 177–183)* identifies hours where the sun is provably up (via zenith, GHI, or fallback hour window) but the reading is ≤ 1e-6. These are re-coded to NaN so they can be properly imputed.

---

### Issue 6 — Extended Daylight Outage Blocks (≥ 3 Consecutive Hours)

**Problem:** Some meters have blocks of 3 or more consecutive daylight hours with zero/missing readings, indicating sustained outages (inverter trips, grid disconnection, etc.). Simple interpolation cannot reliably reconstruct multi-hour outages.

**Resolution:** `detect_outage_blocks()` *(lines 227–252)* identifies consecutive daylight-zero blocks ≥ 3 hours. These are forced to NaN and routed to XGBoost model-based imputation (not median), which uses Solcast weather features (GHI, DNI, DHI, zenith, temperature, cloud opacity) plus cyclical time features to predict expected generation.

In V2, these blocks are flagged as `outage_flag = True` and excluded from analysis via `analysis_valid`, with days having >10% outage rate dropped entirely.

---

### Issue 7 — Unreasonably High Outlier Values

**Problem:** Some meters report extremely high readings that exceed the physical capacity of the solar installation. For example:
- `bun_ltss_pv_lgcmeter` — 874 readings above 300 kWh (max 449 kWh)
- `bun_library` — 411 readings above 300 kWh

**Resolution:** `detect_high_outliers()` *(lines 255–273)* uses a dual-fence approach — the stricter (lower) of the two thresholds is applied:
- IQR fence: Q75 + 3.0 × IQR
- Percentile cap: 99.5th percentile

Values above the threshold are capped to the threshold value. Additionally, a post-cleaning robust cap *(lines 524–530)* computes the 99.99th percentile from clean daylight hours only (GHI > 50, value > 0) and clips the entire series to that value.

---

### Issue 8 — Short Data Gaps (1–2 Hours)

**Problem:** Brief 1–2 hour gaps caused by transient communication failures, brief meter restarts, or data logging glitches.

**Resolution:** `impute_short_gaps_time()` *(lines 303–311)* fills these using time-based linear interpolation with a strict limit of 2 hours. This is physically reasonable since solar output changes smoothly over 1–2 hours.

---

### Issue 9 — Medium Data Gaps (3–24 Hours)

**Problem:** Gaps spanning several hours to a full day, where simple linear interpolation would be unreliable because solar output varies significantly with weather and time of day.

**Resolution:** `impute_medium_gaps_similar_time()` *(lines 314–360)* fills these using the median of the same hour-of-day from nearby days. The donor pool requires GHI > 50 and value > 0 (to avoid using other outage days as donors). If too few donors are found within ±7 days, the window expands adaptively up to ±30 days.

---

### Issue 10 — Long Data Gaps (> 24 Hours) and Outage Blocks

**Problem:** Multi-day gaps and extended outage blocks where no same-hour median approach can work reliably because conditions may have changed significantly.

**Resolution:** `impute_long_gaps_model()` *(lines 388–420)* uses an XGBoost regression model trained on the meter's own clean data. Features include:
- **Solcast weather:** GHI, DNI, DHI, zenith, air temperature, cloud opacity
- **Cyclical time encoding:** sin/cos of hour-of-day and day-of-year

The model learns the meter's specific relationship between weather conditions and power output, then predicts what generation should have been during the gap.

---

### Issue 11 — Negative Values

**Problem:** Any negative meter readings are physically impossible for solar generation.

**Resolution:** After all imputation, `vals.clip(lower=0.0)` *(line 521)* enforces a floor of zero on the entire series.

---

### Issue 12 — Solcast Weather Data Resolution Mismatch

**Problem:** Solcast data arrives at 30-minute resolution while meter data is hourly, making direct merging impossible.

**Resolution:** `resample_solcast_to_hourly()` *(lines 104–111)* resamples Solcast from 30-minute to hourly using mean aggregation, then floors timestamps to the hour boundary for clean joining.

---

### Issue 13 — Inconsistent Column Naming

**Problem:** Input CSV files may have varying column names for the meter identifier and reading value columns.

**Resolution:** `load_meter_data()` *(lines 75–89)* auto-detects and normalises column names — searching for columns containing `"reading"`, `"power"`, or `"meter"` in their names and renaming them to the standard `meter_reading` and `meter`.

---

### Issue 14 — Invalid/Unparseable Timestamps

**Problem:** Some rows may have malformed or missing timestamp values that would cause downstream processing errors.

**Resolution:** Both `load_meter_data()` *(lines 79–80)* and `load_solcast_data()` *(lines 99–100)* parse timestamps with `errors="coerce"` (converting unparseable values to `NaT`) and then drop all rows with null timestamps.

---

### Issue 15 — V2 Analysis Quality Control

**Problem:** For degradation and seasonality analysis, using all data (including imputed values, low-irradiance hours, and outage-contaminated days) can bias results.

**Resolution:** The V2 pipeline (`process_one_meter_v2()`, lines 616–687) adds per-row flags:
- `outage_flag` — True if the hour should have had production but reading is zero
- `analysis_valid` — True if the hour has good irradiance (GHI > 50) and is not an outage

Days with outage rate > 10% of valid daylight hours are flagged for exclusion. Performance Ratio (PR = actual / XGBoost-expected) is only computed on `analysis_valid` hours.

---

## Summary Table

| # | Issue | Function | Lines | Scale |
|---|-------|----------|-------|-------|
| 1 | Solcast time misalignment | `find_best_solcast_shift()` | 114–139 | Up to ±3h offset corrected |
| 2 | Non-zero nighttime readings | `definite_night_mask()` | 186–200, 461–463 | 124,683 values zeroed |
| 3 | Imputation creates night values | Post-imputation enforcement | 522 | Catches all XGBoost night predictions |
| 4 | Stuck/frozen meters | `detect_stuck_meter()` | 203–224, 466–468 | 3,316 hours / 330 blocks |
| 5 | Missing-as-zero (daylight) | `flag_missing_as_zero()` | 177–183, 474–476 | Varies per meter |
| 6 | Extended outage blocks (≥3h) | `detect_outage_blocks()` | 227–252, 478–484 | Hundreds of blocks |
| 7 | Unreasonably high outliers | `detect_high_outliers()` | 255–273, 486–489 | ~1,285 readings > 300 kWh |
| 8 | Short gaps (1–2h) | `impute_short_gaps_time()` | 303–311 | ~1,400 per meter |
| 9 | Medium gaps (3–24h) | `impute_medium_gaps_similar_time()` | 314–360 | Adaptive ±7–30d donors |
| 10 | Long gaps (>24h) + outages | `impute_long_gaps_model()` | 388–420 | XGBoost weather-based |
| 11 | Negative values | `clip(lower=0.0)` | 521 | Safety floor |
| 12 | Solcast resolution mismatch | `resample_solcast_to_hourly()` | 104–111 | 30-min → 1h |
| 13 | Inconsistent column names | `load_meter_data()` | 75–89 | Auto-detection |
| 14 | Invalid timestamps | `pd.to_datetime(errors="coerce")` | 79–80, 99–100 | Drop unparseable rows |
| 15 | V2 analysis quality control | `process_one_meter_v2()` | 616–687 | Per-row flags + day exclusion |


