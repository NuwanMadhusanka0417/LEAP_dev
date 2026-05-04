# Data requirements — PVLib expected power

## Which Solcast file is best?

| | **`solcast_df_cleaned_v2.csv`** | **`solcast_df.csv`** (raw) |
|---|-------------------------------|---------------------------|
| **Timestep** | Hourly | 30-minute |
| **Sites** | Single continuous series (Bundoora-only in your pipeline) | Many campuses per timestamp (BUNDOORA, BENDIGO, MILDURA, …) |
| **Size / use** | ~49k rows — fast, matches hourly meter data | ~490k+ rows — must **filter `campus == BUNDOORA`** (~98k rows) then resample |
| **Recommendation** | **Use this** for Bundoora expected power and alignment with `SolarMeterReadings` | Use if you need **native 30 min** or **other campuses** |

The cleaned file is **hourly**, **one place**, and **QC’d in your workflow**, so it is the **best default** for La Trobe Bundoora PV modelling. The raw file is the multi-site archive; for Bundoora-only analysis it adds merge/resample steps and duplicate timestamps across campuses.

## Weather input (you already have)

| File | Use |
|------|-----|
| **`data/solcast_df_cleaned_v2.csv`** | **Preferred** for Bundoora PV modelling: hourly timestamps, consistent columns, values suited to downstream analysis. |
| `data/solcast_df.csv` | Raw archive: **30-minute** steps, **multiple campuses** per timestamp. Script: `--weather raw --campus BUNDOORA` (resampled to hourly mean). |

Required columns from Solcast (both files, after filtering): **`timestamp`**, **`ghi`**, **`dni`**, **`dhi`**, **`air_temp`**, **`wind_speed_10m`** (optional but used for cell temperature). **`zenith`** is optional; solar position is computed in PVLib from time + location.

## What you should provide for realistic “expected” power

These are **not** in the Solcast CSV. They describe the physical PV system:

| Item | Typical source | Notes |
|------|------------------|--------|
| **Latitude, longitude, elevation** | Fixed for Bundoora campus | Defaults are set in `config.py`; replace if your array is elsewhere. |
| **Array DC nameplate** (kW) | As-built / inverter limits | Scales all expected power linearly. |
| **Tilt** (deg) | Roof / rack design | Strong effect on POA irradiance. |
| **Azimuth** (deg) | Roof orientation | PVLib: **0° = north**, **90° = east**, **180° = south** (Australian **north-facing** arrays are often **~0°**). |
| **Module technology** | Datasheet | Affects temperature coefficient (PVWatts default is fine for a first pass). |
| **Inverter AC limit** (kW) | Nameplate | Optional clipping in AC stage. |
| **Albedo** | Ground cover | Default 0.2; Solcast also has `albedo` column — script can use it. |

## Optional improvements (later)

- **Per-meter** tilt/azimuth if arrays differ (output one column per meter or separate runs).
- **Horizon / shading** — not in standard Solcast; needs site survey.
- **Soiling** — not modelled in baseline “clear expected”; add a loss factor if needed.

## Output of this folder’s script

- CSV with timestamps and modelled **POA irradiance**, **cell temperature**, **DC power**, **AC power** (PVWatts-style), using the weather file you select.
