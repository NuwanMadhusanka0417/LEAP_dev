"""
Clustering-based seasonal degradation analysis
==============================================
- Uses 2020 as IDEAL (baseline) power generation year.
- Computes degradation for every year vs 2020 and plots yearly degradation.
- Clusters all seasons of the year and analyzes seasonal-wise degradation
  and degradation amount per season.
- Configured for a single meter and region (Library meter, BUNDOORA).


- Uses cleaned datasets: SolarMeterReadings1hour_cleaned.csv, solcast_df_cleaned.csv
- Uses common time range (intersection of meter and weather data).
- Night time: zenith >= 90 or GHI below threshold; meter readings set to 0 for analysis.

"""

import os
import pandas as pd
import numpy as np



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths (run from project root or seasonality/)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
RESULTS_DIR = os.path.join(BASE, 'Results_seasonality')
# Cleaned datasets (from data_cleaning.py)
CLEANED_METER_CSV = 'SolarMeterReadings1hour_cleaned.csv'
CLEANED_SOLCAST_CSV = 'solcast_df_cleaned.csv'
# Night detection: sun below horizon (zenith >= 90) or no irradiance (GHI < threshold)
ZENITH_NIGHT_DEG = 90
GHI_NIGHT_THRESHOLD_WM2 = 20.0

os.makedirs(RESULTS_DIR, exist_ok=True)

# Experiment: single meter and region
TARGET_METER = "solar.bun_library#realenergyintotheload#kwh"  # Library meter
TARGET_REGION = "BUNDOORA"  # Bundoora area (meter name "bun_*" implies Bundoora)

# Southern hemisphere seasons (Australia)
SEASON_MAP = {
    12: 'Summer', 1: 'Summer', 2: 'Summer',
    3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
    6: 'Winter', 7: 'Winter', 8: 'Winter',
    9: 'Spring', 10: 'Spring', 11: 'Spring'
}

print("=" * 80)
print("CLUSTERING SEASONAL DEGRADATION ANALYSIS")
print("Meter: Library (solar.bun_library#realenergyintotheload#kwh)")
print("Region: BUNDOORA")
print("Baseline (ideal) year: 2020")
print("=" * 80)

# =============================================================================
# STEP 1: LOAD CLEANED METER + SOLCAST, COMMON TIME RANGE, NIGHT = 0
# =============================================================================
print("\n[STEP 1] Loading cleaned meter and weather (Solcast) data...")

meter_path = os.path.join(DATA_DIR, CLEANED_METER_CSV)
if not os.path.isfile(meter_path):
    meter_path = os.path.join(DATA_DIR, 'SolarMeterReadings1hour.csv')
    print(f"  (Cleaned not found, using {os.path.basename(meter_path)})")
meter_df = pd.read_csv(meter_path)
meter_df['timestamp'] = pd.to_datetime(meter_df['timestamp'], errors='coerce')
meter_df = meter_df.dropna(subset=['timestamp'])

# Filter to Library meter (Bundoora)
meter_col = 'meter' if 'meter' in meter_df.columns else [c for c in meter_df.columns if 'meter' in c.lower()][0]
meter_df = meter_df[meter_df[meter_col].astype(str).str.strip() == TARGET_METER].copy()
if meter_df.empty:
    raise ValueError(f"No data found for meter: {TARGET_METER}. Check meter name and CSV.")

power_col = 'meter_reading' if 'meter_reading' in meter_df.columns else 'Meter_reading'
if power_col not in meter_df.columns:
    power_col = [c for c in meter_df.columns if 'reading' in c.lower() or 'power' in c.lower()][0]
meter_df = meter_df.rename(columns={power_col: 'power'})

# Load cleaned Solcast (hourly)
solcast_path = os.path.join(DATA_DIR, CLEANED_SOLCAST_CSV)
if not os.path.isfile(solcast_path):
    solcast_path = os.path.join(DATA_DIR, 'solcast_df.csv')
    print(f"  (Cleaned Solcast not found, using {os.path.basename(solcast_path)} — will resample to hourly)")
solcast_df = pd.read_csv(solcast_path)
solcast_df['timestamp'] = pd.to_datetime(solcast_df['timestamp'], errors='coerce')
solcast_df = solcast_df.dropna(subset=['timestamp'])
# Ensure hourly: resample to 1h if data is 30-min
if len(solcast_df) > 1:
    deltas = solcast_df['timestamp'].diff().dropna()
    if len(deltas) > 0 and deltas.min() < pd.Timedelta('45min'):
        solcast_df = solcast_df.set_index('timestamp').resample('1h').mean().reset_index()
if 'zenith' not in solcast_df.columns or 'ghi' not in solcast_df.columns:
    raise ValueError("Solcast must contain 'zenith' and 'ghi' columns.")
# Normalise timezone for merge (both must be naive or same tz)
def _strip_tz(ser):
    if getattr(ser.dt, 'tz', None) is not None:
        return ser.dt.tz_localize(None)
    return ser
solcast_df['timestamp'] = _strip_tz(pd.to_datetime(solcast_df['timestamp']))
meter_df['timestamp'] = _strip_tz(pd.to_datetime(meter_df['timestamp']))

# Common time range: inner merge keeps only timestamps present in both
solcast_merge = solcast_df[['timestamp', 'zenith', 'ghi']].copy()
meter_df = meter_df.merge(solcast_merge, on='timestamp', how='inner')
meter_df = meter_df.sort_values('timestamp').reset_index(drop=True)
print(f"  Common time range: {meter_df['timestamp'].min()} to {meter_df['timestamp'].max()} ({len(meter_df)} hourly records)")

# Night: zenith >= 90 (sun below horizon) or GHI below threshold — use 0 for analysis
is_night = (meter_df['zenith'] >= ZENITH_NIGHT_DEG) | (meter_df['ghi'] < GHI_NIGHT_THRESHOLD_WM2)
meter_df.loc[is_night, 'power'] = 0.0
print(f"  Night hours (zenith>={ZENITH_NIGHT_DEG} or GHI<{GHI_NIGHT_THRESHOLD_WM2}) set to 0: {is_night.sum()}")

# One row per timestamp (single meter)
actual_total = meter_df[['timestamp', 'power']].copy()
actual_total.columns = ['timestamp', 'actual_power']
actual_total['year'] = actual_total['timestamp'].dt.year
actual_total['month'] = actual_total['timestamp'].dt.month
actual_total['day_of_year'] = actual_total['timestamp'].dt.dayofyear
actual_total['hour'] = actual_total['timestamp'].dt.hour
actual_total['date'] = actual_total['timestamp'].dt.date

years = sorted(actual_total['year'].unique())
print(f"  Loaded {len(actual_total)} hourly records (Library meter, BUNDOORA), years: {years}")

# =============================================================================
# STEP 2: 2020 AS IDEAL BASELINE
# =============================================================================
print("\n[STEP 2] Building 2020 ideal baseline...")

baseline_2020 = actual_total[actual_total['year'] == 2020].copy()
if baseline_2020.empty:
    raise ValueError("No 2020 data found. Cannot use 2020 as baseline.")

# Baseline by (day_of_year, hour) for fair comparison across years
baseline_doy_hour = baseline_2020.groupby(['day_of_year', 'hour'])['actual_power'].mean().reset_index()
baseline_doy_hour.columns = ['day_of_year', 'hour', 'ideal_power_2020']

# Daily baseline (for daily aggregation)
baseline_daily_2020 = baseline_2020.groupby('day_of_year')['actual_power'].sum().reset_index()
baseline_daily_2020.columns = ['day_of_year', 'ideal_daily_2020']

print(f"  2020 baseline: {len(baseline_2020)} hours, {baseline_2020['date'].nunique()} days")
print(f"  Mean hourly power 2020: {baseline_2020['actual_power'].mean():.2f} kWh")

# =============================================================================
# STEP 3: DEGRADATION PER YEAR (vs 2020)
# =============================================================================
print("\n[STEP 3] Computing degradation for every year vs 2020...")

# Merge each year with 2020 baseline on (day_of_year, hour)
yearly_degradation = []
yearly_totals = []

for year in years:
    df_y = actual_total[actual_total['year'] == year].copy()
    if df_y.empty:
        continue
    merged = df_y.merge(baseline_doy_hour, on=['day_of_year', 'hour'], how='inner')
    if merged.empty:
        continue
    # Degradation: (ideal_2020 - actual) / ideal_2020 * 100 (%), and amount (kWh)
    merged['deficit_kwh'] = merged['ideal_power_2020'] - merged['actual_power']
    merged['deficit_pct'] = np.where(
        merged['ideal_power_2020'] > 1,
        (merged['deficit_kwh'] / merged['ideal_power_2020']) * 100,
        0
    )
    n_hours = len(merged)
    total_ideal = merged['ideal_power_2020'].sum()
    total_actual = merged['actual_power'].sum()
    total_deficit = merged['deficit_kwh'].sum()
    deg_pct = (total_deficit / total_ideal * 100) if total_ideal > 0 else 0
    yearly_degradation.append({
        'year': year,
        'degradation_pct': deg_pct,
        'deficit_kwh': total_deficit,
        'total_ideal_kwh': total_ideal,
        'total_actual_kwh': total_actual,
        'n_hours_matched': n_hours
    })
    yearly_totals.append({
        'year': year,
        'actual_kwh': total_actual,
        'ideal_kwh': total_ideal,
        'deficit_kwh': total_deficit
    })

df_yearly = pd.DataFrame(yearly_degradation)
df_yearly_totals = pd.DataFrame(yearly_totals)
print(df_yearly[['year', 'degradation_pct', 'deficit_kwh', 'total_actual_kwh']].to_string(index=False))

# =============================================================================
# STEP 4: GRAPH - YEARLY DEGRADATION
# =============================================================================
print("\n[STEP 4] Drawing yearly degradation graph...")

fig1, axes = plt.subplots(2, 1, figsize=(10, 8))

ax1 = axes[0]
x = df_yearly['year']
ax1.bar(x - 0.2, df_yearly['degradation_pct'], width=0.4, label='Degradation %', color='coral', alpha=0.9)
ax1.axhline(0, color='black', linestyle='-', lw=0.5)
ax1.set_ylabel('Degradation vs 2020 (%)')
ax1.set_xlabel('Year')
ax1.set_title(f'Yearly Degradation vs 2020 — {TARGET_REGION} Library')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
ax2.bar(x - 0.2, df_yearly['deficit_kwh'] / 1000, width=0.4, label='Deficit (MWh)', color='steelblue', alpha=0.9)
ax2.axhline(0, color='black', linestyle='-', lw=0.5)
ax2.set_ylabel('Deficit (MWh)')
ax2.set_xlabel('Year')
ax2.set_title(f'Yearly Energy Deficit vs 2020 (MWh) — {TARGET_REGION} Library')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig1.savefig(os.path.join(RESULTS_DIR, 'yearly_degradation_vs_2020.png'), dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"  Saved: Results/yearly_degradation_vs_2020.png")

# =============================================================================
# STEP 5: SEASON LABELS AND SEASONAL DEGRADATION
# =============================================================================
print("\n[STEP 5] Assigning seasons and computing seasonal degradation...")

actual_total['season'] = actual_total['month'].map(SEASON_MAP)
# Merge baseline into full data for degradation by season/year
actual_with_baseline = actual_total.merge(
    baseline_doy_hour, on=['day_of_year', 'hour'], how='inner'
)
actual_with_baseline['deficit_kwh'] = (
    actual_with_baseline['ideal_power_2020'] - actual_with_baseline['actual_power']
)
actual_with_baseline['deficit_pct'] = np.where(
    actual_with_baseline['ideal_power_2020'] > 1,
    (actual_with_baseline['deficit_kwh'] / actual_with_baseline['ideal_power_2020']) * 100,
    0
)

season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
seasonal_degradation = []
for year in years:
    for season in season_order:
        sub = actual_with_baseline[(actual_with_baseline['year'] == year) &
                                   (actual_with_baseline['season'] == season)]
        if sub.empty:
            continue
        ideal_s = sub['ideal_power_2020'].sum()
        actual_s = sub['actual_power'].sum()
        deficit_s = sub['deficit_kwh'].sum()
        deg_pct_s = (deficit_s / ideal_s * 100) if ideal_s > 0 else 0
        seasonal_degradation.append({
            'year': year,
            'season': season,
            'degradation_pct': deg_pct_s,
            'deficit_kwh': deficit_s,
            'ideal_kwh': ideal_s,
            'actual_kwh': actual_s,
            'n_hours': len(sub)
        })

df_seasonal = pd.DataFrame(seasonal_degradation)

# =============================================================================
# STEP 6: CLUSTERING SEASONS (KMeans on day-of-year / daily profile)
# =============================================================================
print("\n[STEP 6] Clustering seasonal profiles...")

# Daily aggregates for clustering
daily_agg = actual_total.groupby(['date', 'year', 'day_of_year', 'month']).agg({
    'actual_power': 'sum'
}).reset_index()
daily_agg['season'] = daily_agg['month'].map(SEASON_MAP)
daily_agg = daily_agg.merge(baseline_daily_2020, on='day_of_year', how='inner')
daily_agg['deficit'] = daily_agg['ideal_daily_2020'] - daily_agg['actual_power']
daily_agg['deficit_pct'] = np.where(
    daily_agg['ideal_daily_2020'] > 1,
    (daily_agg['deficit'] / daily_agg['ideal_daily_2020']) * 100,
    0
)

# Features for clustering: day_of_year (cyclic), month, and daily production ratio
daily_agg['doy_sin'] = np.sin(2 * np.pi * daily_agg['day_of_year'] / 365.25)
daily_agg['doy_cos'] = np.cos(2 * np.pi * daily_agg['day_of_year'] / 365.25)
cluster_features = ['doy_sin', 'doy_cos', 'month']
X_cluster = daily_agg[cluster_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

n_clusters = 4  # 4 seasonal clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
daily_agg['cluster'] = kmeans.fit_predict(X_scaled)

# Label clusters by dominant calendar season (month)
cluster_to_season = {}
for c in range(n_clusters):
    months = daily_agg[daily_agg['cluster'] == c]['month'].value_counts()
    dominant_month = months.index[0]
    cluster_to_season[c] = SEASON_MAP.get(dominant_month, f'Cluster_{c}')
daily_agg['cluster_season'] = daily_agg['cluster'].map(cluster_to_season)

print(f"  KMeans clusters: {n_clusters}")
for c in range(n_clusters):
    count = (daily_agg['cluster'] == c).sum()
    print(f"    Cluster {c} ({cluster_to_season[c]}): {count} days")

# Degradation by cluster and year
cluster_yearly = []
for year in years:
    for c in range(n_clusters):
        sub = daily_agg[(daily_agg['year'] == year) & (daily_agg['cluster'] == c)]
        if sub.empty:
            continue
        ideal_c = sub['ideal_daily_2020'].sum()
        actual_c = sub['actual_power'].sum()
        deficit_c = sub['deficit'].sum()
        deg_pct_c = (deficit_c / ideal_c * 100) if ideal_c > 0 else 0
        cluster_yearly.append({
            'year': year,
            'cluster': c,
            'cluster_label': cluster_to_season[c],
            'degradation_pct': deg_pct_c,
            'deficit_kwh': deficit_c,
            'ideal_kwh': ideal_c,
            'actual_kwh': actual_c,
            'n_days': len(sub)
        })

df_cluster_yearly = pd.DataFrame(cluster_yearly)

# =============================================================================
# STEP 7: PLOTS - SEASONAL AND CLUSTER DEGRADATION
# =============================================================================
print("\n[STEP 7] Drawing seasonal and cluster degradation graphs...")

# 7a) Seasonal degradation by year (bar)
fig2, axes = plt.subplots(2, 1, figsize=(12, 9))

ax = axes[0]
pivot_season_pct = df_seasonal.pivot(index='year', columns='season', values='degradation_pct')
pivot_season_pct = pivot_season_pct[[s for s in season_order if s in pivot_season_pct.columns]]
pivot_season_pct.plot(kind='bar', ax=ax, width=0.8)
ax.axhline(0, color='black', linestyle='-', lw=0.5)
ax.set_ylabel('Degradation vs 2020 (%)')
ax.set_xlabel('Year')
ax.set_title(f'Seasonal Degradation by Year (%) — {TARGET_REGION} Library')
ax.legend(title='Season', bbox_to_anchor=(1.02, 1))
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

ax2 = axes[1]
pivot_season_kwh = df_seasonal.pivot(index='year', columns='season', values='deficit_kwh')
pivot_season_kwh = pivot_season_kwh / 1000  # MWh
pivot_season_kwh = pivot_season_kwh[[s for s in season_order if s in pivot_season_kwh.columns]]
pivot_season_kwh.plot(kind='bar', ax=ax2, width=0.8)
ax2.axhline(0, color='black', linestyle='-', lw=0.5)
ax2.set_ylabel('Deficit (MWh)')
ax2.set_xlabel('Year')
ax2.set_title(f'Seasonal Degradation Amount by Year (MWh) — {TARGET_REGION} Library')
ax2.legend(title='Season', bbox_to_anchor=(1.02, 1))
ax2.grid(True, alpha=0.3, axis='y')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR, 'seasonal_degradation_by_year.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved: Results/seasonal_degradation_by_year.png")

# 7b) Cluster-wise degradation by year
fig3, axes = plt.subplots(2, 1, figsize=(12, 9))
ax = axes[0]
for c in range(n_clusters):
    label = cluster_to_season[c]
    df_c = df_cluster_yearly[df_cluster_yearly['cluster'] == c]
    ax.plot(df_c['year'], df_c['degradation_pct'], marker='o', label=label, lw=2)
ax.set_ylabel('Degradation vs 2020 (%)')
ax.set_xlabel('Year')
ax.set_title(f'Cluster-wise Degradation % by Year — {TARGET_REGION} Library')
ax.legend()
ax.grid(True, alpha=0.3)
ax = axes[1]
for c in range(n_clusters):
    label = cluster_to_season[c]
    df_c = df_cluster_yearly[df_cluster_yearly['cluster'] == c]
    ax.plot(df_c['year'], df_c['deficit_kwh'] / 1000, marker='s', label=label, lw=2)
ax.set_ylabel('Deficit (MWh)')
ax.set_xlabel('Year')
ax.set_title(f'Cluster-wise Degradation Amount by Year (MWh) — {TARGET_REGION} Library')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig3.savefig(os.path.join(RESULTS_DIR, 'cluster_seasonal_degradation.png'), dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"  Saved: Results/cluster_seasonal_degradation.png")

# 7c) Combined overview: one figure with yearly + seasonal heatmap
fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
ax = axes[0, 0]
ax.bar(df_yearly['year'], df_yearly['degradation_pct'], color='coral', alpha=0.9, edgecolor='darkred')
ax.axhline(0, color='black', linestyle='-', lw=0.5)
ax.set_ylabel('Degradation (%)')
ax.set_title(f'Yearly Degradation vs 2020 — {TARGET_REGION} Library')
ax.grid(True, alpha=0.3, axis='y')

ax = axes[0, 1]
pivot_hm = df_seasonal.pivot(index='year', columns='season', values='degradation_pct')
pivot_hm = pivot_hm[[s for s in season_order if s in pivot_hm.columns]]
sns.heatmap(pivot_hm, ax=ax, annot=True, fmt='.1f', cmap='RdYlGn_r', center=0, cbar_kws={'label': 'Degradation %'})
ax.set_title(f'Seasonal Degradation % by Year — {TARGET_REGION} Library')
ax.set_xlabel('Season')

ax = axes[1, 0]
pivot_kwh = df_seasonal.pivot(index='year', columns='season', values='deficit_kwh') / 1000
pivot_kwh = pivot_kwh[[s for s in season_order if s in pivot_kwh.columns]]
sns.heatmap(pivot_kwh, ax=ax, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Deficit (MWh)'})
ax.set_title(f'Seasonal Deficit (MWh) by Year — {TARGET_REGION} Library')
ax.set_xlabel('Season')

ax = axes[1, 1]
for c in range(n_clusters):
    label = cluster_to_season[c]
    df_c = df_cluster_yearly[df_cluster_yearly['cluster'] == c]
    ax.plot(df_c['year'], df_c['degradation_pct'], marker='o', label=label, lw=2)
ax.set_ylabel('Degradation (%)')
ax.set_xlabel('Year')
ax.set_title(f'Cluster-wise Degradation % — {TARGET_REGION} Library')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig4.savefig(os.path.join(RESULTS_DIR, 'degradation_overview.png'), dpi=150, bbox_inches='tight')
plt.close(fig4)
print(f"  Saved: Results/degradation_overview.png")

# =============================================================================
# STEP 8: SUMMARY TABLES AND EXPORT
# =============================================================================
print("\n[STEP 8] Saving summary CSVs...")

df_yearly.to_csv(os.path.join(RESULTS_DIR, 'yearly_degradation_summary.csv'), index=False)
df_seasonal.to_csv(os.path.join(RESULTS_DIR, 'seasonal_degradation_summary.csv'), index=False)
df_cluster_yearly.to_csv(os.path.join(RESULTS_DIR, 'cluster_seasonal_degradation_summary.csv'), index=False)

print("  Saved: yearly_degradation_summary.csv")
print("  Saved: seasonal_degradation_summary.csv")
print("  Saved: cluster_seasonal_degradation_summary.csv")

# Print seasonal-wise summary
print("\n" + "=" * 80)
print(f"SEASONAL-WISE DEGRADATION SUMMARY — {TARGET_REGION} Library (vs 2020 baseline)")
print("=" * 80)
print("\nBy calendar season:")
print(df_seasonal.groupby('season').agg({
    'degradation_pct': 'mean',
    'deficit_kwh': 'sum',
    'n_hours': 'sum'
}).round(2).to_string())
print("\nBy cluster (learned seasonal groups):")
print(df_cluster_yearly.groupby('cluster_label').agg({
    'degradation_pct': 'mean',
    'deficit_kwh': 'sum',
    'n_days': 'sum'
}).round(2).to_string())
print("\n" + "=" * 80)
print(f"DONE. Library meter ({TARGET_REGION}) — check {RESULTS_DIR} for graphs and CSVs.")
print("=" * 80)
