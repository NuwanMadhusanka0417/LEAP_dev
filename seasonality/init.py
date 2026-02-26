"""
FINAL SOLAR DEGRADATION ANALYSIS
=================================
Strategy: Use simulation to learn IDEAL weather→power physics,
          then apply to actual data to detect degradation

Key Innovation: Simulation teaches us "perfect" PV behavior
                Actual vs this model reveals real-world degradation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 90)
print("SOLAR DEGRADATION ANALYSIS - SIMULATION-TRAINED MODEL")
print("Learning ideal physics from simulation, detecting degradation in actual data")
print("=" * 90)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading all datasets...")

# Actual meter readings (2020) - aggregate all meters
meter_df = pd.read_csv('../data/SolarMeterReadings1hour.csv')
meter_df['timestamp'] = pd.to_datetime(meter_df['timestamp'])
actual_total = meter_df.groupby('timestamp')['meter_reading'].sum().reset_index()
actual_total.columns = ['timestamp', 'actual_power']

print(f"✓ Actual data (2020): {len(actual_total)} hours, {meter_df['meter'].nunique()} meters aggregated")
print(f"  Power stats: mean={actual_total['actual_power'].mean():.2f}, max={actual_total['actual_power'].max():.2f} kWh")

# Simulation (2021) - ideal performance
sim_df = pd.read_csv('../data/solarsitesimulation.csv')
sim_df['timestamp'] = pd.to_datetime(sim_df['timestamp'])

print(f"✓ Simulation (2021): {len(sim_df)} hours")
print(f"  Grid power: mean={sim_df['grid_power'].mean():.2f}, max={sim_df['grid_power'].max():.2f} kWh")

# Weather (2020)
weather_df = pd.read_csv('../data/solcast_df.csv')
weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
weather_df['timestamp_hour'] = weather_df['timestamp'].dt.floor('H')
weather_hourly = weather_df.groupby('timestamp_hour').agg({
    'ghi': 'mean',
    'dni': 'mean',
    'dhi': 'mean',
    'air_temp': 'mean',
    'relative_humidity': 'mean',
    'cloud_opacity': 'mean',
    'wind_speed_10m': 'mean',
    'zenith': 'mean',
    'azimuth': 'mean'
}).reset_index().rename(columns={'timestamp_hour': 'timestamp'})

print(f"✓ Weather data (2020): {len(weather_hourly)} hours")

# ============================================================================
# STEP 2: TRAIN MODEL ON SIMULATION (LEARN IDEAL PHYSICS)
# ============================================================================
print("\n[STEP 2] Training model on simulation to learn ideal physics...")

# Prepare simulation features
sim_df['hour'] = sim_df['timestamp'].dt.hour
sim_df['day_of_year'] = sim_df['timestamp'].dt.dayofyear
sim_df['day_sin'] = np.sin(2 * np.pi * sim_df['day_of_year'] / 365.25)
sim_df['day_cos'] = np.cos(2 * np.pi * sim_df['day_of_year'] / 365.25)
sim_df['hour_sin'] = np.sin(2 * np.pi * sim_df['hour'] / 24)
sim_df['hour_cos'] = np.cos(2 * np.pi * sim_df['hour'] / 24)

# Solar position features
sim_df['cos_zenith'] = np.cos(np.radians(sim_df['solar_altitude_angle']))
sim_df['sin_altitude'] = np.sin(np.radians(sim_df['solar_altitude_angle']))

# Weather interactions
sim_df['ghi_temp'] = sim_df['global_horizontal_irradiance'] * sim_df['dry_bulb_temperature']
sim_df['temp_efficiency'] = 1 - (sim_df['dry_bulb_temperature'] - 25) * 0.004

# Filter daytime in simulation
sim_day = sim_df[sim_df['global_horizontal_irradiance'] > 10].copy()

# Features for model
features = [
    'global_horizontal_irradiance', 'dry_bulb_temperature', 'windspeed',
    'solar_altitude_angle', 'cos_zenith', 'sin_altitude',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'ghi_temp', 'temp_efficiency'
]

X_sim = sim_day[features].fillna(0)
y_sim = sim_day['grid_power']

print(f"✓ Simulation training set: {len(X_sim)} daytime hours")

# Train model on ideal simulation
ideal_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.9,
    random_state=42
)

ideal_model.fit(X_sim, y_sim)

# Evaluate on simulation
sim_day['predicted_ideal'] = ideal_model.predict(X_sim)
r2_sim = r2_score(y_sim, sim_day['predicted_ideal'])
print(f"✓ Model performance on simulation: R² = {r2_sim:.4f}")

# Feature importance
feat_imp = pd.DataFrame({
    'feature': features,
    'importance': ideal_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 features learned from ideal simulation:")
print(feat_imp.head(10).to_string(index=False))

# ============================================================================
# STEP 3: APPLY TO ACTUAL DATA
# ============================================================================
print("\n[STEP 3] Applying ideal model to actual 2020 data...")

# Merge actual with weather
actual_with_weather = actual_total.merge(weather_hourly, on='timestamp', how='left')

# Create same features
actual_with_weather['hour'] = actual_with_weather['timestamp'].dt.hour
actual_with_weather['day_of_year'] = actual_with_weather['timestamp'].dt.dayofyear
actual_with_weather['day_sin'] = np.sin(2 * np.pi * actual_with_weather['day_of_year'] / 365.25)
actual_with_weather['day_cos'] = np.cos(2 * np.pi * actual_with_weather['day_of_year'] / 365.25)
actual_with_weather['hour_sin'] = np.sin(2 * np.pi * actual_with_weather['hour'] / 24)
actual_with_weather['hour_cos'] = np.cos(2 * np.pi * actual_with_weather['hour'] / 24)

# Map simulation weather features to actual weather features
actual_with_weather['global_horizontal_irradiance'] = actual_with_weather['ghi']
actual_with_weather['dry_bulb_temperature'] = actual_with_weather['air_temp']
actual_with_weather['windspeed'] = actual_with_weather['wind_speed_10m']
actual_with_weather['solar_altitude_angle'] = 90 - actual_with_weather['zenith']
actual_with_weather['cos_zenith'] = np.cos(np.radians(actual_with_weather['zenith']))
actual_with_weather['sin_altitude'] = np.sin(np.radians(actual_with_weather['solar_altitude_angle']))
actual_with_weather['ghi_temp'] = actual_with_weather['ghi'] * actual_with_weather['air_temp']
actual_with_weather['temp_efficiency'] = 1 - (actual_with_weather['air_temp'] - 25) * 0.004

# Filter daytime
actual_day = actual_with_weather[actual_with_weather['ghi'] > 10].copy()
actual_day = actual_day.dropna(subset=features)

print(f"✓ Actual daytime hours: {len(actual_day)}")

# Scale prediction to match actual installation capacity
# Simulation is for single meter, actual is 28 meters aggregated
# Find scaling factor by comparing peak production
sim_peak = sim_day['grid_power'].quantile(0.99)
actual_peak = actual_day['actual_power'].quantile(0.99)
scale_factor = actual_peak / sim_peak

print(f"✓ Capacity scaling factor: {scale_factor:.4f}")
print(f"  (Simulation peak: {sim_peak:.0f}, Actual peak: {actual_peak:.0f})")

# Predict ideal performance for actual conditions
X_actual = actual_day[features]
actual_day['ideal_power'] = ideal_model.predict(X_actual) * scale_factor

# Calculate performance metrics
actual_day['deficit'] = actual_day['ideal_power'] - actual_day['actual_power']
actual_day['deficit_pct'] = (actual_day['deficit'] / (actual_day['ideal_power'] + 0.001)) * 100
actual_day['performance_ratio'] = actual_day['actual_power'] / (actual_day['ideal_power'] + 0.001)

print(f"\n✓ Performance analysis:")
print(f"  Average actual power:   {actual_day['actual_power'].mean():.2f} kWh")
print(f"  Average ideal power:    {actual_day['ideal_power'].mean():.2f} kWh")
print(f"  Performance ratio:      {actual_day['performance_ratio'].mean():.3f}")
print(f"  Average deficit:        {actual_day['deficit'].mean():.2f} kWh ({actual_day['deficit_pct'].mean():.1f}%)")

# ============================================================================
# STEP 4: DEGRADATION TREND ANALYSIS
# ============================================================================
print("\n[STEP 4] Analyzing degradation trend...")

actual_day['days_since_start'] = (actual_day['timestamp'] - actual_day['timestamp'].min()).dt.total_seconds() / 86400

# Daily aggregation
daily = actual_day.groupby(actual_day['timestamp'].dt.date).agg({
    'actual_power': 'sum',
    'ideal_power': 'sum',
    'deficit': 'sum',
    'performance_ratio': 'mean',
    'days_since_start': 'first',
    'ghi': 'mean'
}).reset_index()

daily['deficit_pct'] = (daily['deficit'] / (daily['ideal_power'] + 1)) * 100

# Fit degradation trend
X_trend = daily['days_since_start'].values.reshape(-1, 1)
y_trend = daily['deficit'].values

trend_model = LinearRegression()
trend_model.fit(X_trend, y_trend)
daily['deficit_trend'] = trend_model.predict(X_trend)

deg_per_day = trend_model.coef_[0]
deg_per_year = deg_per_day * 365.25
avg_ideal_daily = daily['ideal_power'].mean()
deg_pct_per_year = (deg_per_year / avg_ideal_daily) * 100

print(f"✓ Degradation metrics:")
print(f"  Daily deficit increase: {deg_per_day:.2f} kWh/day")
print(f"  Annual degradation:     {deg_per_year:.1f} kWh/year")
print(f"  Degradation rate:       {deg_pct_per_year:.2f}% per year")

# Map trend back to hourly
actual_day['deficit_trend'] = np.interp(
    actual_day['days_since_start'],
    daily['days_since_start'],
    daily['deficit_trend']
)

# ============================================================================
# STEP 5: DECOMPOSE RESIDUALS
# ============================================================================
print("\n[STEP 5] Decomposing residual components...")

# Detrended deficit
actual_day['deficit_detrended'] = actual_day['deficit'] - actual_day['deficit_trend']

# Seasonal pattern
seasonal = actual_day.groupby('day_of_year')['deficit_detrended'].mean().reset_index()
seasonal.columns = ['day_of_year', 'seasonal']

if len(seasonal) > 30:
    window = min(51, len(seasonal) if len(seasonal) % 2 == 1 else len(seasonal) - 1)
    seasonal['seasonal'] = savgol_filter(seasonal['seasonal'], window, 3)

actual_day = actual_day.merge(seasonal, on='day_of_year', how='left')

# Events/anomalies
actual_day['events'] = actual_day['deficit_detrended'] - actual_day['seasonal']
threshold = 3 * actual_day['events'].std()
actual_day['is_anomaly'] = np.abs(actual_day['events']) > threshold

print(f"✓ Variance decomposition complete")
print(f"  Seasonal pattern extracted")
print(f"  Anomalies detected: {actual_day['is_anomaly'].sum()} ({100*actual_day['is_anomaly'].mean():.1f}%)")
print(f"  Anomaly threshold: ±{threshold:.1f} kWh")

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================
print("\n[STEP 6] Creating comprehensive visualizations...")

fig = plt.figure(figsize=(22, 18))

# 1. Model performance on simulation
ax1 = plt.subplot(4, 4, 1)
ax1.scatter(sim_day['grid_power'], sim_day['predicted_ideal'], alpha=0.3, s=10)
lim = [0, sim_day['grid_power'].max()]
ax1.plot(lim, lim, 'r--', lw=2)
ax1.set_xlabel('Simulated Power (kWh)')
ax1.set_ylabel('Model Prediction (kWh)')
ax1.set_title(f'Model Training on Simulation\nR² = {r2_sim:.3f}')
ax1.grid(True, alpha=0.3)

# 2. Feature importance
ax2 = plt.subplot(4, 4, 2)
top10 = feat_imp.head(10)
ax2.barh(range(len(top10)), top10['importance'])
ax2.set_yticks(range(len(top10)))
ax2.set_yticklabels(top10['feature'], fontsize=8)
ax2.set_xlabel('Importance')
ax2.set_title('Physics Learned from Simulation')
ax2.grid(True, alpha=0.3, axis='x')

# 3. Actual vs Ideal scatter
ax3 = plt.subplot(4, 4, 3)
sample = actual_day.sample(min(5000, len(actual_day)))
scatter = ax3.scatter(sample['ideal_power'], sample['actual_power'], 
                     alpha=0.3, s=5, c=sample['ghi'], cmap='viridis')
lim = [0, max(actual_day['ideal_power'].max(), actual_day['actual_power'].max())]
ax3.plot(lim, lim, 'r--', lw=2, label='Perfect')
ax3.set_xlabel('Ideal Power (kWh)')
ax3.set_ylabel('Actual Power (kWh)')
ax3.set_title('Actual vs Ideal Performance')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax3, label='GHI')

# 4. Performance ratio distribution
ax4 = plt.subplot(4, 4, 4)
pr_clean = actual_day['performance_ratio'].clip(0, 2)
ax4.hist(pr_clean, bins=100, alpha=0.7, edgecolor='black', linewidth=0.5)
ax4.axvline(pr_clean.median(), color='red', linestyle='--', lw=2, 
           label=f'Median: {pr_clean.median():.3f}')
ax4.axvline(1.0, color='green', linestyle='--', lw=2, label='Ideal')
ax4.set_xlabel('Performance Ratio')
ax4.set_ylabel('Frequency')
ax4.set_title('Performance Ratio Distribution')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# 5. Time series - sample week
ax5 = plt.subplot(4, 4, 5)
week = actual_day[(actual_day['timestamp'] >= '2020-06-15') & 
                  (actual_day['timestamp'] <= '2020-06-21')]
if len(week) > 0:
    ax5.plot(week['timestamp'], week['actual_power'], label='Actual', lw=2, alpha=0.8)
    ax5.plot(week['timestamp'], week['ideal_power'], label='Ideal', lw=2, alpha=0.7)
    ax5.fill_between(week['timestamp'], week['actual_power'], week['ideal_power'], 
                     alpha=0.2, color='red', label='Deficit')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Power (kWh)')
    ax5.set_title('Sample Week (June 2020)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

# 6. Daily performance over time
ax6 = plt.subplot(4, 4, 6)
ax6.scatter(daily['timestamp'], daily['performance_ratio'], alpha=0.4, s=15, color='gray')
window = 7
daily['pr_smooth'] = daily['performance_ratio'].rolling(window, center=True).mean()
ax6.plot(daily['timestamp'], daily['pr_smooth'], color='blue', lw=2.5, label=f'{window}-day avg')
ax6.axhline(1.0, color='green', linestyle='--', lw=1.5, label='Ideal')
ax6.set_xlabel('Date')
ax6.set_ylabel('Performance Ratio')
ax6.set_title('Performance Ratio Over Time')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_ylim([0, 1.5])

# 7. Deficit with trend
ax7 = plt.subplot(4, 4, 7)
ax7.scatter(daily['timestamp'], daily['deficit'], alpha=0.3, s=15, color='gray', label='Daily')
ax7.plot(daily['timestamp'], daily['deficit_trend'], color='red', lw=3, label='Trend')
ax7.axhline(0, color='black', linestyle='--', lw=1)
ax7.set_xlabel('Date')
ax7.set_ylabel('Deficit (kWh/day)')
ax7.set_title(f'Degradation Trend: {deg_pct_per_year:.2f}%/year')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Cumulative deficit
ax8 = plt.subplot(4, 4, 8)
daily['cumulative_deficit'] = daily['deficit'].cumsum()
ax8.plot(daily['timestamp'], daily['cumulative_deficit']/1000, lw=2.5, color='darkred')
ax8.fill_between(daily['timestamp'], 0, daily['cumulative_deficit']/1000, alpha=0.3, color='red')
ax8.set_xlabel('Date')
ax8.set_ylabel('Cumulative Deficit (MWh)')
ax8.set_title(f'Total Energy Loss: {daily["cumulative_deficit"].iloc[-1]/1000:.1f} MWh')
ax8.grid(True, alpha=0.3)

# 9. Monthly comparison
ax9 = plt.subplot(4, 4, 9)
monthly = actual_day.groupby(actual_day['timestamp'].dt.to_period('M')).agg({
    'actual_power': 'sum',
    'ideal_power': 'sum'
}).reset_index()
monthly['timestamp'] = monthly['timestamp'].dt.to_timestamp()
x = np.arange(len(monthly))
width = 0.35
ax9.bar(x - width/2, monthly['actual_power']/1000, width, label='Actual', alpha=0.8)
ax9.bar(x + width/2, monthly['ideal_power']/1000, width, label='Ideal', alpha=0.8)
ax9.set_xlabel('Month')
ax9.set_ylabel('Energy (MWh)')
ax9.set_title('Monthly Production')
ax9.set_xticks(x)
ax9.set_xticklabels([t.strftime('%b') for t in monthly['timestamp']], rotation=45)
ax9.legend()
ax9.grid(True, alpha=0.3, axis='y')

# 10. Performance vs GHI
ax10 = plt.subplot(4, 4, 10)
sample = actual_day.sample(min(5000, len(actual_day)))
scatter = ax10.scatter(sample['ghi'], sample['performance_ratio'].clip(0, 2),
                      alpha=0.3, s=10, c=sample['air_temp'], cmap='coolwarm')
ax10.axhline(1.0, color='green', linestyle='--', lw=2)
ax10.set_xlabel('GHI (W/m²)')
ax10.set_ylabel('Performance Ratio')
ax10.set_title('Performance vs Irradiance')
ax10.grid(True, alpha=0.3)
ax10.set_ylim([0, 1.5])
plt.colorbar(scatter, ax=ax10, label='Temp (°C)')

# 11. Seasonal pattern
ax11 = plt.subplot(4, 4, 11)
ax11.plot(seasonal['day_of_year'], seasonal['seasonal'], lw=2.5, color='purple')
ax11.axhline(0, color='black', linestyle='--', lw=1)
ax11.set_xlabel('Day of Year')
ax11.set_ylabel('Seasonal Deficit Component (kWh)')
ax11.set_title('Seasonal Pattern Beyond Weather')
ax11.grid(True, alpha=0.3)

# 12. Hourly performance pattern
ax12 = plt.subplot(4, 4, 12)
hourly = actual_day.groupby(actual_day['timestamp'].dt.hour).agg({
    'performance_ratio': ['mean', 'std']
}).reset_index()
hourly.columns = ['hour', 'mean', 'std']
ax12.errorbar(hourly['hour'], hourly['mean'], yerr=hourly['std'],
             marker='o', capsize=5, lw=2)
ax12.axhline(1.0, color='green', linestyle='--', lw=2, label='Ideal')
ax12.set_xlabel('Hour of Day')
ax12.set_ylabel('Performance Ratio')
ax12.set_title('Performance by Hour')
ax12.legend()
ax12.grid(True, alpha=0.3)
ax12.set_xlim([5, 19])

# 13. Anomaly detection
ax13 = plt.subplot(4, 4, 13)
normal = actual_day[~actual_day['is_anomaly']]
anomalies = actual_day[actual_day['is_anomaly']]
ax13.scatter(normal['timestamp'], normal['events'], alpha=0.1, s=2, 
            color='gray', label='Normal')
ax13.scatter(anomalies['timestamp'], anomalies['events'], alpha=0.6, s=20,
            color='red', label='Anomalies')
ax13.axhline(threshold, color='red', linestyle='--', lw=1.5)
ax13.axhline(-threshold, color='red', linestyle='--', lw=1.5)
ax13.set_xlabel('Date')
ax13.set_ylabel('Event Component (kWh)')
ax13.set_title(f'Anomalies: {actual_day["is_anomaly"].sum()} events')
ax13.legend()
ax13.grid(True, alpha=0.3)

# 14. Deficit components
ax14 = plt.subplot(4, 4, 14)
monthly_comp = actual_day.groupby(actual_day['timestamp'].dt.to_period('M')).agg({
    'deficit_trend': 'mean',
    'seasonal': 'mean',
    'events': 'mean'
}).reset_index()
monthly_comp['timestamp'] = monthly_comp['timestamp'].dt.to_timestamp()
ax14.plot(monthly_comp['timestamp'], monthly_comp['deficit_trend'], 
         label='Trend', lw=2.5, marker='o')
ax14.plot(monthly_comp['timestamp'], monthly_comp['seasonal'],
         label='Seasonal', lw=2, marker='s')
ax14.axhline(0, color='black', linestyle='--', lw=1)
ax14.set_xlabel('Month')
ax14.set_ylabel('Component (kWh)')
ax14.set_title('Monthly Deficit Components')
ax14.legend()
ax14.grid(True, alpha=0.3)

# 15. Performance vs Temperature
ax15 = plt.subplot(4, 4, 15)
temp_bins = actual_day.groupby(pd.cut(actual_day['air_temp'], bins=15)).agg({
    'performance_ratio': 'mean',
    'air_temp': 'mean'
}).dropna()
ax15.plot(temp_bins['air_temp'], temp_bins['performance_ratio'], 
         marker='o', lw=2.5, markersize=8)
ax15.axhline(1.0, color='green', linestyle='--', lw=2, label='Ideal')
ax15.set_xlabel('Air Temperature (°C)')
ax15.set_ylabel('Performance Ratio')
ax15.set_title('Temperature Impact on Performance')
ax15.legend()
ax15.grid(True, alpha=0.3)

# 16. Deficit distribution
ax16 = plt.subplot(4, 4, 16)
ax16.hist(actual_day['deficit'], bins=100, alpha=0.7, edgecolor='black', linewidth=0.5)
ax16.axvline(actual_day['deficit'].mean(), color='red', linestyle='--', lw=2,
            label=f'Mean: {actual_day["deficit"].mean():.1f} kWh')
ax16.axvline(actual_day['deficit'].median(), color='orange', linestyle='--', lw=2,
            label=f'Median: {actual_day["deficit"].median():.1f} kWh')
ax16.set_xlabel('Deficit (kWh)')
ax16.set_ylabel('Frequency')
ax16.set_title('Deficit Distribution')
ax16.legend()
ax16.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../Results/final_simulation_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: final_simulation_analysis.png")

# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================
print("\n[STEP 7] Saving detailed results...")

# Summary
summary = pd.DataFrame({
    'Metric': [
        'Analysis period',
        'Daytime hours',
        'Model trained on simulation R²',
        'Capacity scale factor',
        'Average actual power (kWh)',
        'Average ideal power (kWh)',
        'Average performance ratio',
        'Median performance ratio',
        'Average deficit (kWh)',
        'Average deficit (%)',
        'Total energy deficit (MWh)',
        'Degradation rate (kWh/day)',
        'Degradation rate (kWh/year)',
        'Degradation rate (%/year)',
        'Anomalies detected',
        'Anomaly rate (%)'
    ],
    'Value': [
        f"{actual_day['timestamp'].min()} to {actual_day['timestamp'].max()}",
        len(actual_day),
        f"{r2_sim:.4f}",
        f"{scale_factor:.4f}",
        f"{actual_day['actual_power'].mean():.2f}",
        f"{actual_day['ideal_power'].mean():.2f}",
        f"{actual_day['performance_ratio'].mean():.4f}",
        f"{actual_day['performance_ratio'].median():.4f}",
        f"{actual_day['deficit'].mean():.2f}",
        f"{actual_day['deficit_pct'].mean():.2f}",
        f"{daily['cumulative_deficit'].iloc[-1]/1000:.2f}",
        f"{deg_per_day:.4f}",
        f"{deg_per_year:.2f}",
        f"{deg_pct_per_year:.2f}",
        int(actual_day['is_anomaly'].sum()),
        f"{100*actual_day['is_anomaly'].mean():.2f}"
    ]
})

summary.to_csv('../Results/final_analysis_summary.csv', index=False)

# Hourly data
hourly_output = actual_day[['timestamp', 'actual_power', 'ideal_power', 'deficit',
                             'deficit_trend', 'seasonal', 'events', 'performance_ratio',
                             'is_anomaly', 'ghi', 'air_temp']].copy()
hourly_output.to_csv('../Results/final_hourly_data.csv', index=False)

# Daily data
daily_output = daily[['timestamp', 'actual_power', 'ideal_power', 'deficit',
                      'deficit_trend', 'performance_ratio', 'cumulative_deficit']].copy()
daily_output.to_csv('../Results/final_daily_data.csv', index=False)

print("✓ Saved all output files")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 90)
print("ANALYSIS COMPLETE - EXECUTIVE SUMMARY")
print("=" * 90)

print(f"\n📊 PERFORMANCE METRICS:")
print(f"   Performance Ratio: {actual_day['performance_ratio'].mean():.3f} (median: {actual_day['performance_ratio'].median():.3f})")
print(f"   → System operates at {actual_day['performance_ratio'].mean()*100:.1f}% of physics-based ideal")
print(f"   Total Energy Loss: {daily['cumulative_deficit'].iloc[-1]/1000:.1f} MWh over {len(daily)} days")

print(f"\n📉 DEGRADATION:")
print(f"   Rate: {deg_pct_per_year:.2f}% per year ({deg_per_year:.1f} kWh/year)")
print(f"   Daily trend: {deg_per_day:.2f} kWh/day increasing deficit")

print(f"\n⚠️  ANOMALIES:")
print(f"   Detected: {actual_day['is_anomaly'].sum()} events ({100*actual_day['is_anomaly'].mean():.1f}% of hours)")

print(f"\n💡 INSIGHTS:")
pr_mean = actual_day['performance_ratio'].mean()
if pr_mean < 0.7:
    print(f"   ⚠️  CRITICAL: {pr_mean*100:.1f}% performance indicates major issues")
    print(f"      → Urgent investigation needed: soiling, shading, equipment failure")
elif pr_mean < 0.85:
    print(f"   ⚠️  LOW PERFORMANCE: {pr_mean*100:.1f}% suggests room for improvement")
    print(f"      → Recommended: cleaning, maintenance, inverter inspection")
elif pr_mean < 0.95:
    print(f"   ✓ MODERATE: {pr_mean*100:.1f}% is acceptable but could be optimized")
else:
    print(f"   ✓ EXCELLENT: {pr_mean*100:.1f}% indicates well-maintained system")

if abs(deg_pct_per_year) > 2:
    print(f"   ⚠️  HIGH DEGRADATION: {abs(deg_pct_per_year):.2f}%/year exceeds normal rates")
    print(f"      → Normal: 0.5-1%/year. Investigate systematic issues.")

print(f"\n📁 OUTPUT FILES:")
print(f"   1. final_simulation_analysis.png    - 16-panel comprehensive visualization")
print(f"   2. final_analysis_summary.csv        - Key metrics and statistics")
print(f"   3. final_hourly_data.csv            - Detailed hourly comparison")
print(f"   4. final_daily_data.csv             - Daily aggregated metrics")
print(f"   5. THIS SCRIPT: simulation_baseline_analysis.py")

print("\n" + "=" * 90)
print("✅ SUCCESS! Simulation-trained model successfully applied to actual data.")
print("   Use these insights to optimize O&M and maximize solar asset value.")
print("=" * 90)