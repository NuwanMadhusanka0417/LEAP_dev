python -m http.server 8080

PV_analysis/4_forecast_7d_pvlib_xgboost.py

cd PV_analysis
python 4_forecast_7d_pvlib_xgboost.py --building-key library
python 4_forecast_7d_pvlib_xgboost.py --building-key library --forward
python 4_forecast_7d_pvlib_xgboost.py --building-key library --forecast-start "2025-11-01 00:00:00"