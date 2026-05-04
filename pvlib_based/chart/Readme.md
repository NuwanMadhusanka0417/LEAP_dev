cd pvlib_based/chart
python prepare_chart_data.py
python -m http.server 8080

http://localhost:8080/library_power_chart.html 