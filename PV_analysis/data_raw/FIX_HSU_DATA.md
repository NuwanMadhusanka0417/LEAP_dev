Confirmed — it's a units bug, not a plotting problem. The "perfect" graph here was my embedded sample data; your real file genuinely contains 34% loss values, and here's why:
pvlib.soiling.hsu expects PM concentrations in g/m³, but Open-Meteo returns them in µg/m³. So your PM values went in 1,000,000× too high, which makes HSU instantly saturate at its maximum loss — and 0.3437 is exactly HSU's saturation ceiling, so your soiling ratio got pinned at 1 − 0.3437 = 0.6563 (the flat floor you saw). The test proves it:

As-is (µg/m³): mean loss 32.8%, pinned at 0.6563 — physically impossible
Fixed (×1e-6 → g/m³): mean loss 0.10%, max 0.85% — realistic for rainy Melbourne

Let me regenerate a corrected CSV from your file and patch both scripts. First the corrected data: