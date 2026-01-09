@echo off
echo Generating Predictions Grid Chart...
call .venv\Scripts\activate
python visualize_predictions.py
echo Done. Check predictions_chart.png
pause
