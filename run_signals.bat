@echo off
echo Generating Signal Chart...
call .venv\Scripts\activate
python visualize_signals.py
echo Done. Check signals_chart.png
pause
