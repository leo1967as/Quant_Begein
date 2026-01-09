@echo off
echo Running Data Transformation...
call .venv\Scripts\activate
python transform_ticks.py
echo Transformation process finished.
pause
