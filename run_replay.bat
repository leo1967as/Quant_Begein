@echo off
echo Starting Replay Simulation...
call .venv\Scripts\activate
python backtest/replay_engine.py
echo Replay finished.
pause
