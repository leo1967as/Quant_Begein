@echo off
echo Training Reversal Prediction Model...
call .venv\Scripts\activate
python train_pipeline.py
echo Training process finished.
pause
