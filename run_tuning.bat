@echo off
echo Starting Hyperparameter Tuning with Optuna...
call .venv\Scripts\activate
python model\tune_hyperparameters.py
echo Tuning finished.
pause
