@echo off
echo Training Dual Classifiers (Top/Bottom)...
call .venv\Scripts\activate
python model\train_classifier.py
echo Training finished.
pause
