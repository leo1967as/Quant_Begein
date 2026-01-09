"""
Test training script with full error output
"""
import sys
import traceback

sys.path.insert(0, ".")

try:
    from model.train_reversal import load_and_prepare_data, prepare_features_and_target
    
    df = load_and_prepare_data("Bar/XAUUSD_mt5_ticks_2018_2019_Train_15m.csv")
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
