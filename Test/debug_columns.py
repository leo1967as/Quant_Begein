"""
Debug script to find duplicate column issues
"""
import polars as pl
import sys
sys.path.insert(0, ".")

# Load data
print("Loading data...")
df = pl.read_csv("Bar/XAUUSD_mt5_ticks_2018_2019_Train_15m.csv")
print(f"Initial columns: {df.columns}")
print(f"Shape: {df.shape}")

# Step 1: Technical Indicators
print("\n1. Adding technical indicators...")
from features.technical_indicators import add_all_indicators
df = add_all_indicators(df)
print(f"Columns after indicators: {df.columns}")

# Step 2: Order Blocks
print("\n2. Calculating OB features...")
from features.order_blocks import calculate_ob_features
df = calculate_ob_features(df)
print(f"Columns after OB: {df.columns}")

# Step 3: S/R
print("\n3. Calculating S/R features...")
from features.support_resistance import calculate_sr_features
df = calculate_sr_features(df)
print(f"Columns after S/R: {df.columns}")

# Step 4: Swing Labels
print("\n4. Creating swing labels...")
from labels.swing_detector import create_swing_labels
df = create_swing_labels(df)
print(f"Columns after labels: {df.columns}")

print("\n[SUCCESS] No duplicate column errors!")
