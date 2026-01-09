"""
Visualize Classification Signals
Plots Candlestick chart with Predicted Arrows vs True ZigZag Dots.
"""
import polars as pl
import xgboost as xgb
import mplfinance as mpf
import pandas as pd
import numpy as np
import os
import sys
import pickle

# Add project root directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
os.chdir(project_root)

from train_pipeline import prepare_data
from features.advanced_features import add_advanced_features
from labels.zigzag import create_classification_labels

def visualize_signals(data_path, model_dir="model", n_bars=300):
    print("Loading data...")
    df = prepare_data(data_path)
    
    print("Adding advanced features...")
    df = add_advanced_features(df)
    
    print("Adding Ground Truth (ZigZag)...")
    df = create_classification_labels(df, atr_multiplier=2.0)
    
    # Load feature names
    with open(os.path.join(model_dir, "classifier_features.pkl"), "rb") as f:
        feature_names = pickle.load(f)
        
    print(f"Features loaded: {len(feature_names)}")
    
    # Select last n_bars
    df_vis = df.tail(n_bars)
    
    # Predict
    X = df_vis.select(feature_names).to_numpy()
    dtest = xgb.DMatrix(X)
    
    # Load Models
    print("Loading Top Classifier...")
    model_top = xgb.Booster()
    model_top.load_model(os.path.join(model_dir, "classifier_top.json"))
    
    print("Loading Bottom Classifier...")
    model_bottom = xgb.Booster()
    model_bottom.load_model(os.path.join(model_dir, "classifier_bottom.json"))
    
    # Get Probabilities
    prob_top = model_top.predict(dtest)
    prob_bottom = model_bottom.predict(dtest)
    
    # Thresholding (0.6 for slightly higher confidence than training)
    threshold = 0.6
    pred_top_raw = (prob_top > threshold).astype(int)
    pred_bottom_raw = (prob_bottom > threshold).astype(int)
    
    # [FILTER] Cooldown Logic
    # Simple loop to remove consecutive signals
    cooldown = 10
    last_top = -cooldown
    last_bot = -cooldown
    
    pred_top = np.zeros_like(pred_top_raw)
    pred_bottom = np.zeros_like(pred_bottom_raw)
    
    for i in range(len(pred_top_raw)):
        if pred_top_raw[i] == 1 and (i - last_top >= cooldown):
            pred_top[i] = 1
            last_top = i
            
        if pred_bottom_raw[i] == 1 and (i - last_bot >= cooldown):
            pred_bottom[i] = 1
            last_bot = i
    
    # Prepare Plot Data
    df_pd = df_vis.select(["datetime", "open", "high", "low", "close", "volume", "is_zigzag_high", "is_zigzag_low"]).to_pandas()
    df_pd["datetime"] = pd.to_datetime(df_pd["datetime"])
    df_pd.set_index("datetime", inplace=True)
    
    # Create Markers for Signals
    # Top Signal -> Place above High
    signal_top = df_pd["high"] * 1.001
    signal_top = signal_top.where(pred_top == 1, np.nan)
    
    # Bottom Signal -> Place below Low
    signal_bottom = df_pd["low"] * 0.999
    signal_bottom = signal_bottom.where(pred_bottom == 1, np.nan)
    
    # Truth Markers
    true_top = df_pd["high"] * 1.002
    true_top = true_top.where(df_pd["is_zigzag_high"] == 1, np.nan)
    
    true_bottom = df_pd["low"] * 0.998
    true_bottom = true_bottom.where(df_pd["is_zigzag_low"] == 1, np.nan)
    
    apds = [
        # Predicted Signals
        mpf.make_addplot(signal_top, type='scatter', markersize=50, marker='v', color='red', panel=0),
        mpf.make_addplot(signal_bottom, type='scatter', markersize=50, marker='^', color='green', panel=0),
        
        # Ground Truth (Small Dots)
        mpf.make_addplot(true_top, type='scatter', markersize=20, marker='.', color='black', panel=0),
        mpf.make_addplot(true_bottom, type='scatter', markersize=20, marker='.', color='black', panel=0),
    ]
    
    output_file = "signals_chart.png"
    print(f"Generating {output_file}...")
    
    mpf.plot(
        df_pd,
        type='candle',
        style='charles',
        title=f'AI Reversal Signals (Red/Green = Pred, Black Dot = True)',
        ylabel='Price',
        addplot=apds,
        volume=True,
        savefig=output_file,
        figscale=1.5
    )
    print("Done.")

if __name__ == "__main__":
    data_file = os.path.join(project_root, "Bar", "XAUUSD_mt5_ticks_2018_2019_Train_15m.csv")
    visualize_signals(data_file)
