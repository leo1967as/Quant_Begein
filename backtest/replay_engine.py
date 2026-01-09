"""
Replay Backtest Engine
Simulates live trading by replaying historical data and AI predictions.
"""
import polars as pl
import xgboost as xgb
import mplfinance as mpf
import pandas as pd
import numpy as np
import os
import sys
import pickle
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from train_pipeline import prepare_data
from features.advanced_features import add_advanced_features
from labels.zigzag import create_classification_labels

class ReplayBacktest:
    def __init__(self, data_path, model_dir="model", initial_balance=10000):
        self.data_path = data_path
        self.model_dir = model_dir
        self.balance = initial_balance
        self.positions = [] # List of active positions
        self.equity_curve = [initial_balance]
        
    def run(self, start_bar=500, n_frames=200):
        """Run the visual replay"""
        print("Preparing Replay Data...")
        
        # 1. Load & Prep
        df = prepare_data(self.data_path)
        df = add_advanced_features(df)
        df = create_classification_labels(df) # For truth comparison
        
        # Load Features
        with open(os.path.join(self.model_dir, "classifier_features.pkl"), "rb") as f:
            feats = pickle.load(f)
            
        # 2. Vectorized Prediction (Cheat for Speed)
        print("Running Predictions...")
        X = df.select(feats).to_numpy()
        dmat = xgb.DMatrix(X)
        
        mdl_top = xgb.Booster()
        mdl_top.load_model(os.path.join(self.model_dir, "classifier_top.json"))
        mdl_bot = xgb.Booster()
        mdl_bot.load_model(os.path.join(self.model_dir, "classifier_bottom.json"))
        
        probs_top = mdl_top.predict(dmat)
        probs_bot = mdl_bot.predict(dmat)
        
        # Threshold
        thresh = 0.6
        pred_top_raw = (probs_top > thresh).astype(int)
        pred_bot_raw = (probs_bot > thresh).astype(int)
        
        # [FILTER] Signal Cooldown Logic
        # Prevent consecutive signals in same direction
        cooldown_bars = 10
        last_top_idx = -cooldown_bars
        last_bot_idx = -cooldown_bars
        
        final_top = np.zeros_like(pred_top_raw)
        final_bot = np.zeros_like(pred_bot_raw)
        
        for i in range(len(df)):
            if pred_top_raw[i] == 1:
                # Check cooldown
                if i - last_top_idx >= cooldown_bars:
                    final_top[i] = 1
                    last_top_idx = i
                    
            if pred_bot_raw[i] == 1:
                # Check cooldown
                if i - last_bot_idx >= cooldown_bars:
                    final_bot[i] = 1
                    last_bot_idx = i
                    
        df = df.with_columns([
            pl.Series("pred_top", final_top),
            pl.Series("pred_bot", final_bot)
        ])
        
        # Convert to Pandas for MPF
        pdf = df.tail(n_frames + start_bar).to_pandas()
        pdf["datetime"] = pd.to_datetime(pdf["datetime"])
        pdf.set_index("datetime", inplace=True)
        
        # Slice for Animation
        # We want to show a moving window
        # But simpler: Show static window but reveal predictions? 
        # Or sliding window? User said "Run graph".
        # Let's do a fixed window of N bars, and iterate through it?
        # No, typically Replay means adding one candle at a time.
        
        data = pdf.iloc[-n_frames:] # Take last N frames for the demo
        
        # Pre-calculate Markers
        # We need lists used for Plotting.
        # Initialize with NaNs
        signal_top = np.full(len(data), np.nan)
        signal_bottom = np.full(len(data), np.nan)
        
        # Setup Plot
        fig = mpf.figure(style='charles', figsize=(12, 8))
        ax1 = fig.add_subplot(2, 1, 1) # Price
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1) # PnL / Vol
        
        def animate(ival):
            if ival >= len(data):
                return
            
            # Current Slice (Growing)
            # Efficient update: Only update the last data point?
            # MPF doesn't support partial update easily. We redraw.
            # To be smooth, we might show a window of 50 bars.
            
            # Current index in 'data'
            idx = ival
            
            # Update Signals (Reveal)
            curr_row = data.iloc[idx]
            if curr_row["pred_top"] == 1:
                signal_top[idx] = curr_row["high"] * 1.001
            if curr_row["pred_bot"] == 1:
                signal_bottom[idx] = curr_row["low"] * 0.999
                
            # Visible Window: Let's keep 50 bars visible
            window = 60
            start_idx = max(0, idx - window)
            end_idx = idx + 1
            
            view_data = data.iloc[start_idx:end_idx]
            view_top = signal_top[start_idx:end_idx]
            view_bot = signal_bottom[start_idx:end_idx]
            
            ax1.clear()
            ax2.clear()
            
            # Custom PnL Logic (Simple)
            # If Top Signal -> Short. If Bottom -> Long.
            # Calculate hypothetical PnL from signals in view.
            # (Skipping complex PnL for now, focussing on Chart Replay)
            
            apds = []
            # Add signals if any exist in view
            if not np.all(np.isnan(view_top)):
                apds.append(mpf.make_addplot(view_top, type='scatter', markersize=50, marker='v', color='red', ax=ax1))
            if not np.all(np.isnan(view_bot)):
                apds.append(mpf.make_addplot(view_bot, type='scatter', markersize=50, marker='^', color='green', ax=ax1))
            
            mpf.plot(view_data, type='candle', ax=ax1, volume=False, addplot=apds)
            ax1.set_title(f"Replay Bar: {idx}/{len(data)}")
            
        print("Starting Animation Window...")
        ani = animation.FuncAnimation(fig, animate, interval=100, frames=len(data), repeat=False)
        mpf.show()

if __name__ == "__main__":
    data_file = "Bar/XAUUSD_mt5_ticks_2018_2019_Train_15m.csv"
    replay = ReplayBacktest(data_file)
    replay.run(n_frames=200) # Replay last 200 bars
