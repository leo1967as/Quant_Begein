"""
Train Reversal Classifiers (XGBoost)
Trains two binary classifiers to detect Swing Highs and Swing Lows.
Uses ZigZag labels and handles class imbalance.
"""
import polars as pl
import xgboost as xgb
import numpy as np
import os
import sys
import pickle
from sklearn.metrics import classification_report, confusion_matrix

# Path setup
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
os.chdir(project_root)

# Imports
from train_pipeline import prepare_data
from features.advanced_features import add_advanced_features
from labels.zigzag import create_classification_labels

def train_dual_classifiers():
    print("Loading Data...")
    # Base features (Technical Indicators, OB, SR)
    df = prepare_data("Bar/XAUUSD_mt5_ticks_2018_2019_Train_15m.csv")
    
    print("Adding Advanced Features...")
    df = add_advanced_features(df)
    
    print("Generating ZigZag Labels (Ground Truth)...")
    # deviation is calculated from ATR inside the function
    df = create_classification_labels(df, atr_multiplier=2.0)
    
    # Check Class Balance
    n_highs = df["is_zigzag_high"].sum()
    n_lows = df["is_zigzag_low"].sum()
    total = len(df)
    
    print(f"\nTotal Rows: {total}")
    print(f"Reversal Highs: {n_highs} ({n_highs/total:.2%})")
    print(f"Reversal Lows:  {n_lows} ({n_lows/total:.2%})")
    
    # Calculate scale_pos_weight for XGBoost
    # weight = negative_samples / positive_samples
    weight_high = (total - n_highs) / n_highs
    weight_low = (total - n_lows) / n_lows
    
    print(f"Scale Pos Weight (High): {weight_high:.2f}")
    print(f"Scale Pos Weight (Low):  {weight_low:.2f}")
    
    # Feature Selection
    feature_columns = [
        "rsi", "atr", "bb_width", "bb_upper", "bb_lower",
        "ema_9", "ema_21", "ema_50",
        "dist_to_bullish_ob", "dist_to_bearish_ob",
        "dist_to_resistance", "dist_to_support",
        "vol_ma_ratio", "buying_pressure",
        "dist_from_ma", "volatility_spike", "rsi_extreme",
        "hour_of_day", "day_of_week", "is_london_session", "is_ny_session"
    ]
    
    # Ensure cols exist
    existing_feats = [c for c in feature_columns if c in df.columns]
    print(f"Training with {len(existing_feats)} features.")
    
    # Prep Data
    df_clean = df.select(existing_feats + ["is_zigzag_high", "is_zigzag_low"]).drop_nulls()
    X = df_clean.select(existing_feats).to_numpy()
    y_high = df_clean["is_zigzag_high"].to_numpy()
    y_low = df_clean["is_zigzag_low"].to_numpy()
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_high_train, y_high_test = y_high[:split_idx], y_high[split_idx:]
    y_low_train, y_low_test = y_low[:split_idx], y_low[split_idx:]
    
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    # ================= TRAIN HIGH MODEL =================
    print("\nTraining TOP Classifier...")
    
    dtrain_high = xgb.DMatrix(X_train, label=y_high_train)
    dtest_high = xgb.DMatrix(X_test, label=y_high_test)
    
    params_high = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "scale_pos_weight": weight_high,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "eval_metric": "aucpr"
    }
    
    model_high = xgb.train(
        params_high,
        dtrain_high,
        num_boost_round=300,
        evals=[(dtest_high, "test")],
        early_stopping_rounds=30,
        verbose_eval=50
    )
    
    # Evaluate
    probs_high = model_high.predict(dtest_high)
    preds_high = (probs_high > 0.5).astype(int)
    print("TOP Model Report:")
    print(classification_report(y_high_test, preds_high))
    
    model_high.save_model(os.path.join(model_dir, "classifier_top.json"))
    
    # ================= TRAIN LOW MODEL =================
    print("\nTraining BOTTOM Classifier...")
    
    dtrain_low = xgb.DMatrix(X_train, label=y_low_train)
    dtest_low = xgb.DMatrix(X_test, label=y_low_test)
    
    params_low = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "scale_pos_weight": weight_low,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "eval_metric": "aucpr" 
    }
    
    model_low = xgb.train(
        params_low,
        dtrain_low,
        num_boost_round=300,
        evals=[(dtest_low, "test")],
        early_stopping_rounds=30,
        verbose_eval=50
    )
    
    # Evaluate
    probs_low = model_low.predict(dtest_low)
    preds_low = (probs_low > 0.5).astype(int)
    print("BOTTOM Model Report:")
    print(classification_report(y_low_test, preds_low))
    
    model_low.save_model(os.path.join(model_dir, "classifier_bottom.json"))

    
    # Save Feature Names
    with open(os.path.join(model_dir, "classifier_features.pkl"), "wb") as f:
        pickle.dump(existing_feats, f)
        
    print("\nDone! Models saved.")

if __name__ == "__main__":
    train_dual_classifiers()
