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
from features.order_flow import add_delta_features
from labels.zigzag import create_classification_labels

def train_dual_classifiers():
    print("Loading Data...")
    # Base features (Technical Indicators, OB, SR)
    df = prepare_data("Bar/XAUUSD_mt5_ticks_2018_2019_Train_15m.csv")
    
    print("Adding Advanced Features (Volume/Session)...")
    df = add_advanced_features(df)
    
    print("Adding Order Flow Features (Delta)...")
    df = add_delta_features(df)
    
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
    
    # Feature Selection - REMOVED RAW PRICES (Non-stationary)
    feature_columns = [
        "rsi", "atr", "bb_width", 
        "dist_to_bullish_ob", "dist_to_bearish_ob",
        "dist_to_resistance", "dist_to_support",
        "vol_ma_ratio", "buying_pressure",
        "dist_from_ma", "volatility_spike", "rsi_extreme",
        "hour_of_day", "day_of_week", "is_london_session", "is_ny_session",
        "delta", "cvd", "is_cvd_div_bearish", "is_cvd_div_bullish"
    ]
    
    # Add Log Return (Stationary Price Info)
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return")
    )
    feature_columns.append("log_return")
    
    # Ensure cols exist
    existing_feats = [c for c in feature_columns if c in df.columns]
    print(f"Training with {len(existing_feats)} features (Non-stationary removed).")
    
    # Prep Data
    df_clean = df.select(existing_feats + ["is_zigzag_high", "is_zigzag_low"]).drop_nulls()
    X = df_clean.select(existing_feats).to_numpy()
    y_high = df_clean["is_zigzag_high"].to_numpy()
    y_low = df_clean["is_zigzag_low"].to_numpy()
    
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    # ================= WALK-FORWARD VALIDATION =================
    print("\nStarting Walk-Forward Validation (5 Splits)...")
    from sklearn.model_selection import TimeSeriesSplit
    
    # Purge size: ZigZag lookahead implies leakage. 
    # If ZigZag needs future N bars, we must drop N bars between Train and Test.
    # Assuming ZigZag deviation implies variable lookahead, but roughly 50 bars?
    purge_gap = 50 
    
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=purge_gap)
    
    scores_top = []
    scores_bot = []
    
    # We will save the model trained on the LAST split (Most recent data)
    # Or train on FULL dataset after validation.
    # Standard practice: Validate to check robustness, then train on full.
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_high_train, y_high_test = y_high[train_idx], y_high[test_idx]
        y_low_train, y_low_test = y_low[train_idx], y_low[test_idx]
        
        # Train Top
        dtrain = xgb.DMatrix(X_train_cv, label=y_high_train)
        dtest = xgb.DMatrix(X_test_cv, label=y_high_test)
        
        params = {
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
        
        model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
        # Score (AUCPR)
        # XGB doesn't return score directly, get from eval? 
        # Easier: predict and use sklearn metric if needed, or just trust the process.
        # Let's print F1 for this fold.
        preds = (model.predict(dtest) > 0.5).astype(int)
        f1 = classification_report(y_high_test, preds, output_dict=True)["1"]["f1-score"]
        scores_top.append(f1)
        
        # Train Bottom
        dtrain_b = xgb.DMatrix(X_train_cv, label=y_low_train)
        dtest_b = xgb.DMatrix(X_test_cv, label=y_low_test)
        params["scale_pos_weight"] = weight_low
        model_b = xgb.train(params, dtrain_b, num_boost_round=200, verbose_eval=False)
        preds_b = (model_b.predict(dtest_b) > 0.5).astype(int)
        f1_b = classification_report(y_low_test, preds_b, output_dict=True)["1"]["f1-score"]
        scores_bot.append(f1_b)
        
        print(f"Fold {i+1}: Top F1={f1:.3f}, Bot F1={f1_b:.3f}")

    print(f"\nAvg Top F1: {np.mean(scores_top):.3f}")
    print(f"Avg Bot F1: {np.mean(scores_bot):.3f}")
    
    # Final Training on ALL Data (or Train+Test of last fold?)
    # Usually retraining on full data is risky without early stopping set.
    # We will use the last split's Test set as the final hold-out for the Saved Model.
    
    print("\nTraining Final Models (Last Split)...")
    # Using the last split from the loop
    
    # Top
    dtrain_final = xgb.DMatrix(X_train_cv, label=y_high_train)
    dtest_final = xgb.DMatrix(X_test_cv, label=y_high_test) # Using last test set for early stopping
    
    params["scale_pos_weight"] = weight_top = weight_high
    model_top = xgb.train(params, dtrain_final, num_boost_round=300, 
                          evals=[(dtest_final, "test")], early_stopping_rounds=30, verbose_eval=50)
                          
    model_top.save_model(os.path.join(model_dir, "classifier_top.json"))

    # Bottom
    dtrain_final_b = xgb.DMatrix(X_train_cv, label=y_low_train)
    dtest_final_b = xgb.DMatrix(X_test_cv, label=y_low_test)
    params["scale_pos_weight"] = weight_low
    model_low = xgb.train(params, dtrain_final_b, num_boost_round=300, 
                          evals=[(dtest_final_b, "test")], early_stopping_rounds=30, verbose_eval=50)

    model_low.save_model(os.path.join(model_dir, "classifier_bottom.json"))

    # Save Feature Names
    with open(os.path.join(model_dir, "classifier_features.pkl"), "wb") as f:
        pickle.dump(existing_feats, f)
        
    print("\nDone! Models saved.")

if __name__ == "__main__":
    train_dual_classifiers()
