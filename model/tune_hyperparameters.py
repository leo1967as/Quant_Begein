"""
Hyperparameter Tuning with Optuna
Optimizes XGBoost 3.0 Quantile Regression parameters
"""
import polars as pl
import xgboost as xgb
import optuna
import numpy as np
import os
import sys

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
os.chdir(project_root)

# Import necessary modules from our package
# We use the train_pipeline logic but adapted for tuning
from train_pipeline import prepare_data

def objective(trial, X_train, y_train, X_test, y_test, quantiles=[0.1, 0.5, 0.9]):
    """
    Optuna objective function.
    Minimizes the combined Quantile Loss (MAE of weighted quantiles).
    """
    
    # Suggest hyperparameters
    params = {
        "objective": "reg:quantileerror",
        "tree_method": "hist",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "seed": 42,
        "n_jobs": -1
    }
    
    total_loss = 0
    
    for q in quantiles:
        # Update quantile alpha
        current_params = params.copy()
        current_params["quantile_alpha"] = q
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Train with early stopping
        model = xgb.train(
            current_params, 
            dtrain, 
            num_boost_round=200,
            evals=[(dtest, "test")],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        # Predict and calculate MAE (Quantile Loss Proxy)
        preds = model.predict(dtest)
        mae = np.mean(np.abs(y_test - preds))
        
        # We can implement proper Pinball Loss if needed, but MAE is good proxy for simple tuning
        # Combined objective: minimize error across all quantiles
        total_loss += mae

    return total_loss / len(quantiles)

def tune_hyperparameters():
    print("Loading and preparing data with ALL features...")
    # Load data
    df = prepare_data("Bar/XAUUSD_mt5_ticks_2018_2019_Train_15m.csv")
    
    # We need to manually add advanced features here because they are not in prepare_data yet
    # Or better: Update prepare_data to include advanced features. 
    # For now, let's just do it here to avoid breaking the pipeline file mid-run.
    from features.advanced_features import add_advanced_features
    print("Adding advanced features (Volume/Sessions)...")
    df = add_advanced_features(df)
    
    # Prepare X, y
    feature_columns = [
        "rsi", "atr", "bb_upper", "bb_lower", "bb_width", "bb_middle",
        "ema_9", "ema_21", "ema_50",
        "dist_to_bullish_ob", "dist_to_bearish_ob",
        "dist_to_resistance", "dist_to_support",
        "vol_ma_ratio", "buying_pressure",
        "hour_of_day", "day_of_week", "is_london_session", "is_ny_session",
        "open", "high", "low", "close", "volume"
    ]
    
    target_column = "next_swing_price"
    
    # Filter existing
    existing = [c for c in feature_columns if c in df.columns]
    print(f"Features used for tuning: {existing}")
    
    df_clean = df.select(existing + [target_column]).drop_nulls()
    X = df_clean.select(existing).to_numpy()
    y = df_clean[target_column].to_numpy()
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train samples: {len(X_train)}")
    
    # Optuna Study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=20)
    
    print("\n" + "="*50)
    print("Tuning Complete!")
    print(f"Best Loss: {study.best_value}")
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("="*50)
    
    return study.best_params

if __name__ == "__main__":
    tune_hyperparameters()
