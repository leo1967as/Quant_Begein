"""
Model Training Script for Reversal Point Prediction
Uses XGBoost 3.0 with QuantileDMatrix for quantile regression
"""
import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os
import sys

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Change working directory to project root
os.chdir(project_root)

from features.technical_indicators import add_all_indicators
from features.order_blocks import calculate_ob_features
from features.support_resistance import calculate_sr_features
from labels.swing_detector import create_swing_labels


def load_and_prepare_data(data_path: str) -> pl.DataFrame:
    """Load OHLC data and add all features"""
    print(f"Loading data from {data_path}...")
    df = pl.read_csv(data_path)
    
    print("Adding technical indicators...")
    df = add_all_indicators(df)
    
    print("Detecting Order Blocks...")
    df = calculate_ob_features(df)
    
    print("Calculating S/R features...")
    df = calculate_sr_features(df)
    
    print("Creating target labels...")
    df = create_swing_labels(df)
    
    return df

def prepare_features_and_target(df: pl.DataFrame) -> tuple:
    """
    Prepare feature matrix X and target vector y.
    
    Features used:
    - RSI, ATR, Bollinger Bands, EMAs
    - OB distances and flags
    - S/R distances
    
    Target:
    - next_swing_price (or distance to next swing)
    """
    feature_columns = [
        # Technical Indicators
        "rsi", "atr", "bb_upper", "bb_lower", "bb_width", "bb_middle",
        "ema_9", "ema_21", "ema_50",
        # Order Blocks
        "dist_to_bullish_ob", "dist_to_bearish_ob",
        # Support/Resistance
        "dist_to_resistance", "dist_to_support",
        # Price context
        "open", "high", "low", "close", "volume"
    ]
    
    # Filter to only existing columns
    existing_features = [c for c in feature_columns if c in df.columns]
    
    # Target: next swing price
    target_column = "next_swing_price"
    
    # Drop rows with null values in features or target
    df_clean = df.select(existing_features + [target_column, "close"]).drop_nulls()
    
    X = df_clean.select(existing_features).to_numpy()
    y = df_clean[target_column].to_numpy()
    current_price = df_clean["close"].to_numpy()
    
    return X, y, current_price, existing_features

def train_xgboost_quantile(X_train, y_train, X_test, y_test, quantiles=[0.1, 0.5, 0.9]):
    """
    Train XGBoost with quantile regression using QuantileDMatrix.
    
    Returns models for different quantiles:
    - 0.1: Lower bound (10th percentile)
    - 0.5: Median prediction
    - 0.9: Upper bound (90th percentile)
    """
    models = {}
    
    for q in quantiles:
        print(f"\nTraining quantile {q}...")
        
        # Create QuantileDMatrix for quantile regression
        dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
        dtest = xgb.QuantileDMatrix(X_test, label=y_test)
        
        params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": q,
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 6,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42
        }
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=[(dtest, "test")],
            early_stopping_rounds=20,
            verbose_eval=50
        )
        
        models[q] = model
        
        # Evaluate
        preds = model.predict(dtest)
        mae = mean_absolute_error(y_test, preds)
        print(f"Quantile {q} - MAE: {mae:.2f}")
    
    return models

def main():
    # Configuration
    data_path = "Bar/XAUUSD_mt5_ticks_2018_2019_Train_15m.csv"
    model_output_dir = "model"
    
    # Load and prepare data
    df = load_and_prepare_data(data_path)
    print(f"\nData shape: {df.shape}")
    
    # Prepare features and target
    X, y, current_price, feature_names = prepare_features_and_target(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features used: {feature_names}")
    
    # Split data (chronological split for time series)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train models
    models = train_xgboost_quantile(X_train, y_train, X_test, y_test)
    
    # Save models
    os.makedirs(model_output_dir, exist_ok=True)
    for q, model in models.items():
        model_path = os.path.join(model_output_dir, f"reversal_q{int(q*100)}.json")
        model.save_model(model_path)
        print(f"Saved model: {model_path}")
    
    # Save feature names
    with open(os.path.join(model_output_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Models saved to: {model_output_dir}/")
    print("="*50)

if __name__ == "__main__":
    main()
