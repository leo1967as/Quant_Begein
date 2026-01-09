"""
All-in-one Data Pipeline
Combines all features and labels into a single module to avoid import issues
"""
import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import pickle
import os

# ============= TECHNICAL INDICATORS =============

def add_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)
    avg_gain = gain.rolling_mean(window_size=period)
    avg_loss = loss.rolling_mean(window_size=period)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return df.with_columns(rsi.alias("rsi"))

def add_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pl.max_horizontal([tr1, tr2, tr3])
    atr = tr.rolling_mean(window_size=period)
    return df.with_columns(atr.alias("atr"))

def add_bollinger_bands(df: pl.DataFrame, period: int = 20, std_dev: float = 2.0) -> pl.DataFrame:
    close = df["close"]
    sma = close.rolling_mean(window_size=period)
    std = close.rolling_std(window_size=period)
    bb_upper = sma + (std * std_dev)
    bb_lower = sma - (std * std_dev)
    bb_width = (bb_upper - bb_lower) / sma
    return df.with_columns([
        bb_upper.alias("bb_upper"),
        bb_lower.alias("bb_lower"),
        bb_width.alias("bb_width"),
        sma.alias("bb_middle")
    ])

def add_ema(df: pl.DataFrame, periods: list = [9, 21, 50]) -> pl.DataFrame:
    close = df["close"]
    for period in periods:
        ema = close.ewm_mean(span=period, adjust=False)
        df = df.with_columns(ema.alias(f"ema_{period}"))
    return df

# ============= ORDER BLOCKS =============

def add_order_blocks(df: pl.DataFrame, atr_multiplier: float = 1.5) -> pl.DataFrame:
    is_bullish = df["close"] > df["open"]
    is_bearish = df["close"] < df["open"]
    
    high = df["high"]
    low = df["low"]
    close = df["close"]
    atr = df["atr"]  # Use already calculated ATR
    
    price_change = close - close.shift(1)
    strong_bullish_move = price_change > (atr * atr_multiplier)
    strong_bearish_move = price_change < -(atr * atr_multiplier)
    
    bullish_ob = is_bearish.shift(1) & strong_bullish_move
    bearish_ob = is_bullish.shift(1) & strong_bearish_move
    
    bullish_ob_high = pl.when(bullish_ob).then(high.shift(1)).otherwise(None)
    bullish_ob_low = pl.when(bullish_ob).then(low.shift(1)).otherwise(None)
    bearish_ob_high = pl.when(bearish_ob).then(high.shift(1)).otherwise(None)
    bearish_ob_low = pl.when(bearish_ob).then(low.shift(1)).otherwise(None)
    
    # Forward fill OB zones
    bullish_ob_high_ff = bullish_ob_high.forward_fill()
    bullish_ob_low_ff = bullish_ob_low.forward_fill()
    bearish_ob_high_ff = bearish_ob_high.forward_fill()
    bearish_ob_low_ff = bearish_ob_low.forward_fill()
    
    dist_to_bullish_ob = (close - bullish_ob_high_ff) / atr
    dist_to_bearish_ob = (bearish_ob_low_ff - close) / atr
    
    return df.with_columns([
        dist_to_bullish_ob.alias("dist_to_bullish_ob"),
        dist_to_bearish_ob.alias("dist_to_bearish_ob")
    ])

# ============= SUPPORT/RESISTANCE =============

def add_support_resistance(df: pl.DataFrame, lookback: int = 5) -> pl.DataFrame:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    atr = df["atr"]
    
    rolling_high = high.rolling_max(window_size=lookback * 2 + 1, center=True)
    rolling_low = low.rolling_min(window_size=lookback * 2 + 1, center=True)
    
    is_swing_high = high == rolling_high
    is_swing_low = low == rolling_low
    
    swing_high_price = pl.when(is_swing_high).then(high).otherwise(None)
    swing_low_price = pl.when(is_swing_low).then(low).otherwise(None)
    
    resistance = swing_high_price.forward_fill()
    support = swing_low_price.forward_fill()
    
    dist_to_resistance = (resistance - close) / atr
    dist_to_support = (close - support) / atr
    
    return df.with_columns([
        dist_to_resistance.alias("dist_to_resistance"),
        dist_to_support.alias("dist_to_support")
    ])

# ============= TARGET LABELS =============

def add_swing_targets(df: pl.DataFrame, lookback: int = 5) -> pl.DataFrame:
    high = df["high"]
    low = df["low"]
    
    rolling_high = high.rolling_max(window_size=lookback * 2 + 1, center=True)
    rolling_low = low.rolling_min(window_size=lookback * 2 + 1, center=True)
    
    is_swing_high = high == rolling_high
    is_swing_low = low == rolling_low
    
    swing_price = pl.when(is_swing_high).then(high).when(is_swing_low).then(low).otherwise(None)
    swing_type = pl.when(is_swing_high).then(1).when(is_swing_low).then(-1).otherwise(None)
    
    next_swing_price = swing_price.shift(-1).backward_fill()
    next_swing_type = swing_type.shift(-1).backward_fill()
    
    close = df["close"]
    dist_to_next_swing = (next_swing_price - close).abs()
    
    return df.with_columns([
        next_swing_price.alias("next_swing_price"),
        next_swing_type.alias("next_swing_type"),
        dist_to_next_swing.alias("dist_to_next_swing")
    ])

# ============= MAIN PIPELINE =============

def prepare_data(data_path: str) -> pl.DataFrame:
    """Load and prepare all data with features and targets"""
    print(f"Loading data from {data_path}...")
    df = pl.read_csv(data_path, try_parse_dates=True)

    
    print("Adding technical indicators...")
    df = add_rsi(df)
    df = add_atr(df)
    df = add_bollinger_bands(df)
    df = add_ema(df)
    
    print("Adding Order Blocks...")
    df = add_order_blocks(df)
    
    print("Adding Support/Resistance...")
    df = add_support_resistance(df)
    
    print("Adding swing targets...")
    df = add_swing_targets(df)
    
    return df

def train_model(df: pl.DataFrame, output_dir: str = "model"):
    """Train XGBoost quantile models"""
    feature_columns = [
        "rsi", "atr", "bb_upper", "bb_lower", "bb_width", "bb_middle",
        "ema_9", "ema_21", "ema_50",
        "dist_to_bullish_ob", "dist_to_bearish_ob",
        "dist_to_resistance", "dist_to_support",
        "open", "high", "low", "close", "volume"
    ]
    
    existing_features = [c for c in feature_columns if c in df.columns]
    target_column = "next_swing_price"
    
    df_clean = df.select(existing_features + [target_column]).drop_nulls()
    
    X = df_clean.select(existing_features).to_numpy()
    y = df_clean[target_column].to_numpy()
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features used: {existing_features}")
    
    # Chronological split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train quantile models
    models = {}
    for q in [0.1, 0.5, 0.9]:
        print(f"\nTraining quantile {q}...")
        
        # For XGBoost 3.0+, use DMatrix for training with quantile objective
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": q,
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42
        }
        
        model = xgb.train(params, dtrain, num_boost_round=200,
                          evals=[(dtest, "test")], early_stopping_rounds=20, verbose_eval=50)
        
        models[q] = model
        preds = model.predict(dtest)
        mae = mean_absolute_error(y_test, preds)
        print(f"Quantile {q} - MAE: {mae:.2f}")

    
    # Save models
    os.makedirs(output_dir, exist_ok=True)
    for q, model in models.items():
        model.save_model(os.path.join(output_dir, f"reversal_q{int(q*100)}.json"))
    
    with open(os.path.join(output_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(existing_features, f)
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Models saved to: {output_dir}/")
    print("="*50)

if __name__ == "__main__":
    df = prepare_data("Bar/XAUUSD_mt5_ticks_2018_2019_Train_15m.csv")
    print(f"\nData shape: {df.shape}")
    train_model(df)
