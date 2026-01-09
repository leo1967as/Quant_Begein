"""
Support and Resistance Detection
Detects S/R zones based on swing highs and lows
"""
import polars as pl

def detect_swing_points(df: pl.DataFrame, lookback: int = 5) -> pl.DataFrame:
    """
    Detect swing highs and lows.
    
    A swing high is a high that is higher than `lookback` bars on both sides.
    A swing low is a low that is lower than `lookback` bars on both sides.
    
    Args:
        df: DataFrame with OHLC data
        lookback: Number of bars to look back/forward
    
    Returns:
        DataFrame with swing detection columns
    """
    high = df["high"]
    low = df["low"]
    
    # Rolling max/min for lookback period
    rolling_high = high.rolling_max(window_size=lookback * 2 + 1, center=True)
    rolling_low = low.rolling_min(window_size=lookback * 2 + 1, center=True)
    
    # Swing high: current high equals the rolling max
    is_swing_high = high == rolling_high
    # Swing low: current low equals the rolling min
    is_swing_low = low == rolling_low
    
    swing_high_price = pl.when(is_swing_high).then(high).otherwise(None)
    swing_low_price = pl.when(is_swing_low).then(low).otherwise(None)
    
    return df.with_columns([
        is_swing_high.alias("sr_is_swing_high"),
        is_swing_low.alias("sr_is_swing_low"),
        swing_high_price.alias("sr_swing_high_price"),
        swing_low_price.alias("sr_swing_low_price")
    ])


def add_support_resistance(df: pl.DataFrame, lookback: int = 5) -> pl.DataFrame:
    """
    Calculate Support/Resistance features.
    
    Features:
    - dist_to_resistance: Distance from current price to nearest resistance (swing high)
    - dist_to_support: Distance from current price to nearest support (swing low)
    """
    # First detect swing points
    df = detect_swing_points(df, lookback)
    
    # Forward fill swing points to track them
    resistance = df["sr_swing_high_price"].forward_fill()
    support = df["sr_swing_low_price"].forward_fill()
    
    close = df["close"]

    
    # Calculate ATR for normalization
    high = df["high"]
    low = df["low"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pl.max_horizontal([tr1, tr2, tr3])
    atr = tr.rolling_mean(window_size=14)
    
    # Distance to S/R (normalized by ATR)
    dist_to_resistance = (resistance - close) / atr
    dist_to_support = (close - support) / atr
    
    return df.with_columns([
        dist_to_resistance.alias("dist_to_resistance"),
        dist_to_support.alias("dist_to_support"),
        resistance.alias("nearest_resistance"),
        support.alias("nearest_support")
    ])
