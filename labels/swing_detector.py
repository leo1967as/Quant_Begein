"""
Swing Point Detector for Target Label Creation
Creates target labels for the AI model
"""
import polars as pl

def create_swing_labels(df: pl.DataFrame, lookback: int = 5, lookahead: int = 20) -> pl.DataFrame:
    """
    Create target labels for reversal prediction.
    
    For each bar, find the next swing point (high or low) within lookahead bars.
    
    Target:
    - next_swing_price: Price of the next swing point
    - next_swing_type: 1 for swing high, -1 for swing low
    - bars_to_swing: Number of bars until the next swing
    
    Args:
        df: DataFrame with OHLC data
        lookback: Number of bars to detect swing points
        lookahead: Maximum bars to look ahead for next swing
    
    Returns:
        DataFrame with target labels
    """
    high = df["high"]
    low = df["low"]
    
    # Detect swing points
    rolling_high = high.rolling_max(window_size=lookback * 2 + 1, center=True)
    rolling_low = low.rolling_min(window_size=lookback * 2 + 1, center=True)
    
    is_swing_high = high == rolling_high
    is_swing_low = low == rolling_low
    
    # Create swing price and type columns
    swing_price = pl.when(is_swing_high).then(high).when(is_swing_low).then(low).otherwise(None)
    swing_type = pl.when(is_swing_high).then(1).when(is_swing_low).then(-1).otherwise(None)
    
    # For target: look ahead to find next swing
    # We'll use backward fill on shifted data to get "next" swing
    # First, shift swing data back by lookahead, then forward fill
    
    # Create row index for calculating bars to swing
    df = df.with_row_index("row_idx")
    
    df = df.with_columns([
        swing_price.alias("swing_price"),
        swing_type.alias("swing_type"),
        is_swing_high.alias("is_swing_high_label"),
        is_swing_low.alias("is_swing_low_label")
    ])
    
    # For each row, find the next swing point
    # We'll use a backward fill approach
    # Shift swing_price backwards by 1, then backward fill to get "next" swing
    next_swing_price = df["swing_price"].shift(-1).backward_fill()
    next_swing_type = df["swing_type"].shift(-1).backward_fill()
    
    df = df.with_columns([
        next_swing_price.alias("next_swing_price"),
        next_swing_type.alias("next_swing_type")
    ])
    
    # Calculate distance to next swing in points
    close = df["close"]
    dist_to_next_swing = (next_swing_price - close).abs()
    
    return df.with_columns([
        dist_to_next_swing.alias("dist_to_next_swing")
    ])
