"""
Order Flow Features
Calculates Delta, Cumulative Delta (CVD), and Divergences.
Requires 'delta' column in the input DataFrame.
"""
import polars as pl

def add_delta_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add Order Flow features.
    
    Features:
    - delta: (Buy Vol - Sell Vol) - already present? Check.
    - cvd: Cumulative Volume Delta
    - is_cvd_divergence_bearish: Price High, CVD Lower High
    - is_cvd_divergence_bullish: Price Low, CVD Higher Low
    """
    
    # Ensure delta exists, if not calculate from buy/sell if available
    if "delta" not in df.columns:
        if "buy_volume" in df.columns and "sell_volume" in df.columns:
             df = df.with_columns((pl.col("buy_volume") - pl.col("sell_volume")).alias("delta"))
        else:
             # Fallback if no raw delta info (shouldn't happen with new transform)
             return df
             
    delta = df["delta"]
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    # 1. Cumulative Volume Delta (CVD)
    cvd = delta.cum_sum()
    
    # 2. Delta Divergence
    # Bearish: Price makes New High, CVD does not (in last N bars)
    lookback = 10
    
    rolling_high_price = high.rolling_max(window_size=lookback)
    rolling_high_cvd = cvd.rolling_max(window_size=lookback)
    
    # Check if current bar is the local high for Price
    is_price_high = high == rolling_high_price
    # Check if current CVD is NOT the local high
    is_cvd_weak = cvd < rolling_high_cvd
    
    div_bearish = is_price_high & is_cvd_weak
    
    # Bullish: Price makes New Low, CVD does not
    rolling_low_price = low.rolling_min(window_size=lookback)
    rolling_low_cvd = cvd.rolling_min(window_size=lookback)
    
    is_price_low = low == rolling_low_price
    is_cvd_strong = cvd > rolling_low_cvd
    
    div_bullish = is_price_low & is_cvd_strong
    
    return df.with_columns([
        cvd.alias("cvd"),
        div_bearish.alias("is_cvd_div_bearish"),
        div_bullish.alias("is_cvd_div_bullish")
    ])
