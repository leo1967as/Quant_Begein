"""
Order Block Detection
Detects Bullish and Bearish Order Blocks based on price structure
"""
import polars as pl

def detect_order_blocks(df: pl.DataFrame, atr_multiplier: float = 1.5) -> pl.DataFrame:
    """
    Detect Order Blocks (OB) in price data.
    
    Bullish OB: Last bearish candle before a strong bullish move
    Bearish OB: Last bullish candle before a strong bearish move
    
    Args:
        df: DataFrame with OHLC data
        atr_multiplier: Minimum move size in ATR units to qualify as "strong"
    
    Returns:
        DataFrame with OB detection columns
    """
    # Calculate candle body
    body = (df["close"] - df["open"]).abs()
    is_bullish = df["close"] > df["open"]
    is_bearish = df["close"] < df["open"]
    
    # Calculate ATR for reference
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pl.max_horizontal([tr1, tr2, tr3])
    atr = tr.rolling_mean(window_size=14)
    
    # Strong move detection (move > atr_multiplier * ATR)
    price_change = close - close.shift(1)
    strong_bullish_move = price_change > (atr * atr_multiplier)
    strong_bearish_move = price_change < -(atr * atr_multiplier)
    
    # Bullish OB: Previous candle was bearish, current candle starts strong bullish move
    bullish_ob = is_bearish.shift(1) & strong_bullish_move
    
    # Bearish OB: Previous candle was bullish, current candle starts strong bearish move
    bearish_ob = is_bullish.shift(1) & strong_bearish_move
    
    # OB zone prices (using previous candle's high/low)
    bullish_ob_high = pl.when(bullish_ob).then(high.shift(1)).otherwise(None)
    bullish_ob_low = pl.when(bullish_ob).then(low.shift(1)).otherwise(None)
    bearish_ob_high = pl.when(bearish_ob).then(high.shift(1)).otherwise(None)
    bearish_ob_low = pl.when(bearish_ob).then(low.shift(1)).otherwise(None)
    
    return df.with_columns([
        bullish_ob.alias("is_bullish_ob"),
        bearish_ob.alias("is_bearish_ob"),
        bullish_ob_high.alias("bullish_ob_high"),
        bullish_ob_low.alias("bullish_ob_low"),
        bearish_ob_high.alias("bearish_ob_high"),
        bearish_ob_low.alias("bearish_ob_low"),
        atr.alias("atr_ob")
    ])

def calculate_ob_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate features based on Order Block proximity.
    
    Features:
    - dist_to_bullish_ob: Distance from current price to nearest bullish OB
    - dist_to_bearish_ob: Distance from current price to nearest bearish OB
    - in_bullish_ob: Whether price is currently in a bullish OB zone
    - in_bearish_ob: Whether price is currently in a bearish OB zone
    """
    # First detect OBs
    df = detect_order_blocks(df)
    
    # Forward fill OB zones to track them over time
    bullish_ob_high_ff = df["bullish_ob_high"].forward_fill()
    bullish_ob_low_ff = df["bullish_ob_low"].forward_fill()
    bearish_ob_high_ff = df["bearish_ob_high"].forward_fill()
    bearish_ob_low_ff = df["bearish_ob_low"].forward_fill()
    
    close = df["close"]
    
    # Distance to OB zones (normalized by ATR)
    atr = df["atr_ob"]
    dist_to_bullish_ob = (close - bullish_ob_high_ff) / atr
    dist_to_bearish_ob = (bearish_ob_low_ff - close) / atr
    
    # Check if price is inside OB zone
    in_bullish_ob = (close <= bullish_ob_high_ff) & (close >= bullish_ob_low_ff)
    in_bearish_ob = (close <= bearish_ob_high_ff) & (close >= bearish_ob_low_ff)
    
    return df.with_columns([
        dist_to_bullish_ob.alias("dist_to_bullish_ob"),
        dist_to_bearish_ob.alias("dist_to_bearish_ob"),
        in_bullish_ob.alias("in_bullish_ob"),
        in_bearish_ob.alias("in_bearish_ob")
    ])
