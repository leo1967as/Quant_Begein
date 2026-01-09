"""
Advanced Features: Volume Profile and Session Times
"""
import polars as pl

def add_volume_features(df: pl.DataFrame, ma_period: int = 20) -> pl.DataFrame:
    """
    Add volume-based features.
    
    - vol_ma_ratio: Relative Volume (Current Vol / MA Vol)
    - buying_pressure: Estimating buying pressure based on candle close relative to high/low
    """
    vol = df["volume"]
    vol_ma = vol.rolling_mean(window_size=ma_period)
    
    # Relative Volume
    vol_ma_ratio = vol / vol_ma
    
    # Buying Pressure (Close relative to range)
    # 1.0 = Close at High (Max buying pressure)
    # 0.0 = Close at Low (Max selling pressure)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    range_len = high - low
    # Avoid division by zero
    buying_pressure = (close - low) / pl.max_horizontal([range_len, pl.lit(0.00001)])
    
    return df.with_columns([
        vol_ma_ratio.alias("vol_ma_ratio"),
        buying_pressure.alias("buying_pressure")
    ])

def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add time-based features (Sessions).
    """
    dt = df["datetime"]
    
    hour = dt.dt.hour()
    day_of_week = dt.dt.weekday()
    
    # Simple session definitions (approximate in UTC)
    # Adjust offsets if data is not UTC. Assuming data is UTC or consistent broker time.
    # London: 08:00 - 16:00
    # NY: 13:00 - 21:00
    
    is_london = (hour >= 8) & (hour < 16)
    is_ny = (hour >= 13) & (hour < 21)
    
    return df.with_columns([
        hour.alias("hour_of_day"),
        day_of_week.alias("day_of_week"),
        is_london.alias("is_london_session"),
        is_ny.alias("is_ny_session")
    ])

def add_anomaly_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add anomaly detection features for reversal prediction.
    """
    close = df["close"]
    atr = df["atr"]
    
    # 1. Mean Reversion (Distance from MA)
    # Normalized by ATR to make it adaptive to volatility
    if "ema_50" in df.columns:
        ma = df["ema_50"]
    else:
        ma = close.ewm_mean(span=50, adjust=False)
        
    dist_from_ma = (close - ma) / atr
    
    # 2. Volatility Spike
    # Ration of current ATR to long-term ATR
    atr_ma = atr.rolling_mean(window_size=20)
    volatility_spike = atr / atr_ma
    
    # 3. RSI Extreme (Distance from 50)
    # Requires RSI to be present
    if "rsi" in df.columns:
        rsi = df["rsi"]
    else:
        # Fallback or error? For now, assume it's there or skip
        rsi = pl.lit(50) 
        
    rsi_extreme = (rsi - 50).abs()
    
    return df.with_columns([
        dist_from_ma.alias("dist_from_ma"),
        volatility_spike.alias("volatility_spike"),
        rsi_extreme.alias("rsi_extreme")
    ])

def add_advanced_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add all advanced features"""
    df = add_volume_features(df)
    df = add_time_features(df)
    df = add_anomaly_features(df)
    return df

