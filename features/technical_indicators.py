"""
Technical Indicators using ta library and Polars
"""
import polars as pl

def add_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Calculate RSI (Relative Strength Index)"""
    delta = df["close"].diff()
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)
    
    avg_gain = gain.rolling_mean(window_size=period)
    avg_loss = loss.rolling_mean(window_size=period)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return df.with_columns(rsi.alias("rsi"))

def add_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Calculate ATR (Average True Range)"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    # True Range is the max of these three
    tr = pl.max_horizontal([tr1, tr2, tr3])
    atr = tr.rolling_mean(window_size=period)
    
    return df.with_columns(atr.alias("atr"))


def add_bollinger_bands(df: pl.DataFrame, period: int = 20, std_dev: float = 2.0) -> pl.DataFrame:
    """Calculate Bollinger Bands"""
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
    """Calculate EMA (Exponential Moving Average) for multiple periods"""
    close = df["close"]
    
    for period in periods:
        ema = close.ewm_mean(span=period, adjust=False)
        df = df.with_columns(ema.alias(f"ema_{period}"))
    
    return df

def add_all_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Add all technical indicators to dataframe"""
    df = add_rsi(df)
    df = add_atr(df)
    df = add_bollinger_bands(df)
    df = add_ema(df)
    return df
