"""
ZigZag Algorithm for Reversal Labeling
Identifies significant peaks and valleys for classification targets.
"""
import polars as pl
import numpy as np

def calculate_zigzag(df: pl.DataFrame, deviation_perc: float = 0.002) -> pl.DataFrame:
    """
    Calculate ZigZag pivots.
    
    Args:
        df: DataFrame with 'high', 'low' columns
        deviation_perc: Minimum percentage change to confirm a reversal (e.g. 0.002 = 0.2%)
        
    Returns:
        DataFrame with 'is_zigzag_high' and 'is_zigzag_low' columns
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    
    # OUTPUT ARRAYS
    # 1 = Peak/Valley, 0 = None
    is_high = np.zeros(n, dtype=int)
    is_low = np.zeros(n, dtype=int)
    
    # ZigZag State
    # 1 = Trend Up (looking for High), -1 = Trend Down (looking for Low)
    trend = 0 
    last_pivot_idx = 0
    last_pivot_val = highs[0]
    
    # Initialize
    # We need to find the first move to determine initial trend
    # Simple init: assume trend up from first bar
    trend = 1
    current_high_val = highs[0]
    current_high_idx = 0
    current_low_val = lows[0]
    current_low_idx = 0
    
    for i in range(1, n):
        # Current bar values
        h = highs[i]
        l = lows[i]
        
        if trend == 1: # Uptrend, looking for higher High or Reversal to Low
            if h > current_high_val:
                # New higher high
                current_high_val = h
                current_high_idx = i
            elif l < current_high_val * (1 - deviation_perc):
                # Reversal detected! The previous high was a Pivot High
                is_high[current_high_idx] = 1
                
                # Switch to Downtrend
                trend = -1
                current_low_val = l
                current_low_idx = i
                last_pivot_idx = current_high_idx
                
        elif trend == -1: # Downtrend, looking for lower Low or Reversal to High
            if l < current_low_val:
                # New lower low
                current_low_val = l
                current_low_idx = i
            elif h > current_low_val * (1 + deviation_perc):
                # Reversal detected! The previous low was a Pivot Low
                is_low[current_low_idx] = 1
                
                # Switch to Uptrend
                trend = 1
                current_high_val = h
                current_high_idx = i
                last_pivot_idx = current_low_idx
                
    # Note: The last pivot is often tentative, we can usually ignore it or mark it.
    # Current code marks a pivot ONLY when confirmed by a reversal. 
    # So the very last leg is incomplete, which is fine for training (we only want confirmed pivots).
    
    return df.with_columns([
        pl.Series("is_zigzag_high", is_high),
        pl.Series("is_zigzag_low", is_low)
    ])

def create_classification_labels(df: pl.DataFrame, atr_multiplier: float = 2.0) -> pl.DataFrame:
    """
    Create classification targets using ZigZag adapted to Volatility (ATR).
    Instead of fixed %, use ATR-based deviation.
    """
    # Calculate ATR first if checking deviation dynamically
    # For simplicity and speed, we will use an average ATR of the whole dataset 
    # or a rolling calculation. But ZigZag is global path dependent.
    # Using a fixed % is standard. Let's infer a good % from ATR.
    
    avg_price = df["close"].mean()
    avg_atr = df["atr"].mean()
    
    # If ATR is missing, calculate it simply
    if avg_atr is None:
        tr = (df["high"] - df["low"]).mean()
        deviation = tr * atr_multiplier / avg_price
    else:
        deviation = avg_atr * atr_multiplier / avg_price
        
    print(f"ZigZag Deviation: {deviation:.4%}")
    
    return calculate_zigzag(df, deviation_perc=deviation)
