import polars as pl
import os
import sys
from datetime import datetime

def transform_ticks(input_file, output_folder, interval="5m"):
    """
    Transforms tick data from CSV to OHLC bars using Polars.
    """
    print(f"Starting transformation: {input_file} -> {interval} bars")
    start_time = datetime.now()

    # Define columns based on sample: Date, Time, Bid, Ask, Last, Vol
    # 20180101,23:00:00,1302.922,1303.428,1302.922,0
    columns = ["date", "time", "bid", "ask", "last", "vol"]

    # Use LazyFrame for memory efficiency with 3.5GB file
    lf = pl.scan_csv(
        input_file,
        has_header=False,
        new_columns=columns,
        separator=",",
        # Optimize parsing
        infer_schema_length=10000,
    )

    # 1. Combine Date and Time into Datetime
    # Date is YYYYMMDD, Time is HH:MM:SS
    # Ensure they are cast to string if infer_schema made them integers
    lf = lf.with_columns([
        pl.col("date").cast(pl.Utf8),
        pl.col("time").cast(pl.Utf8)
    ])
    
    lf = lf.with_columns(
        datetime = (pl.col("date") + " " + pl.col("time")).str.to_datetime("%Y%m%d %H:%M:%S")
    )

    # 2. Resample to OHLC and Calculate Delta (Order Flow)
    # Classify Trade Direction (Buy or Sell Initiative)
    # If Last >= Ask -> Buy Aggressor (Buy Volume)
    # If Last <= Bid -> Sell Aggressor (Sell Volume)
    
    # We need to do this BEFORE groupby.
    # Note: 'vol' might be 0, so we use 1 (tick count) if vol is 0?
    # User requested using 'vol' column logic, but we know vol is 0.
    # So we should use 1 for each tick as volume.
    
    lf = lf.with_columns([
        pl.when(pl.col("last") >= pl.col("ask")).then(pl.lit(1)).otherwise(0).alias("buy_vol"),
        pl.when(pl.col("last") <= pl.col("bid")).then(pl.lit(1)).otherwise(0).alias("sell_vol")
    ])

    ohlc_lf = (
        lf.group_by_dynamic("datetime", every=interval)
        .agg([
            pl.col("bid").first().alias("open"),
            pl.col("bid").max().alias("high"),
            pl.col("bid").min().alias("low"),
            pl.col("bid").last().alias("close"),
            pl.len().alias("volume"), # Use Tick Count as Volume
            pl.col("buy_vol").sum().alias("buy_volume"),
            pl.col("sell_vol").sum().alias("sell_volume"),
            pl.len().alias("tick_count")
        ])
    )
    
    # Calculate Delta
    ohlc_lf = ohlc_lf.with_columns(
        (pl.col("buy_volume") - pl.col("sell_volume")).alias("delta")
    )

    # 3. Collect and Save
    output_basename = os.path.splitext(os.path.basename(input_file))[0]
    output_csv = os.path.join(output_folder, f"{output_basename}_{interval}.csv")
    output_parquet = os.path.join(output_folder, f"{output_basename}_{interval}.parquet")

    print(f"Executing plan and writing to {output_csv}...")
    
    # Materialize the lazy frame
    df = ohlc_lf.collect(streaming=True)
    
    df.write_csv(output_csv)
    df.write_parquet(output_parquet)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Transformation complete!")
    print(f"Rows reduced from millions to {len(df)}")
    print(f"Output saved to: {output_csv} and {output_parquet}")
    print(f"Duration: {duration}")

if __name__ == "__main__":
    input_path = "XAUUSD_mt5_ticks_2018_2019_Train.csv"
    output_dir = "Bar"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Transform to M5 and M15
    for timeframe in ["5m", "15m"]:
        try:
            transform_ticks(input_path, output_dir, interval=timeframe)
        except Exception as e:
            print(f"Error transforming {timeframe}: {e}")
