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

    # 2. Resample to OHLC
    # We use 'bid' as the primary price source
    ohlc_lf = (
        lf.group_by_dynamic("datetime", every=interval)
        .agg([
            pl.col("bid").first().alias("open"),
            pl.col("bid").max().alias("high"),
            pl.col("bid").min().alias("low"),
            pl.col("bid").last().alias("close"),
            pl.col("vol").sum().alias("volume"),
            pl.len().alias("tick_count")
        ])
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
