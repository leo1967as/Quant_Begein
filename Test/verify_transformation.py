import os
import polars as pl
import sys

def verify():
    print("Verifying data transformation...")
    bar_dir = "Bar"
    files_to_check = [
        "XAUUSD_mt5_ticks_2018_2019_Train_5m.csv",
        "XAUUSD_mt5_ticks_2018_2019_Train_5m.parquet",
        "XAUUSD_mt5_ticks_2018_2019_Train_15m.csv"
    ]

    all_passed = True
    for f in files_to_check:
        path = os.path.join(bar_dir, f)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"[PASS] {f} exists ({size:.2f} MB)")
            
            # Basic sanity check on content
            if f.endswith(".csv"):
                df = pl.read_csv(path, n_rows=5)
                print(f"       Sample data from {f}:")
                # Check for standard columns
                expected_cols = ["datetime", "open", "high", "low", "close", "volume"]
                actual_cols = df.columns
                missing_cols = [c for c in expected_cols if c not in actual_cols]
                if missing_cols:
                    print(f"[FAIL] Missing columns in {f}: {missing_cols}")
                    all_passed = False
                else:
                    print(f"       Columns: {actual_cols}")
                    # Check OHLC logic
                    h_ge_l = (df["high"] >= df["low"]).all()
                    if not h_ge_l:
                        print(f"[FAIL] High is not always >= Low in {f}")
                        all_passed = False
                    else:
                        print(f"       OHLC logic check passed (High >= Low)")

        else:
            print(f"[FAIL] {f} NOT found")
            all_passed = False

    if all_passed:
        print("\n[SUCCESS] All verification checks passed!")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some verification checks failed.")
        sys.exit(1)

if __name__ == "__main__":
    # Create Test directory if it doesn't exist
    if not os.path.exists("Test"):
        os.makedirs("Test")
    verify()
