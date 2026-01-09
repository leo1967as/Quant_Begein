"""
Visualize Model Predictions
Plots candlestick chart with predicted reversal zones using mplfinance
"""
import polars as pl
import xgboost as xgb
import mplfinance as mpf
import pandas as pd
import os
import sys

# Add project root directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

    
os.chdir(project_root)

from train_pipeline import prepare_data
from features.advanced_features import add_advanced_features

def visualize(data_path, model_dir="model", n_bars=200):
    print("Loading data...")
    df = prepare_data(data_path)
    print("Adding advanced features...")
    df = add_advanced_features(df)
    
    # Prepare features for prediction
    with open(os.path.join(model_dir, "feature_names.pkl"), "rb") as f:
        import pickle
        feature_names = pickle.load(f)
        
    print(f"Features loaded: {len(feature_names)}")
    
    # Ensure all features exist (some might be from advanced features which were not in initial pickle if we retrained)
    # Actually, if we use the models trained with OLD features, we can't use NEW features.
    # BUT, the user just ran Tuning which used NEW features, but did NOT save the model, only params.
    # The saved models in `model/` are still the OLD ones without advanced features.
    
    # CRITICAL: We should probably retrain the final model with best params and ALL features before visualizing.
    # However, to show *something* now, let's use the EXISTING saved models.
    # The existing models don't use advanced features, so we filter `feature_names` to what they expect.
    
    # Use only last n_bars for visualization to make it readable
    df_vis = df.tail(n_bars)
    
    # Check which features are missing from dataframe vs what model expects
    missing_cols = [c for c in feature_names if c not in df_vis.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
    
    # Create DMatrix
    X = df_vis.select(feature_names).to_numpy()
    dtest = xgb.DMatrix(X)
    
    # Load models
    preds = {}
    for q in [10, 50, 90]:
        model_path = os.path.join(model_dir, f"reversal_q{q}.json")
        if os.path.exists(model_path):
            print(f"Loading {model_path}...")
            model = xgb.Booster()
            model.load_model(model_path)
            preds[f"q{q}"] = model.predict(dtest)
    
    if not preds:
        print("No models found!")
        return

    # Prepare for mplfinance
    # Convert Polars to Pandas
    df_pd = df_vis.select(["datetime", "open", "high", "low", "close", "volume"]).to_pandas()
    df_pd["datetime"] = pd.to_datetime(df_pd["datetime"])
    df_pd.set_index("datetime", inplace=True)
    
    # Create addplots for predictions
    # Predicted Swing Price usually means "Target Price"
    # Overlying it on the *current* bar might be confusing if it predicts the *future*.
    # But usually we visualize "Where the model thinks the Next Swing Is" relative to current price.
    
    apds = [
        mpf.make_addplot(preds["q50"], color='green', linestyle='dotted', width=1.5, panel=0),
        mpf.make_addplot(preds["q90"], color='gray', linestyle='--', width=0.8, panel=0),
        mpf.make_addplot(preds["q10"], color='gray', linestyle='--', width=0.8, panel=0),
    ]
    
    # Fill between Q10 and Q90? mplfinance fill_between uses separate collection
    
    print("Generating chart...")
    output_file = "predictions_chart.png"
    
    mpf.plot(
        df_pd,
        type='candle',
        style='charles',
        title=f'AI Reversal Predictions (Last {n_bars} bars)',
        ylabel='Price',
        addplot=apds,
        fill_between=dict(y1=preds["q10"], y2=preds["q90"], alpha=0.1, color='gray'),
        volume=True,
        savefig=output_file,
        figscale=1.5
    )
    
    print(f"Chart saved to {output_file}")

if __name__ == "__main__":
    data_file = os.path.join(project_root, "Bar", "XAUUSD_mt5_ticks_2018_2019_Train_15m.csv")
    visualize(data_file, n_bars=100)

