# train.py
import sys
import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from model import create_features, prepare_target, train_and_evaluate, TradingModel
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # For saving the model

# --- Configuration ---
OUTPUT_FOLDER_NAME = "modelresult"
DOWNLOAD_TIMEFRAME = "3y"  # Increased for potentially more data
SIGNAL_PLOT_ALPHA = 0.7
BUY_SIGNAL_MARKER = '^'
SELL_SIGNAL_MARKER = 'v'
SIGNAL_MARKER_SIZE = 50
FEATURE_IMPORTANCE_BAR_COLOR = 'skyblue'
RESULTS_FILENAME_FORMAT = "{ticker}_results_{timestamp}.csv"
MODEL_FILENAME_FORMAT = "{ticker}_model_{timestamp}.pkl"
SIGNALS_PLOT_FILENAME = "{ticker}_signals.png"
F1_SCORES_PLOT_FILENAME = "{ticker}_f1_scores.png"
FEATURE_IMPORTANCE_PLOT_FILENAME = "{ticker}_feature_importance.png"

NIFTY50_URL = "https://en.wikipedia.org/wiki/NIFTY_50"
FALLBACK_NIFTY50_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFC.NS", "ICICIBANK.NS", "INFY.NS",
    "HUL.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS",
    "AXISBANK.NS", "MARUTI.NS", "ITC.NS", "HCLTECH.NS", "BAJFINANCE.NS",
    "ASIANPAINT.NS", "NTPC.NS", "POWERGRID.NS", "TITAN.NS", "NESTLEIND.NS",
    "ULTRACEMCO.NS", "ADANIENT.NS", "HDFCLIFE.NS", "WIPRO.NS", "JSWSTEEL.NS",
    "SUNPHARMA.NS", "TATAMOTORS.NS", "BAJAJFINSV.NS", "ADANIPORTS.NS", "SHREECEM.NS",
    "EICHERMOT.NS", "GRASIM.NS", "DIVISLAB.NS", "ONGC.NS", "TECHM.NS",
    "HINDALCO.NS", "CIPLA.NS", "BRITANNIA.NS", "APOLLOHOSP.NS", "HEROMOTOCO.NS",
    "UPL.NS", "M&M.NS", "SBILIFE.NS", "COALINDIA.NS", "VEDL.NS",
    "DMART.NS", "INDIGO.NS", "MRF.NS", "PIDILITIND.NS"
]

def normalize_columns(df):
    """Standardize column names across different yfinance versions"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip('_').lower() for col in df.columns.values]
    else:
        df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]

    # Handle known column name variations
    column_map = {
        'adj_close': 'close',
        'close_price': 'close',
        'vol': 'volume'
    }
    df.columns = [column_map.get(col, col) for col in df.columns]
    return df


def validate_features(df):
    """Ensure all required features exist and are valid"""
    required_features = [
        'returns', 'volatility', 'momentum', 'atr',
        'volume_z', 'volume_change', 'obv',
        'rsi', 'macd', 'bull_div', 'bear_div',
        'returns_lag1', 'returns_lag2', 'volume_z_lag1', 'rsi_lag1'
    ]

    # Check if features exist
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        available = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        print(f"\nERROR: Missing features: {missing}")
        print(f"\nAvailable calculated features: {available or 'None'}")
        return False

    # Check for NaN/inf values
    if df[required_features].isnull().any().any():
        print("\nWARNING: Features contain NaN values:")
        print(df[required_features].isnull().sum())

    if np.isinf(df[required_features].values).any():
        print("\nWARNING: Features contain infinite values")

    return True


def download_stock_data(ticker, timeframe=DOWNLOAD_TIMEFRAME):
    """Download data for a specific stock ticker"""
    methods = [
        {'auto_adjust': True, 'progress': False},
        {'auto_adjust': False, 'progress': False},
        {'method': 'ticker'}  # Fallback to direct ticker access
    ]

    for i, method in enumerate(methods):
        try:
            if method.get('method') == 'ticker':
                data = yf.Ticker(ticker).history(period=timeframe)
            else:
                data = yf.download(ticker, period=timeframe, **method)

            if not data.empty:
                data = normalize_columns(data)
                required = ['open', 'high', 'low', 'close', 'volume']
                if all(col in data.columns for col in required):
                    return data[required].copy()

        except Exception as e:
            if i == len(methods) - 1:  # Last attempt failed
                raise ValueError(f"All download methods failed for {ticker}. Last error: {str(e)}")

    raise ValueError(f"Unable to download data for {ticker} with any method")


def plot_results(df, metrics, importance, threshold, ticker, output_dir):
    """Plotting function to visualize results for a specific ticker and save to a directory"""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label=f'{ticker} Close Price', alpha=SIGNAL_PLOT_ALPHA)
    plt.scatter(df[df['signal'] == 1].index, df['close'][df['signal'] == 1], marker=BUY_SIGNAL_MARKER, color='g', label='Buy Signal', s=SIGNAL_MARKER_SIZE)
    plt.scatter(df[df['signal'] == -1].index, df['close'][df['signal'] == -1], marker=SELL_SIGNAL_MARKER, color='r', label='Sell Signal', s=SIGNAL_MARKER_SIZE)
    plt.title(f'Trading Signals on {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, SIGNALS_PLOT_FILENAME.format(ticker=ticker)))
    plt.close()

    if metrics is not None:
        plt.figure(figsize=(8, 6))
        sns.barplot(x=metrics.index, y=metrics['f1'], color=FEATURE_IMPORTANCE_BAR_COLOR)
        plt.title(f'Walk-Forward Validation F1 Scores for {ticker}')
        plt.xlabel('Fold')
        plt.ylabel('F1 Score')
        plt.savefig(os.path.join(output_dir, F1_SCORES_PLOT_FILENAME.format(ticker=ticker)))
        print(f"\nWalk-Forward Validation Metrics for {ticker}:\n{metrics}")
        plt.close()

    if importance is not None:
        plt.figure(figsize=(10, 6))
        importance.sort_values(ascending=False).plot(kind='bar', color=FEATURE_IMPORTANCE_BAR_COLOR)
        plt.title(f'Feature Importance for {ticker}')
        plt.ylabel('Importance Score')
        plt.xlabel('Feature')
        plt.savefig(os.path.join(output_dir, FEATURE_IMPORTANCE_PLOT_FILENAME.format(ticker=ticker)))
        print(f"\nFeature Importance for {ticker}:\n{importance}")
        plt.close()

    print(f"\nOptimal Probability Threshold for {ticker}: {threshold:.2f}")


def main(tickers):
    desktop = os.path.expanduser("~/Desktop")
    output_folder = os.path.join(desktop, OUTPUT_FOLDER_NAME)
    os.makedirs(output_folder, exist_ok=True)
    print(f"\nSaving results to: {output_folder}")

    for ticker in tickers:
        print(f"\n{'='*30} Processing {ticker} {'='*30}")
        try:
            # 1. Download data
            print(f"\n[1/4] Downloading {ticker} data...")
            data = download_stock_data(ticker)
            print("\nRaw data sample:")
            print(data[['open', 'high', 'low', 'close', 'volume']].head())

            # 2. Create features
            print(f"\n[2/4] Creating technical features for {ticker}...")
            data = create_features(data)
            data = prepare_target(data)

            if not validate_features(data):
                print("\nDebugging steps:")
                print("1. Check model.py has all indicator functions")
                print("2. Verify data has enough history (try longer timeframe)")
                print("3. Inspect raw data above")
                raise ValueError("Feature creation failed")

            print(f"\nSuccessfully created features for {ticker}:")
            print(data.drop(['open', 'high', 'low', 'close', 'volume', 'target'], axis=1).head())

            # 3. Train model
            print(f"\n[3/4] Training model for {ticker}...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results, model, metrics, importance, threshold = train_and_evaluate(
                    data,
                    model_type='xgboost'
                )

            # 4. Save and Plot results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_output_file = os.path.join(output_folder, RESULTS_FILENAME_FORMAT.format(ticker=ticker, timestamp=timestamp))
            results.to_csv(results_output_file)
            print(f"\n[4/4] Success! Results for {ticker} saved to {results_output_file}")

            print(f"\n[Plotting Results for {ticker}...]")
            plot_results(results.copy(), metrics, importance, threshold, ticker, output_folder) # Pass output folder

            # 5. Save the trained model
            model_filename = os.path.join(output_folder, MODEL_FILENAME_FORMAT.format(ticker=ticker, timestamp=timestamp))
            with open(model_filename, 'wb') as file:
                pickle.dump(model.model, file)  # Save the underlying scikit-learn/XGBoost model
            print(f"\nTrained model for {ticker} saved to {model_filename}")

        except ValueError as ve:
            print(f"\nValueError encountered while processing {ticker}: {ve}")
        except Exception as e:
            print(f"\nFATAL ERROR while processing {ticker}: {str(e)}")
            print("\nTROUBLESHOOTING:")
            print("1. Update all packages: pip install --upgrade yfinance pandas numpy scikit-learn xgboost matplotlib seaborn pickle")
            print("2. Check the ticker symbol is correct (ensure '.NS' if needed)")
            print("3. Try a different timeframe (e.g., '{DOWNLOAD_TIMEFRAME}' or '3mo')")
            print("4. Check model.py for missing indicator functions")


if __name__ == "__main__":
    # Fetch NIFTY 50 tickers
    try:
        nifty50_df = pd.read_html(NIFTY50_URL)[1]
        nifty50_tickers = [f"{symbol}.NS" for symbol in nifty50_df['Symbol'].tolist()]
        print("Fetched NIFTY 50 tickers:")
        print(nifty50_tickers)
    except Exception as e:
        print(f"Error fetching NIFTY 50 tickers: {e}")
        nifty50_tickers = FALLBACK_NIFTY50_STOCKS
        print("Using fallback NIFTY 50 tickers:")
        print(nifty50_tickers)

    # Verify dependencies
    try:
        import yfinance as yf
        import pandas as pd
        pd.set_option('display.max_columns', 100)
        import numpy as np
        from model import create_features  # Test import
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pickle # Verify pickle is importable
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install yfinance pandas numpy xgboost scikit-learn matplotlib seaborn pickle")
        sys.exit(1)

    main(nifty50_tickers)