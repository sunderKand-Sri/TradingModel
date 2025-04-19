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
import openpyxl

# --- Configuration ---
OUTPUT_FOLDER_NAME = "modelresult_threshold_filtered"
DOWNLOAD_TIMEFRAME = "3y"  # Increased for potentially more data
RESULTS_FILENAME_FORMAT = "{ticker}_results_{timestamp}.csv"
MODEL_FILENAME_FORMAT = "{ticker}_model_{timestamp}.pkl"
BUY_RECOMMENDATIONS_FILENAME = "buy_recommendations_threshold_filtered_{timestamp}.xlsx"
SELL_RECOMMENDATIONS_FILENAME = "sell_recommendations_threshold_filtered_{timestamp}.xlsx"
OPTIMAL_THRESHOLD_HIGH = 0.85
OPTIMAL_THRESHOLD_LOW = 0.11

NIFTY50_URL = "https://en.wikipedia.org/wiki/NIFTY_50"
FALLBACK_NIFTY50_STOCKS =[
"ETERNAL.NS","SUNPHARMA.NS","ICICIBANK.NS","BHARTIARTL.NS","BAJAJFINSV.NS","KOTAKBANK.NS","SBIN.NS","RELIANCE.NS","SBILIFE.NS","AXISBANK.NS","ADANIPORTS.NS","GRASIM.NS","SHRIRAMFIN.NS","TRENT.NS","JIOFIN.NS","M&M.NS","TITAN.NS","HDFCBANK.NS","ULTRACEMCO.NS","NESTLEIND.NS","BAJFINANCE.NS","NTPC.NS","CIPLA.NS","TATACONSUM.NS","ONGC.NS","INFY.NS","APOLLOHOSP.NS","POWERGRID.NS","EICHERMOT.NS","TATAMOTORS.NS","TCS.NS","ITC.NS","BAJAJ-AUTO.NS","INDUSINDBK.NS","LT.NS","HCLTECH.NS","DRREDDY.NS","HDFCLIFE.NS","BEL.NS","HINDUNILVR.NS","ASIANPAINT.NS","TATASTEEL.NS","MARUTI.NS","ADANIENT.NS","COALINDIA.NS","JSWSTEEL.NS","HEROMOTOCO.NS","TECHM.NS","HINDALCO.NS","WIPRO.NS"
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


def main(tickers):
    desktop = os.path.expanduser("~/Desktop")
    output_folder = os.path.join(desktop, OUTPUT_FOLDER_NAME)
    os.makedirs(output_folder, exist_ok=True)
    print(f"\nSaving results to: {output_folder}")

    buy_recommendations = []
    sell_recommendations = []

    for ticker in tickers:
        print(f"\n{'='*30} Processing {ticker} for Threshold Filtering {'='*30}")
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
                print(f"\nOptimal Probability Threshold for {ticker}: {threshold:.2f}")

                if threshold > OPTIMAL_THRESHOLD_HIGH:
                    if results['signal'].iloc[-1] == -1:
                        sell_recommendations.append({
                            "Ticker": ticker,
                            "Optimal_Threshold": f"{threshold:.2f}",
                            "Last_Signal_Date": results.index[-1],
                            "Last_Price": results['close'].iloc[-1],
                            "F1_Score": f"{metrics['f1'].mean():.2f}"
                        })
                elif threshold < OPTIMAL_THRESHOLD_LOW:
                    if results['signal'].iloc[-1] == 1:
                        buy_recommendations.append({
                            "Ticker": ticker,
                            "Optimal_Threshold": f"{threshold:.2f}",
                            "Last_Signal_Date": results.index[-1],
                            "Last_Price": results['close'].iloc[-1],
                            "F1_Score": f"{metrics['f1'].mean():.2f}"
                        })

            # 4. Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_output_file = os.path.join(output_folder, RESULTS_FILENAME_FORMAT.format(ticker=ticker, timestamp=timestamp))
            results.to_csv(results_output_file)
            print(f"\n[4/4] Success! Results for {ticker} saved to {results_output_file}")

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
            print("1. Update all packages: pip install --upgrade yfinance pandas numpy scikit-learn xgboost matplotlib seaborn pickle openpyxl")
            print("2. Check the ticker symbol is correct (ensure '.NS' if needed)")
            print("3. Try a different timeframe (e.g., '{DOWNLOAD_TIMEFRAME}' or '3mo')")
            print("4. Check model.py for missing indicator functions")

    # Save buy recommendations to Excel
    if buy_recommendations:
        buy_df = pd.DataFrame(buy_recommendations)
        buy_excel_path = os.path.join(output_folder, BUY_RECOMMENDATIONS_FILENAME.format(timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")))
        buy_df.to_excel(buy_excel_path, index=False)
        print(f"\nBuy recommendations (Optimal Threshold < {OPTIMAL_THRESHOLD_LOW:.2f}) saved to: {buy_excel_path}")
    else:
        print(f"\nNo buy recommendations found with Optimal Threshold < {OPTIMAL_THRESHOLD_LOW:.2f}")

    # Save sell recommendations to Excel
    if sell_recommendations:
        sell_df = pd.DataFrame(sell_recommendations)
        sell_excel_path = os.path.join(output_folder, SELL_RECOMMENDATIONS_FILENAME.format(timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")))
        sell_df.to_excel(sell_excel_path, index=False)
        print(f"\nSell recommendations (Optimal Threshold > {OPTIMAL_THRESHOLD_HIGH:.2f}) saved to: {sell_excel_path}")
    else:
        print(f"\nNo sell recommendations found with Optimal Threshold > {OPTIMAL_THRESHOLD_HIGH:.2f}")


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
        import pickle
        import openpyxl # Verify openpyxl is importable
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install yfinance pandas numpy xgboost scikit-learn matplotlib seaborn pickle openpyxl")
        sys.exit(1)

    main(nifty50_tickers)