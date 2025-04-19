# model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# ======================
# Technical Indicators
# ======================

def SMA(series, window):
    """Simple Moving Average"""
    return series.rolling(window=window).mean()


def RSI(series, window=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / (avg_loss + 1e-10)  # Add small constant to avoid division by zero
    return 100 - (100 / (1 + rs))


def MACD(series, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def true_range(df):
    """Calculate True Range"""
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def on_balance_volume(close, volume):
    """Calculate On-Balance Volume"""
    obv = (volume * (~close.diff().le(0) * 2 - 1)).cumsum()
    return obv


def lower_lows(series, length=3):
    """Detect sequence of lower lows"""
    lows = []
    for i in range(len(series)):
        if i < length:
            lows.append(False)
            continue

        current = series.iloc[i]
        prev_lows = series.iloc[i - length:i]

        if (current < prev_lows.min() and
                all(prev_lows.iloc[j] < prev_lows.iloc[j - 1] for j in range(1, len(prev_lows)))):
            lows.append(True)
        else:
            lows.append(False)
    return pd.Series(lows, index=series.index)


def higher_lows(series, length=3):
    """Detect sequence of higher lows"""
    highs = []
    for i in range(len(series)):
        if i < length:
            highs.append(False)
            continue

        current = series.iloc[i]
        prev_highs = series.iloc[i - length:i]

        if (current > prev_highs.max() and
                all(prev_highs.iloc[j] > prev_highs.iloc[j - 1] for j in range(1, len(prev_highs)))):
            highs.append(True)
        else:
            highs.append(False)
    return pd.Series(highs, index=series.index)


def higher_highs(series, length=3):
    """Detect sequence of higher highs"""
    return higher_lows(series, length)


def lower_highs(series, length=3):
    """Detect sequence of lower highs"""
    return lower_lows(series, length)


# ======================
# Feature Engineering
# ======================

def create_features(df, window=20):
    """Create predictive features from price and volume data"""
    df = df.copy()

    # Price features
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window).std()
    df['momentum'] = df['close'] / df['close'].shift(window) - 1
    df['atr'] = true_range(df).rolling(window).mean()

    # Volume features
    df['volume_ma'] = df['volume'].rolling(window).mean()
    df['volume_z'] = (df['volume'] - df['volume_ma']) / (df['volume'].rolling(window).std() + 1e-10)
    df['volume_change'] = df['volume'].pct_change()
    df['obv'] = on_balance_volume(df['close'], df['volume'])

    # Technical indicators
    df['rsi'] = RSI(df['close'], 14)
    macd_val, macd_signal = MACD(df['close'])
    df['macd'] = macd_val
    df['signal_line'] = macd_signal
    df['price_ma'] = SMA(df['close'], 50)

    # Divergence features (using volume for divergence)
    df['bull_div'] = ((lower_lows(df['close'], 3)) &
                      (higher_lows(df['volume'], 3))).astype(int)
    df['bear_div'] = ((higher_highs(df['close'], 3)) &
                      (lower_highs(df['volume'], 3))).astype(int)

    # Lagged features
    for lag in [1, 2]: # Reduced lags for simplicity and to match error
        df[f'returns_lag{lag}'] = df['returns'].shift(lag)
        df[f'volume_z_lag{lag}'] = df['volume_z'].shift(lag)
        df[f'rsi_lag{lag}'] = df['rsi'].shift(lag)

    df = clean_data(df)
    return df.dropna()


def prepare_target(df, forward_periods=5):
    """Create target variable: 1 if price will rise, 0 otherwise"""
    df['target'] = (df['close'].shift(-forward_periods) > df['close']).astype(int)
    return df.dropna()


def clean_data(df):
    """Handle infinite and missing values"""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = np.clip(df[col], -1e8, 1e8)
    return df.dropna()


# ======================
# Modeling Core
# ======================

class TradingModel:
    def __init__(self, model_type='logistic'):
        self.scaler = StandardScaler()
        if model_type == 'logistic':
            self.model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        elif model_type == 'xgboost':
            self.model = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        else:
            raise ValueError("Invalid model type")

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def feature_importance(self, feature_names):
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(self.model.feature_importances_, index=feature_names)
        elif hasattr(self.model, 'coef_'):
            return pd.Series(np.abs(self.model.coef_[0]), index=feature_names)
        else:
            return None


# ======================
# Backtesting & Evaluation
# ======================

def walk_forward_validation(X, y, model_type='logistic', n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []
    feature_importances = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = TradingModel(model_type)
        model.train(X_train, y_train)

        # Store feature importances
        imp = model.feature_importance(X.columns)
        if imp is not None:
            feature_importances.append(imp)

        # Evaluate
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics.append({
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'accuracy': report['accuracy']
        })

    # Aggregate results
    metrics_df = pd.DataFrame(metrics)
    if feature_importances:
        importance_df = pd.DataFrame(feature_importances).mean().sort_values(ascending=False)
    else:
        importance_df = None

    return metrics_df, importance_df


def optimize_threshold(X, y, model):
    probs = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]


# ======================
# Main Training Function
# ======================

def train_and_evaluate(df, model_type='xgboost'):
    """Main function to process data and train model"""
    # Feature engineering
    df = create_features(df)
    df = prepare_target(df, forward_periods=5)

    # Select features
    feature_columns = [
        'returns', 'volatility', 'momentum', 'atr',
        'volume_z', 'volume_change', 'obv',
        'rsi', 'macd', 'bull_div', 'bear_div',
        'returns_lag1', 'returns_lag2', 'volume_z_lag1', 'rsi_lag1' # Ensure all are here
    ]
    X = df[feature_columns]
    y = df['target']

    # Walk-forward validation
    print("\nRunning Walk-Forward Validation...")
    metrics_df, importance_df = walk_forward_validation(X, y, model_type=model_type)

    # Train final model
    print("\nTraining Final Model...")
    final_model = TradingModel(model_type=model_type)
    final_model.train(X, y)

    # Generate signals
    optimal_threshold = optimize_threshold(X, y, final_model)
    print(f"\nOptimal Probability Threshold: {optimal_threshold:.2f}")

    df['signal_prob'] = final_model.predict_proba(X)[:, 1]
    df['signal'] = 0
    df.loc[df['signal_prob'] > optimal_threshold, 'signal'] = 1
    df.loc[df['signal_prob'] < (1 - optimal_threshold), 'signal'] = -1

    return df, final_model, metrics_df, importance_df, optimal_threshold


if __name__ == "__main__":
    print("This module contains the trading model implementation. Import it from another script.")