scratchy10_bot.py

#!/usr/bin/python

import logging
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from alpaca_trade_api.rest import REST, TimeFrame, APIError
from datetime import datetime, timedelta
import smtplib
import time

# --- CONFIGURATION ---
POLYGON_API_KEY = 'your_polygon_api_key'
APCA_API_KEY_ID = 'your_alpaca_api_key_id'
APCA_API_SECRET_KEY = 'your_alpaca_api_secret_key'
BASE_URL = 'https://paper-api.alpaca.markets'

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "your_email@gmail.com"
SMTP_PASSWORD = "your_email_password"
SENDER_EMAIL = "your_email@gmail.com"
RECIPIENT_EMAIL = "recipient_email@example.com"

# Initialize Alpaca API
api = REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, base_url=BASE_URL)

# Date Range for Data
end_date = datetime.now()
start_date = end_date - timedelta(days=90)
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')


# --- FUNCTIONS ---

def get_polygon_data(symbol, start_date, end_date):
    """
    Fetch historical data for a given symbol from Polygon.io.
    """
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={POLYGON_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'results' in data and data['results']:
            df = pd.DataFrame(data['results'])
            df.rename(columns={'c': 'close', 'h': 'high', 'l': 'low', 'o': 'open', 'v': 'volume', 't': 'timestamp'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        else:
            logging.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
    except Exception as e:
        logging.warning(f"Polygon API error for {symbol}: {e}")
        return pd.DataFrame()


def create_features(data):
    """
    Create technical indicators as features for the model.
    """
    required_columns = ['close', 'volume']
    if not all(col in data.columns for col in required_columns):
        missing = [col for col in required_columns if col not in data.columns]
        logging.warning(f"Data missing required columns: {missing}")
        return pd.DataFrame()

    data['close_lag1'] = data['close'].shift(1)
    data['ma5'] = data['close'].rolling(window=5).mean()
    data['rsi'] = 100 - (100 / (1 + data['close'].pct_change().rolling(window=14).mean()))
    data.dropna(inplace=True)

    return data


def create_features_and_labels(data):
    """
    Generate features and labels for model training.
    """
    if isinstance(data, list):
        data = pd.DataFrame(data)

    required_columns = ['close', 'volume']
    if not all(col in data.columns for col in required_columns):
        missing = [col for col in required_columns if col not in data.columns]
        logging.warning(f"Data missing required columns: {missing}")
        return pd.DataFrame(), pd.Series()

    features = create_features(data)
    labels = data['close'].shift(-1)

    combined = pd.concat([features, labels.rename('label')], axis=1)
    combined.dropna(inplace=True)

    features = combined.drop('label', axis=1)
    labels = combined['label']

    return features, labels


def train_model(features, labels):
    """
    Train a Random Forest model and evaluate its performance.
    """
    if features.empty or labels.empty:
        logging.error("Training data is empty.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model Mean Squared Error: {mse}")

    return model


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    symbols = ['AAPL', 'TSLA', 'GOOG', 'ZBRA']
    all_data = {}

    for symbol in symbols:
        data = get_polygon_data(symbol, start_date_str, end_date_str)
        if not data.empty:
            all_data[symbol] = data
        else:
            logging.warning(f"No data for {symbol}")

    for symbol, data in all_data.items():
        features, labels = create_features_and_labels(data)
        if not features.empty:
            print(f"Training model for {symbol}...")
            model = train_model(features, labels)
        else:
            logging.warning(f"Skipping {symbol} due to insufficient data.")