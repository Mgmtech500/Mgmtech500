import alpaca_trade_api as alpaca
import smtplib
import logging
import time
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

# --- Logging Setup ---
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler('three_bar_play_bot.log', maxBytes=2000000, backupCount=5)
logging.basicConfig(level=logging.INFO, handlers=[handler], format='%(asctime)s - %(levelname)s - %(message)s')

# --- Alpaca API Client ---
api = alpaca.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# --- Real-Time Data Fetch (Mock for Testing) ---
def get_real_time_data(symbol):
    data = {
        'open': np.random.uniform(100, 200, 50),
        'high': np.random.uniform(200, 250, 50),
        'low': np.random.uniform(90, 100, 50),
        'close': np.random.uniform(100, 200, 50),
        'volume': np.random.randint(1000, 5000, 50)
    }
    return pd.DataFrame(data)

# --- Feature Engineering ---
def create_features(data):
    data['close_lag1'] = data['close'].shift(1)
    data['close_lag2'] = data['close'].shift(2)
    data['volume_lag1'] = data['volume'].shift(1)
    data['volume_lag2'] = data['volume'].shift(2)

    data['ma5'] = data['close'].rolling(window=5).mean()
    data['ma10'] = data['close'].rolling(window=10).mean()
    data['ma20'] = data['close'].rolling(window=20).mean()

    data['rsi'] = 100 - (100 / (1 + data['close'].diff().fillna(0).rolling(window=14).mean() /
                                data['close'].diff().fillna(0).abs().rolling(window=14).mean()))

    data.dropna(inplace=True)
    return data

# --- Feature and Label Creation ---
def create_features_and_labels(data):
    if data.empty or 'close' not in data.columns:
        logging.error("Data is empty or missing 'close' column.")
        return pd.DataFrame(), pd.Series(dtype=float)

    features = create_features(data)
    features.dropna(inplace=True)

    labels = data['close'].shift(-1)
    labels = labels.loc[features.index]

    valid_idx = labels.notnull()
    features = features[valid_idx]
    labels = labels[valid_idx]

    return features, labels

# --- Model Training and Evaluation ---
def train_and_evaluate_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    logging.info(f"Model Evaluation - MSE: {mse}, R^2: {r2}, MAE: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Mean Absolute Error: {mae}")

    return model

# --- Trade Execution ---
def execute_trade(symbol, side, qty):
    from alpaca_trade_api.rest import APIError
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )
        logging.info(f"Order executed: {order}")
        print(f"Order executed: {order}")
    except APIError as e:
        logging.error(f"Alpaca API error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error executing order: {e}")

# --- Profit Target and Stop Loss ---
def monitor_trade(symbol, buy_price):
    profit_target = buy_price * 1.03  # 3% profit target
    stop_loss = buy_price * 0.98      # 2% stop loss
    max_retries = 60  # Monitor for up to 1 hour

    for attempt in range(max_retries):
        try:
            last_quote = api.get_last_quote(symbol)
            last_price = last_quote.askprice
            if last_price >= profit_target:
                execute_trade(symbol, 'sell', 100)
                logging.info(f"Profit target reached for {symbol}")
                break
            elif last_price <= stop_loss:
                execute_trade(symbol, 'sell', 100)
                logging.info(f"Stop loss triggered for {symbol}")
                break
            time.sleep(60)
        except Exception as e:
            logging.error(f"Error monitoring trade: {e}")
            break

# --- Three Bar Play Detection ---
def detect_three_bar_play(data):
    if len(data) < 3:
        return False

    bar1 = data.iloc[-3]
    bar2 = data.iloc[-2]
    bar3 = data.iloc[-1]

    # Check for Three Bar Play pattern
    if (bar1['close'] > bar1['open'] and
        bar2['high'] < bar1['high'] and
        bar2['low'] > bar1['low'] and
        bar3['close'] > bar1['high']):
        return True

    return False

# --- Candidate Stock Selection ---
def get_candidate_stocks():
    try:
        active_assets = api.list_assets(status='active')
        us_equities = [asset.symbol for asset in active_assets if asset.exchange in ['NASDAQ', 'NYSE']]

        candidate_scores = {}
        for symbol in us_equities[:50]:  # Limit to avoid rate limits
            barset = api.get_barset(symbol, 'day', limit=5)
            bars = barset[symbol]

            close_prices = [bar.c for bar in bars]
            price_change = (close_prices[-1] - close_prices[0]) / close_prices[0]
            volatility = pd.Series(close_prices).pct_change().std()

            score = price_change * volatility
            candidate_scores[symbol] = score

        sorted_candidates = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        top_candidates = [symbol for symbol, score in sorted_candidates[:10]]

        logging.info(f"Top 10 candidate stocks: {top_candidates}")
        return top_candidates

    except Exception as e:
        logging.error(f"Error getting candidate stocks: {e}")
        return []

# --- Main Execution ---
if __name__ == "__main__":
    candidate_stocks = get_candidate_stocks()

    for symbol in candidate_stocks:
        data = get_real_time_data(symbol)
        features, labels = create_features_and_labels(data)

        if not features.empty and not labels.empty:
            model = train_and_evaluate_model(features, labels)

            if detect_three_bar_play(data):
                last_close = data['close'].iloc[-1]
                execute_trade(symbol, 'buy', 100)
                monitor_trade(symbol, last_close)
            else:
                logging.info(f"No Three Bar Play detected for {symbol}")
                print(f"No Three Bar Play detected for {symbol}")
        else:
            logging.error(f"Failed to train model for {symbol} due to insufficient data.")
            print(f"Data issue encountered for {symbol}. Check logs for details.")