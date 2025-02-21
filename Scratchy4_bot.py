Scratchy4

#!/usr/bin/python


# --- Three Bar Play Trading Bot with Machine Learning and Enhanced Features ---

import alpaca_trade_api as tradeapi
from twilio.rest import Client
import smtplib
import logging
import time
import syslog
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Configuration ---

# Alpha Vantage API credentials
ALPHA_VANTAGE_API_KEY = "JBFOYTHV49QBZ9W7"

# Alpaca API credentials
# Alpaca API credentials
APCA_API_KEY_ID = 'PKF7936G7Z9XY2K8DI4O'
APCA_API_SECRET_KEY = 'TUxwrLFCchabEFVRa5pROqIJkT9XpKRZs8wTxbQr'
##lpaca API credentials for paper trading
#API_KEY = 'PKF7936G7Z9XY2K8DI4O'
API_KEY = 'JBFOYTHV49QBZ9W7'
SECRET_KEY = 'TUxwrLFCchabEFVRa5pROqIJkT9XpKRZs8wTxbQr'
BASE_URL = "https://paper-api.alpaca.markets"


ALPACA_API_KEY = "PKF7936G7Z9XY2K8DI4O"

ALPACA_SECRET_KEY = "TUxwrLFCchabEFVRa5pROqIJkT9XpKRZs8wTxbQr"

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"



SMTP_SERVER = "smtp.gmail.com"

SMTP_PORT = 587

SMTP_USERNAME = "mgmtech"

SMTP_PASSWORD = "wqrs opnb snnn dkho"

SENDER_EMAIL = "mgmtech@gmail.com"

RECIPIENT_EMAIL = "michael.a.gallagher@protonmail.com"




# --- Logging Setup ---
logging.basicConfig(filename='three_bar_play_bot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

syslog.openlog(ident='three_bar_play_bot', facility=syslog.LOG_USER)

# --- Alpaca API Client ---
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)


# --- Market Data Source (Polygon.io) ---
def get_real_time_data(symbol):
    """
    Fetches real-time bar data from Polygon.io.

    Args:
        symbol: Stock symbol.

    Returns:
        A pandas DataFrame containing OHLCV data.
    """
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/2023-12-18/2023-12-18?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['results'])
        df = df[['o', 'h', 'l', 'c', 'v']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        return df
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from Polygon.io: {e}")
        send_notification(f"Error fetching data from Polygon.io: {e}")
        return None
# --- Machine Learning Model Training ---
# Assuming you have a function to create features and labels
def create_features_and_labels(data):
    """
    Creates features and labels from OHLCV data for the machine learning model.

    Args:
        data: A pandas DataFrame with OHLCV data.

    Returns:
        A tuple of features and labels DataFrames.
    """
    features = create_features(data)
    labels = data['close'].shift(-1)  # Example label: next day's close price
    features.dropna(inplace=True)
    labels = labels.loc[features.index]
    return features, labels

# Assuming you have a DataFrame named 'data' with your OHLCV data
features, labels = create_features_and_labels(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Create and train the model (example using Random Forest)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")


# --- Feature Engineering ---
def create_features(data):
    """
    Creates features from OHLCV data for the machine learning model.

    Args:
        data: A pandas DataFrame with OHLCV data.

    Returns:
        A pandas DataFrame with engineered features.
    """
    # Lag features
    data['close_lag1'] = data['close'].shift(1)
    data['close_lag2'] = data['close'].shift(2)
    data['volume_lag1'] = data['volume'].shift(1)
    data['volume_lag2'] = data['volume'].shift(2)

    # Moving averages
    data['ma5'] = data['close'].rolling(window=5).mean()
    data['ma10'] = data['close'].rolling(window=10).mean()
    data['ma20'] = data['close'].rolling(window=20).mean()

    # Exponential moving averages
    data['ema5'] = data['close'].ewm(span=5, adjust=False).mean()
    data['ema10'] = data['close'].ewm(span=10, adjust=False).mean()
    data['ema20'] = data['close'].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    data['stddev'] = data['close'].rolling(window=20).std()
    data['upper_band'] = data['ma20'] + (data['stddev'] * 2)
    data['lower_band'] = data['ma20'] - (data['stddev'] * 2)

    # Volume features
    data['volume_ma5'] = data['volume'].rolling(window=5).mean()
    data['volume_ma10'] = data['volume'].rolling(window=10).mean()

    # Momentum
    data['momentum'] = data['close'] - data['close'].shift(4)

    # Rate of Change (ROC)
    data['roc'] = data['close'].pct_change(periods=4)

    # Drop NaN values created by rolling calculations
    data.dropna(inplace=True)

    return data


# --- Three Bar Play Detection (with Machine Learning) ---
def is_three_bar_play(data, model):
    """
    Detects a three-bar play pattern using a machine learning model.

    Args:
        data: A pandas DataFrame with OHLCV data and engineered features.
        model: The trained machine learning model.

    Returns:
        True if a three-bar play pattern is detected, False otherwise.
    """
    try:
        # Create features for the current bar
        features = create_features(data)
        #features = create_features_and_labels(data)
        last_row = features.iloc[-1].values.reshape(1, -1)  # Reshape for prediction

        # Predict the probability of a three-bar play
        prediction = model.predict_proba(last_row)[:, 1]  # Probability of positive class

        # Set a threshold for the prediction (e.g., 0.7)
        if prediction >= 0.7:
            return True
        else:
            return False
    except Exception as e:
        logging.error(f"Error checking for three-bar play: {e}")
        return False


# --- Order Execution ---
def execute_trade(symbol, side, qty):
    """
    Executes a buy or sell order.

    Args:
        symbol: Stock symbol.
        side: 'buy' or 'sell'.
        qty: Quantity of shares to trade.
    """
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )
        logging.info(f"Order executed: {order}")
        send_notification(f"Order executed: {order}")
    except Exception as e:
        logging.error(f"Error executing order: {e}")
        send_notification(f"Error executing order: {e}")


# --- Profit Target and Stop Loss ---
def monitor_trade(symbol, buy_price):
    """
    Monitors the trade for profit target and stop loss.

    Args:
        symbol: Stock symbol.
        buy_price: Purchase price of the stock.
    """
    profit_target = buy_price * 1.03  # 3% profit target
    stop_loss = buy_price * 0.98  # 2% stop loss

    while True:
        try:
            last_price = api.get_last_quote(symbol).askprice
            if last_price >= profit_target:
                execute_trade(symbol, 'sell', 100)
                logging.info(f"Profit target reached for {symbol}")
                send_notification(f"Profit target reached for {symbol}")
                break
            elif last_price <= stop_loss:
                execute_trade(symbol, 'sell', 100)
                logging.info(f"Stop loss triggered for {symbol}")
                send_notification(f"Stop loss triggered for {symbol}")
                break
            time.sleep(60)  # Check every minute
        except Exception as e:
            logging.error(f"Error monitoring trade: {e}")
            send_notification(f"Error monitoring trade: {e}")
            break


# --- Candidate Stock Selection ---
def get_candidate_stocks():
    """
    Generates a list of 10 top candidate stocks for three-bar play
    based on recent price action and volatility.

    Returns:
        A list of 10 stock symbols.
    """
    try:
        # Fetch a list of active assets from Alpaca
        active_assets = api.list_assets(status='active')
        # Filter for US equities
        us_equities = [asset.symbol for asset in active_assets if asset.exchange == 'NASDAQ' or asset.exchange == 'NYSE']

        candidate_scores = {}
        for symbol in us_equities:
            # Fetch historical data for the past 5 days
            barset = api.get_barset(symbol, 'day', limit=5)
            bars = barset[symbol]

            # Calculate price change and volatility
            close_prices = [bar.c for bar in bars]
            price_change = (close_prices[-1] - close_prices[0]) / close_prices[0]
            volatility = pd.Series(close_prices).pct_change().std()

            # Assign a score based on price change and volatility
            score = price_change * volatility

            candidate_scores[symbol] = score

        # Sort candidates by score and select top 10
        sorted_candidates = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        top_candidates = [symbol for symbol, score in sorted_candidates[:10]]

        logging.info(f"Top 10 candidate stocks: {top_candidates}")
        return top_candidates

    except Exception as e:
        logging.error(f"Error getting candidate stocks: {e}")
        send_notification(f"Error getting candidate stocks: {e}")
        return []

# --- Notification Functions ---
def send_notification(message):
    """
    Sends notifications via syslog, SMS, and email.

    Args:
        message: The message to send.
    """
    try:
        # Syslog
        syslog.syslog(syslog.LOG_INFO, message)

        # SMS
        #client = #Client(TWIL#IO_ACCOUN#T_SID, #TWILIO_AU#TH_TOKEN)
        “””sms_message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=RECIPIENT_PHONE_NUMBER
        )
“””
        # Email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            email_message = f"Subject: Three Bar Play Bot Alert\n\n{message}"
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, email_message)

    except Exception as e:
        logging.error(f"Error sending notification: {e}")


# --- Machine Learning Model Training ---
# Assuming you have a DataFrame named 'data' with your OHLCV data
def create_features_and_labels(data):
    """
    Creates features and labels from OHLCV data for the machine learning model.

    Args:
        data: A pandas DataFrame with OHLCV data.

    Returns:
        A tuple of features and labels DataFrames.
    """
    features = create_features(data)
    labels = data['close'].shift(-1)  # Example label: next day's close price
    features.dropna(inplace=True)
    labels = labels.loc[features.index]
    return features, labels

# Assuming you have a DataFrame named 'data' with your OHLCV data
features, labels = create_features_and_labels(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Create and train the model (example using Random Forest)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")


# --- Main Trading Loop ---
if __name__ == "__main__":
    while True:
        candidates = get_candidate_stocks()
        for symbol in candidates:
            try:
                data = get_real_time_data(symbol)
                if data is not None:
                    # Create features for the current bar
                    features = create_features(data)
                    if is_three_bar_play(features, model):  # Use the trained model
                        logging.info(f"Three bar play detected for {symbol}")
                        execute_trade(symbol, 'buy', 100)
                        buy_price = api.get_last_trade(symbol).price
                        monitor_trade(symbol, buy_price)
            except Exception as e:
                logging.error(f"Error processing {symbol}: {e}")
                send_notification(f"Error processing {symbol}: {e}")
        time.sleep(60)  # Check every minute