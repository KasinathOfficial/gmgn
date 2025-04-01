import streamlit as st
import requests
import pandas as pd
import time
import random  # Simulating win probability for now; later, replace with AI model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np

st.set_page_config(page_title="Crypto Explosion Predictor", layout="wide")
st.title("🚀 Crypto Explosion Predictor")

# Placeholder for model loading and training
model_target = None
model_stoploss = None

def fetch_coindcx_data():
    url = "https://api.coindcx.com/exchange/ticker"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return [coin for coin in data if coin['market'].endswith('INR')]  # Filter INR pairs
    except requests.RequestException:
        return []

def fetch_historical_data():
    # Function to fetch or load historical data for training the model (this could be pre-collected data)
    data = pd.read_csv('historical_crypto_data.csv')  # Placeholder for actual historical data CSV
    return data

def train_model(data):
    # Use RandomForestRegressor to train models for predicting Target Price and Stop Loss
    X = data[['price', 'change_24_hour', 'volume', 'volatility']]  # Features
    y_target = data['target_price']  # Target price column in historical data
    y_stoploss = data['stop_loss']  # Stop loss column in historical data

    # Split data into training and testing sets
    X_train, X_test, y_target_train, y_target_test = train_test_split(X, y_target, test_size=0.2, random_state=42)
    X_train, X_test, y_stoploss_train, y_stoploss_test = train_test_split(X, y_stoploss, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest models
    model_target = RandomForestRegressor(n_estimators=100, random_state=42)
    model_target.fit(X_train, y_target_train)

    model_stoploss = RandomForestRegressor(n_estimators=100, random_state=42)
    model_stoploss.fit(X_train, y_stoploss_train)

    # Predict and calculate MAE (Mean Absolute Error) to assess the model's performance
    target_pred = model_target.predict(X_test)
    stoploss_pred = model_stoploss.predict(X_test)

    target_mae = mean_absolute_error(y_target_test, target_pred)
    stoploss_mae = mean_absolute_error(y_stoploss_test, stoploss_pred)

    st.write(f"Target Price Model MAE: {target_mae}")
    st.write(f"Stop Loss Model MAE: {stoploss_mae}")

    # Save the trained models
    joblib.dump(model_target, 'model_target.pkl')
    joblib.dump(model_stoploss, 'model_stoploss.pkl')

def load_models():
    global model_target, model_stoploss
    model_target = joblib.load('model_target.pkl')
    model_stoploss = joblib.load('model_stoploss.pkl')

def calculate_win_probability(change, volume):
    base_win_rate = 50  # Base probability of a trade winning (can be adjusted)
    momentum_boost = min(change * 2, 20)  # More change = Higher probability (Capped at 20%)
    volume_boost = min((volume / 10000000) * 10, 30)  # More volume = Higher probability (Capped at 30%)
    total_probability = base_win_rate + momentum_boost + volume_boost
    return round(min(total_probability, 95), 2)  # Capping at 95% confidence

def analyze_market(data):
    potential_explosions = []
    for coin in data:
        try:
            symbol = coin['market']
            price = float(coin['last_price'])
            volume = float(coin['volume'])
            change = float(coin['change_24_hour'])
            volatility = abs(change) * (1 + (volume / 10000000))

            if change > 5 and volume > 500000:  # Trade filter
                # Predict Target Price and Stop Loss using the ML models
                features = np.array([[price, change, volume, volatility]])
                target_price_pred = model_target.predict(features)[0]
                stop_loss_pred = model_stoploss.predict(features)[0]

                # Calculate win probability
                win_probability = calculate_win_probability(change, volume)

                if win_probability > 80:
                    trade_decision = f"🔥 High Confidence Buy (Win %: {win_probability}%)"
                elif win_probability > 65:
                    trade_decision = f"✅ Strong Buy (Win %: {win_probability}%)"
                else:
                    trade_decision = f"⚠️ Moderate Buy (Win %: {win_probability}%)"

                potential_explosions.append({
                    "Symbol": symbol, "Price": price, "24h Change (%)": change,
                    "Volume": volume, "Volatility (%)": volatility,
                    "Target Price": target_price_pred, "Stop Loss Price": stop_loss_pred,
                    "Win Probability (%)": win_probability, "Trade Decision": trade_decision
                })
        except (ValueError, KeyError):
            continue
    return potential_explosions

placeholder = st.empty()
if not model_target or not model_stoploss:
    train_model(fetch_historical_data())  # Train models if not already loaded

load_models()  # Load pre-trained models for prediction

while True:
    data = fetch_coindcx_data()
    if data:
        analyzed_data = analyze_market(data)
        if analyzed_data:
            df = pd.DataFrame(analyzed_data)
            with placeholder.container():
                st.subheader("📈 Cryptos Likely to Explode Soon")
                st.dataframe(df)
        else:
            with placeholder.container():
                st.info("No potential explosive cryptos detected right now.")
    else:
        with placeholder.container():
            st.error("Failed to retrieve data. Please check API access.")
    
    time.sleep(1)  # Refresh data every second without refreshing the page
