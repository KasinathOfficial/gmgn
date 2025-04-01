import streamlit as st
import requests
import pandas as pd
import time
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Set up Streamlit app
st.set_page_config(page_title="Crypto Explosion Predictor with AI", layout="wide")
st.title("🚀 Crypto Explosion Predictor with Dynamic ML Model")

# Function to fetch live data from CoinDCX
def fetch_coindcx_data():
    url = "https://api.coindcx.com/exchange/ticker"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return [coin for coin in data if coin['market'].endswith('INR')]  # Filter INR pairs
    except requests.RequestException:
        return []

# Function to calculate target price based on price, change, and volume
def calculate_target_price(price, change, volume):
    fib_multiplier = 1.618
    volatility_factor = 1 + (volume / 10000000)
    return round(price * (1 + ((change / 100) * fib_multiplier * volatility_factor)), 2)

# Function to calculate stop loss based on price and change
def calculate_stop_loss(price, change):
    stop_loss_factor = 0.95 if change > 8 else 0.90
    return round(price * stop_loss_factor, 2)

# Function to calculate volatility based on price change and volume
def calculate_volatility(change, volume):
    return round(abs(change) * (1 + (volume / 10000000)), 2)

# Dynamic Model Training
def train_model(data):
    features = []
    labels = []

    for coin in data:
        try:
            price = float(coin['last_price'])
            volume = float(coin['volume'])
            change = float(coin['change_24_hour'])
            target_price = calculate_target_price(price, change, volume)
            stop_loss_price = calculate_stop_loss(price, change)
            volatility = calculate_volatility(change, volume)

            # Creating the feature set for training
            features.append([price, volume, change, volatility])
            label = 1 if change > 10 and volume > 500000 else 0  # Label: 1 for potential explosion, 0 for no
            labels.append(label)

        except (ValueError, KeyError):
            continue
    
    if len(features) > 0:
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Initialize the model (Random Forest)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save the trained model
        joblib.dump(model, 'crypto_explosion_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')

        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
        return model, scaler
    else:
        return None, None

# AI-based prediction function (using the trained model)
def predict_price_explosion(model, scaler, price, volume, change, volatility):
    features = np.array([[price, volume, change, volatility]])
    features = scaler.transform(features)  # Standardize features using the saved scaler
    prediction = model.predict(features)  # Predict whether it will explode (1 or 0)
    probability = model.predict_proba(features)[:, 1]  # Probability of explosion
    return prediction[0], round(probability[0] * 100, 2)

# Analyze the market using live data and apply the AI prediction
def analyze_market(model, scaler, data):
    potential_explosions = []
    for coin in data:
        try:
            symbol = coin['market']
            price = float(coin['last_price'])
            volume = float(coin['volume'])
            change = float(coin['change_24_hour'])

            if change > 5 and volume > 500000:  # Trade filter
                target_price = calculate_target_price(price, change, volume)
                stop_loss_price = calculate_stop_loss(price, change)
                volatility = calculate_volatility(change, volume)

                # AI Prediction
                prediction, probability = predict_price_explosion(model, scaler, price, volume, change, volatility)

                if prediction == 1:
                    trade_decision = f"🔥 High Confidence Buy (Explosive Probability: {probability}%)"
                else:
                    trade_decision = f"⚠️ Low Confidence (Explosive Probability: {probability}%)"

                potential_explosions.append({
                    "Symbol": symbol, "Price": price, "24h Change (%)": change,
                    "Volume": volume, "Volatility (%)": volatility,
                    "Target Price": target_price, "Stop Loss Price": stop_loss_price,
                    "Explosive Probability (%)": probability, "Trade Decision": trade_decision
                })
        except (ValueError, KeyError):
            continue
    return potential_explosions

# Load the model if it exists
try:
    model = joblib.load('crypto_explosion_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    model, scaler = None, None

# Streamlit dynamic updates
placeholder = st.empty()

while True:
    data = fetch_coindcx_data()
    
    # Retrain the model with new data periodically
    if data and model and scaler:
        analyzed_data = analyze_market(model, scaler, data)
        if analyzed_data:
            df = pd.DataFrame(analyzed_data)
            with placeholder.container():
                st.subheader("📈 Cryptos Likely to Explode Soon")
                st.dataframe(df)
        else:
            with placeholder.container():
                st.info("No potential explosive cryptos detected right now.")
    else:
        # If model is not available, train it
        st.write("Training model with live data...")
        model, scaler = train_model(data)
    
    # Sleep to control update frequency
    time.sleep(60)  # Retrain every minute (or set based on your requirements)
