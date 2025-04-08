import streamlit as st
import pandas as pd
import requests
import time

st.set_page_config(page_title="Crypto Explosive Detector", layout="wide")
st.title("1000% Win Rate Crypto Explosion Detector")

# User-configurable settings
min_volume_threshold = 5000000  # Minimum volume for a coin to be considered
price_spike_threshold = 0.10    # 10% price change in 1 min

# Function to fetch tokens dynamically from CoinDCX
@st.cache_data(ttl=300)
def fetch_tokens():
    url = "https://api.coindcx.com/exchange/ticker"
    response = requests.get(url)
    data = response.json()
    tokens = [item['market'] for item in data if item['market'].endswith("INR")]
    return tokens

# Function to simulate price and volume data (Replace this with real-time API for production)
def simulate_data(token):
    prices = [100, 105, 115, 125]  # Simulated prices for the last 4 minutes
    volumes = [4_000_000, 5_000_000, 6_000_000, 7_000_000]  # Simulated volumes
    return prices, volumes

# Function to detect explosive moves
def detect_explosive_moves(token):
    prices, volumes = simulate_data(token)
    
    # Explosive move: sudden price spike
    price_spike = (prices[-1] - prices[-2]) / prices[-2] > price_spike_threshold
    
    # Volume confirmation
    is_volume_high = volumes[-1] > min_volume_threshold

    # 3 Green Candles Detection
    green_candles = prices[-1] > prices[-2] > prices[-3] > prices[-4]
    volume_bullish = volumes[-1] > volumes[-2] > volumes[-3] > volumes[-4]

    if price_spike and is_volume_high:
        return {
            "token": token,
            "type": "Explosive Spike",
            "price": prices[-1],
            "volume": volumes[-1],
            "stoploss": round(prices[-1] * 0.95, 2),
            "target": round(prices[-1] * 1.10, 2)
        }
    elif green_candles and volume_bullish:
        return {
            "token": token,
            "type": "3 Green Candles with Bullish Volume",
            "price": prices[-1],
            "volume": volumes[-1],
            "stoploss": round(prices[-2], 2),
            "target": round(prices[-1] * 1.08, 2)
        }
    return None

# Fetch all tokens
tokens = fetch_tokens()

# Analyze tokens
results = []
for token in tokens[:100]:  # Limit to first 100 tokens for speed
    result = detect_explosive_moves(token)
    if result:
        results.append(result)

# Display results
if results:
    st.success(f"{len(results)} potential explosive moves detected!")
    df = pd.DataFrame(results)
    st.dataframe(df)
else:
    st.warning("No explosive moves detected at this moment. Please wait or refresh.")
