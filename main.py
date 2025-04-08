import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Crypto Explosive Detector", layout="wide")
st.title("1000% Win Rate Crypto Explosion Detector")

# User-configurable settings
min_volume_threshold = 5000000  # Minimum volume
price_spike_threshold = 0.10    # 10% spike in last candle

# 🔁 User-defined number of green candles
green_candle_count = st.sidebar.slider("Number of Consecutive Green Candles", min_value=2, max_value=6, value=3)

@st.cache_data(ttl=300)
def fetch_tokens():
    url = "https://api.coindcx.com/exchange/ticker"
    response = requests.get(url)
    data = response.json()
    tokens = [item['market'] for item in data if item['market'].endswith("INR")]
    return tokens

# Simulated data — replace with real API for production
def simulate_data(token):
    prices = [100 + i * 5 for i in range(green_candle_count + 1)]  # Prices increasing
    volumes = [4_000_000 + i * 1_000_000 for i in range(green_candle_count + 1)]  # Volumes increasing
    return prices, volumes

# Detect explosive or bullish patterns
def detect_explosive_moves(token):
    prices, volumes = simulate_data(token)

    # Check recent price spike
    price_spike = (prices[-1] - prices[-2]) / prices[-2] > price_spike_threshold
    is_volume_high = volumes[-1] > min_volume_threshold

    # Dynamic green candles check
    green_candles = all(prices[i] > prices[i - 1] for i in range(-1, -green_candle_count - 1, -1))
    bullish_volumes = all(volumes[i] > volumes[i - 1] for i in range(-1, -green_candle_count - 1, -1))

    if price_spike and is_volume_high:
        return {
            "token": token,
            "type": "Explosive Spike",
            "price": prices[-1],
            "volume": volumes[-1],
            "stoploss": round(prices[-1] * 0.95, 2),
            "target": round(prices[-1] * 1.10, 2)
        }
    elif green_candles and bullish_volumes:
        return {
            "token": token,
            "type": f"{green_candle_count} Green Candles with Bullish Volume",
            "price": prices[-1],
            "volume": volumes[-1],
            "stoploss": round(prices[-green_candle_count], 2),
            "target": round(prices[-1] * 1.08, 2)
        }
    return None

# Fetch tokens
tokens = fetch_tokens()

# Analyze
results = []
for token in tokens[:100]:
    result = detect_explosive_moves(token)
    if result:
        results.append(result)

# Show output
if results:
    st.success(f"{len(results)} potential explosive moves detected!")
    df = pd.DataFrame(results)
    st.dataframe(df)
else:
    st.warning("No explosive moves detected at this moment. Please wait or refresh.")
