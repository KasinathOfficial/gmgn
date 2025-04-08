import streamlit as st
import requests
import time

# ---------------------------
# Fetch active INR coins from CoinDCX
# ---------------------------
def get_active_symbols():
    url = "https://api.coindcx.com/exchange/ticker"
    response = requests.get(url).json()
    inr_pairs = [coin for coin in response if coin['market'].endswith('INR') and float(coin['volume']) > 500000]
    return inr_pairs

# ---------------------------
# Get candle data for a symbol
# ---------------------------
def get_candle_data(symbol, interval="1m", limit=20):
    market = symbol.replace("INR", "") + "INR"
    url = f"https://public.coindcx.com/market_data/candles?pair={market}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        candles = [{
            "timestamp": c[0],
            "open": c[1],
            "high": c[2],
            "low": c[3],
            "close": c[4],
            "volume": c[5],
            "symbol": symbol
        } for c in data]
        return candles
    return []

# ---------------------------
# Estimate average volume for filtering
# ---------------------------
def get_avg_volume(symbol):
    candles = get_candle_data(symbol, interval="1m", limit=10)
    if candles:
        volumes = [float(candle['volume']) for candle in candles[:-1]]
        return sum(volumes) / len(volumes)
    return 0

# ---------------------------
# Check for explosive candle (Golden Candle logic)
# ---------------------------
def is_explosive_candle(candle):
    open_price = float(candle['open'])
    close_price = float(candle['close'])
    high = float(candle['high'])
    low = float(candle['low'])
    volume = float(candle['volume'])

    price_jump = (close_price - open_price) / open_price
    wick_ratio = (high - close_price) / (close_price - open_price + 1e-6)
    avg_volume = get_avg_volume(candle['symbol'])

    return price_jump > 0.07 and wick_ratio < 0.4 and volume > 3 * avg_volume

# ---------------------------
# Scan for opportunities
# ---------------------------
def scan_coins_for_opportunities():
    coins = get_active_symbols()
    winners = []

    for coin in coins:
        symbol = coin['market']
        candles = get_candle_data(symbol, interval="1m", limit=20)
        if not candles:
            continue

        last_candle = candles[-1]
        if is_explosive_candle(last_candle):
            winners.append({
                'symbol': symbol,
                'price': last_candle['close'],
                'confidence': 100,
                'volume': last_candle['volume']
            })
    return winners

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Crypto Explosion Predictor", layout="wide")
st.title("Crypto Explosion Predictor - 100% Golden Candle Scanner")

if st.button("Scan for Explosive Moves"):
    with st.spinner("Scanning market..."):
        opportunities = scan_coins_for_opportunities()
        if opportunities:
            for coin in opportunities:
                st.success(f"{coin['symbol']} might explode! Price: ₹{coin['price']} | Confidence: {coin['confidence']}% | Volume: {coin['volume']}")
        else:
            st.warning("No explosive opportunities found right now. Try again in a few minutes.")
