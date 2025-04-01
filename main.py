import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up Streamlit
st.set_page_config(page_title="100x Gem Finder", layout="wide")
st.title("🚀 AI-Powered 100x Gem Crypto Finder")

# Step 1: Fetch data from GMGN.ai API
GMGN_API_TRENDING = "https://gmgn.ai/sol/token"

def fetch_data():
    response = requests.get(GMGN_API_TRENDING)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch data. Status Code: {response.status_code}")
        st.write(response.text)  # This will show the error message returned by the API
        return None

data = fetch_data()
if data is None:
    st.error("Error fetching data from GMGN.ai")
    st.stop()

df = pd.DataFrame(data)

# Step 2: Feature Engineering
df["Volatility %"] = (df["high"] - df["low"]) / df["low"] * 100
df["Liquidity Ratio"] = df["volume"] / (df["market_cap"] + 1)  # Normalize liquidity
df["Whale Activity"] = df["whale_transactions"] / (df["transactions"] + 1)
df["Explosion Label"] = (df["price_change_24h"] > 20).astype(int)  # Label for training

# Select relevant features for the model
features = ["Volatility %", "Liquidity Ratio", "Whale Activity"]
X = df[features]
y = df["Explosion Label"]

# Step 3: Train the AI Model (RandomForestClassifier)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Make Predictions with the Model
df["AI Explosive Potential (%)"] = model.predict_proba(df[features])[:, 1] * 100
df["Smart Money Score"] = df["whale_transactions"] / (df["transactions"] + 1)
df["Target Price"] = df["price"] * 1.5  # Example 50% target price
df["Stop Loss"] = df["price"] * 0.9  # Example 10% stop loss

# Step 5: Display the DataFrame in Streamlit
st.subheader("📊 AI-Identified 100x Gem Cryptos")
st.dataframe(df[["name", "Volatility %", "AI Explosive Potential (%)", "Smart Money Score", "Target Price", "Stop Loss"]])

# Step 6: Visualize AI Predictions (Optional)
st.subheader("📈 Volatility vs. AI Prediction")
st.scatter_chart(df[["Volatility %", "AI Explosive Potential (%)"]])

# Step 7: Add Real-Time Alerts (Optional)
if st.button("Get Alerts for Explosive Coins"):
    high_potential_gems = df[df["AI Explosive Potential (%)"] > 70]  # Show only high-potential coins
    st.write("🚀 High Potential 100x Gems:")
    st.dataframe(high_potential_gems[["name", "AI Explosive Potential (%)", "Target Price", "Stop Loss"]])

