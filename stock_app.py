# Trigger rebuild for Streamlit

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import ta
from sklearn.preprocessing import MinMaxScaler

st.title("📈 Stock Price Prediction Dashboard")

# 1️⃣ Upload CSV
uploaded_file = st.file_uploader("Upload stock CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
    st.subheader("Raw Data")
    st.dataframe(df.tail(10))

    # 2️⃣ Compute EMA/MA and RSI
    close_prices = df['Close'].squeeze()
    df['EMA20'] = ta.trend.EMAIndicator(close_prices, window=20).ema_indicator()
    df['EMA50'] = ta.trend.EMAIndicator(close_prices, window=50).ema_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(close_prices, window=14).rsi()

    # Plot indicators
    st.subheader("Close Price with EMA20 & EMA50")
    plt.figure(figsize=(14,5))
    plt.plot(df.index, df['Close'], label='Close', color='blue')
    plt.plot(df.index, df['EMA20'], label='EMA20', color='green')
    plt.plot(df.index, df['EMA50'], label='EMA50', color='orange')
    plt.legend()
    st.pyplot(plt)

    st.subheader("RSI Indicator")
    plt.figure(figsize=(14,3))
    plt.plot(df.index, df['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--')   # Overbought
    plt.axhline(30, color='green', linestyle='--') # Oversold
    plt.legend()
    st.pyplot(plt)

    # 3️⃣ Load trained LSTM model
    model = load_model("lstm_stock_model.h5")
    
    # 4️⃣ Prepare data for prediction
    # Using only 'Close' column for simplicity; normalize
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df[['Close']].values)

    # Create sequences (like your notebook)
    def create_sequences(data, seq_length=50):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    seq_length = 50
    X, y_true = create_sequences(scaled_data, seq_length)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Predict
    y_pred_scaled = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_true.reshape(-1,1))

    # 5️⃣ Plot predicted vs actual prices
    st.subheader("Predicted vs Actual Close Price")
    plt.figure(figsize=(14,5))
    plt.plot(df.index[seq_length:], y_true, label='Actual', color='blue')
    plt.plot(df.index[seq_length:], y_pred, label='Predicted', color='red')
    plt.legend()
    st.pyplot(plt)

    # 6️⃣ Show metrics
    mse = np.mean((y_pred - y_true)**2)
    mae = np.mean(np.abs(y_pred - y_true))
    st.write(f"**Mean Squared Error (MSE):** {mse:.6f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.6f}")
