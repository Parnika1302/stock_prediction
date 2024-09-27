# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import datetime, timedelta

# Streamlit app
st.title("Stock Price Prediction App")

# User input for stock name
stock_name = st.text_input("Enter stock ticker symbol (e.g., AAPL for Apple):", "AAPL")

# Predict button
if st.button("Predict"):
    # 1. Load stock data
    end_date = datetime.now().strftime('%Y-%m-%d')
    stock_data = yf.download(stock_name, start='2010-01-01', end=end_date)

    if not stock_data.empty:
        st.write(f"Showing stock data for {stock_name}")
        st.dataframe(stock_data.tail())  # Show last 5 rows of stock data

        # 2. Preprocess the data
        close_prices = stock_data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # 3. Prepare training data
        train_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_len]
        x_train, y_train = [], []

        for i in range(60, len(train_data) - 7):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i+7, 0])

        x_train = np.array(x_train).reshape(-1, 60, 1)
        y_train = np.array(y_train)

        # 4. Build and train the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=64, epochs=10)

        # 5. Prepare test data and make predictions
        test_data = scaled_data[train_len - 60:]
        x_test = [test_data[i-60:i, 0] for i in range(60, len(test_data))]
        x_test = np.array(x_test).reshape(-1, 60, 1)

        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        st.write("Predicted stock prices:")
        st.line_chart(predicted_prices.flatten())  # Display predicted prices

        # 6. Future prediction (next 7 days)
        x_future = test_data[-60:].reshape(1, 60, 1)
        future_predictions = []
        for _ in range(7):
            pred = model.predict(x_future)
            future_predictions.append(pred[0, 0])
            x_future = np.append(x_future[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
        future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 8)]

        # Show future predictions
        st.write("Future predictions for the next 7 days:")
        future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions})
        st.write(future_df)

    else:
        st.error(f"Could not retrieve data for stock: {stock_name}")
