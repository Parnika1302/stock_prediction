# Install necessary libraries (uncomment if you haven't installed them yet)
# !pip install yfinance tensorflow scikit-learn numpy pandas matplotlib

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# 1. Get user input for stock ticker
stock_name = input("Enter the stock ticker symbol (e.g., AAPL for Apple): ")

# 2. Load stock data until the current date
end_date = datetime.now().strftime('%Y-%m-%d')
stock_data = yf.download(stock_name, start='2010-01-01', end=end_date)

# 3. Preprocess the data: Use only the 'Close' prices for prediction
close_prices = stock_data['Close'].values
close_prices = close_prices.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Create training data: 60 days of data for prediction of the price 7 days ahead
train_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_len]

# Create training sequences and labels (predicting 7 days ahead)
x_train, y_train = [], []
for i in range(60, len(train_data) - 7):
    x_train.append(train_data[i-60:i, 0])  # Use the past 60 days' data
    y_train.append(train_data[i+7, 0])     # Predict the price 7 days ahead

# Convert to NumPy arrays and reshape
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

from tensorflow.keras import Input

# 4. Build and train the LSTM model
model = Sequential()

# Use the Input() layer to specify the shape of the input data
model.add(Input(shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10)

# 5. Prepare test data for the last 60 days
test_data = scaled_data[train_len - 60:]  # Get the last 60 days for prediction
x_test = []
actual_prices = close_prices[train_len:]  # Actual prices for the last portion of the data

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 6. Make predictions for the test data
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)  # Inverse scaling for predictions

# 7. Calculate RMSE for the test data
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
print(f"Root Mean Squared Error: {rmse:.2f}")

# 8. Plot the actual vs predicted prices for the test data (last part)
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index[train_len:], actual_prices, color='blue', label='Actual Stock Price')
plt.plot(stock_data.index[train_len:], predicted_prices, color='red', label='Predicted Stock Price')
plt.title(f'Actual vs Predicted Stock Prices for {stock_name}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()

# 9. Prepare test data for predicting the next week
x_test = test_data[-60:].reshape(1, 60, 1)  # Reshape for LSTM input

# 10. Make predictions for the next week
predictions = []
for _ in range(7):  # Predict for the next 7 days
    pred = model.predict(x_test)
    predictions.append(pred[0, 0])  # Store the prediction
    
    # Update the x_test for the next prediction
    pred_reshaped = pred.reshape(1, 1, 1)  # Reshape predicted value
    x_test = np.append(x_test[:, 1:, :], pred_reshaped, axis=1)  # Shift the input data and append new prediction

# Inverse scale the predictions
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# 11. Prepare dates for the next 7 days starting from tomorrow
future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 8)]

# 12. Print the predicted prices
for i in range(7):
    print(f"Predicted price for {future_dates[i].date()}: ${predictions[i]:.2f}")

# 13. Find the lowest and highest predicted prices for the next week
min_price = predictions.min()
max_price = predictions.max()
min_price_date = future_dates[np.argmin(predictions)]
max_price_date = future_dates[np.argmax(predictions)]

# 14. Print the results
print(f"\nLowest predicted price for {stock_name}: ${min_price:.2f} on {min_price_date.date()}")
print(f"Highest predicted price for {stock_name}: ${max_price:.2f} on {max_price_date.date()}")

# 15. Plot the predicted prices for the next week
plt.figure(figsize=(12, 6))
plt.plot(future_dates, predictions, color='red', marker='o', label='Predicted Stock Price for Next Week')
plt.title(f'Predicted Stock Prices for Next Week of {stock_name}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)  # Rotate dates for better readability
plt.legend()
plt.grid()
plt.show()