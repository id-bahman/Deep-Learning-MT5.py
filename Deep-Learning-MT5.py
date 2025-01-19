import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import time

# Connect to MetaTrader 5
if not mt5.initialize():
    raise RuntimeError("MetaTrader 5 could not be initialized")

# Define the symbol and timeframe
symbol = 'EURUSD'
timeframe = mt5.TIMEFRAME_M15  # 15-minute timeframe
from_date = datetime(2024, 8, 1)  # Start date
to_date = datetime(2024, 12, 1)   # End date

# Retrieve the price data for the specified date range
rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)

# Check if data is retrieved
if rates is None or len(rates) == 0:
    raise ValueError(f"No data available for {symbol} from {from_date} to {to_date}.")

# Convert the data to a DataFrame for analysis
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')  # Convert Unix timestamp to date and time

# Feature engineering: Adding features like moving averages
df['MA_10'] = df['close'].rolling(window=10).mean()
df['MA_20'] = df['close'].rolling(window=20).mean()
df.dropna(inplace=True)

# Prepare data for training and testing
features = ['close', 'MA_10', 'MA_20', 'high', 'low', 'open', 'tick_volume']
data = df[features].values

# Scale the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Scale the target variable (y) as well
y_scaled = scaler.fit_transform(df[['close']])

# Define the time step for LSTM
X, y = [], []
time_step = 60  # Use the previous 60 periods to predict the next one
for i in range(len(scaled_data) - time_step - 1):
    X.append(scaled_data[i:i + time_step, :])
    y.append(y_scaled[i + time_step, 0])  # Predicting the 'close' price

X, y = np.array(X), np.array(y)

# Split the data into training, validation, and testing sets
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.LSTM(100, return_sequences=False),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64)

# Evaluate the model
model.evaluate(X_test, y_test)

# Predict on the test data
predicted_prices = model.predict(X_test)

# Inverse transform to get original prices
predicted_prices_original = scaler.inverse_transform(
    np.hstack([predicted_prices, np.zeros((predicted_prices.shape[0], scaled_data.shape[1] - 1))])
)[:, 0]

# Print actual vs predicted prices
actual_prices = scaler.inverse_transform(
    np.hstack([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1))])
)[:, 0]

print("Actual Prices  Predicted Prices")
for actual, predicted in zip(actual_prices[:10], predicted_prices_original[:10]):
    print(f"{actual:.5f}          {predicted:.5f}")

# Trading logic
# Ensure symbol exists and is ready for trading
symbol_info = mt5.symbol_info(symbol)
if symbol_info is None or not symbol_info.visible:
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Symbol {symbol} is not available or cannot be selected.")

# Function to get current price
def get_current_price(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Could not get tick data for {symbol}.")
    return tick.ask

# Function to place an order
def place_order(order_type, price, stop_loss, take_profit, lot=0.1):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": 10,
        "magic": 123456,
        "comment": "LSTM trade",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.retcode}")
    else:
        print(f"Order placed successfully: {result}")

# Example of using predictions for trading decisions
for i in range(len(predicted_prices_original) - 1):
    current_price = get_current_price(symbol)
    prediction = predicted_prices_original[i]

    # Example trading logic: Buy if prediction is 1% above the current price
    if prediction > current_price * 1.01:
        stop_loss = current_price * 0.99
        take_profit = prediction * 1.01
        place_order(mt5.ORDER_TYPE_BUY, current_price, stop_loss, take_profit)
    elif prediction < current_price * 0.99:
        stop_loss = current_price * 1.01
        take_profit = prediction * 0.99
        place_order(mt5.ORDER_TYPE_SELL, current_price, stop_loss, take_profit)

# Disconnect from MetaTrader 5
mt5.shutdown()
#####################
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# پیش‌بینی‌ها و قیمت‌های واقعی
actual_prices = y_test
predicted_prices = model.predict(X_test)

# محاسبه MSE (Mean Squared Error)
mse = mean_squared_error(actual_prices, predicted_prices)
print(f'Mean Squared Error (MSE): {mse}')

# محاسبه MAE (Mean Absolute Error)
mae = mean_absolute_error(actual_prices, predicted_prices)
print(f'Mean Absolute Error (MAE): {mae}')

# محاسبه R² (Coefficient of Determination)
r2 = r2_score(actual_prices, predicted_prices)
print(f'R² Score: {r2}')