import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import sequential
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers import Dense


# Load stock price data
data = pd.read_csv('stock_data.csv')
prices = data['Close'].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

look_back = 10
X, y = [], []
for i in range(len(prices_scaled) - look_back):
    X.append(prices_scaled[i:i + look_back, 0])
    y.append(prices_scaled[i + look_back, 0])
X, y = np.array(X), np.array(y)

# Reshape input data to be 3D (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = sequential.Sequential()  
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

