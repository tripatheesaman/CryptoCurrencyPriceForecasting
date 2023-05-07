import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout

from sklearn.preprocessing import MinMaxScaler

import math
from sklearn.metrics import mean_squared_error

#Read data
raw_data = pd.read_csv('BTC-USD.csv')

# Select only Closing Price for prediction
data = raw_data["Close"]

# Check for null values
raw_data['Close'].isnull().values.any()

# SCALING DATA
data = np.array(data)
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(data.reshape(-1,1))
scaled_data

# SPLIT TEST AND TRAIN DATA

trainsizesplit = 0.8

train_size = int(len(scaled_data)*trainsizesplit)
test_size = int(len(scaled_data) - train_size)

train_data = scaled_data[0:train_size]
test_data = scaled_data[train_size:]

train_data.shape
test_data.shape

train_data.shape

#FUNCTION TO CONVERT TO DATASETS

def dataset(data, steps):
  x, y = [], []
  for i in range(len(data) - steps - 1):
    x.append(data[i:(i+steps),0])
    y.append(data[i+steps,0])
  return np.array(x), np.array(y)

steps =  100

x_train, y_train = dataset(train_data, steps)
x_test, y_test = dataset(test_data, steps)

x_train =x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)

real_y_value = y_test

x_train.shape

y_train.shape

epoch_no = 100
