#!/usr/bin/env python
# coding: utf-8

# In[7]:


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.dates as mdates
from yahooquery import Ticker
import math
import keras


# In[8]:


ASSET = 'AAPL'
stck_data = Ticker(ASSET).history(interval='1d', period='max')
stck_data


# In[9]:


train_portion = math.floor(len(stck_data['close']) * 0.8)
test_portion = math.floor(len(stck_data['close']) * 0.2)


# In[10]:


training_data = stck_data['close'][:train_portion]
testing_data = stck_data['close'][test_portion:]


# In[13]:


training_set = training_data.values.reshape(-1, 1)
testing_set = testing_data.values.reshape(-1, 1)


# In[14]:


scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(training_set)
testing_set_scaled = scaler.fit_transform(testing_set)


# In[15]:


def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
X_train, y_train = create_sequences(training_set_scaled)
X_validation, y_validation = create_sequences(testing_set_scaled)


# In[16]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))


# In[17]:


model = keras.Sequential()


# In[18]:


model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=50, return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=50, return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(units=50))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(units=1))


# In[19]:


model.compile(optimizer=keras.optimizers.Adam(), loss="mean_squared_error")


# In[20]:


history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_validation, y_validation))


# In[21]:


plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[37]:


test_df = Ticker(ASSET).history(interval='1d', period='max')
test_df


# In[38]:


test_df = test_df.reset_index(level='symbol')
test_df


# In[39]:


test_df.index


# In[23]:


real_price = test_df['close'].values.reshape(-1, 1)


# In[24]:


data_total = pd.concat((stck_data['close'], test_df['close']), axis=0)


# In[25]:


inpts = data_total[len(data_total) - len(test_df) - 60:].values
inpts = inpts.reshape(-1, 1)
inpts = scaler.transform(inpts)
X_test = []
for i in range(60, len(inpts)):
    X_test.append(inpts[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[26]:


predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


# In[51]:


# Prepare dates for the predicted stock prices
date_range = pd.date_range(start='1980-12-12', periods=len(predicted_stock_price), freq='B')  # 'B' for business day frequency

# Visualizing Results with Month and Year on X-axis
plt.figure(figsize=(10, 6))
plt.plot(test_df.index, real_price, color='blue', label='Stock Price')
plt.plot(date_range, predicted_stock_price, color='green', label='Predicted Stock Price')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=36))  # Show tick marks for every 3 months
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format x-axis labels as 'Jan 2023', 'Feb 2023', etc.
plt.title('SPY Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('SPY Stock Price')
plt.legend()
plt.show()


# In[ ]:




