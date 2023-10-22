import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import streamlit as st
import os
import pickle

# Convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def trading_advice(actual, prediction):
    if prediction > actual:
        return "Based on the predictions, you might consider buying."
    elif prediction < actual:
        return "Based on the predictions, you might consider selling."
    else:
        return "The price seems stable. You might consider holding."

# Streamlit app
def main():
    st.title("Cryptocurrency Price Prediction and Trading Bot")

    csv_path = 'https://github.com/Surya152002/Intel-Task-8/blob/main/crypto_dataset%20(9).csv'
    
    # Read and preprocess the input data
    df = pd.read_csv(csv_path, parse_dates=['Timestamp'], index_col='Timestamp')
    
    coins = ['BTC-USD Close', 'ETH-USD Close', 'LTC-USD Close']
    coin_choice = st.selectbox("Select a cryptocurrency", coins)

    if coin_choice == "BTC-USD Close":
        coin_model_path = 'btc-usd_close_model.pkl'
    elif coin_choice == "ETH-USD Close":
        coin_model_path = 'eth-usd_close_model.pkl'
    else:
        coin_model_path = 'ltc-usd_close_model.pkl'
    
    values = df[coin_choice].values.reshape(-1, 1)
    
    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # Frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    
    # Split into train and test sets
    values = reframed.values
    n_train = int(len(values) * 0.8)
    train, test = values[:n_train, :], values[n_train:, :]

    # Split into input and output
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # Reshape for LSTM [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # Check if model exists, if not, train it
    if not os.path.exists(coin_model_path):
        # Fit the model
        model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=0, shuffle=False)

        # Save model
        model.save(coin_model_path)
    else:
        # Load the model
        model = keras.models.load_model(coin_model_path)
    
    # Predict on the entire data
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # Invert scaling
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # Display predictions
    st.header("Price Predictions")
    st.write(inv_yhat)

    # TODO: Rest of the streamlit interface

if __name__ == "__main__":
    main()
