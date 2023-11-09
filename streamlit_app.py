import streamlit as st
import yfinance as yf
import backtrader as bt
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load your trained LSTM model
model = load_model('./best_model.h5')  # Uncomment this line to load your trained model

# Define the LSTM prediction function
def lstm_predict(model, data, scaler, look_back=60):
    # Preprocess data
    last_values = scaler.transform(data[-look_back:].reshape(-1, 1))
    X_test = np.reshape(last_values, (1, look_back, 1))
    
    # Make prediction
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0, 0]

# Define the trading strategy using LSTM predictions
class LSTMStrategy(bt.Strategy):
    params = dict(
        model=None,  # LSTM model
        scaler=None,  # Scaler object for data normalization
        look_back=60  # Look back period for LSTM input
    )

    def __init__(self):
        self.order = None
        self.price = self.datas[0].close
        self.predicted_price = None

    def next(self):
        if self.order:
            return  # pending order execution
        
        if not self.position:  # if not in the market
            # Predict the next closing price
            self.predicted_price = lstm_predict(
                self.params.model,
                np.array(self.price.get(size=self.params.look_back)),
                self.params.scaler
            )
            if self.predicted_price > self.price[0]:  # if the price is predicted to rise
                self.order = self.buy()
        else:
            if self.predicted_price < self.price[0]:  # if the price is predicted to fall
                self.order = self.sell()

# Function to fetch data
def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to run backtest
def run_backtest(data, strategy, model, scaler):
    cerebro = bt.Cerebro()
    feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(feed)
    cerebro.addstrategy(strategy, model=model, scaler=scaler)
    cerebro.broker.set_cash(10000)
    cerebro.run()
    return cerebro

# Streamlit interface
st.title('Cryptocurrency Trading Bot with LSTM')

# User inputs for the prediction and backtesting
ticker = st.selectbox('Select Ticker', ('BTC-USD', 'ETH-USD', 'LTC-USD'))
start_date = st.date_input('Start Date', value=pd.to_datetime('2021-01-01'))
end_date = st.date_input('End Date', value=pd.to_datetime('2021-12-31'))
look_back = st.slider('Look Back Period', min_value=10, max_value=100, value=60)

# Button to make prediction and run the strategy
if st.button('Run Trading Strategy'):
    data = get_data(ticker, start_date, end_date)
    scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize the MinMaxScaler
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Mock model prediction (replace this with actual model.predict call)
    # For demonstration, we assume next price is same as last closing price
    next_price = data['Close'].iloc[-1]
    
    # Display the prediction
    st.write(f'The predicted next closing price is ${next_price}')

    # Run backtest with LSTM Strategy
    cerebro = run_backtest(data, LSTMStrategy, model=None, scaler=scaler)  # Pass the actual model instead of None

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(data['Close'], label='Close Price')
    st.pyplot(fig)
    
    # Show the performance
    st.write(f'Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}')
