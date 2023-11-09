import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
import numpy as np
import base64

# Function to make predictions using the LSTM model
def make_predictions(model, data, scaler, look_back, prediction_days):
    predictions = []
    scaled_data = scaler.transform(data[-look_back:].reshape(-1, 1))
    scaled_data = scaled_data.reshape((1, look_back, 1))
    
    for _ in range(prediction_days):
        # Make the prediction
        predicted_price = model.predict(scaled_data)
        # Inverse transform to get the actual price
        predicted_price = scaler.inverse_transform(predicted_price)
        predictions.append(predicted_price.ravel()[0])
        
        # Append prediction to scaled_data for making subsequent predictions
        next_input = scaler.transform(predicted_price.reshape(-1, 1))
        scaled_data = np.append(scaled_data[:, 1:, :], [next_input], axis=1)
        
    return predictions


# Function to download prediction data as a CSV file
def get_table_download_link(df):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download CSV file</a>'
    return href


# Streamlit app layout
st.title('Real-time Crypto Prediction with LSTM')

# User input parameters (no longer in the sidebar)
st.header('User Input Parameters')
ticker = st.selectbox('Select Ticker', ('BTC-USD', 'ETH-USD', 'LTC-USD'))
start_date = st.date_input('Start date', pd.to_datetime('2021-01-01'))
end_date = st.date_input('End date', pd.to_datetime('today'))
look_back = st.number_input('Look Back Period',value=60)
prediction_days = st.slider('Days to Predict', 1, 30, 5)

# Load pre-trained LSTM model and scaler
with st.spinner('Loading model and scaler...'):
    model = load_model('./best_model.h5')  # make sure to use the correct path to your model
    scaler = MinMaxScaler(feature_range=(0, 1))  # Assuming the scaler was fitted to the training data

# Fetch historical data from yfinance
with st.spinner(f'Downloading {ticker} data...'):
    data = yf.download(ticker, start=start_date, end=end_date)
st.success('Downloaded data successfully!')

# Prepare data for prediction
data_close = data['Close'].values.reshape(-1, 1)
scaler.fit(data_close)

# Make real-time prediction
if len(data_close) >= look_back:
    with st.spinner('Making predictions...'):
        predictions = make_predictions(model, data_close, scaler, look_back, prediction_days)
        # Convert predictions to a list of Python floats
        predictions_list = [float(pred) for pred in predictions]
    st.success('Predictions made successfully!')

    # Plot the results using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines+markers', name='Close Price'))
    # Add predicted prices to the plot
    prediction_dates = pd.date_range(start=data.index[-1], periods=prediction_days+1, closed='right')
    fig.add_trace(go.Scatter(x=prediction_dates, y=predictions_list, mode='lines+markers', name='Predicted Price', marker=dict(color='red')))
    fig.update_layout(title='Cryptocurrency Price Prediction', xaxis_title='Date', yaxis_title='Price', legend_title='Legend')
    st.plotly_chart(fig, use_container_width=True)

    


    # Display download link for predictions with actual prices included
    prediction_dates = pd.date_range(start=data.index[-1], periods=prediction_days+1, closed='right')
    prediction_df = pd.DataFrame(index=prediction_dates)
    prediction_df['Predicted Price'] = predictions_list
    prediction_df['Actual Price'] = data['Close'][-prediction_days:].values
    st.markdown(get_table_download_link(prediction_df), unsafe_allow_html=True)

    # Display the last prediction
    st.write(f'Last Predicted Closing Price for {ticker}: ${predictions_list[-1]:.2f}')
    
    # Simulate investment results based on predictions
    # Simulate investment results based on the last prediction
    initial_account_balance = 1000000  
    account_balance_predicted = initial_account_balance
    account_balance_actual = initial_account_balance

    # Get the last actual closing price and the last predicted price
    last_invest_price = data['Close'].iloc[-prediction_days-1]  # Price at which we "buy"
    last_sell_price_predicted = predictions_list[-1]  # Predicted price at which we "sell"
    last_sell_price_actual = data['Close'].iloc[-prediction_days] if prediction_days <= len(data['Close']) else last_sell_price_predicted

    # Update account balances based on the last day's prediction and actual price
    account_balance_predicted += (last_sell_price_predicted - last_invest_price) * (account_balance_predicted / last_invest_price)
    account_balance_actual += (last_sell_price_actual - last_invest_price) * (account_balance_actual / last_invest_price)


    # Assuming investing at the closing price and selling at the next day's predicted/actual price
    for i in range(1, prediction_days + 1):
        invest_price = data['Close'].iloc[-i-1]  # Price at which we "buy"
        sell_price_predicted = predictions_list[-i]  # Predicted price at which we "sell"
        sell_price_actual = data['Close'].iloc[-i] if i <= len(data['Close']) else sell_price_predicted

        # Update account balances
        account_balance_predicted += (sell_price_predicted - invest_price) * (account_balance_predicted / invest_price)
        account_balance_actual += (sell_price_actual - invest_price) * (account_balance_actual / invest_price)

    # Display simulated investment results
    st.write(f"Initial Account Balance: ${initial_account_balance:,.2f}")
    st.write(f"Account Balance based on Predictions: ${account_balance_predicted:,.2f}")
    st.write(f"Account Balance based on Actual Prices: ${account_balance_actual:,.2f}")

else:
    st.error('Not enough data to make a prediction.')

# Refresh the page to update the predictions
if st.button("Refresh Predictions"):
    st.experimental_rerun()
