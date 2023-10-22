import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import streamlit as st
import os
import openai
import nltk
from nltk.corpus import stopwords

# Global setup for OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

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

def chatbot_response(user_input):
    predefined_responses = [
        f"The predicted price for the next period is {predictions[-1]:.2f}.",
        trading_advice(actual_price[-1], predictions[-1]),
        "I am here to help with your cryptocurrency trading decisions.",
        "Can you specify your query?"
    ]
    return predefined_responses[np.random.randint(0, len(predefined_responses))]

# Streamlit app
def main():
    st.title("Cryptocurrency Price Prediction and Trading Bot")

    csv_path = 'https://github.com/Surya152002/Intel-Task-8/blob/main/crypto_dataset%20(9).csv'

    # Read and preprocess the input data
    df = pd.read_csv(csv_path, parse_dates=['Timestamp'], index_col='Timestamp')
    coins = ['BTC-USD Close', 'ETH-USD Close', 'LTC-USD Close']
    coin_choice = st.selectbox("Select a cryptocurrency", coins)

    # LSTM model preparation and prediction
    if coin_choice == "BTC-USD Close":
        coin_model_path = 'btc-usd_close_model.h5'
    elif coin_choice == "ETH-USD Close":
        coin_model_path = 'eth-usd_close_model.h5'
    else:
        coin_model_path = 'ltc-usd_close_model.h5'

    values = df[coin_choice].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, 1, 1)

    values = reframed.values
    n_train = int(len(values) * 0.8)
    train, test = values[:n_train, :], values[n_train:, :]

    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    if not os.path.exists(coin_model_path):
        model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=0, shuffle=False)
        model.save(coin_model_path)
    else:
        model.load_weights(coin_model_path)

    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    predictions = inv_yhat[:, 0]
    actual_price = scaler.inverse_transform(test)

    st.header("Price Predictions")
    st.write(predictions)

    # Display predictions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index[-len(predictions):], actual_price[:, 0], label='Actual')
    ax.plot(df.index[-len(predictions):], predictions, label='Predicted', linestyle='--')
    ax.set_title(f'{coin_choice} Price Prediction with LSTM')
    ax.legend()
    st.pyplot(fig)

    # Chatbot live interaction
    st.header("Chat with Trading Bot")
    user_message = st.text_input("You: ")
    if user_message:
        bot_reply = chatbot_response(user_message)
        st.write(f"Bot: {bot_reply}")

if __name__ == "__main__":
    main()
