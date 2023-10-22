import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os  # Added for file path handling
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]


# Download the necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess data
def preprocess_data(df):
    # Handle missing values by forward filling
    df = df.fillna(method='ffill')

    # Optionally, handle any remaining missing values by backward filling
    df = df.fillna(method='bfill')

    # Optionally, standardize (scale) the data to have zero mean and unit variance
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df

def trading_advice(actual, prediction):
    """
    Provide trading advice based on prediction and actual prices.
    """
    if prediction > actual:
        return "Based on the predictions, you might consider buying."
    elif prediction < actual:
        return "Based on the predictions, you might consider selling."
    else:
        return "The price seems stable. You might consider holding."

def chatbot_response(user_input, predefined_responses):
    # Tokenize and vectorize user input and predefined responses
    vectorizer = TfidfVectorizer(tokenizer=lambda text: nltk.word_tokenize(text, language='english'),
                                 stop_words=stopwords.words('english'))
    vectors = vectorizer.fit_transform([user_input] + predefined_responses)

    # Calculate cosine similarities
    cosine_matrix = cosine_similarity(vectors)

    # Find the most similar predefined response to the user's input
    response_idx = np.argmax(cosine_matrix[0][1:])

    return predefined_responses[response_idx]

def get_gpt3_response(user_input):
    prompt = f"Answer questions related to cryptocurrency.\n\nUser: {user_input}\nBot:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50  # You can adjust this based on response length
    )
    return response.choices[0].text


# Streamlit app
def main():
    st.title("Cryptocurrency Price Prediction and Trading Bot")

    # Specify the path to the CSV file
    csv_path = 'https://github.com/Surya152002/Intel-Task-8/blob/main/crypto_dataset%20(9).csv'  # Update with your specific file path

    # Check if the CSV file exists at the specified path
    if not os.path.exists(csv_path):
        st.error("The specified CSV file does not exist. Please provide a valid file path.")
        return

    # Read and preprocess the input data
    input_data = pd.read_csv(csv_path, parse_dates=['Timestamp'], index_col='Timestamp')
    input_data = preprocess_data(input_data)

    # List of coins
    coins = ['BTC-USD Close', 'ETH-USD Close', 'LTC-USD Close']

    # Get user's choice of cryptocurrency
    coin_choice = st.selectbox("Select a cryptocurrency", coins)

    # Train ARIMA model if needed
    if coin_choice == "BTC-USD Close":
        coin_model_path = 'btc-usd_close_model.pkl'  # Update with your desired model path
        coin_column = 'BTC-USD Close'
    elif coin_choice == "ETH-USD Close":
        coin_model_path = 'eth-usd_close_model.pkl'  # Update with your desired model path
        coin_column = 'ETH-USD Close'
    else:
        coin_model_path = 'ltc-usd_close_model.pkl'  # Update with your desired model path
        coin_column = 'LTC-USD Close'

    # Check if the model exists, if not, train it
    if not os.path.exists(coin_model_path):
        # Split the data into training and testing sets
        train_size = int(len(input_data) * 0.8)
        train, test = input_data[:train_size], input_data[train_size:]

        # Build and train the ARIMA model
        model = ARIMA(train[coin_column], order=(5, 1, 0))  # Example order, you can tune this
        model_fit = model.fit()

        # Save the trained model using pickle
        with open(coin_model_path, 'wb') as model_file:
            pickle.dump(model_fit, model_file)

    # Load the model using pickle
    with open(coin_model_path, 'rb') as model_file:
        model_fit = pickle.load(model_file)

    # Make predictions
    predictions = model_fit.forecast(steps=len(input_data))

    # Display predictions
    st.header("Price Predictions")
    st.write(predictions)

    # Visualize results
    st.header("Price Prediction Chart")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(input_data.index, input_data[coin_column], label='Actual')
    ax.plot(input_data.index, predictions, label='Predicted', linestyle='--')
    ax.set_title(f'{coin_choice} Price Prediction')
    ax.legend()
    st.pyplot(fig)

    # Chatbot live interaction

    st.header("Chat with Trading Bot")

    user_message = st.text_input("You: ")
    predefined_responses = [
        "The predicted price for the next period is {}.".format(predictions.iloc[-1]),
        trading_advice(input_data[coin_column].iloc[-1], predictions.iloc[-1]),
        "I am here to help with your cryptocurrency trading decisions.",
        "Can you specify your query?"
    ]

    if user_message:
        if "predict" in user_message.lower() or "forecast" in user_message.lower():
            bot_reply = trading_advice(input_data[coin_column].iloc[-1], predictions.iloc[-1])
        else:
            bot_reply = get_gpt3_response(user_message)
        st.write(f"Bot: {bot_reply}")

if __name__ == "__main__":
    main()

