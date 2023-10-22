# pip install pandas numpy matplotlib streamlit pystan fbprophet cryptocmd plotly
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
import pandas_datareader as data
import fbprophet
import requests
from fbprophet import Prophet
from datetime import date
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from streamlit_option_menu import option_menu
#from keras.models import load_model
from PIL import Image


img=Image.open('logo.jpeg')
st.set_page_config(page_title='predict crypto price',page_icon=img)
hide_menu_style="""
 <style>
 #MainMenu {visibility:hidden;}
 footer{visibility:hidden;}
 </style>
 """
st.markdown(hide_menu_style,unsafe_allow_html=True)


selected=option_menu(
   menu_title=None,
   options=["Home","Time Series Analysis","Indicator","Plan my purchase"],
   icons=["house","clock-history","bar-chart-fill","currency-bitcoin"],
   menu_icon="cast",
   default_index=0,
   orientation="horizontal",
)

start='2011-01-01'
end='2022-01-01'
today=date.today().strftime("%Y-%m-%d")

if selected=="Home":
 st.title("TILL DATE DATA")
 cryptos=("BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "USDC-USD", "SOL-USD", "ADA-USD", "XRP-USD", "LUNA1-USD", "HEX-USD", "AVAX-USD", "DOT-USD", "DOGE-USD", "SHIB-USD", "MATIC-USD", "ATOM-USD", "LTC-USD")
 targetcrypto=st.selectbox("ENTER THE CRYPTO TO BE PREDICTED",cryptos)
 def load_data(ticker):
    data=yf.download(ticker,start,today)
    data.reset_index(inplace=True)
    return data
 data_load_state=st.text("loading data...")
 data=load_data(targetcrypto)
 data_load_state.text("data loaded sucessfully....")
 st.subheader("DATA FROM 2011 TO CURRENT DAY")
 st.write(data.tail())
 def org_graph():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    fig.layout.update(title_text="TILL DATE DATA",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
 org_graph()
 st.header("Lets understand the data")
 st.subheader("Open")
 st.write("Open indicates the price of crypto currency at the begining of the day")
 st.subheader("Close")
 st.write("Close indicates the price of crypto currency at the end of the day")
 st.subheader("Low")
 st.write("Low indicates the least price of the crypto currency recorded in the whole day")
 st.subheader("High")
 st.write("High indicates the highest price of the crypto currency recorded in the whole day")
 st.subheader("Volume")
 st.write("Volume indicates the amount of crypto recorded at the end of the day")


if selected=="Time Series Analysis":
 cryptos=("BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "USDC-USD", "SOL-USD", "ADA-USD", "XRP-USD", "LUNA1-USD", "HEX-USD", "AVAX-USD", "DOT-USD", "DOGE-USD", "SHIB-USD", "MATIC-USD", "ATOM-USD", "LTC-USD")
 targetcrypto=st.selectbox("ENTER THE CRYPTO TO BE PREDICTED",cryptos)
 def load_data(ticker):
    data=yf.download(ticker,start,today)
    data.reset_index(inplace=True)
    return data
 data_load_state=st.text("loading data...")
 data=load_data(targetcrypto)
 data_load_state.text("data loaded sucessfully....")
 timeline=("WEEKS","MONTHS","YEARS")
 duration=st.selectbox("ENTER THE DURATION TO PREDICT",timeline)
 N=st.slider("ENTER NUMBER",0,10)
 if(duration=="WEEKS"):
    no_of_days=N*7
 if(duration=="MONTHS"):
    no_of_days=N*30
 if(duration=="YEARS"):
    no_of_days=N*365
 df_train=data[['Date', 'Close']]
 df_train=df_train.rename(columns={"Date":"ds","Close":"y"})
 m= Prophet()
 m.fit(df_train)
 future=m.make_future_dataframe(periods=no_of_days)
 forecast=m.predict(future)
 st.subheader('PREDICTED DATA')
 st.write(forecast.tail())
 st.subheader('FORECAST DATA')
 fig1=plot_plotly(m,forecast)
 st.plotly_chart(fig1)
 st.write("FORECAST COMPONENTS")
 fig2=m.plot_components(forecast)
 st.write(fig2)
 st.subheader("We can observe Wednesday has been recorded as the highest price day of the week in crypto market")
 st.subheader("Whereas Thursday has been recorded for the lowest price in the week")
 st.subheader("While in year May has been recorded as all time high")
 st.subheader("july has been recorded as the all time low in the year")


if selected=="Indicator":
 cryptos=("BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "USDC-USD", "SOL-USD", "ADA-USD", "XRP-USD", "LUNA1-USD", "HEX-USD", "AVAX-USD", "DOT-USD", "DOGE-USD", "SHIB-USD", "MATIC-USD", "ATOM-USD", "LTC-USD")
 targetcrypto=st.selectbox("ENTER THE CRYPTO TO BE PREDICTED",cryptos)
 def load_data(ticker):
    data=yf.download(ticker,start='2019-01-01')
    return data
 data_load_state=st.text("loading data...")
 data=load_data(targetcrypto)
 data_load_state.text("data loaded sucessfully....")
 data['MA20'] = data['Adj Close'].rolling(20).mean()
 data['MA50'] = data['Adj Close'].rolling(50).mean()
 data['MA100'] = data['Adj Close'].rolling(100).mean()
 data = data[['Adj Close','MA20','MA50','MA100']]
 data = data.dropna()
 st.write(data)
 st.subheader("SMA")
 st.write("Here we make use of one of the famous indicator known as SMA i.e Simple Moving Average here the past prices are considered and the average btw the values is calculated for the period of time and this process continues")

 def sma_graph():
    fig1=go.Figure()
    fig1.add_trace(go.Scatter(x=data['MA20'].index,y=data['MA20'],name='MA20'))
    fig1.add_trace(go.Scatter(x=data['MA50'].index,y=data['MA50'],name='MA50'))
    fig1.add_trace(go.Scatter(x=data['MA100'].index,y=data['MA100'], name='MA100'))
    fig1.add_trace(go.Scatter(x=data['Adj Close'].index,y=data['Adj Close'], name='Adj Close'))
    fig1.layout.update(title_text="Moving average",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)
 sma_graph()
 st.subheader("CONCLUSION")
 st.write("We observe that considering 20 days moving average shows relatively close result than 50 days moving average and 100 days moving average.hence if the period is less for calculating moving average the more accurate results will be drawn")

if selected=="Plan my purchase":
 cryptos=("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "LUNA1USDT", "HEXUSDT", "AVAXUSDT", "DOTUSDT", "DOGEUSDT", "SHIBUSDT", "MATICUSDT", "ATOMUSDT", "LTCUSDT")
 targetcrypto=st.selectbox("ENTER THE CRYPTO TO BE PREDICTED",cryptos)
 key = "https://api.binance.com/api/v3/ticker/price?symbol="
 url = key+targetcrypto
 data= requests.get(url)
 data= data.json()
 currprice= float(data['price'])
 money=st.number_input("ENTER THE MONEY YOU WANT TO INVEST IN DOLLARS",min_value=10.00,step=1.00)
 amount=money/currprice
 st.write("YOU WILL HAVE ",amount," OF ",targetcrypto," IN YOUR WALLET")
 
