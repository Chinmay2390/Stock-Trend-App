import streamlit as st
import yfinance as yf
import pandas as pd
import json
st.set_page_config(
    page_title="Stock Price Prediction App",
    page_icon="🕸",
)
st.title("Contact Us")


df1 = yf.download(tickers='AAPL', period='1d', interval='1d')
df2 = yf.download(tickers='AMZN', period='1d', interval='1d')
df3 = yf.download(tickers='GOOGL', period='1d', interval='1d')
df4 = yf.download(tickers='UBER', period='1d', interval='1d')
df5 = yf.download(tickers='TSLA', period='1d', interval='1d')
df6 = yf.download(tickers='TWTR', period='1d', interval='1d')

df1.insert(0, "Ticker", "AAPL")
df2.insert(0, "Ticker", "AMZN")
df3.insert(0, "Ticker", "GOOGL")
df4.insert(0, "Ticker", "UBER")
df5.insert(0, "Ticker", "TSLA")
df6.insert(0, "Ticker", "TWTR")

df = pd.concat([df1, df2, df3, df4, df5, df6], axis=0)
df.reset_index(level=0, inplace=True)
df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
convert_dict = {'Date': object}
df = df.astype(convert_dict)
df.drop('Date', axis=1, inplace=True)

json_records = df.reset_index().to_json(orient='records')
recent_stocks = []
recent_stocks = json.loads(json_records)

# Streamlit app
def index():
    st.title("Recent Stocks Data")
    st.write(pd.DataFrame(recent_stocks))

if __name__ == '__main__':
    index()
    
#st.write(st.session_state['my_input'])