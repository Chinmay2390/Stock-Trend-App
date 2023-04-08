import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(
    page_title="Stock Price Prediction App",
    page_icon="🕸",
)
# st.title("Contact Us")
st.title("Active Stocks")
#st.write(st.session_state['my_input'])


# The Home page when Server loads up
def index():
    # ================================================= Left Card Plot =========================================================
    # Here we use yf.download function
    data = yf.download(
        
        # passes the ticker
        tickers=['AAPL', 'AMZN', 'QCOM', 'META', 'NVDA', 'JPM'],
        
        group_by = 'ticker',
        
        threads=True, # Set thread value to true
        
        # used for access data[ticker]
        period='1mo', 
        interval='1d'
    
    )

    data.reset_index(level=0, inplace=True)



    fig_left = go.Figure()
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['AAPL']['Adj Close'], name="AAPL")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['AMZN']['Adj Close'], name="AMZN")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['QCOM']['Adj Close'], name="QCOM")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['META']['Adj Close'], name="META")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['NVDA']['Adj Close'], name="NVDA")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['JPM']['Adj Close'], name="JPM")
            )
    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    st.plotly_chart(fig_left)


index()

import streamlit as st
import pandas as pd
import yfinance as yf
import json

# # Download stock data
# df1 = yf.download(tickers='AAPL', period='1d', interval='1d')
# df2 = yf.download(tickers='AMZN', period='1d', interval='1d')
# df3 = yf.download(tickers='GOOGL', period='1d', interval='1d')
# df4 = yf.download(tickers='UBER', period='1d', interval='1d')
# df5 = yf.download(tickers='TSLA', period='1d', interval='1d')
# df6 = yf.download(tickers='TWTR', period='1d', interval='1d')

# # Add ticker column to each DataFrame
# df1.insert(0, "Ticker", "AAPL")
# df2.insert(0, "Ticker", "AMZN")
# df3.insert(0, "Ticker", "GOOGL")
# df4.insert(0, "Ticker", "UBER")
# df5.insert(0, "Ticker", "TSLA")
# df6.insert(0, "Ticker", "TWTR")

# # Combine DataFrames and format
# df = pd.concat([df1, df2, df3, df4, df5, df6], axis=0)
# df.reset_index(level=0, inplace=True)
# df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
# convert_dict = {'Date': object}
# df = df.astype(convert_dict)
# df.drop('Date', axis=1, inplace=True)

# # Convert DataFrame to JSON and load as list of dictionaries
# json_records = df.reset_index().to_json(orient ='records')
# recent_stocks = json.loads(json_records)

# # Assume 'df' is the DataFrame you want to display as a table

# # Define the CSS style for the header row
# header_style = {'selector': 'th',
#                 'props': [('background-color', 'orange'), 
#                           ('color', 'white'), 
#                           ('font-weight', 'bold'), 
#                           ('padding', '10px')]}
# df.style.apply(lambda _: [header_style]*len(df.columns))

# # Display the table with the header row style applied
# st.write(df.style.apply(lambda _: header_style, axis=1))
# # Display table in Streamlit app
# st.table(df)


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
