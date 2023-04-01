import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from pandas_datareader import data as pdr
import yfinance as yf




st.set_page_config(
    page_title="Stock Price Prediction App",
    page_icon="🕸",
)


st.title("Indicators")

user_input = st.text_input("Enter Stock Ticker","AAPL")

# yf.pdr_override()
# start_date = "2010-01-01"
# end_date = "2023-02-28"
# df = yf.download(user_input, start_date, end_date)

# st.write(df.describe())

df = pd.read_csv("NFLX.csv")
st.write(df.describe())


indicator = st.selectbox("Indicators: ",['Simple Moving Average', 'RSI', 'Exponential Moving Average','Moving Average Convergance/Divergance'])

df = df.set_index(pd.DatetimeIndex(df['Date'].values))
st.write(df.head())   


#creating the sma
def smacalci(data, period = 30, column = 'Close'):
    return data[column].rolling(window = period).mean()

#creating the ema
def emacalci(data, period = 20, column = 'Close'):
    return data[column].ewm(span = period, adjust = False).mean()


#Calculate the Moving Average Convergence/Divergence (MACD)
def MACD(data, period_long=26, period_short=12, period_signal= 9, column='Close'): 

    #Calculate the Short Term Exonential Moving Average
    ShortEMA = emacalci(data, period_short, column=column)
    #Calculate the Long Term Exponential Moving Average 
    LongEMA = emacalci(data, period_long, column = column)
    #Calculate the Moving Average Convergence/Divergence (MACD)
    data['MACD'] = ShortEMA - LongEMA
    #Calculate the signal line
    data['Signal_Line'] = emacalci(data, period_signal, column='MACD') 
    return data



#Create a function to compute the Relative Strength Index (RSI) 
# def RSI (data, period=14, column='Close'): 

#     delta=data[column].diff(1)
#     delta=delta[1:]
#     up=delta.copy()
#     down=delta.copy()
#     up[up <0] = 0
#     down [down>0] = 0
#     data['up'] = up
#     data['down'] = down
#     AVG_Gain= smacalci (data, period, column = 'up')
#     AVG_Loss = abs (smacalci(data, period, column = 'down'))
#     RS = AVG_Gain / AVG_Loss
#     RSI = 100.0 (100.0/(1.0+ RS))
#     data['RSI'] = RSI
#     return data


#calling functions
MACD(df)
#RSI(df)
df['SMA'] = smacalci(df)
df['EMA'] = emacalci(df)

if indicator == 'Moving Average Convergance/Divergance':
   
    column_list = ['MACD', 'Signal_Line'] 
    df1 = df[column_list]
    figure = plt.figure(figsize=(12.2, 6.4)) 
    plt.plot(df1['MACD'],'r')
    plt.plot(df1['Signal_Line'],'g')
    plt.title('MACD for NETFLIX (NFLX)')
    plt.ylabel('USD Price')
    plt.xlabel('Date')
    st.pyplot(figure)

elif indicator == 'Simple Moving Average':
    column_list = ['SMA', 'Close'] 
    figure = plt.figure(figsize=(12.2, 6.4)) 
    plt.plot(df['SMA'],'r') 
    plt.plot(df['Close'],'g')
    plt.title('SMA for NETFLIX (NFLX)')
    plt.ylabel('USD Price')
    plt.xlabel('Date')
    st.pyplot(figure)

elif indicator == 'Exponential Moving Average':
    column_list = ['EMA', 'Close'] 
    figure = plt.figure(figsize=(12.2, 6.4)) 
    plt.plot(df['EMA'],'r') 
    plt.plot(df['Close'],'g')
    plt.title('EMA for NETFLIX (NFLX)')
    plt.ylabel('USD Price')
    plt.xlabel('Date')
    st.pyplot(figure)

elif indicator == 'RSI':
    figure = column_list = ['RSI'] 
    figure = plt.figure(figsize=(12.2, 6.4))
    plt.plot(df['RSI'],'r')
    plt.title('RSI for NETFLIX (NFLX)')
    plt.ylabel('USD Price')
    plt.xlabel('Date')
    st.pyplot(figure)

else:
    pass


#st.write(st.session_state['my_input'])