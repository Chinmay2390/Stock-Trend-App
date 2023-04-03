import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from keras.models import load_model
import takeSP500 as sp
from sklearn.preprocessing import MinMaxScaler


import streamlit as st


st.title("Stock Trend App")
user_input = st.text_input("Enter Stock Ticker","AAPL")
if(st.button("Submit")):
    st.subheader("Data from 2010-01-01 to 2023-02-28")

    start_date = "2010-01-01"
    end_date = "2023-02-28"

    # df = data.DataReader('AAPL','yahoo',start,end)
    # df.head()
    # dataSTOCK = web.DataReader("TSLA", 'yahoo', start_date, end_date)

    yf.pdr_override()

    df = yf.download(user_input, start_date, end_date)
    # st.table(df.head())
    st.table(sp.makeTickerDF())
    st.write(df.describe())

    st.subheader("Closing Price vs Time Chart")
    figure = plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(figure)


    st.subheader("Closing Price vs Time Chart with ma100")
    ma100 = df['Close'].rolling(100).mean()
    figure = plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    plt.plot(ma100)
    st.pyplot(figure)


    st.subheader("Closing Price vs Time Chart with ma100 and ma200")
    ma200 = df['Close'].rolling(200).mean()
    figure = plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    plt.plot(ma100)
    plt.plot(ma200)
    st.pyplot(figure)


    #splitting data into training and testing
    training_data = pd.DataFrame(df['Close'][0 : int(len(df)*0.70)])
    testing_data = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])



    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(training_data)


    #loading my model
    model = load_model('keras_model1.h5')

    #testing model
    past100days = training_data.tail(100)
    last_df = past100days.append(testing_data,ignore_index = True)
    input_data = scaler.fit_transform(last_df)
    x_test = []
    y_test = []
    for i in range(100,input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])
        
    x_test , y_test = np.array(x_test), np.array(y_test)

    #predicting
    y_predicted = model.predict(x_test)

    #scaling-backword
    scalervar = scaler.scale_
    scaling_factor = 1/scalervar[0]
    y_predicted = y_predicted*scaling_factor
    y_test = y_test*scaling_factor

    #printing result
    st.subheader("Predictions:")
    figure = plt.figure(figsize=(12,6))
    plt.plot(y_predicted,'r',label = 'Predicted Value')
    plt.plot(y_test,'b',label = 'Actual Value')
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(figure)