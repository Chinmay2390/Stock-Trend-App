import json
from requests import request
from sklearn import model_selection, preprocessing
import streamlit as st
import yfinance as yf
import streamlit.components.v1 as components
from django.shortcuts import render
import plotly.graph_objects as go
from plotly.offline import plot
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import tickers
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime,timedelta
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


ticker_value = st.text_input("Enter Stock Ticker","AAPL")
number_of_days = st.text_input("Enter number of days","10")
number_of_days = number_of_days
# print(type(number_of_days))
model = st.selectbox("Models: ", ['LSTM', 'Decision Tree', 'Linear Regression'])
df = pd.DataFrame()

def basic(ticker_value, number_of_days):
    try:
        # ticker_value = request.POST.get('ticker')
        ticker_value = ticker_value.upper()
        df = yf.download(tickers = ticker_value, period='1d', interval='1m')
        print("Downloaded ticker = {} successfully".format(ticker_value))
    except:
        # return render(request, 'API_Down.html', {})
        st.write("API is down")

    try:
        # number_of_days = request.POST.get('days')
        number_of_days = int(number_of_days)
    except:
        # return render(request, 'Invalid_Days_Format.html', {})
        st.write("Please enter valid date")

    if ticker_value not in tickers.Valid_Ticker:
        # return render(request, 'Invalid_Ticker.html', {})
        st.write("This is Invalid Ticker")
    
    if number_of_days < 0:
        # return render(request, 'Negative_Days.html', {})
        st.write("Please enter positive number of days.")
    
    if number_of_days > 365:
        # return render(request, 'Overflow_days.html', {})
        st.write("We can only predict upto 1 year.")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'], name = 'market data'))
    fig.update_layout(
                        title='{} live share price evolution'.format(ticker_value),
                        yaxis_title='Stock Price (USD per Shares)')
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
        )
    )
    fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div = plot(fig, auto_open=False, output_type='div')
    st.plotly_chart(fig)
    # return df
def predictLR(ticker_value, number_of_days):
    try:
        df_ml = yf.download(tickers=ticker_value, period='3mo', interval='1h')
    except:
        ticker_value = 'AAPL'
        df_ml = yf.download(tickers=ticker_value, period='3mo', interval='1m')
    
    df_ml = df_ml[['Adj Close']]
    forecast_out = int(number_of_days)
    df_ml['Prediction'] = df_ml[['Adj Close']].shift(-forecast_out)
    # Splitting data for Test and Train
    X = np.array(df_ml.drop(['Prediction'],1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df_ml['Prediction'])
    y = y[:-forecast_out]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
    # Applying Linear Regression
    clf = LinearRegression()
    clf.fit(X_train,y_train)
    # Prediction Score
    confidence = clf.score(X_test, y_test)
    # Predicting for 'n' days stock data
    forecast_prediction = clf.predict(X_forecast)
    forecast = forecast_prediction.tolist()
    # st.write(f'Ticker Symbol: {ticker_value}')
    # st.write(f'Number of Days to Forecast: {number_of_days}')
    # st.write(f'Confidence Score: {confidence}')
    # st.write(f'Forecast: {forecast}')

    pred_dict = {"Date": [], "Prediction": []}
    for i in range(0, len(forecast)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast[i])
    
    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')
    
    st.plotly_chart(pred_fig)
def predictDT(ticker_value, number_of_days):
    start_date = "2010-01-01"
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        df = yf.download(ticker_value,start_date,end_date)
    except:
        st.write("sorry could not load dataset.")

    df = df[['Close']]
    future_days = int(number_of_days)
    df['Prediction'] = df[['Close']].shift(-future_days)
    # df = df.reset_index()
    # df = df.drop(['Date'], axis=1)
    X = np.array(df.drop(['Prediction'],1))
    X = X[:-future_days]
    y = np.array(df['Prediction'])[:-future_days]
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.25)
    tree = DecisionTreeRegressor().fit(x_train, y_train)
    x_feature = df[:-future_days]
    x_feature = x_feature[-future_days:]
    st.write(x_feature.shape)
    x_feature = x_feature.drop(['Prediction'],axis=1)
    tree_prediction = tree.predict(x_feature)
    # tree_prediction = tree.predict(x_feature[:, 0].reshape(-1, 1))


    # valid = df[X.shape[0]:]
    # valid["Prediction"] = tree_prediction

    #plot predictions

    # forecast_prediction = clf.predict(X_forecast)
    forecast = tree_prediction.tolist()
    # st.write(f'Ticker Symbol: {ticker_value}')
    # st.write(f'Number of Days to Forecast: {number_of_days}')
    # st.write(f'Confidence Score: {confidence}')
    # st.write(f'Forecast: {forecast}')

    pred_dict = {"Date": [], "Prediction": []}
    for i in range(0, len(forecast)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast[i])
    
    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')
    
    st.plotly_chart(pred_fig)
    
def predictLSTM(ticker_value, number_of_days):
    number_of_days = int (number_of_days)
    # Download the data for AAPL
    now = datetime.now()
    end_date = now.strftime("%Y-%m-%d")
    start_date = (now - pd.Timedelta(days=1000)).strftime("%Y-%m-%d")
    df = yf.download(ticker_value, start_date, end_date)

    # Preprocess the data
    scaler = MinMaxScaler(feature_range=(0,1))
    df['Close_scaled'] = scaler.fit_transform(df['Close'].values.reshape(-1,1))

    # Split the data into training and testing sets
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size,:]
    test_data = df.iloc[train_size:,:]

    def create_dataset(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    time_steps = 100

    X_train, y_train = create_dataset(train_data[['Close_scaled']], train_data['Close_scaled'], time_steps)
    X_test, y_test = create_dataset(test_data[['Close_scaled']], test_data['Close_scaled'], time_steps)

    from keras.models import load_model

    model = load_model('keras_model_GPT.h5')

    # y_pred = model.predict(X_test)
    # Inverse transform the data to get actual stock prices
    # y_pred_actual = scaler.inverse_transform(y_pred)
    # y_test_actual = scaler.inverse_transform(y_test)
    # import numpy as np
    # Reshape y_test into a 2D array with a single column
    # y_test = y_test.reshape(-1, 1)
    # Inverse transform the data to get actual stock prices
    # y_test_actual = scaler.inverse_transform(y_test)
    # Calculate the root mean squared error
    # rmse = np.sqrt(np.mean((y_pred_actual - y_test_actual)**2))
    # print(f"RMSE: {rmse:.2f}")

    past_100_days = df.tail(100)
    past_100_days = past_100_days.drop(['Open','High','Low','Close','Adj Close','Volume'],axis=1)
    past_100_days['Close_scaled'] = past_100_days['Close_scaled'].astype(float)

    for i in range(1,number_of_days):
        # past_100_days
        xc_test = np.array([past_100_days])
        xc_test = xc_test.reshape(xc_test.shape[0], xc_test.shape[1], 1)
        y_predicted = model.predict(xc_test)

        past_100_days = past_100_days.iloc[1:] # remove the first row
        last_index = past_100_days.index[-1]
        next_day = last_index + timedelta(days=1)
        y_predicted = y_predicted.ravel()
        past_100_days.loc[next_day] = y_predicted

    
    past_100_days = past_100_days.reset_index()
    last_n_records = past_100_days['Close_scaled'].tail(number_of_days)
    last_n_records = np.array(last_n_records).reshape(-1, 1)
    last_n_records = scaler.inverse_transform(last_n_records)
    forecast = last_n_records.reshape(-1)

    pred_dict = {"Date": [], "Prediction": []}
    for i in range(0, len(forecast)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast[i])
    
    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')

    st.plotly_chart(pred_fig)
# scale = scaler.scale_[0]
# scaling_factor = 1/scale
# print(y_predicted)
# y_predicted = scaler.inverse_transform(y_predicted)




    ####################################################################
def main():
    # Load the data
        ticker = pd.read_csv('data/Tickers.csv')

        # Set column names
        ticker.columns = ['Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change', 'Market_Cap',
                        'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry']

        # Create a text input for the user to enter a ticker symbol
        to_search = ticker_value

        # Loop through the rows of the DataFrame and find the matching ticker symbol
        found_ticker = None
        for i in range(ticker.shape[0]):
            if ticker.Symbol[i] == to_search:
                found_ticker = ticker.iloc[i]
                break

        # If a matching ticker is found, display its details
        if found_ticker is not None:
            st.write("\n\n\n")
            st.write(f"Symbol: {found_ticker.Symbol}")
            st.write(f"Name: {found_ticker.Name}")
            st.write(f"Last Sale: {found_ticker.Last_Sale}")
            st.write(f"Net Change: {found_ticker.Net_Change}")
            st.write(f"Percent Change: {found_ticker.Percent_Change}")
            st.write(f"Market Cap: {found_ticker.Market_Cap}")
            st.write(f"Country: {found_ticker.Country}")
            st.write(f"IPO Year: {found_ticker.IPO_Year}")
            st.write(f"Volume: {found_ticker.Volume}")
            st.write(f"Sector: {found_ticker.Sector}")
            st.write(f"Industry: {found_ticker.Industry}")
        
        else:
            st.write("Ticker symbol not found.")

# if __name__ == "__main__":
#     main()

    ###########################################################################
#     def my_view(request):
#         data = {
#             'plot_div': plot_div,
#             'confidence': confidence,
#             'forecast': forecast,
#             'ticker_value': ticker_value,
#             'number_of_days': number_of_days,
#             'plot_div_pred': plot_div,
#             'Symbol': Symbol,
#             'Name': Name,
#             'Last_Sale': Last_Sale,
#             'Net_Change': Net_Change,
#             'Percent_Change': Percent_Change,
#             'Market_Cap': Market_Cap,
#             'Country': Country,
#             'IPO_Year': IPO_Year,
#             'Volume': Volume,
#             'Sector': Sector,
#             'Industry': Industry,
#         }
#         return data
# ################################################################
#     data = my_view(None)

#     # Display the data in Streamlit
#     st.write(data['forecast'])
#     st.write(data['confidence'])
#############################################################
if st.button('Submit'):
    # main()
    basic(ticker_value,number_of_days)
    if model == "Linear Regression":
        predictLR(ticker_value,number_of_days)
    elif model == "LSTM":
        predictLSTM(ticker_value,number_of_days)
    elif model == "Decision Tree":
        predictDT(ticker_value,number_of_days)
    else:
        pass
