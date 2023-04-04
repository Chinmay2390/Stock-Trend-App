import streamlit as st
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go 
st.set_page_config(
    page_title="Stock Price Prediction App",
    page_icon="🕸",
)
# st.title("Project Work")

#st.write(st.session_state['my_input'])

background_color= "#808080"
ticker_info = st.container()
df = pd.read_csv('file2.csv')
df =df.dropna()

with ticker_info:
    st.title("All Tickers")
    fig = go.Figure(data=go.Table(
    header=dict(values=list(df[['Ticker', 'Full_Name']].columns), 
                fill_color='#fd8e72', 
                align='center'), 
    cells=dict(values=[df.Ticker, df.Full_Name], 
               fill_color='#e5ecf6',
               align='left')))

    fig.update_layout(margin=dict(l=0,r=0,b=0,t=0), 
                      paper_bgcolor=background_color)

    st.write(fig)
