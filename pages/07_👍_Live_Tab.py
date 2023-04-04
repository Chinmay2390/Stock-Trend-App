import streamlit as st
import yfinance as yf
import streamlit.components.v1 as components

import plotly.graph_objects as go
from plotly.offline import plot
import datetime as dt
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

user_input1 = st.text_input("Enter Stock Ticker","AAPL")
user_input2 = st.text_input("Enter number of days"," ")

if st.button('Submit'):
    st.write('Output')

    html_string = '''
    <head>
    <meta charset="utf-8">
    <title>Stock Market Prediction</title>
    <meta content="width=device-width, initial-scale=1.0" name="viewport">
    <meta content="" name="keywords">
    <meta content="" name="description">

    
    </head>
    '''
    components.html(html_string)  # JavaScript works

    st.markdown(html_string, unsafe_allow_html=True)  # JavaScript doesn't work

