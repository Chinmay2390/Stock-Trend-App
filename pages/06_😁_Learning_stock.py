import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Machine Learning for Stock Market Prediction With Step-by-Step Implementation",
    page_icon="🕸",
)
st.title("Learn Stock Market")

# st.header("Introduction")
st.subheader("What is a Stock Market?")
# st.subheader("")
st.write("\tA stock market is a marketplace where buyers and sellers meet to trade i.e. buy and sell shares of publicly listed companies. A stock market is fondly known as a share market, equity market or share bazaar.\n\nIn simple terms, if Ram wants to sell 10 shares of Reliance Industries at Rs 1990/ share, he will place a sell order on the stock exchange. The stock exchange will then find a buyer who wants to buy 10 shares of Reliance Industries at Rs 1990/ share. So, the stock market is a virtual market where the buyers and sellers meet to trade shares. ")
img = Image.open("images/img1.png")
st.image(img)
st.subheader("The two types of stock markets are:")
st.markdown(
"""
- Primary markets
- Secondary markets
"""
)
st.write("Primary market is a marketplace where companies raise capital for the very first time. The process of issuing shares to the general public for the first time is known as an Initial Public Offering, or IPO.\n\nOnce the shares are issued in the primary market, they are traded i.e. bought and sold in the secondary market via a stock exchange. ")
st.subheader("Stock exchanges are further divided as:")
st.markdown(
"""
- National stock exchanges 
- Regional stock exchanges. 
"""
)