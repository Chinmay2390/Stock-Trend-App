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
st.write("**Primary market** is a marketplace where companies raise capital for the very first time. The process of issuing shares to the general public for the first time is known as an Initial Public Offering, or IPO.\n\nOnce the shares are issued in the primary market, they are traded i.e. bought and sold in the **secondary market** via a stock exchange. ")
st.subheader("Stock exchanges are further divided as:")
st.markdown(
"""
- National stock exchanges 
- Regional stock exchanges. 
"""
)
st.write("The Bombay Stock Exchange and National Stock Exchange are the only two national stock exchanges in India, with the BSE being the oldest stock exchange in Asia. The BSE is also the 10th largest stock exchange in the world with a market capitalisation of 2.1 Trillion Dollars.\n\nThe movement of the market is mapped using the 2 primary indices, SENSEX and NIFTY. Sensex is the index used by the BSE to gauge the movement of the 30 biggest indian companies whereas NIFTY is the index used by the NSE to measure the movement of the 50 largest companies in India. ")
st.subheader("Stock Chart Types:")
st.write("The simple line chart is used for long-term trend assessment; the OHLC and Candlestick charts show the open, high, low, and close prices. VAP and Equivolume charts add volume data to the chart analysis. Finally, Market Profile and Point and Figure charts remove the timeline X-axis from chart analysis.\n\nThe Line Chart is the simplest, depicting only the closing price. The High Low Close Chart shows the price high low & close. As we move to OHLC, Japanese Candlestick, and Point & Figure Charts, a new world of understanding supply and demand is unleashed.")
st.markdown(
"""
- Line Stock Charts
- High Low Close Bar Stock Charts (HLC)
- Open High Low Close Bar Stock Charts (OHLC)
- Japanese Candlestick Charts
- Volume at Price (VAP) Stock Chart
- EquiVolume Stock Charts
- Point and Figure (P&F) Charts
- Market Profile Stock Charts
"""
)

