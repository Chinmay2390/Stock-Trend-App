import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Machine Learning for Stock Market Prediction With Step-by-Step Implementation",
    page_icon="🕸",
)
st.title("Learn Stock")

# st.header("Introduction")
st.subheader("Introduction")
# st.subheader("")
st.write("\tStock market prediction and analysis are some of the most difficult jobs to complete. There are numerous causes for this, including market volatility and a variety of other dependent and independent variables that influence the value of a certain stock in the market. These variables make it extremely difficult for any stock market expert to anticipate the rise and fall of the market with great precision. Considered among the most potent tree-based techniques, Random Forest can predict the stock process as it can also solvereg ression-based problems.\n\nHowever, this tutorial, with the introduction of Data Science, Machine Learning, and artificial intelligence\n and its strong algorithms, the most recent market research, and Stock price Prediction advancements have begun to include such approaches in analyzing stock market data.")
img = Image.open("images/img1.png")
st.image(img)
