import streamlit as st
import requests

# Define function to get stock news
def get_stock_news(company):
    # Set the API endpoint and parameters
    endpoint = "https://finance.yahoo.com/_finance_api/api/newsfeed/query"
    params = {
        "category": "general",
        "isCount": "false",
        "symbols": company,
        "count": 10,
        "lang": "en-US",
        "region": "US"
    }
    # Send GET request to Yahoo Finance API and retrieve news articles
    response = requests.get(endpoint, params=params)
    articles = response.json()["data"]["stream_items"]
    # Display article titles and links
    for article in articles:
        title = article["title"]
        url = article["url"]
        st.write(f"- [{title}]({url})")

# Create the news tab
def news_tab():
    st.title("Stock News")
    st.write("Enter a stock symbol to get the latest news:")
    company = st.text_input("Company", "AAPL")
    if st.button("Search"):
        get_stock_news(company)

if __name__ == '__main__':
    news_tab()