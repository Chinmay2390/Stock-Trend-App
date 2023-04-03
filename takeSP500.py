import bs4 as bs
import requests
import yfinance as yf
import datetime
import pandas as pd

resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})



tickers = []

for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)


tickers = [s.replace('\n', '') for s in tickers]
# print(tickers)

# ticker = input("Enter the stock ticker\n")
ticker = 'AAPL'
tickerInfo = yf.Ticker(ticker)
# print(tickerInfo.info["exchangeTimezoneName"])

def makeTickerDF():
    tickerdf = pd.Series(tickerInfo.info,index = ["market", "symbol"])
    # print(tickerdf)
    return tickerdf