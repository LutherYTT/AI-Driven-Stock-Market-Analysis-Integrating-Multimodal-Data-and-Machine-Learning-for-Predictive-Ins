import yfinance as yf
import pandas as pd
import numpy as np

# Get historical stock data from Yahoo Finance 
# Parameters:
#    ticker (str): ticker symbol 
#    years (int): span of years for which to get data
# Return:
#    pd.DataFrame: DataFrame containing OHLCV and other data
def fetch_stock_data(ticker, years=10):
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)

    data = yf.download(
        tickers=ticker,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        progress=False
    )

    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)

    return data.set_index('Date')