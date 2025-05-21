import argparse
import yfinance as yf
import numpy as np
import pandas as pd
from src.data_fetch import *
from src.preprocessing import *
from src.statistics import *
from src.visualization import *
from src.advanced_analysis import *

def parse_args():
    parser = argparse.ArgumentParser(description='Stock Analysis Tool')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol (e.g., 01810.HK)')
    parser.add_argument('--years', type=int, default=10, help='Number of years of historical data to fetch')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ticker_symbol = args.ticker.upper()

    stock_id = ticker_symbol.split('.')[0].zfill(5)

    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    pd.set_option('display.max_columns', None)

    df = fetch_stock_data(ticker_symbol, years=args.years)
    df.columns = df.columns.get_level_values(0)  # Keep only the first level
    print(f"Amount of data captured: {len(df)}")
    # print(df.head())

    df = calculate_technical_indicators(df)
    create_interactive_chart(df, stock_id)

    print("\nKey statistical indicators:")
    statistics = calculate_statistics(df)
    print(statistics.to_string())

    plot_return_distribution(df, stock_id)

    if not os.path.exists(f"./output/{stock_id}"):
        os.makedirs(f"./output/{stock_id}")
        print(f"Folder '{stock_id}' created.")
    else:
        print(f"Folder './output/{stock_id}' already exists.")
    # df.to_csv(f"./output/{stock_id}/time_series_analysis_{datetime.now().strftime('%Y%m%d')}.csv", index=True)
    
    ####################
    # Advance Analysis #
    ####################
    volatility_clustering(df, stock_id)
    run_backtest(df, stock_id)
    anomaly_detection(df, stock_id)
    # Save the updated df
    df.to_csv(f"./output/{stock_id}/enhanced_time_series_analysis_{datetime.now().strftime('%Y%m%d')}.csv", index=True)