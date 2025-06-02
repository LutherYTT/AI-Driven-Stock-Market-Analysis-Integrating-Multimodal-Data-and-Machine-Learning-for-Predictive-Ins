import argparse
from src.call_deepseek import *

def parse_args():
    parser = argparse.ArgumentParser(description='Stock Analysis Tool')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol (e.g., 01810.HK)')
    parser.add_argument('--model', type=str, required=True, choices=['deepseek-chat','deepseek-reasoner'], help='Model type to use')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    ticker_symbol = args.ticker.upper()
    stock_id = ticker_symbol.split('.')[0].zfill(5)

    api_key = 'put-your-deepseek-api-key-here'
    model_type = args.model
    # model_type = 'deepseek-reasoner'
    main(stock_id, api_key, model_type)
