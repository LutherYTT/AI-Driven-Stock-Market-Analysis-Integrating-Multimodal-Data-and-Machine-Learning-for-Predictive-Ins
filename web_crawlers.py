import argparse
from src.news_scraper import *

def parse_args():
    parser = argparse.ArgumentParser(description='Stock Analysis Tool')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol (e.g., 01810.HK)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    ticker_symbol = args.ticker.upper()

    stock_id = ticker_symbol.split('.')[0].zfill(5)
    
    get_google_news(stock_id)
    
    get_usa_google_news(stock_id)

    scraper = YahooFinanceScraper()
    news_data_yahoo = scraper.scrape_news(ticker_symbol, max_news=15)
    scraper.save_to_csv(news_data_yahoo, ticker_symbol)
    # if news_data_yahoo:
    #     print("\nYahoo Finance Latest News：")
    #     for idx, news in enumerate(news_data_yahoo, 1):
    #         print(f"{idx}. [{news['publish_date']}] {news['title']}")

    news_data_sina = scrape_sina_stock_news(stock_id, save_csv=True)
    # print("\nSina Latest News：")
    # for i, item in enumerate(news_data_sina[:], 1):
    #     print(f"{i}. [{item['time']}] {item['title']}")

    news_data_aastocks = scrape_aastocks_news(stock_id, save_csv=True, delay=1)
    # print("\nAastocks Latest News：")
    # for i, item in enumerate(news_data_aastocks[:], 1):
    #     print(f"{i}. [{item['time']}] {item['title']}")
    #     print(f"   Source：{item['source']}")
    #     print(f"   Detail Content：{item['detail_content'][:100]}...\n")
