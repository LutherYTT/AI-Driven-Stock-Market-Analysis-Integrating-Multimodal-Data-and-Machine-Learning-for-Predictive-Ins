import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import re

def preprocess_data(google_news, yahoo_news, sina_news, aastocks_news, time_series):
    # Function to extract summary (first 50 characters)
    def extract_summary(text, max_length=50):
        if pd.isna(text):
            return ""
        return text[:max_length] + '...' if len(text) > max_length else text
    
    # Preprocess each news source
    google_summaries = google_news.apply(
        lambda row: f"{row['title']}: {extract_summary(row['description'])}", axis=1
    ).tolist()
    
    yahoo_summaries = yahoo_news.apply(
        lambda row: f"{row['title']}: {extract_summary(row['content'])}", axis=1
    ).tolist()
    
    sina_summaries = sina_news.apply(
        lambda row: f"{row['title']}: (No content available)", axis=1
    ).tolist()  # Sina only has titles and timestamps
    
    aastocks_summaries = aastocks_news.apply(
        lambda row: f"{row['title']}: {extract_summary(row['detail_content'])}", axis=1
    ).tolist()
    
    # Combine all news summaries
    news_text = "\n".join(google_summaries + yahoo_summaries + sina_summaries + aastocks_summaries)

    # Use the last 30 days (assuming data is sorted by date)
    recent_data = time_series.tail(30)
    avg_price = recent_data['Close'].mean()
    max_price = recent_data['Close'].max()
    min_price = recent_data['Close'].min()
    volatility = recent_data['Close'].std()
    
    # Get the most recent day's data
    latest_data = time_series.iloc[-1]

    # Format stock data
    stock_data = f"""
    Last 30 Days Stock Stats:
    Average Price: {avg_price:.2f}
    Max Price: {max_price:.2f}
    Min Price: {min_price:.2f}
    Volatility: {volatility:.2f}

    Latest Day's Data:
    Date: {latest_data['Date']},
    Close: {latest_data['Close']},
    High: {latest_data['High']},
    Low: {latest_data['Low']},
    Volume: {latest_data['Volume']},
    MA20: {latest_data['MA20']},
    MA50: {latest_data['MA50']},
    MA200: {latest_data['MA200']},
    Volatility: {latest_data['Volatility']},
    RSI_14: {latest_data['RSI_14']}
    """
    
    return stock_data, news_text

# Load a CSV file containing news articles and stock price data.
def load_data(stock_id):
    # Load news CSV files
    google_news = pd.read_csv(f"./output/{stock_id}/google_news_{datetime.now().strftime('%Y%m%d')}.csv")
    yahoo_news = pd.read_csv(f"./output/{stock_id}/yahoo_news_{stock_id}_{datetime.now().strftime('%Y%m%d')}.csv")
    sina_news = pd.read_csv(f"./output/{stock_id}/sina_stock_news_{stock_id}_{datetime.now().strftime('%Y%m%d')}.csv")
    aastocks_news = pd.read_csv(f"./output/{stock_id}/aastocks_news_full_{stock_id}_{datetime.now().strftime('%Y%m%d')}.csv")
    
    # Load and preprocess stock price data
    time_series = pd.read_csv(f"./output/{stock_id}/enhanced_time_series_analysis_{datetime.now().strftime('%Y%m%d')}.csv")
    
    return google_news, yahoo_news, sina_news, aastocks_news, time_series

# Constructs the prompts used by the DeepSeek API.
def construct_prompt(stock_id, stock_data, news_text):
    prompt = f"""
    As a professional stock analyst, you are asked to analyze the following stock data and news articles to provide trading recommendations for the stock {stock_id}. Please use the provided stock price data and news articles from Google News, Sina and AAStocks. Stock data includes historical prices, volume and technical indicators (e.g. moving averages, RSI, etc.). News articles include company announcements, industry news and market analysis.
    Stock price data:
    {stock_data}

    News Article:
    {news_text}
    
    You are advised to use technical analysis (based on price and technical indicators), fundamental analysis (based on company and industry information), and sentiment analysis (based on news sentiment and considering the impact of news on general investor sentiment) to consider market trends over the short term (approximately 5 days) and medium term (approximately 30 days). Please note that the stock market is subject to volatility and uncertainty. The following analysis is for reference only and investment decisions should be made with caution.

    Please export your analysis results in the following format:

    Stock risk (1-5): [1-5]
    Short-term up/down forecast (~30 days): [up/small up/stable/small down/down]
    Medium-term up/down forecast (~30 days): [up/small up/stable/small down/down]
    Stock Price Trend: [Up/Down]
    Recommended Buy Price: [price]
    Recommended Sell Price: [price]

    Sentiment Analysis: [analysis]
    Short comment: [brief comment]
    """
    return prompt

# Call the DeepSeek API.
def call_deepseek_api(prompt, api_key, model_type):
    api_url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    payload = {
        'model': model_type,
        'messages': [
            {"role": "system", "content": "As a professional stock analyst, you are asked to analyze the following stock data and news articles to provide trading recommendations for specific stocks. Please use the provided stock price data and news articles from Google News, Sina and AAStocks. Stock data includes historical prices, volume and technical indicators (e.g. moving averages, RSI, etc.). News articles include company announcements, industry news and market analysis."},
            {"role": "user", "content": prompt}
        ],
        'max_tokens': 1000,
        'temperature': 0.8
    }
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=300  # 增加超时设置
        )
        # 添加响应状态检查
        if response.status_code != 200:
            return {
                "error": f"API request failed, status:{response.status_code}",
                "response": response.text
            }
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return {"error": f"API call exceptions:{str(e)}"}

def parse_response(text):
    # patterns = {
    #     'risk_level': r'Stock risk\(1-5\):\s*(\d)',
    #     'short_term': r'Short-term up/down forecast\(~5 days\):\s*([^\n]+)',
    #     'medium_term': r'Medium-term up/down forecast\(~30 days\):\s*([^\n]+)',
    #     'trend': r'Stock Price Trend:\s*([^\n]+)',
    #     'buy_price': r'Recommended Buy Price:\s*([^\n]+)',
    #     'sell_price': r'Recommended Sell Price:\s*([^\n]+)',
    #     'comment': r'Short comment:\s*([^\n]+)'
    # }
    # result = {}
    # for key, pattern in patterns.items():
    #     match = re.search(pattern, text)
    #     if match:
    #         result[key] = match.group(1).strip()
    #     else:
    #         result[key] = 'N/A'
    result = text
    return result

def main(stock_id, api_key, model_type):
    # Load data
    google_news, yahoo_news, sina_news, aastocks_news, time_series = load_data(stock_id)
    # Preprocess
    stock_data, news_text = preprocess_data(google_news, yahoo_news, sina_news, aastocks_news, time_series)

    # print("Stock Data:\n", stock_data)
    # print("\nNews Summaries:\n", news_text)

    # Bulid prompt
    prompt = construct_prompt(stock_id, stock_data, news_text)
    # Calling the API
    response_text = call_deepseek_api(prompt, api_key, model_type)

    # Parsing Response
    result = parse_response(response_text)

    # output
    print(result)

    print("[WARNING] This answer was generated by AI and is for informational purposes only, so please screen it carefully.")
    with open(f"./output/{stock_id}/Deepseet_Output.txt", "w") as text_file:
        text_file.write(result)