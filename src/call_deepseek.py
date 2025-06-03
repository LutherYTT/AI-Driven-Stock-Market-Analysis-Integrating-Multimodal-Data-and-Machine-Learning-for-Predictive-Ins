import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import re

def preprocess_data(google_news, google_USA_news, yahoo_news, sina_news, aastocks_news, time_series):
    # Function to extract summary (first 50 characters)
    def extract_summary(text, max_length=500):
        if pd.isna(text):
            return ""
        return text[:max_length] + '...' if len(text) > max_length else text
    
    # Preprocess each news source
    google_summaries = google_news.apply(
        lambda row: f"{row['title']}: {extract_summary(row['description'])}", axis=1
    ).tolist()
    
    google_USA_summaries = google_USA_news.apply(
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
    recent_30_data = time_series.tail(30)
    # Use the last 3 days (assuming data is sorted by date)
    recent_3_data = time_series.tail(3)

    avg_30_price = recent_30_data['Close'].mean()
    max_30_price = recent_30_data['Close'].max()
    min_30_price = recent_30_data['Close'].min()
    volatility_30 = recent_30_data['Close'].std()
    avg_30_volume = recent_30_data['Volume'].mean()
    avg_30_rsi_14 = recent_30_data['RSI_14'].mean()
    avg_30_cci_20 = recent_30_data['CCI_20'].mean()
    avg_30_macd = recent_30_data['MACD'].mean()
    avg_30_adx_14 = recent_30_data['ADX_14'].mean()
    avg_30_kdj = recent_30_data['%K'].mean()

    avg_3_price = recent_3_data['Close'].mean()
    max_3_price = recent_3_data['Close'].max()
    min_3_price = recent_3_data['Close'].min()
    volatility_3 = recent_3_data['Close'].std()
    avg_3_volume = recent_3_data['Volume'].mean()
    avg_3_rsi_14 = recent_3_data['RSI_14'].mean()
    avg_3_cci_20 = recent_3_data['CCI_20'].mean()
    avg_3_macd = recent_3_data['MACD'].mean()
    avg_3_adx_14 = recent_3_data['ADX_14'].mean()
    avg_3_kdj = recent_3_data['%K'].mean()

    # Get the most recent day's data
    latest_data = time_series.iloc[-1]

    # Format stock data
    stock_data = f"""
    Last 30 Days Stock Stats:
    Average Price: {avg_30_price:.2f}
    Max Price: {max_30_price:.2f}
    Min Price: {min_30_price:.2f}
    Volatility: {volatility_30:.2f}
    Average Volume: {avg_30_volume:.4f}
    Average RSI_14: {avg_30_rsi_14:.4f}
    Average CCI_20: {avg_30_cci_20:.4f}
    Average MACD: {avg_30_macd:.4f}
    Average ADX_14: {avg_30_adx_14:.4f}
    Average %K(KDJ): {avg_30_kdj:.4f}

    Last 3 Days Stock Stats:
    Average Price: {avg_3_price:.2f}
    Max Price: {max_3_price:.2f}
    Min Price: {min_3_price:.2f}
    Volatility: {volatility_3:.2f}
    Average Volume: {avg_3_volume:.4f}
    Average RSI_14: {avg_3_rsi_14:.4f}
    Average CCI_20: {avg_3_cci_20:.4f}
    Average MACD: {avg_3_macd:.4f}
    Average ADX_14: {avg_3_adx_14:.4f}
    Average %K(KDJ): {avg_3_kdj:.4f}

    Yesterday Stock Stats:
    Date: {latest_data['Date']},
    Close: {latest_data['Close']},
    High: {latest_data['High']},
    Low: {latest_data['Low']},
    Volume: {latest_data['Volume']},
    MA20: {latest_data['MA20']},
    MA50: {latest_data['MA50']},
    MA200: {latest_data['MA200']},
    Volatility: {latest_data['Volatility']},
    RSI_14: {latest_data['RSI_14']},
    CCI_20: {latest_data['CCI_20']},
    MACD: {latest_data['MACD']},
    ADX_14: {latest_data['ADX_14']},
    %K(KDJ): {latest_data['%K']}
    """
    
    return stock_data, news_text

# Load a CSV file containing news articles and stock price data.
def load_data(stock_id):
    # Load news CSV files
    google_news = pd.read_csv(f"./output/{stock_id}/google_news_{datetime.now().strftime('%Y%m%d')}.csv")
    google_USA_news = pd.read_csv(f"./output/{stock_id}/google_USA_news_{datetime.now().strftime('%Y%m%d')}.csv")
    yahoo_news = pd.read_csv(f"./output/{stock_id}/yahoo_news_{stock_id}_{datetime.now().strftime('%Y%m%d')}.csv")
    sina_news = pd.read_csv(f"./output/{stock_id}/sina_stock_news_{stock_id}_{datetime.now().strftime('%Y%m%d')}.csv")
    aastocks_news = pd.read_csv(f"./output/{stock_id}/aastocks_news_full_{stock_id}_{datetime.now().strftime('%Y%m%d')}.csv")
    
    # Load and preprocess stock price data
    time_series = pd.read_csv(f"./output/{stock_id}/enhanced_time_series_analysis_{datetime.now().strftime('%Y%m%d')}.csv")
    
    return google_news, google_USA_news, yahoo_news, sina_news, aastocks_news, time_series

# Constructs the prompts used by the DeepSeek API.
def construct_prompt(stock_id, stock_data, news_text):
    prompt = f"""
    Role and Task:
    You are a professional Hong Kong stock market analyst focused on providing objective, evidence-based investment recommendations to clients. Based on the provided stock price data and news articles, conduct a comprehensive analysis of the stock {stock_id} and provide clear short-term (approximately 5 days) and mid-term (approximately 30 days) buy/sell recommendations (e.g., Strong Buy, Buy, Hold, Reduce, Sell).

    Data Sources:
        1. Hong Kong Stock Price Data: {stock_data}
        2. {stock_id} News Articles: {news_text}
    
    Required Analysis Methods (Integrated Approach):
        You must integrate the following three analysis methods, clearly reflecting each in your analysis:
            1. Technical Analysis (Based on Stock Price Data):
                Interpret the current price relative to key moving averages (e.g., 20-day, 50-day, 200-day) to identify support/resistance levels.
                Analyze volume changes (e.g., increasing/decreasing volume) and their alignment with price trends.
                Evaluate signals from key technical indicators:
                    RSI: Is it overbought (>70) or oversold (<30)? What is the trend?
                    MACD: Position of fast/slow lines? Histogram changes (momentum increase/decrease)? Bullish/bearish crossover signals?
                    Other provided indicator signals.
            Identify chart patterns (if data is clear, e.g., head-and-shoulders, triangles, flags).
            Conclusion: Assess the strength of the current technical trend (uptrend, downtrend, consolidation) and identify key support/resistance levels.
            
            2. Fundamental Analysis (Based on Company and Industry Information in News):
                Company Level: Interpret key announcements (e.g., earnings outperformance/underperformance, profit growth/decline, margin changes, major investments/acquisitions/divestitures, management changes, share issuance/buyback plans).
                Industry Level: Analyze industry news (e.g., significant policy tailwinds/headwinds, technological breakthroughs, raw material price fluctuations, competitive landscape changes, leading company developments, industry outlook).
                Macro Level: Evaluate relevant macroeconomic news (e.g., interest rate policies, inflation data, GDP growth, exchange rate movements) and their potential impact on the industry/company.
                Conclusion: Assess the company’s fundamental health (financial stability, growth prospects) and industry outlook (positive/negative/neutral).
            
            3. News Sentiment Analysis (Based on News Articles):
                Analyze the overall sentiment from Google News, Sina, and AAStocks (e.g., very optimistic, optimistic, neutral, pessimistic, very pessimistic).
                Identify key events or themes driving sentiment (e.g., strong earnings = optimistic; regulatory crackdowns = pessimistic).
                Evaluate the potential impact of news on broader investor sentiment (e.g., boosting confidence, triggering panic).
                Conclusion: Synthesize sentiment from the three sources and assess its potential influence on short-term market sentiment.
            
    Comprehensive Evaluation and Recommendations:
        Integrated Analysis: Cross-validate signals from technical, fundamental, and sentiment analyses. Are they aligned (e.g., positive news + technical breakout + improving fundamentals) or conflicting (e.g., negative news but oversold technicals + low valuation)?
        Short-Term Outlook (Approx. 5 Days):
            Based on technical indicators (especially short-term signals like 5-day MA, short-term RSI, MACD histogram), immediate news impacts (e.g., recent earnings release, sudden policy changes), and market sentiment reactions, evaluate price movements and trading opportunities over the next 5 days.
            Provide clear short-term recommendations and key trigger points (e.g., buy on breakout above resistance, stop-loss on breakdown below support).
        Mid-Term Outlook (Approx. 30 Days):
            Based on mid-to-long-term technical trends (e.g., 50-day/200-day MA), sustained fundamental impacts (e.g., industry policy effects, quarterly earnings trends), and evolving macro/industry sentiment, assess potential trends over the next month.
            Provide clear mid-term recommendations with core rationale (e.g., buy and hold based on expected earnings recovery and improving industry conditions).
        Risk Warning: Clearly outline key uncertainties and downside risks (e.g., macroeconomic shocks, industry policy risks, company-specific execution risks, technical signal failure, sentiment reversal risks).
    
    Output Requirements:
        Structure: Organize the report as follows:
            Summary and Core Recommendations: Clearly list short-term and mid-term recommendations (prioritize key information).
            Technical Analysis Key Points: Summarize key technical signals and trend conclusions.
            Fundamental Analysis Key Points: Summarize key company, industry, and macro developments and their impact.
            News Sentiment Analysis Key Points: Summarize sentiment from the three news sources and assess market sentiment impact.
            Detailed Comprehensive Evaluation: Explain how the three analyses support your short-term and mid-term recommendations.
            Key Risk Warnings: Highlight major risks.
            
    Language: Professional, objective, and unambiguous. Use standard investment analysis terminology. **In English.**
    Stance: Provide independent recommendations based solely on the provided data and analysis framework, prioritizing client interests.
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
    google_news, google_USA_news, yahoo_news, sina_news, aastocks_news, time_series = load_data(stock_id)
    # Preprocess
    stock_data, news_text = preprocess_data(google_news, google_USA_news, yahoo_news, sina_news, aastocks_news, time_series)

    # print("Stock Data:\n", stock_data)
    # print("\nNews Summaries:\n", news_text)

    # Bulid prompt
    prompt = construct_prompt(stock_id, stock_data, news_text)
    # Calling the API
    response_text = call_deepseek_api(prompt, api_key, model_type)

    # Parsing Response
    result = parse_response(response_text)
    
    warning_text = "[WARNING] This answer was generated by AI and is for informational purposes only, does not constitute investment advice, so please screen it carefully."
    
    result = result + "\n" + warning_text
    # output
    print(result)

    with open(f"./output/{stock_id}/{stock_id}_Deepseek_Output_{datetime.now().strftime('%Y%m%d')}.txt", "w") as text_file:
        text_file.write(result)
