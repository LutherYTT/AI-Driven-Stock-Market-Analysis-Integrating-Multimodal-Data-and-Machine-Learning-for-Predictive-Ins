from gnews import GNews
from datetime import datetime, timedelta
import csv

def get_google_news(stock_id, max_news=15):
    # Initialize GNews with various parameters, including proxy
    google_news = GNews(
        language='zh-tw',
        country='HK',
        period='7d',
        start_date=None,
        end_date=None,
        max_results=max_news
    )

    google_news_result = google_news.get_news_by_topic("BUSINESS")
    # print(google_news_result)

    # Write the data to a CSV file
    csv_filename = f"./output/{stock_id}/google_news_{datetime.now().strftime('%Y%m%d')}.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["title", "description", "published date", "url", "publisher"])
        writer.writeheader()
        writer.writerows(google_news_result)

def get_usa_google_news(stock_id, max_news=15):
    # Initialize GNews with various parameters, including proxy
    google_news = GNews(
        language='zh-tw',
        country='US',
        period='7d',
        start_date=None,
        end_date=None,
        max_results=max_news
    )

    google_usa_news_result = google_news.get_news_by_topic("BUSINESS")
    # print(google_usa_news_result)

    # Write the data to a CSV file
    csv_filename = f"./output/{stock_id}/google_USA_news_{datetime.now().strftime('%Y%m%d')}.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["title", "description", "published date", "url", "publisher"])
        writer.writeheader()
        writer.writerows(google_usa_news_result)

    print(f"CSV file '{csv_filename}' has been saved successfully!")

import csv
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin
import html
from datetime import datetime, timedelta
import os
import re

class YahooFinanceScraper:
    def __init__(self):
        self.base_url = "https://hk.finance.yahoo.com"
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "zh-HK,zh;q=0.9,en-US;q=0.8,en;q=0.7"
        }
        self.today = datetime.now()

    def _retry_request(self, url, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt
                    print(f"Request failed, retry after {sleep_time} seconds... ({url})")
                    time.sleep(sleep_time)
                else:
                    raise e
        return None

    # Convert relative time to absolute date
    def _parse_time(self, raw_time):
        try:
            # Cleaning of raw data
            clean_time = re.sub(r'\s+', ' ', raw_time.strip())

            # Match different time formats
            if 'min' in clean_time:
                mins = int(re.search(r'(\d+)', clean_time).group(1))
                return (self.today - timedelta(minutes=mins)).strftime('%Y-%m-%d %H:%M')
            elif 'hr' in clean_time:
                hours = int(re.search(r'(\d+)', clean_time).group(1))
                return (self.today - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M')
            elif 'yesterday' in clean_time:
                return (self.today - timedelta(days=1)).strftime('%Y-%m-%d')
            elif 'day before yesterday' in clean_time:
                return (self.today - timedelta(days=2)).strftime('%Y-%m-%d')
            elif 'days' in clean_time:
                days = int(re.search(r'(\d+)', clean_time).group(1))
                return (self.today - timedelta(days=days)).strftime('%Y-%m-%d')
            elif 'month' in clean_time and '日' in clean_time:
                # Chinese date format processed: October 20
                return datetime.strptime(clean_time, '%Y年%m月%d日').strftime('%Y-%m-%d')
            else:
                return clean_time  # Keeping the original format
        except:
            return "Time format anomaly"

    def _clean_text(self, text):
        return html.unescape(text).strip()

    def _clean_content(self, soup):
        for element in soup.find_all(class_=['wrapper', 'ad-container', 'advertisement']):
            element.decompose()

        content = []
        for p in soup.find_all('p', class_='yf-1090901'):
            text = self._clean_text(p.get_text())
            if text and not text.startswith('More from MegaMan'):
                content.append(text)
        return '\n\n'.join(content)

    def scrape_news(self, ticker_symbol, max_news=10):
        news_list = []
        news_url = urljoin(self.base_url, f"/quote/{ticker_symbol}/news/")

        try:
            response = self._retry_request(news_url)
            soup = BeautifulSoup(response.text, 'lxml')
            items = soup.find_all('li', class_='stream-item story-item yf-1drgw5l')[:max_news]

            for idx, item in enumerate(items, 1):
                print(f"Processing news item {idx}/{len(items)}...")

                # Extracting metadata
                title_elem = item.find('h3')
                link_elem = item.find('a', href=True)
                time_elem = item.find('div', class_='publishing')

                if not all([title_elem, link_elem, time_elem]):
                    continue

                # Extraction and processing time
                raw_time = ' '.join(time_elem.stripped_strings).split('•')[-1].strip()
                pub_date = self._parse_time(raw_time)

                news_item = {
                    'title': self._clean_text(title_elem.get_text()),
                    'link': urljoin(self.base_url, link_elem['href']),
                    'publish_date': pub_date,
                    'content': ''
                }

                # Get details
                try:
                    detail_response = self._retry_request(news_item['link'])
                    detail_soup = BeautifulSoup(detail_response.text, 'lxml')
                    content_wrapper = detail_soup.find('div', class_='atoms-wrapper')
                    news_item['content'] = self._clean_content(content_wrapper) if content_wrapper else "内容未找到"
                except Exception as e:
                    print(f"Failed to process detail page: {str(e)}")
                    news_item['content'] = "Failed to get content"

                news_list.append(news_item)
                time.sleep(1)

        except Exception as e:
            print(f"Mainstream process anomaly: {str(e)}")

        return news_list

    def save_to_csv(self, data, ticker_symbol):
        if not data:
            print("No data to save")
            return

        # Generate Stock Code Catalog
        stock_id = ticker_symbol.split('.')[0].zfill(5)

        # Creating filenames with timestamps
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"./output/{stock_id}/yahoo_news_{stock_id}_{timestamp}.csv"

        try:
            with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                fieldnames = ['title', 'link', 'publish_date', 'content']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for item in data:
                    writer.writerow({
                        'title': item['title'],
                        'link': item['link'],
                        'publish_date': item['publish_date'],
                        'content': item['content'].replace('\ufeff', '')
                    })
            print(f"Successfully saved {len(data)} data to {filename}.")
        except Exception as e:
            print(f"CSV save failure: {str(e)}")

import os
import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime

def scrape_sina_stock_news(stock_id, save_csv=False):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    url = f"https://stock.finance.sina.com.cn/hkstock/go.php/CompanyNews/page/1/code/{stock_id}/.phtml"

    try:
        # Send HTTP request
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        response.encoding = 'gbk'

        soup = BeautifulSoup(response.text, 'html.parser')
        news_list = []

        ul = soup.find('ul', {'class': 'list01', 'id': 'js_ggzx'})
        if not ul:
            return []

        for li in ul.find_all('li'):
            if not li.text.strip() or li.text.strip() == '\xa0':
                continue

            a_tag = li.find('a')
            title = a_tag.get('title', '').strip() if a_tag else ''

            time_span = li.find('span', {'class': 'rt'})
            time = time_span.text.strip() if time_span else ''

            if title and time:
                news_list.append({
                    'title': title,
                    'time': time
                })

        # Save CSV
        if save_csv and news_list:
            filename = f"./output/{stock_id}/sina_stock_news_{stock_id}_{datetime.now().strftime('%Y%m%d')}.csv"
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=['title', 'time'])
                writer.writeheader()
                writer.writerows(news_list)
            print(f"The data has been saved to {filename}")

        return news_list

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return []
    except Exception as e:
        print(f"Parsing Error: {e}")
        return []


import requests
from bs4 import BeautifulSoup
import csv
import re
from datetime import datetime
import time

def fetch_news_detail(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        response.encoding = 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')

        # Get details
        detail_div = soup.find('div', {'class': 'newscontent5', 'id': 'spanContent'})
        detail_content = ''
        if detail_div:
            # Clean up irrelevant content such as voting
            for div in detail_div.find_all('div', {'id': 'newsVoting'}):
                div.decompose()
            detail_content = detail_div.get_text('\n', strip=True)

        # Get news source
        source_text = ''
        source_pattern = re.compile(r'新聞來源\s*\(不包括新聞圖片\):\s*(.+)')
        source_match = source_pattern.search(response.text)
        if source_match:
            source_text = source_match.group(1).strip()

        return {
            'detail_content': detail_content,
            'source': source_text
        }

    except Exception as e:
        print(f"Failed to capture detail page: {url} - {str(e)}")
        return {'detail_content': '', 'source': ''}

def scrape_aastocks_news(stock_id, save_csv=False, delay=1):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7'
    }

    url = f"https://www.aastocks.com/tc/stocks/analysis/stock-aafn/{stock_id}/0/hk-stock-news/1"

    try:
        # Get List Page
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        news_list = []

        content_div = soup.find('div', {'id': 'aafn-search-c1'})
        if not content_div:
            return []

        # Get news items
        for item in content_div.find_all('div', recursive=False):
            if 'videoNewsList' in item.get('class', []):
                continue

            # Basic Information
            title_tag = item.find('div', class_='newshead4')
            title = title_tag.get_text(strip=True) if title_tag else ''

            # Get detail page link
            link_tag = title_tag.find('a') if title_tag else None
            detail_url = link_tag['href'] if link_tag else ''
            if detail_url and not detail_url.startswith('http'):
                detail_url = f'https://www.aastocks.com{detail_url}'

            # time handling
            time_tag = item.find('div', class_='newstime4')
            time_text = ''
            if time_tag:
                script_tag = time_tag.find('script')
                if script_tag:
                    match = re.search(r"dt:'(.*?)'", script_tag.string)
                    if match:
                        raw_time = match.group(1)
                        time_text = datetime.strptime(raw_time, '%Y/%m/%d %H:%M').strftime('%Y-%m-%d %H:%M:%S')
                else:
                    time_text = time_tag.get_text(strip=True).split('CST')[0].strip()

            if title and time_text:
                news_item = {
                    'title': title,
                    'time': time_text,
                    'detail_url': detail_url,
                    'detail_content': '',
                    'source': ''
                }

                # Get details
                if detail_url:
                    time.sleep(delay)  # Add request interval
                    detail_data = fetch_news_detail(detail_url)
                    news_item.update(detail_data)

                news_list.append(news_item)

        # Save CSV
        if save_csv and news_list:
            filename = f"./output/{stock_id}/aastocks_news_full_{stock_id}_{datetime.now().strftime('%Y%m%d')}.csv"
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                fieldnames = ['time', 'title', 'source', 'detail_url', 'detail_content']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(news_list)
            print(f"The data has been saved to {filename}")

        return news_list

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return []
    except Exception as e:
        print(f"Parsing Error: {e}")
        return []
