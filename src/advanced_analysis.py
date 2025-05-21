import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import backtrader as bt
from sklearn.ensemble import IsolationForest
import os
from datetime import datetime

matplotlib.use('Agg')
plt.rcParams['figure.figsize'] = [15, 12]
plt.rcParams.update({'font.size': 12})

# Volatility cluster analysis
def volatility_clustering(df, stock_id):
    volatility_data = df['Volatility_20D'].dropna().values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(volatility_data)
    df['Volatility_Cluster'] = np.nan
    df.loc[df['Volatility_20D'].dropna().index, 'Volatility_Cluster'] = kmeans.labels_

    # Visualization
    plt.figure(figsize=(12, 6))
    for cluster in range(3):
        cluster_data = df[df['Volatility_Cluster'] == cluster]
        plt.scatter(cluster_data.index, cluster_data['Close'], label=f'Cluster {cluster}')
    plt.title('Volatility Clustering')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.savefig(f"./output/{stock_id}/volatility_clustering_{datetime.now().strftime('%Y%m%d')}.png")
    plt.close()

# Backtesting 
class MA_CrossStrategy(bt.Strategy):
    def __init__(self):
        self.ma20 = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
        self.ma50 = bt.indicators.SimpleMovingAverage(self.data.close, period=50)

    def next(self):
        if self.ma20 > self.ma50 and self.ma20[-1] <= self.ma50[-1]:
            self.buy()
        elif self.ma20 < self.ma50 and self.ma20[-1] >= self.ma50[-1]:
            self.sell()

def run_backtest(df, stock_id):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(MA_CrossStrategy)
    cerebro.broker.setcash(100000.0)
    cerebro.run()

    fig = cerebro.plot(
        style='candlestick',
        figsize=(12, 6),
        iplot=False,
        volume=True
    )

    # Save Backtest Chart
    img_name = f"./output/{stock_id}/backtest_result_{datetime.now().strftime('%Y%m%d')}.png"
    fig[0][0].savefig(img_name)

    # Display the saved image
    # from IPython.display import Image
    # Image(filename=img_name)


# Anomaly Detection
def anomaly_detection(df, stock_id):
    features = ['Close', 'Volume']
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    anomalies = iso_forest.fit_predict(df[features])
    df['Anomaly'] = anomalies
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.scatter(df[df['Anomaly'] == -1].index, df[df['Anomaly'] == -1]['Close'], color='red', label='Anomaly')
    plt.title('Anomaly Detection')
    plt.legend()
    plt.savefig(f"./output/{stock_id}/anomaly_detection_{datetime.now().strftime('%Y%m%d')}.png")
    plt.close()
