import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def create_interactive_chart(df, stock_id):
    fig = make_subplots(
                    rows=8, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.3, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
                    specs=[[{"secondary_y": True}]]*8,
                    subplot_titles=(
                            "Price and Moving Averages",
                            "Volume",
                            "MACD",
                            "Volatility",
                            "RSI (14)",
                            "CCI (20)",
                            "ADX",
                            "KDJ"))


    # PriceTrend and Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20 Day MA',
                           line=dict(color='orange', width=1)),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50 Day MA',
                           line=dict(color='green', width=1)),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], name='200 Day MA',
                           line=dict(color='red', width=1)),
                 row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='green'),
                 row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                           line=dict(color='blue', width=1)),
                 row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal Line'], name='Signal Line',
                           line=dict(color='red', width=1)),
                 row=3, col=1)

    # Volatilities
    fig.add_trace(go.Scatter(x=df.index, y=df['Volatility'], name='Volatility',
                           line=dict(color='purple', width=1)),
                 row=4, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name='RSI 14',
                     line=dict(color='cyan', width=1)), row=5, col=1)
    fig.add_hline(y=30, row=5, col=1, line=dict(color='gray', dash='dot'))
    fig.add_hline(y=70, row=5, col=1, line=dict(color='gray', dash='dot'))

    # CCI
    fig.add_trace(go.Scatter(x=df.index, y=df['CCI_20'], name='CCI 20',
                     line=dict(color='magenta', width=1)), row=6, col=1)
    fig.add_hline(y=100, row=6, col=1, line=dict(color='gray', dash='dot'))
    fig.add_hline(y=-100, row=6, col=1, line=dict(color='gray', dash='dot'))

    # ADX
    fig.add_trace(go.Scatter(x=df.index, y=df['ADX_14'], name="ADX"), row=7, col=1)
    fig.add_hline(y=25, line_dash="dot", line_color="gray", annotation_text="弱趋势", row=7, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="强趋势", row=7, col=1)

    # Stochastics (KDJ)
    fig.add_trace(go.Scatter(x=df.index, y=df['%K'], name="%K"), row=8, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['%D'], name="%D"), row=8, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="gray", row=8, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="gray", row=8, col=1)

    fig.update_layout(height=2000) 

    fig.write_html(f"./output/{stock_id}/interactive_chart.html") # Saves the chart as an interactive HTML file

    fig.show()


def plot_return_distribution(df, stock_id):
    plt.figure(figsize=(12, 6))
    plt.hist(df['Daily Return'].dropna(), bins=100, alpha=0.7,
            color='blue', density=True)
    plt.title('Daily Returns Distribution')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f"./output/{stock_id}/return_distribution_{datetime.now().strftime('%Y%m%d')}.png")
    plt.show()