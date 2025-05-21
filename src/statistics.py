import pandas as pd
import numpy as np

# Calculation of key statistical indicators
def calculate_statistics(df):
    stats = {
        'Mean Daily Return': df['Daily Return'].mean(),
        'Standard Deviation': df['Daily Return'].std(),
        'Annualized Volatility': df['Daily Return'].std() * np.sqrt(252),
        'Sharpe Ratio': (df['Daily Return'].mean() / df['Daily Return'].std()) * np.sqrt(252),
        'Max Drawdown': (df['Close'] / df['Close'].cummax() - 1).min(),
        'Average Volume': df['Volume'].mean(),
        'ADX Average': df['ADX_14'].mean(),
        'Overbought Ratio': (df['%K'] > 80).mean(),
        'Oversold Ratio': (df['%K'] < 20).mean()
    }

    return pd.Series(stats)