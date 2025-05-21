import pandas as pd
import numpy as np

def compute_adx(high, low, close, window=14):
    # Calculate directional motion
    plus_dm = high.diff()
    minus_dm = -low.diff()

    # Filtering effective directional movement
    plus_dm[(plus_dm <= minus_dm) | (plus_dm < 0)] = 0
    minus_dm[(minus_dm <= plus_dm) | (minus_dm < 0)] = 0

    # Calculate the true amplitude
    tr = pd.DataFrame({
        'h-l': high - low,
        'h-pc': abs(high - close.shift()),
        'l-pc': abs(low - close.shift())
    }).max(axis=1)

    # Smoothing process
    alpha = 1/window
    plus_dm_smoothed = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smoothed = minus_dm.ewm(alpha=alpha, adjust=False).mean()
    tr_smoothed = tr.ewm(alpha=alpha, adjust=False).mean()

    # Calculation of the directional index
    plus_di = (plus_dm_smoothed / tr_smoothed) * 100
    minus_di = (minus_dm_smoothed / tr_smoothed) * 100

    # Calculate ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx, plus_di, minus_di


def compute_stochastic_oscillator(high, low, close, k_window=14, d_window=3):
    lowest_low = low.rolling(k_window).min()
    highest_high = high.rolling(k_window).max()

    k = (close - lowest_low) / (highest_high - lowest_low) * 100
    d = k.rolling(d_window).mean()
    j = 3*k - 2*d
    return k, d, j


def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_cci(data, window=20):
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    sma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    return (tp - sma) / (0.015 * mad)


def calculate_technical_indicators(df):
    # Original Price Features
    df['Daily Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily Return'].rolling(window=20).std() * np.sqrt(252)

    # Moving average
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    # Volatility index
    df['Volatility_20D'] = df['Daily Return'].rolling(window=20).std() * np.sqrt(252)
    df['Volatility_60D'] = df['Daily Return'].rolling(window=60).std() * np.sqrt(252)

    # Bollinger bands (using 20-day MA)
    rolling_std = df['Close'].rolling(20).std()
    df['Upper_Bollinger'] = df['MA20'] + 2 * rolling_std
    df['Lower_Bollinger'] = df['MA20'] - 2 * rolling_std

    # MACD（12/26/9）
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI (14-day cycle)
    df['RSI_14'] = compute_rsi(df['Close'])

    # CCI（20-day cycle）
    df['CCI_20'] = compute_cci(df)

    # ADX
    df['ADX_14'], df['Plus_DI'], df['Minus_DI'] = compute_adx(df['High'], df['Low'], df['Close'])

    # Stochastics (KDJ)
    df['%K'], df['%D'], df['%J'] = compute_stochastic_oscillator(df['High'], df['Low'], df['Close'])

    return df.dropna()


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