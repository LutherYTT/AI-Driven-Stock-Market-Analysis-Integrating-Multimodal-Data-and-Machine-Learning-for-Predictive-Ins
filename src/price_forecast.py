import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import backtrader as bt
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import IsolationForest
import os
from datetime import datetime

# Machine Learning Price Forecast
def lstm_price_prediction(df, stock_id):
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_dataset(data, time_step=60):
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step), 0]
            X.append(a)
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Save model and scaler
    model.save(f'./output/{stock_id}/lstm_model_{stock_id}.h5')  
    joblib.dump(scaler, f'./output/{stock_id}/{stock_id}_scaler.pkl') 

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Actual Price')
    plt.plot(df.index[time_step:len(train_predict) + time_step], train_predict, label='Train Predict')
    plt.plot(df.index[len(train_predict) + (time_step * 2) + 1:len(df) - 1], test_predict, label='Test Predict')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.savefig(f"./output/{stock_id}/price_prediction_{datetime.now().strftime('%Y%m%d')}.png")
    plt.show()

def predict_future(stock_id, last_60days_data, predict_days=30):
    # last_60days_data: Data containing the closing price for at least the last 60 daysFrame
    # predict_days: Number of days to be predicted
    # Load model and scaler
    model = load_model(f'./output/{stock_id}/lstm_model_{stock_id}.h5')
    scaler = joblib.load(f'./output/{stock_id}/{stock_id}_scaler.pkl')
    
    # Data Preprocessing
    scaled_data = scaler.transform(last_60days_data['Close'].values.reshape(-1, 1))
    
    # Creation of a predictive sequence
    predictions = []
    current_batch = scaled_data[-60:].reshape(1, 60, 1)  # Take the last 60 days of data
    
    for _ in range(predict_days):
        current_pred = model.predict(current_batch)
        predictions.append(current_pred[0])
        # Update batch data
        current_batch = np.append(current_batch[:, 1:, :], [current_pred], axis=1)
    
    # Denormalization
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Generate date index
    last_date = last_60days_data.index[-1]
    date_list = [last_date + pd.Timedelta(days=x) for x in range(1, predict_days+1)]
    
    # Preparation of forecast maps
    plt.figure(figsize=(12,6))
    plt.plot(date_list, predictions, marker='o', linestyle='--', color='r')
    plt.title(f'{predict_days} Days Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price')
    plt.grid(True)
    plt.savefig(f"./output/{stock_id}/future_prediction_{datetime.now().strftime('%Y%m%d')}.png")
    plt.show()
    
    return pd.Series(predictions.flatten(), index=date_list)


# Multi-feature LSTM training function
def lstm_price_prediction_multifeature(
    df: pd.DataFrame,
    stock_id,
    feature_cols: list,
    target_col: str = 'Close',
    time_step: int = 60,
    epochs: int = 5,
    batch_size: int = 32,
    model_dir: str = None
):

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df[feature_cols])

    train_size = int(len(data_scaled) * 0.8)
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size - time_step:]

    def create_dataset(data, time_step):
        X, Y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), :])
            Y.append(data[i + time_step, feature_cols.index(target_col)])
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    n_features = len(feature_cols)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_step, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        model.save(f"./output/{model_dir}/lstm_model_multifeature.h5")
        joblib.dump(scaler, f"./output/{model_dir}/scaler_multifeature.pkl")

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    def inv_transform(preds):
        tmp = np.zeros((len(preds), n_features))
        tmp[:, feature_cols.index(target_col)] = preds.flatten()
        return scaler.inverse_transform(tmp)[:, feature_cols.index(target_col)]

    train_pred_inv = inv_transform(train_pred)
    test_pred_inv = inv_transform(test_pred)
    actual = df[target_col].values

    # Visualization
    plt.figure(figsize=(12, 6))
    train_len = len(train_pred_inv)
    test_len = len(test_pred_inv)
    train_index = df.index[time_step:time_step + train_len]
    test_index = df.index[time_step + train_len:time_step + train_len + test_len]

    plt.plot(train_index, actual[time_step:time_step + train_len], label='Train Actual')
    plt.plot(train_index, train_pred_inv, label='Train Predict')
    plt.plot(test_index, actual[time_step + train_len:time_step + train_len + test_len], label='Test Actual')
    plt.plot(test_index, test_pred_inv, label='Test Predict')
    plt.title('Multivariate LSTM Stock Prediction')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.legend()
    plt.show()

    return model, scaler

# Load a multi-feature model and make future price predictions and plot the predictions
def predict_future_multifeature(
    model_path: str,
    scaler_path: str,
    last_data: pd.DataFrame,
    feature_cols: list,
    target_col: str = 'Close',
    predict_days: int = 30,
    time_step: int = 60
) -> pd.Series:

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    data_scaled = scaler.transform(last_data[feature_cols])
    batch = data_scaled[-time_step:].reshape(1, time_step, len(feature_cols))
    preds = []

    for _ in range(predict_days):
        pred = model.predict(batch)[0, 0]
        preds.append(pred)
        last_step = batch[0, -1, :].copy()
        last_step[feature_cols.index(target_col)] = pred
        batch = np.concatenate([batch[:, 1:, :], last_step.reshape(1, 1, -1)], axis=1)

    tmp = np.zeros((len(preds), len(feature_cols)))
    tmp[:, feature_cols.index(target_col)] = preds
    inv_preds = scaler.inverse_transform(tmp)[:, feature_cols.index(target_col)]

    # Generate date index
    last_date = last_data.index[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, predict_days + 1)]
    series_preds = pd.Series(inv_preds, index=future_dates)

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, inv_preds, marker='o', linestyle='--')
    plt.title(f'Future {predict_days}-Day Price Prediction')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.grid(True)
    plt.show()

    return series_preds