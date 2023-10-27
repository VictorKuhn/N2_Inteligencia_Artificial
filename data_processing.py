import pandas as pd


def load_data(filename):
    data = pd.read_csv(filename)
    return data


def calculate_daily_returns(data):
    data['Daily_Return'] = data['Close'].pct_change()
    return data.dropna()

