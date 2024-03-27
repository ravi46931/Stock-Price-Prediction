import numpy as np
from src.constants import *
from datetime import datetime, timedelta


def time():
    current_date = datetime.now()
    current_time = current_date.strftime("%Y-%m-%d")
    current_time = datetime.strptime(current_time, "%Y-%m-%d")

    ten_years_ago = current_date - timedelta(days=DAYS)
    ten_years_ago = ten_years_ago.strftime("%Y-%m-%d")
    ten_years_ago = datetime.strptime(ten_years_ago, "%Y-%m-%d")

    return current_time, ten_years_ago


def dataset(df, time_steps=30):
    X_tr = []
    y_tr = []
    for i in range(len(df) - time_steps - 1):
        x = df[i : i + time_steps, 0]
        y = df[i + time_steps : i + time_steps + 1, 0]
        X_tr.append(x)
        y_tr.append(y)
    return np.array(X_tr), np.array(y_tr)
