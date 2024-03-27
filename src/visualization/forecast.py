import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from src.constants import *
from src.exception import CustomException


def plot_forecast(y_train, y_test, scaler, fore_ind, forecast_p, image_name):
    try:
        x_test = np.arange(len(y_test))
        x_train = np.arange(len(y_train))
        fore_ind = np.array(fore_ind)
        forecast_p = np.array(forecast_p)
        plt.figure()
        plt.plot(
            x_train,
            scaler.inverse_transform(y_train),
            label="Actual price of train set",
        )
        plt.plot(
            x_test + len(x_train),
            scaler.inverse_transform(y_test),
            label="Actual price of test set",
        )
        plt.plot(
            fore_ind + len(x_train),
            forecast_p + 4.5,
            label=f"Predict for next {PREDICTION_DAYS} days",
        )
        plt.title("Forecasting of the stock price")
        plt.xlabel("Days")
        plt.ylabel("Stock price")
        plt.legend()

        image_path = os.path.join(STATIC_DIR, IMAGE_DIR)
        os.makedirs(image_path, exist_ok=True)
        forecast_path = os.path.join(image_path, image_name)

        plt.savefig(forecast_path)

    except Exception as e:
        raise CustomException(e, sys)
