import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from src.constants import *
from src.exception import CustomException


def plot_pred_actual(y, pred, scaler, type, image_name):
    try:
        x = np.arange(len(y))
        plt.figure()
        plt.figure(figsize=(8, 4))
        plt.plot(x, scaler.inverse_transform(y), label="Actual")
        plt.plot(x, pred, label="Predicted")
        plt.legend()
        plt.title(f"Predicted vs Actual price of the stock on {type} set")
        plt.xlabel("Days")
        plt.ylabel("Stock price")

        image_path = os.path.join(STATIC_DIR, IMAGE_DIR)
        os.makedirs(image_path, exist_ok=True)
        pred_actual = os.path.join(image_path, image_name)
        plt.savefig(pred_actual)

    except Exception as e:
        raise CustomException(e, sys)


def custom_plot(y_train, train_pred, scaler, image_name, days=45):
    try:
        x_train = np.arange(len(y_train))
        plt.figure()
        plt.figure(figsize=(8, 4))
        plt.plot(
            x_train[len(x_train) - days :],
            scaler.inverse_transform(y_train[len(y_train) - days :]),
            label="Actual",
        )
        plt.plot(
            x_train[len(x_train) - days :],
            train_pred[len(train_pred) - days :],
            label="Predicted",
        )
        plt.title(f"Stock price of last {days} days of the train set")
        plt.xlabel("Days")
        plt.ylabel("Stock price")
        plt.legend()


        image_path = os.path.join(STATIC_DIR, IMAGE_DIR)
        os.makedirs(image_path, exist_ok=True)
        custom_pred_actual = os.path.join(image_path, image_name)

        plt.savefig(custom_pred_actual)

    except Exception as e:
        raise CustomException(e, sys)
