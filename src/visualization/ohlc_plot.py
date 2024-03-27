import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from src.constants import *
from src.exception import CustomException

def save_ohlc(df):
    try:
        df["Date"] = df["Date"].apply(lambda x: x.split("-")[0])
        df.set_index("Date", inplace=True)
        plt.figure()
        plt.figure(figsize=OHLC_FIGSIZE)

        plt.subplot(2, 2, 1)
        df["Open"].plot(c="r", linewidth=0.6)
        plt.legend(["Open"])

        plt.subplot(2, 2, 2)
        df["High"].plot(c="b", linewidth=0.6)
        plt.legend(["High"])

        plt.subplot(2, 2, 3)
        df["Low"].plot(c="g", linewidth=0.6)
        plt.legend(["Low"])

        plt.subplot(2, 2, 4)
        df["Close"].plot(c="deeppink", linewidth=0.6)
        plt.legend(["Close"])

        plt.suptitle("OHLC Plot")
        plt.tight_layout(pad=2)

        image_path = os.path.join(STATIC_DIR, IMAGE_DIR)
        os.makedirs(image_path, exist_ok=True)
        ohlc_path = os.path.join(image_path, OHLC_IMAGE_NAME)
        plt.savefig(ohlc_path)

    except Exception as e:
        raise CustomException(e, sys)
