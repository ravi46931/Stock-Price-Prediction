# Importing libraries
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D
from src.constants import *


# creating model
def create_model():
    model = Sequential()
    model.add(
        Conv1D(
            filters=FILTERS,
            kernel_size=KERNEL_SIZE,
            strides=STRIDES,
            activation=ACTIVATION,
            padding=PADDING,
            input_shape=[TIME_STEPS, 1],
        )
    )
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(30, activation=ACTIVATION))
    model.add(Dense(1))

    return model
