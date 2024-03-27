import os
import sys
import matplotlib.pyplot as plt

from src.constants import *
from src.exception import CustomException


def plot_mae_loss(model_history):
    try:
        # Get mae and loss from history log
        mae = model_history["mae"]
        loss = model_history["loss"]
        # Get number of epochs
        epochs_val = range(len(loss))

        plt.figure()
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_val, mae, c="green")
        plt.xlabel("Epoch")
        plt.ylabel("mae")
        plt.subplot(1, 2, 2)
        plt.plot(epochs_val, loss, c="r")
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.suptitle("Epoch vs mae and training loss curve")
        plt.tight_layout(pad=2, w_pad=3)

        image_path = os.path.join(STATIC_DIR, IMAGE_DIR)
        os.makedirs(image_path, exist_ok=True)
        mae_loss = os.path.join(image_path, MAE_LOSS)
        plt.savefig(mae_loss)

        # Plotting the last 80% of the epochs i.e. from epoch 20 to 100 if there are 100 epochs
        split = int(epochs_val[-1] * 0.2)
        plt.figure()
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_val[split:], mae[split:], c="green")
        plt.xlabel("Epoch")
        plt.ylabel("mae")
        plt.subplot(1, 2, 2)
        plt.plot(epochs_val[split:], loss[split:], c="r")
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.suptitle("Epoch vs mae and training loss curve (Zoom)")
        plt.tight_layout(pad=2, w_pad=3)
        mae_loss_zoom = os.path.join(image_path, MAE_LOSS_ZOOM)
        plt.savefig(mae_loss_zoom)

    except Exception as e:
        raise CustomException(e, sys)
