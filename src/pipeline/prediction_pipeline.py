import sys
import numpy as np
from joblib import load

from src.constants import *
from src.logger import logging
from src.exception import CustomException
from tensorflow.keras.models import load_model  # type: ignore
from src.visualization.forecast import plot_forecast
from src.entity.artifacts_entity import (
    DataPreprocessingArtifacts,
    ModelTrainerArtifacts,
)


class PredictionPipeline:
    def __init__(
        self,
        data_preprocessing_artifacts: DataPreprocessingArtifacts,
        model_trainer_artifacts: ModelTrainerArtifacts,
    ):
        self.data_preprocessing_artifacts = data_preprocessing_artifacts
        self.model_trainer_artifacts = model_trainer_artifacts
        self.forecast = []

    def prediction(self):
        try:
            logging.info("Forecasting started...")
            X_test = np.load(self.data_preprocessing_artifacts.x_test_file_path)
            val = X_test[len(X_test) - 1]
            pred_set = val.reshape(1, TIME_STEPS, 1)
            model = load_model(self.model_trainer_artifacts.model_file_path)
            for _ in range(PREDICTION_DAYS):
                pred_val = model.predict(pred_set)
                pred_set = pred_set[0, 1:]
                pred_set = pred_set.tolist()
                pred_val = pred_val.tolist()[0]
                self.forecast.append(pred_val)
                pred_set.append(pred_val)
                pred_set = np.array(pred_set)
                pred_set = pred_set.reshape(1, TIME_STEPS, 1)

            logging.info("Forecasting completed...")
        except Exception as e:
            raise CustomException(e, sys)

    def start_forecast_plot(self):
        try:
            logging.info("Start plotting the forecast...")
            scaler = load(self.data_preprocessing_artifacts.scaler_file_path)
            forecast_p = scaler.inverse_transform(self.forecast)
            y_test = np.load(self.data_preprocessing_artifacts.y_test_file_path)
            x_test = np.arange(len(y_test))
            ind = x_test[len(x_test) - 1]
            fore_ind = [i + ind for i in range(PREDICTION_DAYS)]
            y_train = np.load(self.data_preprocessing_artifacts.y_train_file_path)

            plot_forecast(
                y_train, y_test, scaler, fore_ind, forecast_p, image_name=FORECAST_PLOT
            )
            logging.info("Forecast plotting completed...")

        except Exception as e:
            raise CustomException(e, sys)

    def run_prediction_pipeline(self):
        try:
            # Start prediction
            logging.info("Start running prediction pipeline...")

            self.prediction()
            self.start_forecast_plot()

            logging.info("Prediction pipeline run successfully...")

        except Exception as e:
            raise CustomException(e, sys)
