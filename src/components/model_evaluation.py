import os
import sys
import json
import numpy as np
from joblib import load
from sklearn.metrics import mean_squared_error

from src.constants import *
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifacts_entity import (
    DataPreprocessingArtifacts,
    ModelTrainerArtifacts,
    ModelEvaluationArtifacts,
)

from tensorflow.keras.models import load_model  # type: ignore


class ModelEvaluation:

    def __init__(
        self,
        data_preprocessing_artifacts: DataPreprocessingArtifacts,
        model_trainer_artifacts: ModelTrainerArtifacts,
        model_evaluation_config: ModelEvaluationConfig,
    ):
        self.data_preprocessing_artifacts = data_preprocessing_artifacts
        self.model_trainer_artifacts = model_trainer_artifacts
        self.model_evaluation_config = model_evaluation_config

    def model_evaluation(self):
        try:
            # Load model
            logging.info("Loading model and other data...")

            model = load_model(self.model_trainer_artifacts.model_file_path)

            X_train = np.load(self.data_preprocessing_artifacts.x_train_file_path)
            y_train = np.load(self.data_preprocessing_artifacts.y_train_file_path)
            X_test = np.load(self.data_preprocessing_artifacts.x_test_file_path)
            y_test = np.load(self.data_preprocessing_artifacts.y_test_file_path)

            # Load the scaler object from the file
            scaler = load(self.data_preprocessing_artifacts.scaler_file_path)

            # Model prediction
            logging.info("Predicting on the train and test set ")
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)

            ##Transform back to the original form
            train_pred = scaler.inverse_transform(train_predict)
            test_pred = scaler.inverse_transform(test_predict)

            rmse_train = round(np.sqrt(mean_squared_error(y_train, train_pred)), 2)
            print("Root mean squared error of training set: {}".format(rmse_train))

            rmse_test = round(np.sqrt(mean_squared_error(y_test, test_pred)), 2)
            print("Root mean squared error of test set: {}".format(rmse_test))

            metrics = {
                "RMSE TEST DATA SET": rmse_test,
                "RMSE TRAIN DATA SET": rmse_train,
            }

            return metrics, train_pred, test_pred

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self):
        try:
            logging.info("Initiating model evaluation...")
            metrics, train_pred, test_pred = self.model_evaluation()

            os.makedirs(
                self.model_evaluation_config.MODEL_EVALUATION_ARTIFACTS_DIR,
                exist_ok=True,
            )
            # Convert dictionary to JSON string
            json_string = json.dumps(metrics)

            # Write JSON string to a file
            with open(self.model_evaluation_config.METRICS_FILE_PATH, "w") as json_file:
                json_file.write(json_string)

            np.save(self.model_evaluation_config.TRAIN_PRED_FILE_PATH, train_pred)
            np.save(self.model_evaluation_config.TEST_PRED_FILE_PATH, test_pred)

            model_evaluation_artifacts = ModelEvaluationArtifacts(
                self.model_evaluation_config.METRICS_FILE_PATH,
                self.model_evaluation_config.TRAIN_PRED_FILE_PATH,
                self.model_evaluation_config.TEST_PRED_FILE_PATH,
            )
            logging.info("Model evaluation completed...")

            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e, sys)
