import sys
import pickle
import pandas as pd
import numpy as np
from joblib import load

from src.constants import *
from src.logger import logging
from src.exception import CustomException
from src.visualization.ohlc_plot import save_ohlc
from src.visualization.mae_loss import plot_mae_loss
from src.visualization.pred_act import plot_pred_actual, custom_plot
from src.entity.artifacts_entity import (
    DataIngestionArtifacts,
    DataPreprocessingArtifacts,
    ModelEvaluationArtifacts,
    ModelTrainerArtifacts,
)


class VisualPipeline:
    def __init__(
        self,
        data_ingestion_artifacts: DataIngestionArtifacts,
        data_preprocessing_artifacts: DataPreprocessingArtifacts,
        model_trainer_artifacts: ModelTrainerArtifacts,
        model_evaluation_artifacts: ModelEvaluationArtifacts,
    ):

        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_preprocessing_artifacts = data_preprocessing_artifacts
        self.model_trainer_artifacts = model_trainer_artifacts
        self.model_evaluation_artifacts = model_evaluation_artifacts

    def start_ohlc_plot(self):
        try:
            logging.info("Entered the start_ohlc_plot method of TrainPipeline class")
            df = pd.read_csv(self.data_ingestion_artifacts.data_file_path)
            # Function calling for plot
            save_ohlc(df)
            logging.info("Exited the start_ohlc_plot method of TrainPipeline class")

        except Exception as e:
            raise CustomException(e, sys)

    def start_mae_loss_plot(self):
        try:
            logging.info(
                "Entered the start_mae_loss_plot method of TrainPipeline class"
            )
            model_history_path = self.model_trainer_artifacts.model_history_file_path
            # Load the saved history
            with open(model_history_path, "rb") as file:
                model_history = pickle.load(file)
            # Function calling for plot
            plot_mae_loss(model_history)
            logging.info("Exited the start_mae_loss_plot method of TrainPipeline class")

        except Exception as e:
            raise CustomException(e, sys)

    def start_pred_actual_plot(self):
        try:
            logging.info(
                "Entered the start_pred_actual_plot method of TrainPipeline class"
            )
            # Training prediction plot
            y_train = np.load(self.data_preprocessing_artifacts.y_train_file_path)
            train_pred = np.load(self.model_evaluation_artifacts.train_pred_file_path)
            scaler = load(self.data_preprocessing_artifacts.scaler_file_path)
            plot_pred_actual(
                y_train, train_pred, scaler, type="training", image_name=PRED_ACT_TRAIN
            )

            # Training prediction plot
            y_test = np.load(self.data_preprocessing_artifacts.y_test_file_path)
            test_pred = np.load(self.model_evaluation_artifacts.test_pred_file_path)
            scaler = load(self.data_preprocessing_artifacts.scaler_file_path)
            plot_pred_actual(
                y_test, test_pred, scaler, type="test", image_name=PRED_ACT_TEST
            )

            logging.info(
                "Exited the start_pred_actual_plot method of TrainPipeline class"
            )

        except Exception as e:
            raise CustomException(e, sys)

    def start_custom_plot(self):
        try:
            logging.info("Entered the start_custom_plot method of TrainPipeline class")
            y_train = np.load(self.data_preprocessing_artifacts.y_train_file_path)
            train_pred = np.load(self.model_evaluation_artifacts.train_pred_file_path)
            scaler = load(self.data_preprocessing_artifacts.scaler_file_path)
            custom_plot(
                y_train, train_pred, scaler, image_name=CUSTOM_PLOT, days=RECENT_DAYS
            )

            logging.info("Exited the start_custom_plot method of TrainPipeline class")

        except Exception as e:
            raise CustomException(e, sys)

    def run_visual_pipeline(self):
        try:
            logging.info("Start running visual pipeline...")
            self.start_ohlc_plot()
            self.start_mae_loss_plot()
            self.start_pred_actual_plot()
            self.start_custom_plot()
            logging.info("VIsual pipeline run successfully...")
        except Exception as e:
            raise CustomException(e, sys)
