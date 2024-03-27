import os
import sys
import pickle
import numpy as np
import tensorflow as tf

from src.constants import *
from src.logger import logging
from src.ml.model import create_model
from src.exception import CustomException
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifacts_entity import (
    DataPreprocessingArtifacts,
    ModelTrainerArtifacts,
)


class ModelTrainer:
    def __init__(
        self,
        data_preprocessing_artifacts: DataPreprocessingArtifacts,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_preprocessing_artifacts = data_preprocessing_artifacts
        self.model_trainer_config = model_trainer_config

    def training(self):
        try:
            model = create_model()
            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

            # Set the training parameters
            model.compile(
                optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=[METRICS]
            )

            X_train = np.load(self.data_preprocessing_artifacts.x_train_file_path)
            y_train = np.load(self.data_preprocessing_artifacts.y_train_file_path)
            # Train the model
            model_history = model.fit(
                X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE
            )

            return model_history, model

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self):
        try:
            logging.info("Initiating model training..")
            model_history, model = self.training()
            os.makedirs(
                self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True
            )

            logging.info("Saving model...")
            model.save(self.model_trainer_config.MODEL_FILE_PATH)

            logging.info("Saving model history...")
            with open(self.model_trainer_config.MODEL_HISTORY_FILE_PATH, "wb") as file:
                pickle.dump(model_history.history, file)

            model_trainer_artifacts = ModelTrainerArtifacts(
                self.model_trainer_config.MODEL_FILE_PATH,
                self.model_trainer_config.MODEL_HISTORY_FILE_PATH,
            )
            logging.info("Model training completed..")

            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys)
